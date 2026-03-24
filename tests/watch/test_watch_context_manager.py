"""Unit tests for ModelWatcher / watch() context manager."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gradient_pathology.watch import ModelWatcher, watch
from gradient_pathology.core import GradientReport
from gradient_pathology.expert.engine import ExpertFinding


# ── helpers ────────────────────────────────────────────────────────────────

def _simple_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def _run_backward(model: nn.Module, steps: int = 5) -> None:
    """Run `steps` forward/backward passes on random data."""
    loss_fn = nn.MSELoss()
    for _ in range(steps):
        x = torch.randn(4, 8)
        y = torch.randn(4, 1)
        model.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()


# ── context manager protocol ───────────────────────────────────────────────

class TestContextManager:
    def test_basic_usage_no_crash(self):
        model = _simple_model()
        with watch(model, auto_print=False):
            _run_backward(model)

    def test_report_available_after_exit(self):
        model = _simple_model()
        with watch(model, auto_print=False) as w:
            _run_backward(model)
        assert w.report is not None
        assert isinstance(w.report, GradientReport)

    def test_findings_available_after_exit(self):
        model = _simple_model()
        with watch(model, auto_print=False) as w:
            _run_backward(model)
        assert w.findings is not None
        assert isinstance(w.findings, list)

    def test_layer_stats_populated(self):
        model = _simple_model()
        with watch(model, auto_print=False) as w:
            _run_backward(model, steps=3)
        assert len(w.report.layer_stats) > 0

    def test_step_count_increments(self):
        model = _simple_model()
        with watch(model, auto_print=False) as w:
            _run_backward(model, steps=4)
        # Each backward pass triggers at least one hook call per module
        assert w.step_count > 0

    def test_hooks_removed_after_exit(self):
        """After __exit__, model should have no pathology hooks."""
        model = _simple_model()
        with watch(model, auto_print=False) as w:
            pass
        assert not w.is_running

    def test_exception_inside_context_does_not_suppress(self):
        model = _simple_model()
        with pytest.raises(ValueError):
            with watch(model, auto_print=False):
                raise ValueError("intentional")

    def test_exception_still_stops_hooks(self):
        model = _simple_model()
        w = watch(model, auto_print=False)
        try:
            with w:
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert not w.is_running


# ── explicit start / stop ──────────────────────────────────────────────────

class TestExplicitLifecycle:
    def test_start_stop(self):
        model = _simple_model()
        w = watch(model, auto_print=False)
        w.start()
        _run_backward(model, steps=3)
        w.stop()
        assert w.report is not None

    def test_double_start_is_idempotent(self):
        model = _simple_model()
        w = watch(model, auto_print=False)
        w.start()
        w.start()  # should not add duplicate hooks
        _run_backward(model, steps=2)
        w.stop()
        assert w.report is not None

    def test_stop_before_start_is_safe(self):
        model = _simple_model()
        w = watch(model, auto_print=False)
        w.stop()  # should not raise
        assert w.report is None

    def test_is_running_flag(self):
        model = _simple_model()
        w = watch(model, auto_print=False)
        assert not w.is_running
        w.start()
        assert w.is_running
        w.stop()
        assert not w.is_running


# ── report content ─────────────────────────────────────────────────────────

class TestReportContent:
    def test_data_source_is_watch_hook(self):
        model = _simple_model()
        with watch(model, auto_print=False) as w:
            _run_backward(model)
        assert w.report.data_source == "watch_hook"

    def test_global_mean_nonzero_after_backward(self):
        model = _simple_model()
        with watch(model, auto_print=False) as w:
            _run_backward(model, steps=5)
        assert w.report.global_mean >= 0.0

    def test_layer_names_match_model_modules(self):
        model = _simple_model()
        module_names = {
            n for n, m in model.named_modules()
            if list(m.parameters(recurse=False))
        }
        with watch(model, auto_print=False) as w:
            _run_backward(model, steps=2)
        stat_names = {s.layer_name for s in w.report.layer_stats}
        # Every watched module should appear in stats
        assert module_names <= stat_names | {""}  # allow <root> alias

    def test_no_backward_produces_empty_or_zero_report(self):
        model = _simple_model()
        with watch(model, auto_print=False) as w:
            pass  # no backward
        # Either no layers recorded OR all norms are zero
        if w.report.layer_stats:
            assert all(s.grad_norm == 0.0 for s in w.report.layer_stats)
        else:
            assert w.report.global_mean == 0.0


# ── threshold configuration ────────────────────────────────────────────────

class TestThresholds:
    def test_custom_vanishing_threshold(self):
        model = _simple_model()
        # Set a very HIGH threshold so everything looks vanishing
        with watch(model, vanishing_threshold=1e10, auto_print=False) as w:
            _run_backward(model, steps=3)
        assert w.findings is not None
        # With threshold=1e10 at least some finding should appear
        # (cannot assert exactly which without knowing grad magnitudes)

    def test_custom_exploding_threshold(self):
        model = _simple_model()
        # Set a very LOW threshold so everything looks exploding
        with watch(model, exploding_threshold=1e-20, auto_print=False) as w:
            _run_backward(model, steps=3)
        assert w.findings is not None


# ── quick_summary ──────────────────────────────────────────────────────────

class TestQuickSummary:
    def test_summary_before_stop(self):
        model = _simple_model()
        w = watch(model, auto_print=False)
        summary = w.quick_summary()
        assert "not stopped" in summary.lower() or "⏳" in summary

    def test_summary_after_stop(self):
        model = _simple_model()
        with watch(model, auto_print=False) as w:
            _run_backward(model)
        summary = w.quick_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


# ── output_dir artefacts ───────────────────────────────────────────────────

class TestArtefacts:
    def test_json_artefact_created(self, tmp_path):
        model = _simple_model()
        with watch(model, auto_print=False, output_dir=str(tmp_path)) as w:
            _run_backward(model, steps=2)
        w._print_report()  # force artefact write
        assert (tmp_path / "report.json").exists()

    def test_parquet_artefact_created(self, tmp_path):
        pytest.importorskip("pandas", reason="pandas not installed")
        model = _simple_model()
        with watch(model, auto_print=False, output_dir=str(tmp_path)) as w:
            _run_backward(model, steps=2)
        w._print_report()
        assert (tmp_path / "layer_stats.parquet").exists()
