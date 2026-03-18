"""Tests for Phase-4: LiveGradientBridge, StreamlitCallback, ExpertEngine.

Streamlit-dependent rendering tests are skipped when Streamlit is absent.
Plotly-dependent tests are skipped when Plotly is absent.
"""

from __future__ import annotations

import time
import threading
from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn

from gradient_pathology.core import (
    GradientPathology,
    GradientReport,
    LayerGradientStats,
    LayerGroup,
)
from gradient_pathology.monitor.bridge import (
    LiveGradientBridge,
    get_global_bridge,
    reset_global_bridge,
)
from gradient_pathology.monitor.callback import StreamlitCallback
from gradient_pathology.expert.engine import (
    ExpertEngine,
    ExpertFinding,
    _safe_norm,
)

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

requires_plotly = pytest.mark.skipif(
    not _PLOTLY_AVAILABLE, reason="plotly not installed"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_stats(n: int = 6, vanishing: bool = False, exploding: bool = False) -> List[LayerGradientStats]:
    groups = [
        LayerGroup.EMBEDDING, LayerGroup.ATTENTION, LayerGroup.LAYER_NORM,
        LayerGroup.FFN,       LayerGroup.LAYER_NORM, LayerGroup.HEAD,
    ]
    stats = []
    for i in range(n):
        if vanishing and i == 0:
            mean, gn = 1e-9, 1e-9
        elif exploding and i == n - 1:
            mean, gn = 2e3, 2e4
        else:
            mean, gn = 1e-3 * (i + 1), 1e-2 * (i + 1)
        s = LayerGradientStats(
            layer_name=f"model.layer_{i}.weight",
            layer_index=i, mean=mean,
            std=abs(mean) * 0.1, min=-abs(mean), max=abs(mean),
            median=mean, num_zeros=0, total_params=64,
            layer_type="Linear", depth=i,
            group=groups[i % len(groups)], grad_norm=gn,
        )
        stats.append(s)
    return stats


def _make_report(n=6, vanishing=False, exploding=False) -> GradientReport:
    stats = _make_stats(n=n, vanishing=vanishing, exploding=exploding)
    return GradientReport(
        layer_stats=stats,
        global_mean=float(np.mean([abs(s.mean) for s in stats])),
        global_std=float(np.std([abs(s.mean) for s in stats])),
        num_steps=10,
        data_source="synthetic",
    )


# ---------------------------------------------------------------------------
# LiveGradientBridge
# ---------------------------------------------------------------------------

class TestLiveGradientBridge:
    def setup_method(self):
        reset_global_bridge()

    def teardown_method(self):
        reset_global_bridge()

    def test_push_step_increments_total(self):
        b = LiveGradientBridge(max_steps=100)
        b.push_step(0, 1.0, {"fc.weight": {"mean": 1e-3, "std": 1e-4, "max": 2e-3}})
        assert b.total_steps == 1

    def test_ring_buffer_caps_at_max_steps(self):
        b = LiveGradientBridge(max_steps=5)
        for i in range(10):
            b.push_step(i, float(i), {})
        assert len(b.step_history) == 5
        assert len(b.loss_history) == 5

    def test_snapshot_returns_copy(self):
        b = LiveGradientBridge()
        b.push_step(0, 0.5, {})
        snap1 = b.snapshot()
        b.push_step(1, 0.4, {})
        snap2 = b.snapshot()
        assert len(snap1["steps"]) == 1
        assert len(snap2["steps"]) == 2

    def test_push_alert_stores_message(self):
        b = LiveGradientBridge()
        b.push_alert("test alert")
        assert "test alert" in b.snapshot()["alerts"]

    def test_pop_alerts_clears_queue(self):
        b = LiveGradientBridge()
        b.push_alert("a")
        b.push_alert("b")
        popped = b.pop_alerts()
        assert len(popped) == 2
        assert b.pop_alerts() == []

    def test_signal_done_sets_is_training_false(self):
        b = LiveGradientBridge()
        b.push_step(0, 1.0, {})
        assert b.snapshot()["is_training"] is True
        b.signal_done()
        assert b.snapshot()["is_training"] is False

    def test_clear_resets_all_buffers(self):
        b = LiveGradientBridge()
        b.push_step(0, 1.0, {"x": {"mean": 1e-3, "std": 0.0, "max": 1e-3}})
        b.push_alert("msg")
        b.clear()
        snap = b.snapshot()
        assert snap["steps"] == []
        assert snap["alerts"] == []
        assert snap["total_steps"] == 0

    def test_thread_safety(self):
        """Concurrent push_step calls must not corrupt the bridge."""
        b = LiveGradientBridge(max_steps=200)
        errors: List[Exception] = []

        def worker(start: int) -> None:
            try:
                for i in range(50):
                    b.push_step(start + i, float(i), {})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i * 50,)) for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors, f"Thread safety errors: {errors}"
        assert b.total_steps == 200

    def test_global_bridge_singleton(self):
        b1 = get_global_bridge()
        b2 = get_global_bridge()
        assert b1 is b2

    def test_inject_session_state(self):
        b = LiveGradientBridge()
        b.push_step(0, 0.5, {})
        fake_state: dict = {}
        b.inject_session_state(fake_state)
        assert "live_steps" in fake_state
        assert fake_state["live_is_training"] is True


# ---------------------------------------------------------------------------
# StreamlitCallback
# ---------------------------------------------------------------------------

class TestStreamlitCallback:
    def setup_method(self):
        reset_global_bridge()

    def teardown_method(self):
        reset_global_bridge()

    def _simple_model(self) -> nn.Module:
        return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

    def _run_backward(self, model: nn.Module) -> float:
        x   = torch.randn(4, 8)
        y   = torch.randn(4, 4)
        out = model(x)
        loss = nn.functional.mse_loss(out, y)
        loss.backward()
        return loss.item()

    def test_on_batch_end_pushes_to_bridge(self):
        model    = self._simple_model()
        bridge   = LiveGradientBridge()
        callback = StreamlitCallback(model, bridge=bridge)
        loss_val = self._run_backward(model)
        callback.on_batch_end(loss=loss_val, step=0)
        assert bridge.total_steps == 1

    def test_vanishing_gradient_triggers_alert(self):
        model  = self._simple_model()
        bridge = LiveGradientBridge()
        # Manually zero all gradients to simulate vanishing
        loss_val = self._run_backward(model)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.fill_(1e-10)
        callback = StreamlitCallback(
            model, bridge=bridge, alert_threshold=1e-7
        )
        callback.on_batch_end(loss=loss_val, step=0)
        snap = bridge.snapshot()
        assert len(snap["alerts"]) > 0
        assert any("VANISHING" in a for a in snap["alerts"])

    def test_on_train_end_sets_is_training_false(self):
        model    = self._simple_model()
        bridge   = LiveGradientBridge()
        callback = StreamlitCallback(
            model, bridge=bridge, report_every_n_steps=1
        )
        loss_val = self._run_backward(model)
        callback.on_batch_end(loss=loss_val, step=0)
        callback.on_train_end()
        assert bridge.snapshot()["is_training"] is False

    def test_internal_step_counter_increments(self):
        model    = self._simple_model()
        bridge   = LiveGradientBridge()
        callback = StreamlitCallback(model, bridge=bridge)
        for i in range(5):
            self._run_backward(model)
            callback.on_batch_end(loss=0.1)
        assert callback._step_count == 5

    def test_uses_global_bridge_when_none_passed(self):
        model    = self._simple_model()
        callback = StreamlitCallback(model, bridge=None)
        assert callback.bridge is get_global_bridge()


# ---------------------------------------------------------------------------
# ExpertEngine
# ---------------------------------------------------------------------------

class TestExpertEngineSafeNorm:
    def test_returns_grad_norm_when_positive(self):
        s = _make_stats(n=1)[0]
        s.grad_norm = 0.77
        assert _safe_norm(s) == pytest.approx(0.77)

    def test_falls_back_to_abs_mean(self):
        s = _make_stats(n=1)[0]
        s.grad_norm = 0.0
        s.mean = -0.5
        assert _safe_norm(s) == pytest.approx(0.5 + 1e-12)


class TestExpertEngineAnalyze:
    def test_empty_report_returns_no_findings(self):
        empty   = GradientReport(layer_stats=[], global_mean=0.0, global_std=0.0, num_steps=0)
        engine  = ExpertEngine()
        assert engine.analyze(empty) == []

    def test_healthy_report_returns_info_finding(self):
        report  = _make_report(n=4)
        engine  = ExpertEngine(vanishing_threshold=1e-12)
        findings = engine.analyze(report)
        # Should have at most one info finding (global health ok)
        severities = {f.severity for f in findings}
        assert severities <= {"info"}

    def test_vanishing_report_produces_critical_finding(self):
        report  = _make_report(n=4, vanishing=True)
        engine  = ExpertEngine(vanishing_threshold=1e-7)
        findings = engine.analyze(report)
        rule_ids = {f.rule_id for f in findings}
        assert "vanishing_layers" in rule_ids

    def test_exploding_report_produces_critical_finding(self):
        report  = _make_report(n=4, exploding=True)
        engine  = ExpertEngine(exploding_threshold=1e3)
        findings = engine.analyze(report)
        rule_ids = {f.rule_id for f in findings}
        assert "exploding_layers" in rule_ids

    def test_findings_sorted_critical_first(self):
        report  = _make_report(n=6, vanishing=True, exploding=True)
        engine  = ExpertEngine()
        findings = engine.analyze(report)
        if len(findings) >= 2:
            assert findings[0].severity in ("critical", "warning")

    def test_analyze_layer_filters_by_layer(self):
        report  = _make_report(n=6, vanishing=True)
        engine  = ExpertEngine(vanishing_threshold=1e-7)
        # First layer is vanishing
        van_name = next(
            s.layer_name for s in report.layer_stats
            if s.diagnose() == GradientPathology.VANISHING
        )
        layer_findings = engine.analyze_layer(van_name, report)
        assert len(layer_findings) >= 1
        assert all(van_name in f.affected_layers for f in layer_findings)

    def test_top_finding_returns_most_severe(self):
        report = _make_report(n=6, vanishing=True)
        engine = ExpertEngine(vanishing_threshold=1e-7)
        top    = engine.top_finding(report)
        assert top is not None
        assert top.severity == "critical"

    def test_top_finding_returns_none_for_empty(self):
        empty  = GradientReport(layer_stats=[], global_mean=0.0, global_std=0.0, num_steps=0)
        engine = ExpertEngine()
        assert engine.top_finding(empty) is None


class TestExpertFindingProperties:
    def test_severity_emoji(self):
        f = ExpertFinding(
            rule_id="test", severity="critical",
            headline="test", detail="",
        )
        assert f.severity_emoji == "🚨"

    def test_severity_color_critical(self):
        f = ExpertFinding(rule_id="x", severity="critical", headline="", detail="")
        assert f.severity_color == "#E74C3C"

    def test_severity_color_warning(self):
        f = ExpertFinding(rule_id="x", severity="warning", headline="", detail="")
        assert f.severity_color == "#F39C12"


class TestExpertEngineRules:
    def test_dead_neurons_rule_fires_on_high_zero_ratio(self):
        stats = _make_stats(n=4)
        # Force one layer to have high zero_ratio
        stats[0].num_zeros   = 62
        stats[0].total_params = 64
        report = GradientReport(
            layer_stats=stats,
            global_mean=1e-3, global_std=1e-4, num_steps=5
        )
        engine   = ExpertEngine()
        findings = engine.analyze(report)
        rule_ids = {f.rule_id for f in findings}
        assert "dead_neurons" in rule_ids

    def test_structural_bottleneck_rule_fires_on_sharp_drop(self):
        stats = _make_stats(n=4)
        # Force a huge drop: layer 0 norm = 100, layer 1 norm = 0.001
        stats[3].grad_norm = 100.0   # depth 3 = shallowest after reverse sort
        stats[2].grad_norm = 0.001
        report = GradientReport(
            layer_stats=stats,
            global_mean=1e-3, global_std=1e-4, num_steps=5
        )
        engine   = ExpertEngine(bottleneck_drop_ratio=0.3)
        findings = engine.analyze(report)
        rule_ids = {f.rule_id for f in findings}
        assert "structural_bottleneck" in rule_ids

    def test_attention_collapse_rule_fires_for_attn_vanishing(self):
        stats = _make_stats(n=4)
        # Make an ATTENTION layer vanishing
        for s in stats:
            if s.group == LayerGroup.ATTENTION:
                s.grad_norm = 1e-10
                s.mean      = 1e-10
                break
        report = GradientReport(
            layer_stats=stats, global_mean=1e-3, global_std=1e-4, num_steps=5
        )
        engine   = ExpertEngine(vanishing_threshold=1e-7)
        findings = engine.analyze(report)
        rule_ids = {f.rule_id for f in findings}
        assert "attention_collapse" in rule_ids

    def test_all_findings_have_rule_id(self):
        report   = _make_report(n=6, vanishing=True, exploding=True)
        engine   = ExpertEngine()
        findings = engine.analyze(report)
        for f in findings:
            assert f.rule_id, "Finding must have a non-empty rule_id"

    def test_all_critical_findings_have_recommendations(self):
        report   = _make_report(n=6, vanishing=True, exploding=True)
        engine   = ExpertEngine()
        findings = engine.analyze(report)
        for f in findings:
            if f.severity == "critical":
                assert len(f.recommendations) > 0

    def test_all_critical_findings_have_code_snippets(self):
        report   = _make_report(n=6, vanishing=True)
        engine   = ExpertEngine(vanishing_threshold=1e-7)
        findings = engine.analyze(report)
        for f in findings:
            if f.severity == "critical":
                assert len(f.code_snippets) > 0


# ---------------------------------------------------------------------------
# Integration: StreamlitCallback → bridge → ExpertEngine
# ---------------------------------------------------------------------------

class TestPhase4Integration:
    def setup_method(self):
        reset_global_bridge()

    def teardown_method(self):
        reset_global_bridge()

    def test_full_pipeline_no_crash(self):
        """Train for 3 steps, build a report, run expert engine."""
        model  = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        bridge = LiveGradientBridge()
        cb     = StreamlitCallback(
            model, bridge=bridge, report_every_n_steps=2
        )

        for step in range(3):
            x    = torch.randn(4, 8)
            y    = torch.randn(4, 4)
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            cb.on_batch_end(loss=loss.item(), step=step)

        cb.on_train_end()

        snap = bridge.snapshot()
        assert snap["total_steps"] == 3
        assert snap["is_training"] is False

        # If a report was built, run the expert engine on it
        if snap["current_report"] is not None:
            engine   = ExpertEngine()
            findings = engine.analyze(snap["current_report"])
            assert isinstance(findings, list)
