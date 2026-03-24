"""Unit + integration tests for gradient_pathology.runner (pathology-run)."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from gradient_pathology.runner import (
    _find_nn_modules,
    _NNModuleTracker,
    _build_runner_parser,
    run_script,
)


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture()
def simple_train_script(tmp_path: Path) -> Path:
    """A minimal training script that creates a model and runs a few steps."""
    code = textwrap.dedent("""\
        import torch
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        loss_fn = nn.MSELoss()
        for _ in range(3):
            x = torch.randn(4, 8)
            y = torch.randn(4, 1)
            model.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
    """)
    p = tmp_path / "train.py"
    p.write_text(code)
    return p


@pytest.fixture()
def no_model_script(tmp_path: Path) -> Path:
    """A script that defines no nn.Module."""
    p = tmp_path / "no_model.py"
    p.write_text("x = 1 + 1\n")
    return p


@pytest.fixture()
def crashing_script(tmp_path: Path) -> Path:
    """A script that raises an exception after creating a model."""
    code = textwrap.dedent("""\
        import torch.nn as nn
        model = nn.Linear(4, 2)
        raise RuntimeError("intentional crash")
    """)
    p = tmp_path / "crash.py"
    p.write_text(code)
    return p


@pytest.fixture()
def sysexit_script(tmp_path: Path) -> Path:
    """A script that calls sys.exit(0)."""
    code = textwrap.dedent("""\
        import sys
        import torch.nn as nn
        model = nn.Linear(4, 2)
        sys.exit(0)
    """)
    p = tmp_path / "exit_script.py"
    p.write_text(code)
    return p


# ── _find_nn_modules ───────────────────────────────────────────────────────

class TestFindNNModules:
    def test_finds_module(self):
        import torch.nn as nn
        ns = {"model": nn.Linear(4, 2), "x": 42, "s": "hello"}
        found = _find_nn_modules(ns)
        assert len(found) == 1
        assert isinstance(found[0], nn.Module)

    def test_empty_namespace(self):
        assert _find_nn_modules({}) == []

    def test_no_modules(self):
        assert _find_nn_modules({"a": 1, "b": "str"}) == []

    def test_multiple_modules(self):
        import torch.nn as nn
        ns = {"m1": nn.Linear(2, 2), "m2": nn.Linear(4, 4)}
        assert len(_find_nn_modules(ns)) == 2


# ── _NNModuleTracker ───────────────────────────────────────────────────────

class TestNNModuleTracker:
    def test_install_uninstall(self):
        import torch.nn as nn
        orig = nn.Module.__init__
        tracker = _NNModuleTracker()
        tracker.install()
        assert nn.Module.__init__ is not orig
        tracker.uninstall()
        assert nn.Module.__init__ is orig

    def test_tracks_new_modules(self):
        import torch.nn as nn
        tracker = _NNModuleTracker()
        tracker.install()
        try:
            _ = nn.Linear(4, 4)
        finally:
            tracker.uninstall()
        assert len(tracker.models) >= 1

    def test_uninstall_twice_safe(self):
        tracker = _NNModuleTracker()
        tracker.uninstall()  # no-op, must not raise
        tracker.uninstall()


# ── argument parser ────────────────────────────────────────────────────────

class TestRunnerParser:
    def test_script_required(self):
        p = _build_runner_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])

    def test_script_parsed(self, tmp_path):
        p = _build_runner_parser()
        args = p.parse_args([str(tmp_path / "x.py")])
        assert args.script.endswith(".py")

    def test_report_format_default(self, tmp_path):
        p = _build_runner_parser()
        args = p.parse_args([str(tmp_path / "x.py")])
        assert args.report_format == "markdown"

    def test_quiet_flag(self, tmp_path):
        p = _build_runner_parser()
        args = p.parse_args([str(tmp_path / "x.py"), "--quiet"])
        assert args.quiet is True

    def test_invalid_format_exits(self, tmp_path):
        p = _build_runner_parser()
        with pytest.raises(SystemExit):
            p.parse_args([str(tmp_path / "x.py"), "--report-format", "html"])


# ── run_script ─────────────────────────────────────────────────────────────

class TestRunScript:
    def test_simple_script_returns_zero(self, simple_train_script, tmp_path):
        code = run_script(str(simple_train_script), output_dir=str(tmp_path))
        assert code == 0

    def test_report_json_created(self, simple_train_script, tmp_path):
        run_script(str(simple_train_script), output_dir=str(tmp_path))
        assert (tmp_path / "report.json").exists()

    def test_report_json_structure(self, simple_train_script, tmp_path):
        run_script(str(simple_train_script), output_dir=str(tmp_path))
        data = json.loads((tmp_path / "report.json").read_text())
        assert "global" in data
        assert "layers" in data
        assert len(data["layers"]) > 0

    def test_data_source_script_run(self, simple_train_script, tmp_path):
        run_script(str(simple_train_script), output_dir=str(tmp_path))
        data = json.loads((tmp_path / "report.json").read_text())
        assert data["global"]["data_source"] == "script_run"

    def test_nonexistent_script_returns_one(self, tmp_path):
        code = run_script("/nonexistent/train.py", output_dir=str(tmp_path))
        assert code == 1

    def test_non_py_file_returns_one(self, tmp_path):
        f = tmp_path / "model.txt"
        f.write_text("hello")
        code = run_script(str(f), output_dir=str(tmp_path))
        assert code == 1

    def test_no_model_script_returns_one(self, no_model_script, tmp_path):
        code = run_script(str(no_model_script), output_dir=str(tmp_path))
        assert code == 1

    def test_crashing_script_does_not_raise(self, crashing_script, tmp_path):
        """A crashing script should still return an int, not raise."""
        code = run_script(str(crashing_script), output_dir=str(tmp_path))
        # May return 0 (model found) or 1 (no model) — not a Python exception
        assert isinstance(code, int)

    def test_sysexit_script_handled(self, sysexit_script, tmp_path):
        """Scripts that call sys.exit() should be handled gracefully."""
        code = run_script(str(sysexit_script), output_dir=str(tmp_path))
        assert isinstance(code, int)

    def test_json_format_output(self, simple_train_script, tmp_path, capsys):
        run_script(
            str(simple_train_script),
            output_dir=str(tmp_path),
            report_format="json",
        )
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "global" in parsed

    def test_quiet_output(self, simple_train_script, tmp_path, capsys):
        run_script(str(simple_train_script), output_dir=str(tmp_path), quiet=True)
        out = capsys.readouterr().out
        assert len(out.strip()) > 0  # at least the summary line

    def test_parquet_created(self, simple_train_script, tmp_path):
        pytest.importorskip("pandas")
        run_script(str(simple_train_script), output_dir=str(tmp_path))
        assert (tmp_path / "layer_stats.parquet").exists()
