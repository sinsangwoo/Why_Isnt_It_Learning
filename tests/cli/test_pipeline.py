"""Integration tests for the full CLI pipeline (run_pipeline helper)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gradient_pathology.cli.main import run_pipeline


# ── Happy path ────────────────────────────────────────────────────────────

def test_pipeline_defaults(tmp_path: Path):
    """Default config produces exit code 0 and creates report.json."""
    code = run_pipeline(
        num_steps=5,
        output_dir=str(tmp_path),
    )
    assert code == 0
    report_file = tmp_path / "report.json"
    assert report_file.exists(), "report.json must be created"


def test_pipeline_json_structure(tmp_path: Path):
    """report.json must contain 'global', 'layers', and 'findings' keys."""
    run_pipeline(num_steps=5, output_dir=str(tmp_path))
    data = json.loads((tmp_path / "report.json").read_text())
    assert set(data.keys()) >= {"global", "layers", "findings"}
    assert data["global"]["num_steps"] == 5


def test_pipeline_has_layers(tmp_path: Path):
    """The demo model should produce at least 1 layer entry."""
    run_pipeline(num_steps=5, output_dir=str(tmp_path))
    data = json.loads((tmp_path / "report.json").read_text())
    assert len(data["layers"]) > 0


def test_pipeline_parquet_created(tmp_path: Path):
    """Parquet file is written when pandas/pyarrow are available."""
    pytest.importorskip("pandas", reason="pandas not available")
    run_pipeline(num_steps=5, output_dir=str(tmp_path))
    assert (tmp_path / "layer_stats.parquet").exists()


def test_pipeline_json_format(tmp_path: Path, capsys):
    """--report-format json prints the JSON to stdout."""
    run_pipeline(
        num_steps=5,
        output_dir=str(tmp_path),
        report_format="json",
    )
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert "global" in parsed


def test_pipeline_custom_steps(tmp_path: Path):
    """num_steps is reflected in the saved report."""
    run_pipeline(num_steps=7, output_dir=str(tmp_path))
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["global"]["num_steps"] == 7


def test_pipeline_custom_threshold(tmp_path: Path):
    """Passing a custom threshold should not crash the pipeline."""
    code = run_pipeline(
        num_steps=5,
        output_dir=str(tmp_path),
        threshold=1e-5,
    )
    assert code == 0


# ── Config file integration ────────────────────────────────────────────────

def test_pipeline_with_json_config(tmp_path: Path):
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({"num_steps": 6, "batch_size": 8}))
    code = run_pipeline(
        config_path=str(config),
        output_dir=str(tmp_path / "out"),
    )
    assert code == 0
    data = json.loads((tmp_path / "out" / "report.json").read_text())
    assert data["global"]["num_steps"] == 6


def test_pipeline_cli_overrides_config(tmp_path: Path):
    """CLI num_steps overrides the config-file value."""
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({"num_steps": 99}))
    run_pipeline(
        config_path=str(config),
        num_steps=4,
        output_dir=str(tmp_path / "out"),
    )
    data = json.loads((tmp_path / "out" / "report.json").read_text())
    assert data["global"]["num_steps"] == 4


# ── Error cases ───────────────────────────────────────────────────────────

def test_pipeline_bad_config_path(tmp_path: Path):
    """A missing config file should return exit code 1."""
    code = run_pipeline(
        config_path="/nonexistent/path.json",
        output_dir=str(tmp_path),
    )
    assert code == 1


def test_pipeline_invalid_num_steps(tmp_path: Path):
    """num_steps=0 should return exit code 1."""
    code = run_pipeline(
        num_steps=0,
        output_dir=str(tmp_path),
    )
    assert code == 1


def test_pipeline_invalid_report_format(tmp_path: Path):
    """An unsupported report_format should return exit code 1."""
    code = run_pipeline(
        num_steps=5,
        output_dir=str(tmp_path),
        report_format="html",
    )
    assert code == 1
