"""Unit tests for cli.report — markdown/JSON/Parquet rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gradient_pathology.core import GradientPathology, GradientReport, LayerGradientStats, LayerGroup
from gradient_pathology.expert.engine import ExpertEngine, ExpertFinding
from gradient_pathology.cli.report import (
    render_markdown,
    save_json_report,
    save_parquet_report,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_layer(name: str, norm: float, mean: float = 1e-3) -> LayerGradientStats:
    return LayerGradientStats(
        layer_name=name,
        layer_index=0,
        mean=mean,
        std=1e-4,
        min=-1e-3,
        max=1e-3,
        median=1e-4,
        num_zeros=0,
        total_params=100,
        layer_type="Linear",
        depth=0,
        group=LayerGroup.OTHER,
        grad_norm=norm,
    )


def _make_report(layers=None) -> GradientReport:
    if layers is None:
        layers = [_make_layer("layer_0", 1e-3)]
    return GradientReport(
        layer_stats=layers,
        global_mean=1e-3,
        global_std=1e-4,
        num_steps=10,
        data_source="synthetic",
    )


# ── render_markdown ───────────────────────────────────────────────────────

def test_render_markdown_healthy(tmp_path: Path):
    report = _make_report()
    findings: list = []
    text = render_markdown(report, findings, tmp_path)
    assert "GRADIENT PATHOLOGY REPORT" in text
    assert "layer_0" in text
    assert str(tmp_path) in text


def test_render_markdown_with_finding(tmp_path: Path):
    report = _make_report()
    f = ExpertFinding(
        rule_id="vanishing_layers",
        severity="critical",
        title="Test vanishing",
        detail="Some detail",
        layers=["layer_0"],
        confidence=0.9,
    )
    text = render_markdown(report, [f], tmp_path)
    assert "EXPERT ENGINE FINDINGS" in text
    assert "Test vanishing" in text


def test_render_markdown_no_layers(tmp_path: Path):
    """Empty layer list should not crash."""
    report = _make_report(layers=[])
    text = render_markdown(report, [], tmp_path)
    assert "GRADIENT PATHOLOGY REPORT" in text


def test_render_markdown_long_layer_name(tmp_path: Path):
    """Layer names > 40 chars are truncated with '..'."""
    long_name = "a" * 50
    report = _make_report([_make_layer(long_name, 1e-3)])
    text = render_markdown(report, [], tmp_path)
    assert ".." in text


# ── save_json_report ──────────────────────────────────────────────────────

def test_save_json_report(tmp_path: Path):
    report = _make_report()
    engine = ExpertEngine()
    findings = engine.analyse(report)
    dest = save_json_report(report, findings, tmp_path)
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert "global" in data
    assert "layers" in data
    assert "findings" in data
    assert data["global"]["num_steps"] == 10
    assert len(data["layers"]) == 1


def test_save_json_report_layer_fields(tmp_path: Path):
    """Each layer entry must have the expected keys."""
    report = _make_report()
    dest = save_json_report(report, [], tmp_path)
    data = json.loads(dest.read_text())
    layer = data["layers"][0]
    for key in ("name", "index", "depth", "group", "layer_type",
                "grad_norm", "mean", "std", "zero_ratio", "status"):
        assert key in layer, f"Missing key: {key}"


def test_save_json_report_vanishing(tmp_path: Path):
    """Vanishing layer produces a critical finding and is reflected in JSON."""
    layers = [_make_layer("bad_layer", norm=1e-10, mean=1e-10)]
    report = _make_report(layers)
    engine = ExpertEngine()
    findings = engine.analyse(report)
    dest = save_json_report(report, findings, tmp_path)
    data = json.loads(dest.read_text())
    assert any(f["severity"] == "critical" for f in data["findings"])


# ── save_parquet_report ───────────────────────────────────────────────────

def test_save_parquet_report(tmp_path: Path):
    pytest.importorskip("pandas", reason="pandas not installed")
    report = _make_report()
    dest = save_parquet_report(report, tmp_path)
    assert dest.exists()
    import pandas as pd
    df = pd.read_parquet(dest)
    assert len(df) == 1
    assert "grad_norm" in df.columns
    assert "status" in df.columns


def test_save_parquet_no_pandas(tmp_path: Path, monkeypatch):
    """ImportError is raised when pandas is not available."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    report = _make_report()
    with pytest.raises(ImportError, match="pandas"):
        save_parquet_report(report, tmp_path)
