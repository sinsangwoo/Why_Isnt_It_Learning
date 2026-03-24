"""Smoke-tests for the ``pathology-diagnose`` CLI entrypoint.

These tests invoke the parser and ``run_pipeline`` without spawning a
subprocess, keeping the test suite fast and CI-friendly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from gradient_pathology.cli.main import _build_parser, run_pipeline


# ── Parser tests ──────────────────────────────────────────────────────────

def test_parser_defaults():
    p = _build_parser()
    args = p.parse_args([])
    assert args.config is None
    assert args.num_steps is None
    assert args.quiet is False


def test_parser_short_flags():
    p = _build_parser()
    args = p.parse_args(["-n", "20", "-q"])
    assert args.num_steps == 20
    assert args.quiet is True


def test_parser_config_flag():
    p = _build_parser()
    args = p.parse_args(["--config", "cfg.yaml"])
    assert args.config == "cfg.yaml"


def test_parser_report_format_choices():
    p = _build_parser()
    args = p.parse_args(["--report-format", "json"])
    assert args.report_format == "json"


def test_parser_invalid_format_exits():
    p = _build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--report-format", "html"])


# ── run_pipeline smoke tests ──────────────────────────────────────────────

def test_run_pipeline_smoke(tmp_path: Path):
    code = run_pipeline(num_steps=3, output_dir=str(tmp_path))
    assert code == 0


def test_run_pipeline_quiet(tmp_path: Path, capsys):
    run_pipeline(num_steps=3, output_dir=str(tmp_path), quiet=True)
    out = capsys.readouterr().out
    # Quiet mode should still produce some output (quick_summary)
    assert len(out.strip()) > 0


def test_run_pipeline_output_dir_created(tmp_path: Path):
    out_dir = tmp_path / "nested" / "output"
    run_pipeline(num_steps=3, output_dir=str(out_dir))
    assert out_dir.exists()


def test_run_pipeline_report_json_valid(tmp_path: Path):
    run_pipeline(num_steps=3, output_dir=str(tmp_path), report_format="json")
    report = json.loads((tmp_path / "report.json").read_text())
    assert "global" in report
    assert "layers" in report
