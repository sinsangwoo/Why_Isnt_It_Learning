"""Unit tests for cli.config — DiagnoseConfig loading and validation."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from gradient_pathology.cli.config import DiagnoseConfig


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def json_config_file(tmp_path: Path) -> Path:
    data = {
        "num_steps": 20,
        "threshold": 1e-6,
        "output_dir": str(tmp_path / "out"),
        "input_shape": [8],
        "batch_size": 16,
        "device": "cpu",
        "report_format": "markdown",
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture()
def yaml_config_file(tmp_path: Path) -> Path:
    content = textwrap.dedent("""\
        num_steps: 30
        threshold: 2.0e-7
        output_dir: /tmp/yaml_out
        input_shape: [16, 16]
        batch_size: 8
        device: cpu
        report_format: json
    """)
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p


# ── Defaults ──────────────────────────────────────────────────────────────

def test_defaults():
    cfg = DiagnoseConfig()
    assert cfg.num_steps == 50
    assert cfg.threshold is None
    assert cfg.output_dir == "pathology_output"
    assert cfg.input_shape == (10,)
    assert cfg.batch_size == 32
    assert cfg.device == "cpu"
    assert cfg.report_format == "markdown"


# ── JSON loading ──────────────────────────────────────────────────────────

def test_from_json(json_config_file: Path):
    cfg = DiagnoseConfig.from_file(str(json_config_file))
    assert cfg.num_steps == 20
    assert cfg.threshold == pytest.approx(1e-6)
    assert cfg.input_shape == (8,)   # list -> tuple coercion
    assert cfg.batch_size == 16
    assert cfg.report_format == "markdown"


def test_json_partial_override(tmp_path: Path):
    """Only overriding num_steps should leave other fields at defaults."""
    p = tmp_path / "partial.json"
    p.write_text(json.dumps({"num_steps": 5}))
    cfg = DiagnoseConfig.from_file(str(p))
    assert cfg.num_steps == 5
    assert cfg.device == "cpu"        # default preserved
    assert cfg.batch_size == 32       # default preserved


def test_json_unknown_keys_ignored(tmp_path: Path):
    """Unknown fields in the JSON should be silently ignored."""
    p = tmp_path / "extra.json"
    p.write_text(json.dumps({"num_steps": 7, "unknown_key": "oops"}))
    cfg = DiagnoseConfig.from_file(str(p))
    assert cfg.num_steps == 7


# ── YAML loading ──────────────────────────────────────────────────────────

def test_from_yaml(yaml_config_file: Path):
    pytest.importorskip("yaml", reason="pyyaml not installed")
    cfg = DiagnoseConfig.from_file(str(yaml_config_file))
    assert cfg.num_steps == 30
    assert cfg.report_format == "json"
    assert cfg.input_shape == (16, 16)


# ── Error cases ───────────────────────────────────────────────────────────

def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        DiagnoseConfig.from_file("/nonexistent/path/config.json")


def test_unsupported_extension(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("")
    with pytest.raises(ValueError, match="Unsupported config format"):
        DiagnoseConfig.from_file(str(p))


# ── Validation ────────────────────────────────────────────────────────────

def test_validate_ok():
    DiagnoseConfig(num_steps=10, batch_size=4).validate()  # should not raise


@pytest.mark.parametrize("field,value", [
    ("num_steps",     0),
    ("num_steps",    -1),
    ("batch_size",    0),
    ("threshold",    -1.0),
    ("threshold",     0.0),
    ("report_format", "html"),
])
def test_validate_invalid(field, value):
    cfg = DiagnoseConfig()
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        cfg.validate()


def test_validate_invalid_input_shape():
    cfg = DiagnoseConfig(input_shape=())
    with pytest.raises(ValueError):
        cfg.validate()


def test_validate_invalid_input_shape_negative():
    cfg = DiagnoseConfig(input_shape=(-1,))
    with pytest.raises(ValueError):
        cfg.validate()
