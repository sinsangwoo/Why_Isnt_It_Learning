"""Config schema and loader for the pathology-diagnose CLI.

Supports YAML (requires PyYAML) and JSON.  All fields are optional with
sensible defaults; unknown fields are silently ignored so old configs remain
forward-compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisConfig:
    """Full configuration for a single diagnosis run.

    Attributes
    ----------
    model_path:
        Path to a .pt / .pth file saved with ``torch.save(model)`` or
        ``torch.save(model.state_dict())``.  When ``None`` a small synthetic
        MLP is created for demonstration.
    output_dir:
        Directory where reports and artefacts are written.
    num_steps:
        Number of forward/backward passes.
    threshold:
        Vanishing-gradient threshold passed to ``ExpertEngine``.
    batch_size:
        Mini-batch size for synthetic data mode.
    device:
        Torch device string (``'cpu'`` or ``'cuda'``).
    input_shape:
        Input tensor shape *excluding* the batch dimension.
    save_parquet:
        Whether to save per-layer stats as a Parquet file.
    save_json:
        Whether to save the full report as JSON.
    quiet:
        Suppress progress bars and info messages.
    """

    model_path:   Optional[str] = None
    output_dir:   str           = "./gradient_pathology_reports"
    num_steps:    int           = 100
    threshold:    float         = 1e-7
    batch_size:   int           = 32
    device:       str           = "cpu"
    input_shape:  List[int]     = field(default_factory=lambda: [10])
    save_parquet: bool          = True
    save_json:    bool          = True
    quiet:        bool          = False

    # ── Validation ───────────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Return a list of validation error messages (empty = valid)."""
        errors: List[str] = []

        if self.num_steps < 1:
            errors.append(f"num_steps must be >= 1, got {self.num_steps}")
        if self.num_steps > 100_000:
            errors.append(f"num_steps={self.num_steps} is suspiciously large (> 100 000)")

        if self.threshold <= 0:
            errors.append(f"threshold must be > 0, got {self.threshold}")
        if self.threshold >= 1:
            errors.append(f"threshold must be < 1, got {self.threshold}")

        if self.batch_size < 1:
            errors.append(f"batch_size must be >= 1, got {self.batch_size}")
        if self.batch_size > 65536:
            errors.append(f"batch_size={self.batch_size} is suspiciously large (> 65536)")

        if self.device not in ("cpu", "cuda"):
            errors.append(f"device must be 'cpu' or 'cuda', got {self.device!r}")

        if not self.input_shape:
            errors.append("input_shape must have at least one dimension")
        else:
            for i, dim in enumerate(self.input_shape):
                if dim < 1:
                    errors.append(f"input_shape[{i}]={dim} must be >= 1")

        if self.model_path is not None and not Path(self.model_path).exists():
            errors.append(f"model_path not found: {self.model_path}")

        return errors

    # ── Serialisation helpers ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict suitable for JSON/YAML serialisation."""
        return {
            "model_path":   self.model_path,
            "output_dir":   self.output_dir,
            "num_steps":    self.num_steps,
            "threshold":    self.threshold,
            "batch_size":   self.batch_size,
            "device":       self.device,
            "input_shape":  self.input_shape,
            "save_parquet": self.save_parquet,
            "save_json":    self.save_json,
            "quiet":        self.quiet,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagnosisConfig":
        """Build a config from a plain dict (unknown keys are ignored)."""
        known = {
            "model_path", "output_dir", "num_steps", "threshold",
            "batch_size", "device", "input_shape", "save_parquet",
            "save_json", "quiet",
        }
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: Path) -> DiagnosisConfig:
    """Load a :class:`DiagnosisConfig` from a YAML or JSON file.

    YAML is attempted first (requires ``pyyaml``); falls back to JSON.

    Raises
    ------
    ValueError
        If the file extension is unrecognised or content cannot be parsed.
    """
    suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8")

    if suffix in (".yaml", ".yml"):
        data = _load_yaml(raw_text, path)
    elif suffix == ".json":
        data = json.loads(raw_text)
    else:
        # Try YAML first, then JSON for extension-less or unknown files
        try:
            data = _load_yaml(raw_text, path)
        except Exception:  # noqa: BLE001
            data = json.loads(raw_text)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping, got {type(data).__name__}")

    return DiagnosisConfig.from_dict(data)


def _load_yaml(text: str, path: Path) -> Dict[str, Any]:
    """Parse YAML text, raising a clear error if PyYAML is missing."""
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load YAML configs.  Install it with:\n"
            "  pip install pyyaml"
        ) from exc
    return yaml.safe_load(text)  # type: ignore[no-any-return]
