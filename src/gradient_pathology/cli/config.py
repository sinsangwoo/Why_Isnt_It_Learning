"""Config loader: YAML/JSON -> DiagnoseConfig dataclass."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DiagnoseConfig:
    """Validated configuration for a single CLI diagnosis run.

    Fields
    ------
    num_steps:
        Number of forward/backward passes in synthetic mode.
    threshold:
        Override the default vanishing-gradient threshold.  When ``None``
        the ExpertEngine default (``1e-7``) is used.
    output_dir:
        Directory where Parquet + JSON artefacts are written.
        Defaults to ``./pathology_output``.
    input_shape:
        Synthetic mode input shape, excluding the batch dimension.
    batch_size:
        Synthetic mode mini-batch size.
    device:
        PyTorch device string (``'cpu'`` / ``'cuda'`` / ``'mps'``).
    report_format:
        ``'markdown'`` (default) or ``'json'``.
    """

    num_steps: int = 50
    threshold: Optional[float] = None
    output_dir: str = "pathology_output"
    input_shape: tuple = (10,)
    batch_size: int = 32
    device: str = "cpu"
    report_format: str = "markdown"

    # ------------------------------------------------------------------ #
    # Factories                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_file(cls, path: str) -> "DiagnoseConfig":
        """Load config from a YAML or JSON file.

        YAML support is optional (requires ``pyyaml``).  If the file has a
        ``.yaml`` / ``.yml`` extension and PyYAML is not installed, a helpful
        ``ImportError`` is raised instead of a cryptic parse failure.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        suffix = p.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required for YAML config files.  "
                    "Install it with:  pip install pyyaml"
                ) from exc
            with p.open() as fh:
                data = yaml.safe_load(fh) or {}
        elif suffix == ".json":
            with p.open() as fh:
                data = json.load(fh)
        else:
            raise ValueError(
                f"Unsupported config format '{suffix}'.  Use .yaml, .yml, or .json."
            )

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "DiagnoseConfig":
        """Build a config from a plain dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}

        # Coerce input_shape to tuple (JSON stores it as a list)
        if "input_shape" in filtered:
            filtered["input_shape"] = tuple(filtered["input_shape"])

        return cls(**filtered)

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """Raise ``ValueError`` if any field is out of range."""
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.threshold is not None and self.threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {self.threshold}")
        if self.report_format not in ("markdown", "json"):
            raise ValueError(
                f"report_format must be 'markdown' or 'json', got '{self.report_format}'"
            )
        if not self.input_shape or any(d < 1 for d in self.input_shape):
            raise ValueError(
                f"input_shape must be a non-empty tuple of positive ints, got {self.input_shape}"
            )
