"""Orchestration layer: ties config → analyzer → expert engine → store → reporter.

This module is the single place that knows about all sub-systems.  The CLI
``main.py`` only talks to :func:`run_diagnosis`; tests can call it directly
or mock individual sub-systems.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from gradient_pathology.analyzer import GradientAnalyzer
from gradient_pathology.cli.config import DiagnosisConfig
from gradient_pathology.cli.reporter import render_report
from gradient_pathology.cli.snapshot_store import GradientSnapshotStore
from gradient_pathology.expert.engine import ExpertEngine


def run_diagnosis(config: DiagnosisConfig) -> int:
    """Execute a full diagnosis run and return a process exit code.

    Exit codes
    ----------
    0  — success, no critical issues
    1  — success, but critical findings were detected
    2  — configuration / runtime error (handled upstream in main.py)

    Parameters
    ----------
    config:
        A validated :class:`~gradient_pathology.cli.config.DiagnosisConfig`.

    Returns
    -------
    int
        Process exit code.
    """
    import torch
    import torch.nn as nn

    # ── 1. Load or build model ───────────────────────────────────────────────
    if not config.quiet:
        print("⏳ Loading model…", file=sys.stderr)

    model = _load_model(config)

    # ── 2. Run gradient analysis ─────────────────────────────────────────────
    if not config.quiet:
        print("⏳ Running gradient analysis…", file=sys.stderr)

    analyzer = GradientAnalyzer(model, device=config.device)
    report = analyzer.diagnose(
        num_steps=config.num_steps,
        batch_size=config.batch_size,
        input_shape=tuple(config.input_shape),
    )

    # ── 3. Expert engine ─────────────────────────────────────────────────────
    engine = ExpertEngine(vanishing_threshold=config.threshold)
    findings = engine.analyse(report)

    # ── 4. Terminal report ───────────────────────────────────────────────────
    rendered = render_report(report, findings, quiet=config.quiet)
    print(rendered)

    # ── 5. Persist artefacts ─────────────────────────────────────────────────
    if config.save_parquet or config.save_json:
        if not config.quiet:
            print(f"\n💾 Saving artefacts to {config.output_dir}…", file=sys.stderr)
        store = GradientSnapshotStore(
            output_dir=config.output_dir,
            save_parquet=config.save_parquet,
            save_json=config.save_json,
        )
        run_dir = store.save(report, findings)
        if not config.quiet:
            print(f"✅ Artefacts saved → {run_dir}", file=sys.stderr)

    # ── 6. Exit code ─────────────────────────────────────────────────────────
    has_critical = any(f.severity == "critical" for f in findings)
    return 1 if has_critical else 0


def _load_model(
    config: DiagnosisConfig,
) -> "torch.nn.Module":  # type: ignore[name-defined]
    """Load a model from ``config.model_path``, or create a demo MLP."""
    import torch
    import torch.nn as nn

    if config.model_path is None:
        # Demo: small synthetic MLP whose input size matches config.input_shape
        in_features = 1
        for dim in config.input_shape:
            in_features *= dim
        return nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    path = Path(config.model_path)
    obj = torch.load(str(path), map_location="cpu", weights_only=False)

    if isinstance(obj, nn.Module):
        return obj

    # Assume it's a state_dict — wrap in a placeholder and load
    raise ValueError(
        f"Loaded object from {path} is a {type(obj).__name__}, not an nn.Module.\n"
        "To analyse a state_dict, reconstruct your model first and pass the\n"
        "nn.Module instance to GradientAnalyzer directly."
    )
