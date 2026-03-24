"""CLI entrypoint: ``pathology-diagnose``.

Usage examples
--------------
# Minimal — synthetic mode with defaults:
pathology-diagnose

# With a YAML config:
pathology-diagnose --config config.yaml

# Override individual fields:
pathology-diagnose --num-steps 100 --output-dir ./out --device cuda

# JSON report format:
pathology-diagnose --report-format json

# Quiet (suppress per-layer table, show only findings):
pathology-diagnose --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pathology-diagnose",
        description="Gradient Pathology — CLI diagnostic pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  pathology-diagnose                       # quick sanity check\n"
            "  pathology-diagnose --config cfg.yaml     # use a config file\n"
            "  pathology-diagnose --num-steps 200 --device cuda\n"
        ),
    )

    # Config file
    p.add_argument(
        "--config", "-c",
        metavar="PATH",
        help="Path to a YAML or JSON config file.",
    )

    # Per-field overrides (take precedence over config file)
    p.add_argument(
        "--num-steps", "-n",
        type=int,
        metavar="N",
        help="Number of forward/backward passes (default: 50).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        metavar="FLOAT",
        help="Vanishing-gradient threshold override (default: 1e-7).",
    )
    p.add_argument(
        "--output-dir", "-o",
        metavar="DIR",
        help="Output directory for artefacts (default: pathology_output).",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="PyTorch device: cpu / cuda / mps (default: cpu).",
    )
    p.add_argument(
        "--report-format",
        choices=["markdown", "json"],
        default=None,
        metavar="FMT",
        help="Output format: markdown (default) or json.",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-layer table; print only expert findings.",
    )

    return p


def _load_config(args: argparse.Namespace):
    """Load config from file (if given) then apply CLI overrides."""
    from gradient_pathology.cli.config import DiagnoseConfig

    cfg = DiagnoseConfig.from_file(args.config) if args.config else DiagnoseConfig()

    # CLI args override config-file values
    if args.num_steps is not None:
        cfg.num_steps = args.num_steps
    if args.threshold is not None:
        cfg.threshold = args.threshold
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.device is not None:
        cfg.device = args.device
    if args.report_format is not None:
        cfg.report_format = args.report_format

    cfg.validate()
    return cfg


def _build_demo_model():
    """Return a simple demo model for the no-model-path synthetic mode."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def run_pipeline(
    config_path: Optional[str] = None,
    num_steps: Optional[int] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    report_format: Optional[str] = None,
    threshold: Optional[float] = None,
    quiet: bool = False,
) -> int:
    """Programmatic entry-point for the CLI pipeline (also used by tests).

    Returns the exit code (0 = success).
    """
    import types
    # Build a fake Namespace so _load_config can work without argparse
    fake_args = types.SimpleNamespace(
        config=config_path,
        num_steps=num_steps,
        threshold=threshold,
        output_dir=output_dir,
        device=device,
        report_format=report_format,
    )
    try:
        cfg = _load_config(fake_args)  # type: ignore[arg-type]
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"❌  Config error: {exc}", file=sys.stderr)
        return 1

    # ── Step 1: build demo model ─────────────────────────────────────
    model = _build_demo_model()

    # ── Step 2: run GradientAnalyzer ────────────────────────────────
    from gradient_pathology.analyzer import GradientAnalyzer
    analyzer = GradientAnalyzer(model, device=cfg.device)
    report = analyzer.diagnose(
        num_steps=cfg.num_steps,
        input_shape=cfg.input_shape,
        batch_size=cfg.batch_size,
    )

    # ── Step 3: run ExpertEngine ─────────────────────────────────────
    from gradient_pathology.expert.engine import ExpertEngine
    engine_kwargs = {}
    if cfg.threshold is not None:
        engine_kwargs["vanishing_threshold"] = cfg.threshold
    engine = ExpertEngine(**engine_kwargs)
    findings = engine.analyse(report)

    # ── Step 4: save artefacts ───────────────────────────────────────
    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    from gradient_pathology.cli.report import save_json_report, save_parquet_report, render_markdown

    save_json_report(report, findings, output_path)
    try:
        save_parquet_report(report, output_path)
    except ImportError:
        pass  # pandas/pyarrow optional

    # ── Step 5: print report ─────────────────────────────────────────
    if cfg.report_format == "json":
        import json
        from gradient_pathology.cli.report import save_json_report
        dest = output_path / "report.json"
        print(dest.read_text())
    else:
        if not quiet:
            print(render_markdown(report, findings, output_path))
        else:
            # Quiet mode: just print expert summary
            from gradient_pathology.expert.engine import ExpertEngine as _EE
            _e = ExpertEngine(**engine_kwargs)
            print(_e.quick_summary(report))

    return 0


def main() -> None:
    """CLI entrypoint registered in pyproject.toml."""
    parser = _build_parser()
    args = parser.parse_args()

    import types
    try:
        cfg = _load_config(args)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"❌  Config error: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Step 1: demo model ───────────────────────────────────────────
    model = _build_demo_model()

    # ── Step 2: GradientAnalyzer ─────────────────────────────────────
    from gradient_pathology.analyzer import GradientAnalyzer
    analyzer = GradientAnalyzer(model, device=cfg.device)
    report = analyzer.diagnose(
        num_steps=cfg.num_steps,
        input_shape=cfg.input_shape,
        batch_size=cfg.batch_size,
    )

    # ── Step 3: ExpertEngine ─────────────────────────────────────────
    from gradient_pathology.expert.engine import ExpertEngine
    engine_kwargs = {}
    if cfg.threshold is not None:
        engine_kwargs["vanishing_threshold"] = cfg.threshold
    engine = ExpertEngine(**engine_kwargs)
    findings = engine.analyse(report)

    # ── Step 4: artefacts ────────────────────────────────────────────
    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    from gradient_pathology.cli.report import save_json_report, save_parquet_report, render_markdown

    save_json_report(report, findings, output_path)
    try:
        save_parquet_report(report, output_path)
    except ImportError:
        pass

    # ── Step 5: print ────────────────────────────────────────────────
    if cfg.report_format == "json":
        dest = output_path / "report.json"
        print(dest.read_text())
    elif args.quiet:
        print(engine.quick_summary(report))
    else:
        print(render_markdown(report, findings, output_path))

    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
