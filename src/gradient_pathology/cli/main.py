"""pathology-diagnose — CLI entrypoint.

Usage examples
--------------
# Minimal (synthetic mode, no config file needed)
pathology-diagnose --output-dir ./reports

# Full: YAML config + saved model checkpoint
pathology-diagnose --config config.yaml --output-dir ./reports

# JSON config
pathology-diagnose --config config.json --output-dir ./reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from gradient_pathology.cli.config import DiagnosisConfig, load_config
from gradient_pathology.cli.runner import run_diagnosis


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="pathology-diagnose",
        description=(
            "Gradient Pathology CLI — automated gradient diagnostics for PyTorch models.\n"
            "Outputs a Markdown report to stdout and saves Parquet/JSON artefacts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  pathology-diagnose --output-dir ./reports\n"
            "  pathology-diagnose --config cfg.yaml --output-dir ./reports\n"
            "  pathology-diagnose --config cfg.json --num-steps 50 --threshold 1e-6\n"
        ),
    )

    # ── Config / override flags ──────────────────────────────────────────────
    parser.add_argument(
        "--config", "-c",
        metavar="PATH",
        help="Path to YAML or JSON config file (optional; CLI flags override file values).",
    )
    parser.add_argument(
        "--model-path", "-m",
        metavar="PATH",
        help="Path to a saved model file (torch.save / state_dict .pt/.pth).  "
             "When omitted a small synthetic MLP is used for demonstration.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        metavar="DIR",
        default="./gradient_pathology_reports",
        help="Directory where Parquet + JSON artefacts are written (default: %(default)s).",
    )

    # ── Diagnosis parameters (override config file) ──────────────────────────
    parser.add_argument(
        "--num-steps", "-n",
        type=int,
        metavar="N",
        help="Number of forward/backward passes (default: 100).",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        metavar="FLOAT",
        help="Vanishing-gradient threshold (default: 1e-7).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        metavar="N",
        help="Batch size for synthetic data (default: 32).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device to run analysis on (default: cpu).",
    )
    parser.add_argument(
        "--input-shape",
        metavar="DIM[,DIM,...]",
        help="Comma-separated input shape excluding batch dim, e.g. '64' or '3,32,32'.",
    )

    # ── Output flags ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        default=False,
        help="Skip Parquet artefact generation.",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        default=False,
        help="Skip JSON artefact generation.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress progress bars and info messages; only print the final report.",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 1.0.0 (Phase 1)",
    )

    return parser


def _parse_input_shape(raw: str) -> tuple[int, ...]:
    """Parse '64' or '3,32,32' into a tuple of ints."""
    try:
        return tuple(int(x.strip()) for x in raw.split(","))
    except ValueError as exc:
        raise ValueError(
            f"--input-shape must be comma-separated integers, got: {raw!r}"
        ) from exc


def app(argv: Optional[list[str]] = None) -> int:
    """Main CLI entrypoint.  Returns process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── Load base config ────────────────────────────────────────────────────
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}", file=sys.stderr)
            return 2
        try:
            config = load_config(config_path)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"❌ Failed to parse config: {exc}", file=sys.stderr)
            return 2
    else:
        config = DiagnosisConfig()

    # ── Apply CLI overrides ─────────────────────────────────────────────────
    if args.model_path:
        config.model_path = args.model_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    if args.threshold is not None:
        config.threshold = args.threshold
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.device:
        config.device = args.device
    if args.input_shape:
        try:
            config.input_shape = list(_parse_input_shape(args.input_shape))
        except ValueError as exc:
            print(f"❌ {exc}", file=sys.stderr)
            return 2
    if args.no_parquet:
        config.save_parquet = False
    if args.no_json:
        config.save_json = False
    config.quiet = args.quiet

    # ── Validate ────────────────────────────────────────────────────────────
    errors = config.validate()
    if errors:
        for err in errors:
            print(f"❌ Config error: {err}", file=sys.stderr)
        return 2

    # ── Run ─────────────────────────────────────────────────────────────────
    try:
        exit_code = run_diagnosis(config)
    except KeyboardInterrupt:
        print("\n⚡ Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Unexpected error: {exc}", file=sys.stderr)
        raise

    return exit_code


def main() -> None:
    """Console-scripts entrypoint."""
    sys.exit(app())


if __name__ == "__main__":
    main()
