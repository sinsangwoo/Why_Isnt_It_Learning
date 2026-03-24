"""Rendering helpers: GradientReport + ExpertFindings -> terminal markdown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from gradient_pathology.core import GradientPathology, GradientReport
from gradient_pathology.expert.engine import ExpertFinding


# ANSI colours (disabled automatically when output is not a TTY)
import sys

_BOLD   = "\033[1m"  if sys.stdout.isatty() else ""
_RESET  = "\033[0m"  if sys.stdout.isatty() else ""
_RED    = "\033[31m" if sys.stdout.isatty() else ""
_YELLOW = "\033[33m" if sys.stdout.isatty() else ""
_GREEN  = "\033[32m" if sys.stdout.isatty() else ""
_CYAN   = "\033[36m" if sys.stdout.isatty() else ""


def render_markdown(
    report: GradientReport,
    findings: List[ExpertFinding],
    output_path: Path,
) -> str:
    """Render a full markdown CLI report string.

    Parameters
    ----------
    report:
        The :class:`~gradient_pathology.core.GradientReport` produced by
        :class:`~gradient_pathology.analyzer.GradientAnalyzer`.
    findings:
        Sorted list of :class:`~gradient_pathology.expert.engine.ExpertFinding`
        objects produced by :class:`~gradient_pathology.expert.engine.ExpertEngine`.
    output_path:
        Directory where artefacts were saved (shown in the footer).

    Returns
    -------
    str
        The complete report string, suitable for ``print()``.
    """
    lines: List[str] = []
    W = 68  # report width

    # ── Header ────────────────────────────────────────────────────────
    lines += [
        _BOLD + "=" * W + _RESET,
        _BOLD + "  🔬  GRADIENT PATHOLOGY REPORT" + _RESET,
        _BOLD + "=" * W + _RESET,
        f"  Data source : {report.data_source}",
        f"  Steps       : {report.num_steps}",
        f"  Layers      : {len(report.layer_stats)}",
        f"  Global mean : {report.global_mean:.3e}",
        f"  Global std  : {report.global_std:.3e}",
        _BOLD + "-" * W + _RESET,
    ]

    # ── Per-layer table ────────────────────────────────────────────────
    lines.append(_BOLD + f"  {'Layer':<40} {'Status':<12} {'Norm':>10}" + _RESET)
    lines.append("  " + "-" * (W - 2))
    for s in report.layer_stats:
        pathology = s.diagnose()
        if pathology == GradientPathology.HEALTHY:
            sym = _GREEN + "✓ HEALTHY   " + _RESET
        elif pathology == GradientPathology.VANISHING:
            sym = _RED + "✗ VANISHING " + _RESET
        elif pathology == GradientPathology.EXPLODING:
            sym = _RED + "✗ EXPLODING " + _RESET
        elif pathology == GradientPathology.DEAD_NEURONS:
            sym = _YELLOW + "⚠ DEAD      " + _RESET
        else:
            sym = _YELLOW + "⚠ UNSTABLE  " + _RESET
        name = s.layer_name[:38] + ".." if len(s.layer_name) > 40 else s.layer_name
        lines.append(f"  {name:<40} {sym} {s.grad_norm:>10.3e}")

    # ── Expert findings ────────────────────────────────────────────────
    if findings:
        lines += [
            "",
            _BOLD + "=" * W + _RESET,
            _BOLD + "  🧠  EXPERT ENGINE FINDINGS" + _RESET,
            _BOLD + "=" * W + _RESET,
        ]
        for f in findings:
            colour = _RED if f.severity == "critical" else (
                _YELLOW if f.severity == "warning" else _CYAN
            )
            lines.append(colour + _BOLD + f"  {f.emoji}  [{f.severity.upper()}] {f.title}" + _RESET)
            # Indent detail lines
            for detail_line in f.detail.splitlines():
                lines.append(f"      {detail_line}")
            if f.layers:
                affected = ", ".join(f.layers[:5])
                if len(f.layers) > 5:
                    affected += f" … (+{len(f.layers) - 5} more)"
                lines.append(f"      Affected : {affected}")
            if f.code_hint:
                lines.append("      Code hint:")
                for hint_line in f.code_hint.splitlines():
                    lines.append(f"        {hint_line}")
            lines.append("")
    else:
        lines += [
            "",
            _GREEN + _BOLD + "  ✅  No issues detected — all layers healthy." + _RESET,
        ]

    # ── Footer ────────────────────────────────────────────────────────
    lines += [
        "",
        _BOLD + "=" * W + _RESET,
        f"  📁  Artefacts saved to: {output_path}",
        _BOLD + "=" * W + _RESET,
    ]
    return "\n".join(lines)


def save_json_report(
    report: GradientReport,
    findings: List[ExpertFinding],
    output_path: Path,
) -> Path:
    """Serialise report + findings to a JSON file.

    Returns
    -------
    Path
        The path of the written file.
    """
    payload = {
        "global": {
            "data_source": report.data_source,
            "num_steps": report.num_steps,
            "global_mean": report.global_mean,
            "global_std": report.global_std,
        },
        "layers": [
            {
                "name": s.layer_name,
                "index": s.layer_index,
                "depth": s.depth,
                "group": s.group.value,
                "layer_type": s.layer_type,
                "grad_norm": s.grad_norm,
                "mean": s.mean,
                "std": s.std,
                "min": s.min,
                "max": s.max,
                "zero_ratio": s.zero_ratio,
                "status": s.diagnose().value,
            }
            for s in report.layer_stats
        ],
        "findings": [
            {
                "rule_id": f.rule_id,
                "severity": f.severity,
                "title": f.title,
                "detail": f.detail,
                "layers": f.layers,
                "confidence": f.confidence,
            }
            for f in findings
        ],
    }
    dest = output_path / "report.json"
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return dest


def save_parquet_report(report: GradientReport, output_path: Path) -> Path:
    """Save per-layer stats to a Parquet file (requires pandas + pyarrow).

    Returns
    -------
    Path
        The path of the written file.

    Raises
    ------
    ImportError
        If ``pandas`` or ``pyarrow`` are not installed.
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pandas is required for Parquet output.  "
            "Install it with:  pip install gradient-pathology[storage]"
        ) from exc

    rows = [
        {
            "layer_name": s.layer_name,
            "layer_index": s.layer_index,
            "depth": s.depth,
            "group": s.group.value,
            "layer_type": s.layer_type,
            "grad_norm": s.grad_norm,
            "mean": s.mean,
            "std": s.std,
            "min": s.min,
            "max": s.max,
            "zero_ratio": s.zero_ratio,
            "status": s.diagnose().value,
        }
        for s in report.layer_stats
    ]
    df = pd.DataFrame(rows)
    dest = output_path / "layer_stats.parquet"
    df.to_parquet(dest, index=False)
    return dest
