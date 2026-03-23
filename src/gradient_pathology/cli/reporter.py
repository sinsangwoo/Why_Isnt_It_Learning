"""Terminal reporter — renders the diagnosis result as rich Markdown to stdout.

Designed to be completely independent of the storage layer so that it can
also be used in notebook / interactive contexts.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from gradient_pathology.core import GradientPathology, GradientReport
from gradient_pathology.expert.engine import ExpertFinding


# ANSI colour codes (disabled when not a TTY)
_IS_TTY = sys.stdout.isatty()
_RESET  = "\033[0m"  if _IS_TTY else ""
_BOLD   = "\033[1m"  if _IS_TTY else ""
_RED    = "\033[91m" if _IS_TTY else ""
_YELLOW = "\033[93m" if _IS_TTY else ""
_GREEN  = "\033[92m" if _IS_TTY else ""
_CYAN   = "\033[96m" if _IS_TTY else ""
_DIM    = "\033[2m"  if _IS_TTY else ""


def render_report(
    report: GradientReport,
    findings: Optional[List[ExpertFinding]] = None,
    quiet: bool = False,
) -> str:
    """Render *report* + *findings* as a Markdown string for terminal output.

    Parameters
    ----------
    report:
        The :class:`~gradient_pathology.core.GradientReport` from the analyser.
    findings:
        Optional list of :class:`~gradient_pathology.expert.engine.ExpertFinding`
        objects produced by :class:`~gradient_pathology.expert.engine.ExpertEngine`.
    quiet:
        When ``True`` only the critical/warning findings are included; the
        per-layer table is omitted.

    Returns
    -------
    str
        The rendered Markdown / ANSI string ready for ``print()``.
    """
    lines: List[str] = []
    SEP = "═" * 68

    # ── Header ───────────────────────────────────────────────────────────────
    lines.append(f"{_BOLD}{'═' * 68}{_RESET}")
    lines.append(f"{_BOLD}  🔬 GRADIENT PATHOLOGY REPORT{_RESET}")
    lines.append(f"{_BOLD}{SEP}{_RESET}")
    lines.append(f"  Data source : {_CYAN}{report.data_source}{_RESET}")
    lines.append(f"  Steps       : {report.num_steps}")
    lines.append(f"  Global mean : {_fmt_sci(report.global_mean)}")
    lines.append(f"  Global std  : {_fmt_sci(report.global_std)}")

    if report.data_source == "synthetic":
        lines.append("")
        lines.append(
            f"  {_YELLOW}⚠  Gradients computed on SYNTHETIC data.{_RESET}\n"
            f"     Pass a real DataLoader for actionable diagnostics."
        )

    # ── Expert findings ──────────────────────────────────────────────────────
    if findings:
        lines.append("")
        lines.append(f"{_BOLD}{'─' * 68}{_RESET}")
        lines.append(f"{_BOLD}  📋 EXPERT FINDINGS{_RESET}")
        lines.append(f"{_BOLD}{'─' * 68}{_RESET}")

        for f in findings:
            colour = _RED if f.severity == "critical" else (
                _YELLOW if f.severity == "warning" else _DIM
            )
            lines.append(f"  {colour}{f.emoji}  [{f.severity.upper()}] {f.title}{_RESET}")
            lines.append(f"     Rule: {f.rule_id}  |  Confidence: {f.confidence:.0%}")
            if f.layers:
                preview = ", ".join(f.layers[:3])
                more = f" … +{len(f.layers) - 3}" if len(f.layers) > 3 else ""
                lines.append(f"     Layers: {preview}{more}")
            lines.append("")
    else:
        lines.append("")
        lines.append(f"  {_GREEN}✅  No issues detected — all layers healthy!{_RESET}")

    # ── Per-layer table (skipped in quiet mode) ───────────────────────────────
    if not quiet:
        lines.append("")
        lines.append(f"{_BOLD}{'─' * 68}{_RESET}")
        lines.append(f"{_BOLD}  📊 PER-LAYER DIAGNOSTICS{_RESET}")
        lines.append(f"{_BOLD}{'─' * 68}{_RESET}")
        lines.append(
            f"  {'Layer':<38} {'Type':<12} {'Norm':>10}  Status"
        )
        lines.append(f"  {'─' * 66}")
        for s in report.layer_stats:
            pathology = s.diagnose()
            if pathology == GradientPathology.HEALTHY:
                status = f"{_GREEN}✓ healthy{_RESET}"
            elif pathology == GradientPathology.VANISHING:
                status = f"{_RED}✗ vanishing{_RESET}"
            elif pathology == GradientPathology.EXPLODING:
                status = f"{_RED}✗ exploding{_RESET}"
            elif pathology == GradientPathology.DEAD_NEURONS:
                status = f"{_YELLOW}✗ dead neurons{_RESET}"
            else:
                status = f"{_YELLOW}✗ unstable{_RESET}"

            name = s.layer_name
            if len(name) > 37:
                name = "…" + name[-36:]
            lines.append(
                f"  {name:<38} {s.layer_type:<12} {s.grad_norm:>10.2e}  {status}"
            )

    # ── Footer ───────────────────────────────────────────────────────────────
    lines.append("")
    lines.append(f"{_BOLD}{SEP}{_RESET}")

    return "\n".join(lines)


def _fmt_sci(value: float) -> str:
    """Format a float in scientific notation, coloured by magnitude."""
    formatted = f"{value:.2e}"
    if abs(value) < 1e-7 and abs(value) > 0:
        return f"{_RED}{formatted}{_RESET}"
    if abs(value) > 1e3:
        return f"{_RED}{formatted}{_RESET}"
    return f"{_GREEN}{formatted}{_RESET}"
