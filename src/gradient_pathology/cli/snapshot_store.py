"""GradientSnapshotStore — persists diagnosis results to Parquet and JSON.

The store is intentionally decoupled from the diagnosis logic so that the
CLI, future API servers, and tests can use it independently.

File layout (under ``output_dir``)
-----------------------------------
<output_dir>/
    run_<timestamp>/
        layer_stats.parquet   # per-layer gradient statistics
        report.json           # full serialised GradientReport + findings
        summary.md            # human-readable Markdown summary
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from gradient_pathology.core import GradientReport
from gradient_pathology.expert.engine import ExpertFinding


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class GradientSnapshotStore:
    """Persists a :class:`~gradient_pathology.core.GradientReport` to disk.

    Parameters
    ----------
    output_dir:
        Root directory.  A timestamped sub-directory is created per run.
    save_parquet:
        Write a ``layer_stats.parquet`` file (requires ``pandas`` + ``pyarrow``).
    save_json:
        Write a ``report.json`` file.
    """

    def __init__(
        self,
        output_dir: str | Path,
        save_parquet: bool = True,
        save_json: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.save_parquet = save_parquet
        self.save_json = save_json
        self._run_dir: Optional[Path] = None

    # ── Public API ───────────────────────────────────────────────────────────

    def save(
        self,
        report: GradientReport,
        findings: Optional[List[ExpertFinding]] = None,
        run_tag: Optional[str] = None,
    ) -> Path:
        """Persist *report* (and optional *findings*) to a timestamped directory.

        Returns
        -------
        Path
            The run directory that was created.
        """
        run_dir = self._make_run_dir(run_tag)
        self._run_dir = run_dir

        if self.save_parquet:
            self._write_parquet(report, run_dir)

        if self.save_json:
            self._write_json(report, findings or [], run_dir)

        self._write_markdown(report, findings or [], run_dir)

        return run_dir

    @property
    def run_dir(self) -> Optional[Path]:
        """The directory created by the most recent :meth:`save` call."""
        return self._run_dir

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _make_run_dir(self, tag: Optional[str]) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"run_{ts}" if not tag else f"run_{ts}_{tag}"
        run_dir = self.output_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    # ── Parquet ──────────────────────────────────────────────────────────────

    def _write_parquet(self, report: GradientReport, run_dir: Path) -> None:
        try:
            import pandas as pd  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pandas is required to write Parquet files.\n"
                "Install it with: pip install gradient-pathology[storage]"
            ) from exc

        rows = [
            {
                "layer_name":   s.layer_name,
                "layer_index":  s.layer_index,
                "layer_type":   s.layer_type,
                "group":        s.group.value,
                "depth":        s.depth,
                "mean":         s.mean,
                "std":          s.std,
                "min":          s.min,
                "max":          s.max,
                "median":       s.median,
                "grad_norm":    s.grad_norm,
                "zero_ratio":   s.zero_ratio,
                "num_zeros":    s.num_zeros,
                "total_params": s.total_params,
                "pathology":    s.diagnose().value,
            }
            for s in report.layer_stats
        ]
        df = pd.DataFrame(rows)
        out_path = run_dir / "layer_stats.parquet"
        df.to_parquet(out_path, index=False)

    # ── JSON ─────────────────────────────────────────────────────────────────

    def _write_json(  # noqa: WPS217
        self,
        report: GradientReport,
        findings: List[ExpertFinding],
        run_dir: Path,
    ) -> None:
        payload: Dict[str, Any] = {
            "meta": {
                "data_source": report.data_source,
                "num_steps":   report.num_steps,
                "global_mean": report.global_mean,
                "global_std":  report.global_std,
            },
            "layer_stats": [
                {
                    "layer_name":   s.layer_name,
                    "layer_index":  s.layer_index,
                    "layer_type":   s.layer_type,
                    "group":        s.group.value,
                    "depth":        s.depth,
                    "mean":         s.mean,
                    "std":          s.std,
                    "min":          s.min,
                    "max":          s.max,
                    "median":       s.median,
                    "grad_norm":    s.grad_norm,
                    "zero_ratio":   s.zero_ratio,
                    "pathology":    s.diagnose().value,
                }
                for s in report.layer_stats
            ],
            "findings": [
                {
                    "rule_id":    f.rule_id,
                    "severity":   f.severity,
                    "title":      f.title,
                    "detail":     f.detail,
                    "layers":     f.layers,
                    "code_hint":  f.code_hint,
                    "confidence": f.confidence,
                }
                for f in findings
            ],
        }
        out_path = run_dir / "report.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Markdown ─────────────────────────────────────────────────────────────

    def _write_markdown(  # noqa: WPS217
        self,
        report: GradientReport,
        findings: List[ExpertFinding],
        run_dir: Path,
    ) -> None:
        lines: List[str] = []
        lines.append("# Gradient Pathology Report\n")
        lines.append(f"- **Data source**: `{report.data_source}`")
        lines.append(f"- **Steps**: {report.num_steps}")
        lines.append(f"- **Global mean**: `{report.global_mean:.2e}`")
        lines.append(f"- **Global std**: `{report.global_std:.2e}`")
        lines.append("")

        if findings:
            lines.append("## Expert Findings\n")
            for f in findings:
                lines.append(f"### {f.emoji} {f.title}")
                lines.append(f"- **Rule**: `{f.rule_id}`")
                lines.append(f"- **Severity**: {f.severity}")
                lines.append(f"- **Confidence**: {f.confidence:.0%}")
                if f.layers:
                    lines.append(f"- **Affected layers**: {', '.join(f.layers[:5])}")
                lines.append("")
                lines.append(f.detail)
                if f.code_hint:
                    lines.append("")
                    lines.append("```python")
                    lines.append(f.code_hint)
                    lines.append("```")
                lines.append("")
        else:
            lines.append("## ✅ No issues detected\n")

        lines.append("## Per-Layer Statistics\n")
        lines.append("| Layer | Type | Group | Norm | Mean | Status |")
        lines.append("|-------|------|-------|------|------|--------|")
        for s in report.layer_stats:
            pathology = s.diagnose().value
            icon = "✓" if pathology == "healthy" else "✗"
            lines.append(
                f"| `{s.layer_name}` | {s.layer_type} | {s.group.value} "
                f"| `{s.grad_norm:.2e}` | `{s.mean:.2e}` | {icon} {pathology} |"
            )

        out_path = run_dir / "summary.md"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
