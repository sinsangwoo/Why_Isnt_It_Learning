"""Phase-4 Expert System panel for the Streamlit dashboard.

Three rendering contexts are supported:

1. **Notification banner** (:func:`render_expert_banner`) — a compact
   coloured banner shown at the top of *every* tab whenever critical
   findings exist.  Includes a one-liner summary and a button to expand.

2. **Full popup** (:func:`render_expert_popup`) — a ``st.expander`` block
   that shows all :class:`~gradient_pathology.expert.engine.ExpertFinding`
   objects with their detail text, affected layer list, and ready-to-paste
   code hints.

3. **Layer click panel** (:func:`render_layer_expert_panel`) — filters
   findings to only those that mention the selected layer, used inside
   the Sankey tab’s deep-dive section.
"""

from __future__ import annotations

from typing import List, Optional

from gradient_pathology.core import GradientReport
from gradient_pathology.expert.engine import ExpertEngine, ExpertFinding

try:
    import streamlit as st
    _ST = True
except ImportError:
    _ST = False


# Module-level singleton — recreated on each Streamlit rerun (cheap).
_ENGINE = ExpertEngine()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_findings(report: GradientReport) -> List[ExpertFinding]:
    """Run the ExpertEngine and return sorted findings."""
    return _ENGINE.analyse(report)


def render_expert_banner(
    report: GradientReport,
    key_prefix: str = "expert_banner",
) -> None:
    """Render a compact status banner at the top of a tab.

    * **Green** when no findings.
    * **Orange** when only warnings / info.
    * **Red** when at least one critical finding.

    A **🔍 Details** expander is appended if any findings exist.
    """
    if not _ST:
        return
    findings = get_findings(report)
    if not findings:
        st.success("✅ Expert System: all layers healthy — no issues detected.")
        return

    crit = [f for f in findings if f.severity == "critical"]
    warn = [f for f in findings if f.severity == "warning"]
    info = [f for f in findings if f.severity == "info"]

    summary = _ENGINE.quick_summary(report)
    if crit:
        st.error(summary)
    elif warn:
        st.warning(summary)
    else:
        st.info(summary)

    with st.expander("🔍 Expert System details", expanded=bool(crit)):
        render_expert_popup(
            findings=findings,
            key_prefix=key_prefix,
            show_header=False,
        )


def render_expert_popup(
    report: Optional[GradientReport] = None,
    findings: Optional[List[ExpertFinding]] = None,
    key_prefix: str = "expert_popup",
    show_header: bool = True,
) -> None:
    """Render the full Expert System findings panel.

    Either *report* or *findings* must be supplied.

    Parameters
    ----------
    report:
        If supplied, findings are computed from this report.
    findings:
        Pre-computed findings list (takes precedence over *report*).
    key_prefix:
        Streamlit widget key prefix for uniqueness.
    show_header:
        Whether to render the ``## Expert System Diagnostics`` heading.
    """
    if not _ST:
        return

    if findings is None:
        if report is None:
            st.info("No report supplied.")
            return
        findings = get_findings(report)

    if not findings:
        st.success("✅ No issues detected by the Expert System.")
        return

    if show_header:
        st.subheader("🧠 Expert System Diagnostics")

    for severity in ("critical", "warning", "info"):
        group = [f for f in findings if f.severity == severity]
        if not group:
            continue

        label_map = {
            "critical": ("🚨 Critical issues", True),
            "warning":  ("⚠️ Warnings",         False),
            "info":     ("ℹ️ Suggestions",      False),
        }
        section_label, default_open = label_map[severity]

        with st.expander(f"{section_label} ({len(group)})", expanded=default_open):
            for i, finding in enumerate(group):
                _render_single_finding(
                    finding,
                    key=f"{key_prefix}_{severity}_{i}",
                )
            if severity != "info":
                st.divider()


def render_layer_expert_panel(
    layer_name: str,
    report: GradientReport,
    key_prefix: str = "layer_expert",
) -> None:
    """Render Expert findings that mention *layer_name*.

    Used in the Sankey tab’s layer deep-dive section.  Shows findings
    filtered to findings whose ``.layers`` list contains *layer_name*.
    Falls back to showing all findings when none mention the layer.
    """
    if not _ST:
        return

    findings = get_findings(report)
    relevant = [
        f for f in findings
        if any(layer_name in ln for ln in f.layers)
    ]

    if not relevant:
        # Show global findings if nothing is layer-specific
        relevant = findings

    if not relevant:
        st.success("✅ No Expert System issues for this layer.")
        return

    st.markdown(f"**Expert diagnosis for** `{layer_name.split('.')[-1]}`")
    for i, finding in enumerate(relevant[:5]):  # cap at 5 to avoid clutter
        _render_single_finding(finding, key=f"{key_prefix}_{i}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _render_single_finding(
    finding: ExpertFinding,
    key: str = "finding",
) -> None:
    """Render one :class:`ExpertFinding` as a styled card."""
    # Header line
    st.markdown(
        f"**{finding.emoji} {finding.title}**  "
        f"&nbsp;&nbsp;<sub>confidence: {finding.confidence:.0%}</sub>",
        unsafe_allow_html=True,
    )

    # Detail markdown
    st.markdown(finding.detail)

    # Affected layers
    if finding.layers:
        with st.expander(
            f"Affected layers ({len(finding.layers)})",
            expanded=len(finding.layers) <= 3,
        ):
            for ln in finding.layers:
                st.code(ln, language="text")

    # Code hint
    if finding.code_hint:
        with st.expander("💡 Fix — copy & paste", expanded=False):
            st.code(finding.code_hint, language="python")

    st.divider()
