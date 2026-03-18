"""Phase-4: ExpertSystem popup panel for the Streamlit dashboard.

This module renders the :class:`~gradient_pathology.expert.engine.ExpertEngine`
diagnostics as an interactive Streamlit component:

* A **notification banner** at the top of the dashboard listing critical
  findings with one-click expand.
* An **Expert Diagnosis** panel (full-width expander) that shows every
  finding in detail: severity badge, headline, Markdown explanation,
  itemised recommendations, and copy-pasteable code snippets.
* A **Layer-focused panel** (``render_layer_expert_panel``) that shows
  only findings relevant to a specific layer — used from the Heatmap and
  Sankey tabs when a user clicks/selects a layer node.
"""

from __future__ import annotations

from typing import List, Optional

from gradient_pathology.core import GradientReport
from gradient_pathology.expert.engine import ExpertEngine, ExpertFinding

try:
    import streamlit as st
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module-level engine instance (shared, stateless)
# ---------------------------------------------------------------------------

_ENGINE = ExpertEngine()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def render_expert_banner(
    report: GradientReport,
    key_prefix: str = "expert_banner",
    vanishing_threshold: float = 1e-7,
    exploding_threshold: float = 1e3,
) -> None:
    """Render a compact notification strip at the top of the dashboard.

    Shows critical finding headlines as ``st.error`` banners and warning
    headlines as ``st.warning`` banners.  Clicking the expander reveals
    the full expert panel.

    Parameters
    ----------
    report:
        The current :class:`~gradient_pathology.core.GradientReport`.
    key_prefix:
        Streamlit widget key prefix.
    vanishing_threshold / exploding_threshold:
        Forwarded to :class:`~gradient_pathology.expert.engine.ExpertEngine`.
    """
    if not _ST_AVAILABLE:
        return

    engine   = ExpertEngine(
        vanishing_threshold=vanishing_threshold,
        exploding_threshold=exploding_threshold,
    )
    findings = engine.analyze(report)

    if not findings:
        return

    critical = [f for f in findings if f.severity == "critical"]
    warnings = [f for f in findings if f.severity == "warning"]

    # Compact banners
    for f in critical:
        st.error(f"{f.severity_emoji} **{f.headline}** — {len(f.affected_layers)} layer(s) affected")
    for f in warnings[:3]:  # cap at 3 to avoid clutter
        st.warning(f"{f.severity_emoji} {f.headline}")


def render_expert_panel(
    report: GradientReport,
    key_prefix: str = "expert",
    vanishing_threshold: float = 1e-7,
    exploding_threshold: float = 1e3,
    expanded: bool = False,
) -> None:
    """Render the full expert diagnostics in an expandable panel.

    Parameters
    ----------
    report:
        The current :class:`~gradient_pathology.core.GradientReport`.
    key_prefix:
        Streamlit widget key prefix.
    vanishing_threshold / exploding_threshold:
        Forwarded to the engine.
    expanded:
        Whether the panel starts expanded.  Defaults to ``False``
        (collapsed) to avoid overwhelming first-time users.
    """
    if not _ST_AVAILABLE:
        return

    engine   = ExpertEngine(
        vanishing_threshold=vanishing_threshold,
        exploding_threshold=exploding_threshold,
    )
    findings = engine.analyze(report)
    critical = [f for f in findings if f.severity == "critical"]
    n_issues = sum(1 for f in findings if f.severity != "info")

    label = (
        f"🧠 Expert Diagnosis — {n_issues} issue(s) found"
        if n_issues > 0
        else "🧠 Expert Diagnosis — all healthy"
    )

    with st.expander(label, expanded=expanded or len(critical) > 0):
        if not findings:
            st.success("✅ No issues detected by the expert engine.")
            return

        for i, f in enumerate(findings):
            _render_finding_card(f, key_prefix=f"{key_prefix}_f{i}")
            if i < len(findings) - 1:
                st.markdown("---")


def render_layer_expert_panel(
    layer_name: str,
    report: GradientReport,
    key_prefix: str = "layer_expert",
) -> None:
    """Render findings relevant to a specific *layer_name*.

    This is the **popup panel** triggered when a user clicks/selects a
    node in the Heatmap or Sankey diagrams.

    Parameters
    ----------
    layer_name:
        Fully-qualified parameter name (must match
        :attr:`~gradient_pathology.core.LayerGradientStats.layer_name`).
    report:
        Full gradient report.
    key_prefix:
        Streamlit widget key prefix.
    """
    if not _ST_AVAILABLE:
        return

    engine   = ExpertEngine()
    findings = engine.analyze_layer(layer_name, report)

    if not findings:
        st.info(f"✅ No expert findings for `{layer_name}`.")
        return

    st.markdown(f"**Expert findings for** `{layer_name}`")
    for i, f in enumerate(findings):
        _render_finding_card(f, key_prefix=f"{key_prefix}_lf{i}", compact=True)


def render_realtime_alerts(
    alerts: List[str],
    key_prefix: str = "rt_alerts",
) -> None:
    """Render real-time alert messages from the :class:`~gradient_pathology.monitor.bridge.LiveGradientBridge`.

    Parameters
    ----------
    alerts:
        List of alert strings popped from ``bridge.pop_alerts()``.
    key_prefix:
        Streamlit widget key prefix.
    """
    if not _ST_AVAILABLE or not alerts:
        return

    with st.expander(f"🔔 {len(alerts)} real-time alert(s)", expanded=True):
        for alert in alerts[-10:]:  # show last 10
            if "VANISHING" in alert or "EXPLODING" in alert:
                st.error(alert)
            else:
                st.warning(alert)


# ---------------------------------------------------------------------------
# Internal render helpers
# ---------------------------------------------------------------------------

def _render_finding_card(
    finding: ExpertFinding,
    key_prefix: str = "",
    compact: bool = False,
) -> None:
    """Render one :class:`ExpertFinding` as a styled Streamlit card."""
    # Header row
    severity_fn = {
        "critical": st.error,
        "warning":  st.warning,
        "info":     st.info,
    }.get(finding.severity, st.info)

    severity_fn(f"{finding.severity_emoji} **{finding.headline}**  "
                f"*(confidence {finding.confidence:.0%})*")

    if compact:
        # Just the top recommendation in compact mode
        if finding.recommendations:
            st.caption(f"Suggestion: {finding.recommendations[0]}")
        return

    # Detail block
    st.markdown(finding.detail)

    # Recommendations
    if finding.recommendations:
        st.markdown("**Recommendations:**")
        for rec in finding.recommendations:
            st.markdown(f"- {rec}")

    # Code snippets
    if finding.code_snippets:
        st.markdown("**Code fixes:**")
        for label, code in finding.code_snippets.items():
            st.markdown(f"*{label}*")
            st.code(code, language="python")

    # Affected layers
    if finding.affected_layers:
        with st.expander(
            f"Affected layers ({len(finding.affected_layers)})",
            expanded=False,
        ):
            for ln in finding.affected_layers:
                st.text(ln)
