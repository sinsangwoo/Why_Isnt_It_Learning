"""Streamlit helper: render the Sankey tab inside the existing dashboard.

Usage inside ``dashboard.py``::

    from gradient_pathology.sankey.dashboard_tab import render_sankey_tab

    with tab_sankey:
        render_sankey_tab(report)

Interactivity model
-------------------
Because Streamlit does not natively forward Plotly click events back to
Python, we use a **selectbox** that mirrors the Sankey nodes: the user can
pick any layer from a dropdown and instantly see the
:class:`~gradient_pathology.sankey.detail_panel.LayerDetailPanel` for it.

This gives a full diagnostic deep-dive on any layer the Sankey highlights.
The selectbox is pre-seeded with the *worst* layer (highest loss_fraction)
so there is always something meaningful shown on first render.
"""

from __future__ import annotations

from typing import Optional

from gradient_pathology.core import GradientReport
from gradient_pathology.sankey.flow import FlowStrategy
from gradient_pathology.sankey.renderer import GradientSankeyRenderer
from gradient_pathology.sankey.detail_panel import LayerDetailPanel

try:
    import streamlit as st
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    import plotly.graph_objects as go  # noqa: F401
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


def render_sankey_tab(
    report: GradientReport,
    key_prefix: str = "sankey",
) -> None:
    """Render the full Sankey tab in the Streamlit dashboard.

    Layout
    ------
    1. **Settings expander** — flow strategy, thresholds, toggles.
    2. **Sankey figure** (Plotly, full width).
    3. **Info-loss summary** — bottleneck / vanishing counts + table.
    4. **Layer detail panel** — selectbox-driven deep-dive with a 2×2
       subplot (radar, bar, peer-rank, diagnosis card).

    Parameters
    ----------
    report:
        :class:`~gradient_pathology.core.GradientReport` to visualise.
    key_prefix:
        Streamlit widget key prefix (avoids collisions in multi-tab apps).
    """
    if not _ST_AVAILABLE:
        raise ImportError("streamlit must be installed to use render_sankey_tab.")

    if not report.layer_stats:
        st.info("No layer statistics available — run Analyze first.")
        return

    # ── 1. Settings ────────────────────────────────────────────────────────
    with st.expander("⚙️ Sankey settings", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            strategy_choice = st.selectbox(
                "Flow strategy",
                options=["LOG", "NORMALISED", "RELATIVE", "SQRT", "RAW"],
                index=0,
                key=f"{key_prefix}_strategy",
                help=(
                    "LOG = log₁₀ normalised (best general purpose). "
                    "RELATIVE = fraction of peak. "
                    "NORMALISED = linear min-max."
                ),
            )
            strategy = FlowStrategy[strategy_choice]

        with col2:
            van_thresh = st.select_slider(
                "Vanishing threshold",
                options=[1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
                value=1e-7,
                format_func=lambda v: f"{v:.0e}",
                key=f"{key_prefix}_van_thresh",
            )
            bottleneck_ratio = st.slider(
                "Bottleneck drop ratio",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
                key=f"{key_prefix}_bn_ratio",
                help="Relative flow drop that qualifies as a bottleneck.",
            )

        with col3:
            group_by_layer = st.checkbox(
                "Merge weight/bias pairs",
                value=True,
                key=f"{key_prefix}_merge",
                help="Combine weight+bias parameters of the same module into one node.",
            )
            show_group_colors = st.checkbox(
                "Colour nodes by group",
                value=True,
                key=f"{key_prefix}_group_colors",
            )

    # ── 2. Build renderer + flow ───────────────────────────────────────────
    renderer = GradientSankeyRenderer(
        report,
        strategy=strategy,
        vanishing_threshold=van_thresh,
        bottleneck_drop_ratio=bottleneck_ratio,
        show_group_colours=show_group_colors,
        group_by_layer=group_by_layer,
    )

    if not _PLOTLY_AVAILABLE:
        st.warning(
            "Plotly is not installed — cannot render Sankey diagram.\n"
            "Install with: `pip install plotly`"
        )
        return

    fig = renderer.build()
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")

    # ── 3. Info-loss summary ───────────────────────────────────────────────
    flow = renderer.flow
    n_van  = len(flow.vanishing_links)
    n_bn   = len(flow.bottleneck_links)
    max_lf = flow.max_loss_fraction

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Vanishing links",  n_van,  delta=None if n_van  == 0 else f"-{n_van}",  delta_color="inverse")
    m2.metric("Bottleneck links", n_bn,   delta=None if n_bn   == 0 else f"-{n_bn}",   delta_color="inverse")
    m3.metric("Max info loss",    f"{max_lf * 100:.1f}%")

    # Bottleneck table
    if flow.bottleneck_links or flow.vanishing_links:
        st.subheader("⚠️ Critical Information-Loss Zones")
        issue_links = [
            lk for lk in flow.links
            if lk.zone.value in ("vanishing", "bottleneck", "exploding")
        ]
        for lk in sorted(issue_links, key=lambda l: l.loss_fraction, reverse=True)[:8]:
            src  = flow.node_layer_names[lk.source_idx]
            dst  = flow.node_layer_names[lk.target_idx]
            pct  = lk.loss_fraction * 100
            zone = lk.zone.value.upper()
            color_map = {
                "VANISHING":  "error",
                "BOTTLENECK": "warning",
                "EXPLODING":  "error",
            }
            fn = getattr(st, color_map.get(zone, "info"))
            fn(
                f"**{zone}** | `{src}` → `{dst}` | "
                f"info loss: **{pct:.1f}%** | "
                f"src_norm: `{lk.raw_source_norm:.2e}` | "
                f"dst_norm: `{lk.raw_target_norm:.2e}`"
            )

    # ── 4. Layer detail panel ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Layer Deep-Dive")
    st.caption(
        "Select any layer to see its full diagnostic breakdown. "
        "The default selection is the layer with the highest information loss."
    )

    panel = LayerDetailPanel(report)
    all_names = panel.all_layer_names()

    # Pre-select the worst layer by loss_fraction
    default_name = _worst_layer_name(flow, all_names)
    default_idx  = all_names.index(default_name) if default_name in all_names else 0

    selected = st.selectbox(
        "Layer",
        options=all_names,
        index=default_idx,
        key=f"{key_prefix}_layer_select",
        format_func=lambda n: f"{n.split('.')[-1]}  —  {n}",
    )

    if selected:
        d = panel.build_dict(selected)
        if d.get("found"):
            # Show the Plotly 2×2 panel
            detail_fig = panel.build_plotly(selected)
            st.plotly_chart(detail_fig, use_container_width=True,
                            key=f"{key_prefix}_detail")
        else:
            st.info(f"Layer '{selected}' not found in report.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _worst_layer_name(
    flow: "SankeyFlow",  # type: ignore[name-defined]
    all_names: list,
) -> Optional[str]:
    """Return the name of the layer with the highest loss_fraction."""
    if not flow.links:
        return all_names[0] if all_names else None
    worst_link = max(flow.links, key=lambda lk: lk.loss_fraction)
    idx = worst_link.target_idx
    if idx < len(flow.node_layer_names):
        return flow.node_layer_names[idx]
    return all_names[0] if all_names else None
