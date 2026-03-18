"""Streamlit helper: render the Heatmap tab inside the existing dashboard.

Usage inside ``dashboard.py``::

    from gradient_pathology.heatmap.dashboard_tab import render_heatmap_tab

    with tab_heatmap:
        render_heatmap_tab(report)
"""

from __future__ import annotations

from typing import Optional

from gradient_pathology.core import GradientReport
from gradient_pathology.heatmap.colormap import ColorScheme
from gradient_pathology.heatmap.layout import LayoutStrategy
from gradient_pathology.heatmap.renderer import (
    GradientHeatmapRenderer,
    VANISHING_THRESHOLD,
    EXPLODING_THRESHOLD,
)

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


def render_heatmap_tab(
    report: GradientReport,
    key_prefix: str = "heatmap",
) -> None:
    """Render the full interactive Heatmap tab in the Streamlit dashboard.

    The tab exposes user-controllable parameters in an expander so the main
    chart area stays uncluttered:

    * Colormap selector (Viridis / RdYlGn)
    * Layout strategy (Sequential / Grouped / Spring)
    * Vanishing threshold slider
    * Show/hide edges toggle

    Parameters
    ----------
    report:
        The :class:`~gradient_pathology.core.GradientReport` to visualise.
    key_prefix:
        Streamlit widget key prefix to avoid key collisions when the function
        is called multiple times in the same app.
    """
    if not _ST_AVAILABLE:
        raise ImportError("streamlit must be installed to use render_heatmap_tab.")

    if not report.layer_stats:
        st.info("No layer statistics available — run Analyze first.")
        return

    # ---- Controls -----------------------------------------------------------
    with st.expander("⚙️ Heatmap settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            scheme_choice = st.selectbox(
                "Colormap",
                options=["Viridis", "RdYlGn"],
                index=0,
                key=f"{key_prefix}_scheme",
                help="Viridis = intensity (dark→bright). RdYlGn = health (red=bad, green=healthy).",
            )
            scheme = ColorScheme.VIRIDIS if scheme_choice == "Viridis" else ColorScheme.RDYLGN

        with col2:
            layout_choice = st.selectbox(
                "Layout",
                options=["Sequential", "Grouped", "Spring"],
                index=0,
                key=f"{key_prefix}_layout",
                help="Sequential = vertical stack. Grouped = columns by layer type. Spring = force-directed.",
            )
            strategy_map = {
                "Sequential": LayoutStrategy.SEQUENTIAL,
                "Grouped":    LayoutStrategy.GROUPED,
                "Spring":     LayoutStrategy.SPRING,
            }
            strategy = strategy_map[layout_choice]

        with col3:
            show_edges = st.checkbox(
                "Show gradient flow edges",
                value=True,
                key=f"{key_prefix}_edges",
            )

        van_thresh = st.select_slider(
            "Vanishing threshold",
            options=[1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            value=VANISHING_THRESHOLD,
            format_func=lambda v: f"{v:.0e}",
            key=f"{key_prefix}_van_thresh",
        )

    # ---- Render -------------------------------------------------------------
    renderer = GradientHeatmapRenderer(
        report,
        scheme=scheme,
        layout_strategy=strategy,
        vanishing_threshold=van_thresh,
        show_edges=show_edges,
    )

    if _PLOTLY_AVAILABLE:
        fig = renderer.build()
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")
    else:
        st.warning(
            "Plotly is not installed — showing static Matplotlib fallback.\n"
            "Install with: `pip install plotly`"
        )
        mpl_fig = renderer.build_static()
        st.pyplot(mpl_fig)

    # ---- Warning summary ----------------------------------------------------
    vanishing_layers = [
        s for s in report.layer_stats
        if s.diagnose().value == "vanishing"
    ]
    exploding_layers = [
        s for s in report.layer_stats
        if s.diagnose().value == "exploding"
    ]

    if vanishing_layers or exploding_layers:
        st.markdown("---")
        st.subheader("⚠️ Detected Issues")

    if vanishing_layers:
        with st.expander(
            f"🔴 Vanishing gradients — {len(vanishing_layers)} layers",
            expanded=True,
        ):
            for s in vanishing_layers:
                gn = getattr(s, "grad_norm", abs(s.mean))
                st.error(
                    f"**{s.layer_name}** | type={s.layer_type} | "
                    f"group={s.group.value} | grad_norm={gn:.2e}\n\n"
                    "Suggestions: Add LayerNorm · Use GELU/ReLU · He init"
                )

    if exploding_layers:
        with st.expander(
            f"🟠 Exploding gradients — {len(exploding_layers)} layers",
            expanded=True,
        ):
            for s in exploding_layers:
                gn = getattr(s, "grad_norm", abs(s.mean))
                st.warning(
                    f"**{s.layer_name}** | type={s.layer_type} | "
                    f"group={s.group.value} | grad_norm={gn:.2e}\n\n"
                    "Suggestions: Gradient clipping · Reduce LR · Check init"
                )
