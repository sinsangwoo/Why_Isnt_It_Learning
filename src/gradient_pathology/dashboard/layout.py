"""Phase-4: Master dashboard layout — orchestrates all four tabs.

Tabs
----
1. 📊 Live Monitor   — real-time loss + grad-norm charts + alert feed
2. 🌊 Sankey Flow    — information-loss Sankey + layer deep-dive (Phase 3)
3. 🌡️  Heatmap         — architecture node-graph heatmap (Phase 2)
4. 📝 Classic Report  — original bar chart + text report (Phase 1 baseline)

Expert system integration
-------------------------
* A compact :func:`~gradient_pathology.dashboard.expert_panel.render_expert_banner`
  is rendered **above** the tabs so critical findings are always visible.
* Each tab can invoke
  :func:`~gradient_pathology.dashboard.expert_panel.render_expert_panel`
  or
  :func:`~gradient_pathology.dashboard.expert_panel.render_layer_expert_panel`
  for deeper drill-down.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch.nn as nn

from gradient_pathology.analyzer import GradientAnalyzer
from gradient_pathology.core import GradientPathology, GradientReport
from gradient_pathology.experiments import create_deep_network
from gradient_pathology.heatmap.dashboard_tab import render_heatmap_tab
from gradient_pathology.sankey.dashboard_tab import render_sankey_tab
from gradient_pathology.dashboard.expert_panel import (
    render_expert_banner,
    render_expert_panel,
)
from gradient_pathology.dashboard.realtime_tab import render_realtime_tab


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_dashboard(
    bridge: Optional[Any] = None,
) -> None:
    """Launch the Phase-4 Gradient Pathology dashboard.

    Parameters
    ----------
    bridge:
        Optional :class:`~gradient_pathology.monitor.bridge.LiveGradientBridge`
        instance.  When supplied, the *Live Monitor* tab receives real-time
        data from a running training loop.
    """
    st.set_page_config(
        page_title="Gradient Pathology Dashboard",
        page_icon="🔬",
        layout="wide",
    )

    # ── Header ───────────────────────────────────────────────────────────
    st.title("🔬 Gradient Pathology Monitor")
    st.markdown(
        """
    **Production-grade gradient diagnostics** • Phase 4: Real-time monitoring + Expert System  
    *From high school curiosity to production ML diagnostics*
    """
    )

    # ── Sidebar ──────────────────────────────────────────────────────────
    _render_sidebar(bridge)

    # ── Expert banner (always visible above tabs) ────────────────────────
    if "report" in st.session_state:
        report = st.session_state["report"]
        render_expert_banner(
            report,
            vanishing_threshold=st.session_state.get("van_threshold", 1e-7),
            exploding_threshold=st.session_state.get("exp_threshold", 1e3),
        )

    # ── Global metrics strip ────────────────────────────────────────────
    if "report" in st.session_state:
        _render_metrics_strip(st.session_state["report"])

    # ── Four-tab layout ──────────────────────────────────────────────────
    tab_live, tab_sankey, tab_heatmap, tab_classic = st.tabs([
        "📊 Live Monitor",
        "🌊 Sankey Flow",
        "🌡️ Architecture Heatmap",
        "📝 Classic Report",
    ])

    report = st.session_state.get("report")

    with tab_live:
        render_realtime_tab(
            bridge=bridge,
            static_report=report,
        )

    with tab_sankey:
        if report:
            render_sankey_tab(report)
            st.markdown("---")
            render_expert_panel(report, key_prefix="sankey_exp", expanded=False)
        else:
            st.info("👈 Run analysis first to see the Sankey diagram.")

    with tab_heatmap:
        if report:
            render_heatmap_tab(report)
            st.markdown("---")
            render_expert_panel(report, key_prefix="heatmap_exp", expanded=False)
        else:
            st.info("👈 Run analysis first to see the Heatmap.")

    with tab_classic:
        if report:
            _render_classic_tab(report)
        else:
            st.info("👈 Configure your model in the sidebar and click 'Analyze Gradients'.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(bridge: Optional[Any]) -> None:
    """Render the sidebar: model config, analysis settings, threshold controls."""
    st.sidebar.header("🛠️ Model Configuration")
    depth       = st.sidebar.slider("Network Depth",  5, 100, 20)
    activation  = st.sidebar.selectbox(
        "Activation", ["relu", "sigmoid", "tanh", "gelu"], index=0
    )
    hidden_size = st.sidebar.slider("Hidden Size", 32, 512, 64)
    use_norm    = st.sidebar.checkbox("Use LayerNorm", value=False)

    st.sidebar.header("🔍 Analysis Settings")
    num_steps = st.sidebar.slider("Gradient Samples", 10, 200, 50)

    st.sidebar.header("⚠️ Detection Thresholds")
    van_threshold = st.sidebar.select_slider(
        "Vanishing threshold",
        options=[1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
        value=1e-7,
        format_func=lambda v: f"{v:.0e}",
        key="van_threshold",
    )
    exp_threshold = st.sidebar.select_slider(
        "Exploding threshold",
        options=[1e1, 1e2, 1e3, 1e4],
        value=1e3,
        format_func=lambda v: f"{v:.0e}",
        key="exp_threshold",
    )

    if bridge is not None:
        st.sidebar.header("🟢 Live Training")
        snap = bridge.snapshot()
        st.sidebar.metric("Steps",   snap["total_steps"])
        st.sidebar.metric("Alerts",  len(snap["alerts"]))
        is_training = snap.get("is_training", False)
        st.sidebar.markdown(
            "🟢 **Training active**" if is_training else "⏹️ **Idle**"
        )
        if st.sidebar.button("🗑️ Clear bridge"):
            bridge.clear()
            st.rerun()

    if st.sidebar.button("🚀 Analyze Gradients", type="primary"):
        with st.spinner("Building model and analyzing gradients..."):
            model = create_deep_network(
                depth=depth,
                activation=activation,
                hidden_size=hidden_size,
                use_norm=use_norm,
            )
            analyzer = GradientAnalyzer(model, device="cpu")
            report   = analyzer.diagnose(num_steps=num_steps)

            st.session_state["report"]       = report
            st.session_state["model_config"] = {
                "depth": depth,
                "activation": activation,
                "hidden_size": hidden_size,
                "use_norm": use_norm,
            }
        st.rerun()


# ---------------------------------------------------------------------------
# Metrics strip
# ---------------------------------------------------------------------------

def _render_metrics_strip(report: GradientReport) -> None:
    """Render the four global KPI metrics across the top."""
    n          = len(report.layer_stats)
    n_bad      = len(report.get_problematic_layers())
    n_good     = n - n_bad
    health_pct = n_good / n * 100 if n > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Global Mean Gradient", f"{report.global_mean:.2e}")
    with c2:
        st.metric("Global Std Gradient",  f"{report.global_std:.2e}")
    with c3:
        st.metric(
            "Problematic Layers",
            f"{n_bad}/{n}",
            delta=f"-{n_bad}" if n_bad > 0 else None,
            delta_color="inverse",
        )
    with c4:
        st.metric("Health Score", f"{health_pct:.1f}%")


# ---------------------------------------------------------------------------
# Classic tab
# ---------------------------------------------------------------------------

def _render_classic_tab(report: GradientReport) -> None:
    """Render the original bar/pie chart + text report tab."""
    st.subheader("📊 Gradient Analysis")

    fig = _plot_gradient_distribution(report)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("📋 Detailed Layer-by-Layer Report"):
        st.code(report.summary(), language="text")

    render_expert_panel(report, key_prefix="classic_exp", expanded=False)

    if report.get_problematic_layers():
        st.subheader("💡 Recommendations")
        for layer in report.get_problematic_layers():
            pathology = layer.diagnose()
            if pathology == GradientPathology.VANISHING:
                st.error(
                    f"**{layer.layer_name}**: Vanishing (mean={layer.mean:.2e})\n"
                    "- Try: ReLU/GELU · He/Xavier init · LayerNorm"
                )
            elif pathology == GradientPathology.EXPLODING:
                st.error(
                    f"**{layer.layer_name}**: Exploding (mean={layer.mean:.2e})\n"
                    "- Use gradient clipping · Reduce LR · Check init"
                )
            elif pathology == GradientPathology.UNSTABLE:
                st.warning(
                    f"**{layer.layer_name}**: Unstable (std={layer.std:.2e})\n"
                    "- Consider: Gradient clipping · LayerNorm"
                )
    else:
        st.success("✅ All layers show healthy gradient flow!")

    config = st.session_state.get("model_config", {})
    if config:
        with st.expander("🏗️ Model Architecture"):
            st.json(config)


def _plot_gradient_distribution(report: GradientReport) -> "plt.Figure":
    """Reproduce the original bar + pie chart (Classic tab)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    layer_indices = [s.layer_index for s in report.layer_stats]
    layer_means   = [abs(s.mean)   for s in report.layer_stats]
    colors = [
        "red"    if s.diagnose() == GradientPathology.VANISHING  else
        "orange" if s.diagnose() == GradientPathology.EXPLODING  else
        "yellow" if s.diagnose() == GradientPathology.UNSTABLE   else
        "green"
        for s in report.layer_stats
    ]

    ax1.bar(layer_indices, layer_means, color=colors, alpha=0.7)
    ax1.axhline(y=1e-7, color="red",    linestyle="--", label="Vanishing threshold")
    ax1.axhline(y=1e2,  color="orange", linestyle="--", label="Exploding threshold")
    ax1.set_yscale("log")
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Mean Gradient (log scale)")
    ax1.set_title("Gradient Flow Across Layers")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    pathology_counts: Dict[str, int] = {}
    for stats in report.layer_stats:
        p = stats.diagnose()
        pathology_counts[p.value] = pathology_counts.get(p.value, 0) + 1

    ax2.pie(
        pathology_counts.values(),
        labels=pathology_counts.keys(),
        autopct="%1.1f%%",
        colors=["green", "red", "orange", "gray", "yellow"],
    )
    ax2.set_title("Gradient Pathology Distribution")
    plt.tight_layout()
    return fig
