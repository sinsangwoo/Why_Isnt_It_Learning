"""Phase-4 Streamlit tab: Live Monitor.

Renders real-time gradient-norm and loss curves that update automatically
as the training loop pushes data into the shared :class:`LiveGradientBridge`.

Layout
------
1. **Status row** — last step, last loss, global grad-mean, alert count.
2. **Loss curve** — step vs. loss (Plotly, updates on every rerun).
3. **Grad-norm trend** — step vs. global_mean with vanishing/exploding
   threshold bands.
4. **Per-layer heatmap** — a compact horizontal bar chart of the
   latest-snapshot layer norms (top-N layers ranked by norm).
5. **Alert feed** — most recent 10 alerts from the bridge.
6. **Expert banner** — if a GradientReport is available (post-analysis).

Auto-refresh
------------
Streamlit reruns the page whenever a widget changes.  To achieve
continuous refresh while training is running the tab uses
``st.empty()`` containers and ``time.sleep`` inside a loop *only when
the user has ticked the “auto-refresh” checkbox*; otherwise a
**🔄 Refresh now** button is shown instead.  This avoids blocking the
browser when the user is not on this tab.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from gradient_pathology.monitor.bridge import LiveGradientBridge
from gradient_pathology.dashboard.expert_panel import render_expert_banner

try:
    import streamlit as st
    _ST = True
except ImportError:
    _ST = False

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

try:
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


def render_realtime_tab(
    bridge: LiveGradientBridge,
    report: Optional[object] = None,   # GradientReport | None
    key_prefix: str = "live",
    top_n_layers: int = 20,
) -> None:
    """Render the Live Monitor tab.

    Parameters
    ----------
    bridge:
        Shared :class:`LiveGradientBridge` receiving training-loop data.
    report:
        Optional :class:`~gradient_pathology.core.GradientReport` (shown
        in the Expert banner if available).
    key_prefix:
        Widget key prefix for uniqueness.
    top_n_layers:
        How many layers to show in the per-layer norm bar chart.
    """
    if not _ST:
        return

    # ---- Expert banner (if report available) ----------------------------
    if report is not None:
        render_expert_banner(report, key_prefix=f"{key_prefix}_banner")
        st.divider()

    # ---- Bridge empty state ---------------------------------------------
    if bridge.is_empty:
        st.info(
            "📶 Waiting for training data…\n\n"
            "Connect a :class:`StreamlitCallback` to your training loop:\n"
            "```python\n"
            "from gradient_pathology.monitor import StreamlitCallback, LiveGradientBridge\n"
            "bridge   = LiveGradientBridge.from_session_state()\n"
            "callback = StreamlitCallback(model, bridge)\n"
            "# … in your loop: callback.on_batch_end(step, loss)\n"
            "```"
        )
        return

    snaps = bridge.all_snapshots()
    latest = snaps[-1]

    # ---- Status row ------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Step",            str(latest.step))
    c2.metric("Loss",            f"{latest.loss:.4f}"   if math.isfinite(latest.loss) else "N/A")
    c3.metric("Grad mean",       f"{latest.global_mean:.2e}" if math.isfinite(latest.global_mean) else "N/A")
    total_alerts = sum(len(s.alerts) for s in snaps)
    c4.metric("Total alerts",    str(total_alerts),
              delta=None if total_alerts == 0 else f"+{len(latest.alerts)}",
              delta_color="inverse")

    st.divider()

    # ---- Charts ----------------------------------------------------------
    steps_loss,   losses     = bridge.metrics_series("loss")
    steps_mean,   means      = bridge.metrics_series("global_mean")

    # Filter NaN
    loss_pairs = [(s, v) for s, v in zip(steps_loss, losses)   if math.isfinite(v)]
    mean_pairs = [(s, v) for s, v in zip(steps_mean, means)    if math.isfinite(v)]

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Training Loss**")
        if _PLOTLY and loss_pairs:
            fig = _plotly_line(
                x=[p[0] for p in loss_pairs],
                y=[p[1] for p in loss_pairs],
                name="loss",
                color="#4C9BE8",
                ylog=True,
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"{key_prefix}_loss_chart")
        elif _MPL and loss_pairs:
            fig = _mpl_line(
                x=[p[0] for p in loss_pairs],
                y=[p[1] for p in loss_pairs],
                ylabel="Loss (log)",
            )
            st.pyplot(fig)
        else:
            st.caption("No loss data yet.")

    with chart_col2:
        st.markdown("**Global Gradient Norm Trend**")
        if _PLOTLY and mean_pairs:
            fig = _plotly_line_with_thresholds(
                x=[p[0] for p in mean_pairs],
                y=[p[1] for p in mean_pairs],
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"{key_prefix}_grad_chart")
        elif _MPL and mean_pairs:
            fig = _mpl_line(
                x=[p[0] for p in mean_pairs],
                y=[p[1] for p in mean_pairs],
                ylabel="grad_mean (log)",
            )
            st.pyplot(fig)
        else:
            st.caption("No gradient data yet.")

    # ---- Per-layer bar chart (latest snapshot) ---------------------------
    if latest.layer_norms:
        st.divider()
        st.markdown(f"**Per-layer gradient norms (latest step, top {top_n_layers})**")
        sorted_layers = sorted(
            latest.layer_norms.items(), key=lambda kv: kv[1], reverse=True
        )[:top_n_layers]
        names  = [kv[0].split(".")[-1][:22] for kv in sorted_layers]
        values = [kv[1] for kv in sorted_layers]

        if _PLOTLY:
            colors = [
                "#E74C3C" if v < 1e-7
                else "#E67E22" if v > 1e3
                else "#2ECC71"
                for v in values
            ]
            fig = go.Figure(go.Bar(
                x=values, y=names,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.2e}" for v in values],
                textposition="outside",
            ))
            fig.update_xaxes(type="log", gridcolor="#2A2D3A")
            fig.update_layout(
                height=max(250, len(names) * 22 + 60),
                plot_bgcolor="#0F1117",
                paper_bgcolor="#0F1117",
                font=dict(color="#CCCCCC", size=10),
                margin=dict(l=10, r=80, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"{key_prefix}_layer_bars")

    # ---- Alert feed -------------------------------------------------------
    all_alerts = []
    for s in reversed(snaps):
        all_alerts.extend(reversed(s.alerts))
        if len(all_alerts) >= 10:
            break

    if all_alerts:
        st.divider()
        st.markdown("**🔔 Recent alerts (latest first)**")
        for alert in all_alerts[:10]:
            if "Vanishing" in alert or "Exploding" in alert:
                st.error(alert)
            else:
                st.warning(alert)

    # ---- Refresh controls ------------------------------------------------
    st.divider()
    col_refresh, col_count = st.columns([2, 3])
    with col_refresh:
        if st.button("🔄 Refresh now", key=f"{key_prefix}_refresh"):
            st.rerun()
    with col_count:
        st.caption(
            f"Buffer: {len(snaps)}/{bridge._max_steps} steps · "
            f"Total pushed: {bridge.total_pushed}"
        )


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _plotly_line(
    x: list, y: list, name: str, color: str, ylog: bool = False
) -> "go.Figure":
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=color, width=1.8),
        name=name,
        fill="tozeroy",
        fillcolor=color.replace("#", "rgba(") + ",0.08)" if "#" in color else color,
    ))
    fig.update_layout(
        height=260,
        plot_bgcolor="#0F1117",
        paper_bgcolor="#0F1117",
        font=dict(color="#CCCCCC", size=10),
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False,
    )
    if ylog:
        fig.update_yaxes(type="log", gridcolor="#2A2D3A")
    else:
        fig.update_yaxes(gridcolor="#2A2D3A")
    fig.update_xaxes(gridcolor="#2A2D3A", title_text="step")
    return fig


def _plotly_line_with_thresholds(x: list, y: list) -> "go.Figure":
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color="#6ECE58", width=1.8),
        name="grad mean",
    ))
    # Vanishing band
    fig.add_hrect(y0=0, y1=1e-7,
                  fillcolor="rgba(231,76,60,0.10)",
                  line_width=0,
                  annotation_text="vanishing",
                  annotation_position="top left",
                  annotation_font_size=9,
                  annotation_font_color="#E74C3C")
    # Exploding band
    fig.add_hrect(y0=1e3, y1=max(max(y) * 2, 1e4),
                  fillcolor="rgba(230,126,34,0.10)",
                  line_width=0,
                  annotation_text="exploding",
                  annotation_position="bottom right",
                  annotation_font_size=9,
                  annotation_font_color="#E67E22")
    fig.update_layout(
        height=260,
        plot_bgcolor="#0F1117",
        paper_bgcolor="#0F1117",
        font=dict(color="#CCCCCC", size=10),
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False,
    )
    fig.update_yaxes(type="log", gridcolor="#2A2D3A")
    fig.update_xaxes(gridcolor="#2A2D3A", title_text="step")
    return fig


def _mpl_line(x: list, y: list, ylabel: str) -> "plt.Figure":
    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor="#0F1117")
    ax.set_facecolor("#0F1117")
    ax.plot(x, y, color="#4C9BE8", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel, color="#CCCCCC", fontsize=8)
    ax.set_xlabel("step", color="#CCCCCC", fontsize=8)
    ax.tick_params(colors="#CCCCCC", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.grid(True, alpha=0.2, color="#444")
    plt.tight_layout()
    return fig
