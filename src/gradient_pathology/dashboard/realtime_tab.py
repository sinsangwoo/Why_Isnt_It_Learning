"""Phase-4: Real-time monitoring tab for the Streamlit dashboard.

This tab renders a **live training dashboard** that updates as the
training loop pushes data to a :class:`~gradient_pathology.monitor.bridge.LiveGradientBridge`:

* Loss curve (rolling window).
* Per-layer gradient norm trend chart (top N most volatile layers).
* Alert feed from the bridge.
* Health score sparkline.
* Auto-refresh control (uses ``st.rerun`` on a timer).

When no live bridge is attached (static mode) the tab falls back to
showing the most recent :class:`~gradient_pathology.core.GradientReport`
passed explicitly.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from gradient_pathology.core import GradientReport

try:
    import streamlit as st
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


def render_realtime_tab(
    bridge: Optional[Any] = None,
    static_report: Optional[GradientReport] = None,
    key_prefix: str = "rt",
    auto_refresh_secs: int = 3,
) -> None:
    """Render the live monitoring tab.

    Parameters
    ----------
    bridge:
        A :class:`~gradient_pathology.monitor.bridge.LiveGradientBridge`
        instance.  When ``None`` the tab operates in static mode using
        *static_report*.
    static_report:
        Fallback :class:`~gradient_pathology.core.GradientReport` shown
        when *bridge* is ``None`` or has no data yet.
    key_prefix:
        Streamlit widget key prefix.
    auto_refresh_secs:
        Seconds between auto-refresh cycles when training is active.
    """
    if not _ST_AVAILABLE:
        return

    # ── Pull data from bridge or session_state ────────────────────────────
    snap: Optional[Dict[str, Any]] = None

    if bridge is not None:
        snap = bridge.snapshot()
    else:
        # Try session_state (populated by StreamlitCallback.inject)
        if "live_steps" in st.session_state:
            snap = {
                "steps":          st.session_state.get("live_steps", []),
                "losses":         st.session_state.get("live_losses", []),
                "grad_snapshots": st.session_state.get("live_grad_snapshots", []),
                "current_report": st.session_state.get("live_report"),
                "alerts":         st.session_state.get("live_alerts", []),
                "is_training":    st.session_state.get("live_is_training", False),
                "total_steps":    st.session_state.get("live_total_steps", 0),
            }

    has_live_data = snap is not None and len(snap.get("steps", [])) > 0
    report        = (
        snap.get("current_report") if snap else None
    ) or static_report

    # ── Status bar ───────────────────────────────────────────────────────
    is_training = snap.get("is_training", False) if snap else False
    total_steps = snap.get("total_steps", 0)     if snap else 0

    sc1, sc2, sc3 = st.columns([1, 1, 2])
    with sc1:
        if is_training:
            st.success("🟢 Training active")
        elif has_live_data:
            st.info("⏹️ Training completed")
        else:
            st.warning("⏳ Awaiting training data...")
    with sc2:
        st.metric("Steps recorded", total_steps)
    with sc3:
        if is_training:
            refresh_interval = st.slider(
                "Auto-refresh (s)",
                min_value=1, max_value=30,
                value=auto_refresh_secs,
                key=f"{key_prefix}_refresh",
            )
        else:
            refresh_interval = None

    # ── Real-time alerts ─────────────────────────────────────────────────
    if snap:
        alerts = snap.get("alerts", [])
        if alerts:
            from gradient_pathology.dashboard.expert_panel import render_realtime_alerts
            render_realtime_alerts(alerts, key_prefix=key_prefix)

    # ── No live data yet — show static report if available ───────────────
    if not has_live_data:
        if report is not None:
            st.info(
                "👆 No live training data yet. Showing the most recent snapshot analysis below.\n"
                "Connect :class:`StreamlitCallback` to your training loop to enable live updates."
            )
            _render_static_summary(report)
        else:
            _render_setup_guide()
        return

    # ── Live charts ──────────────────────────────────────────────────────
    assert snap is not None
    steps           = snap["steps"]
    losses          = snap["losses"]
    grad_snapshots  = snap["grad_snapshots"]

    if _PLOTLY_AVAILABLE:
        _render_live_plotly(steps, losses, grad_snapshots, key_prefix)
    else:
        _render_live_matplotlib(steps, losses, grad_snapshots, key_prefix)

    # ── Expert panel from live report ────────────────────────────────────
    if report is not None:
        st.markdown("---")
        from gradient_pathology.dashboard.expert_panel import render_expert_panel
        render_expert_panel(report, key_prefix=f"{key_prefix}_exp", expanded=False)

    # ── Auto-refresh ─────────────────────────────────────────────────────
    if is_training and refresh_interval:
        time.sleep(refresh_interval)
        st.rerun()


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _render_live_plotly(
    steps: List[int],
    losses: List[float],
    grad_snapshots: List[Dict],
    key_prefix: str,
) -> None:
    """Render live loss + gradient-norm charts using Plotly."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Training Loss", "Layer Gradient Norms (top 8)"],
        horizontal_spacing=0.08,
    )

    # Loss curve
    if losses:
        fig.add_trace(
            go.Scatter(
                x=list(steps),
                y=list(losses),
                mode="lines",
                name="loss",
                line=dict(color="#4C9BE8", width=2),
            ),
            row=1, col=1,
        )

    # Per-layer grad-norm trends (pick top 8 layers by variance)
    if grad_snapshots:
        all_layers = list(grad_snapshots[-1].keys())

        # Compute variance of mean-grad across history for each layer
        layer_variance: Dict[str, float] = {}
        for name in all_layers:
            series = [
                snap[name]["mean"] for snap in grad_snapshots
                if name in snap
            ]
            layer_variance[name] = float(
                __import__("numpy").var(series) if series else 0.0
            )

        top_layers = sorted(
            all_layers,
            key=lambda n: layer_variance.get(n, 0.0),
            reverse=True,
        )[:8]

        palette = [
            "#4C9BE8", "#F5A623", "#7ED321", "#E74C3C",
            "#9B59B6", "#2ECC71", "#E67E22", "#95A5A6",
        ]
        for j, name in enumerate(top_layers):
            series = [
                snap[name]["mean"] for snap in grad_snapshots
                if name in snap
            ]
            fig.add_trace(
                go.Scatter(
                    x=list(steps)[-len(series):],
                    y=series,
                    mode="lines",
                    name=name.split(".")[-1][:20],
                    line=dict(color=palette[j % len(palette)], width=1.5),
                    opacity=0.85,
                ),
                row=1, col=2,
            )

    fig.update_yaxes(type="log", row=1, col=2, gridcolor="#2A2D3A")
    fig.update_layout(
        height=360,
        plot_bgcolor="#0F1117",
        paper_bgcolor="#0F1117",
        font=dict(color="#FAFAFA", size=11),
        legend=dict(
            bgcolor="#1A1D24",
            font=dict(size=9, color="#CCCCCC"),
            orientation="v",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor="#2A2D3A")
    fig.update_yaxes(gridcolor="#2A2D3A", row=1, col=1)

    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_live_chart")


def _render_live_matplotlib(
    steps: List[int],
    losses: List[float],
    grad_snapshots: List[Dict],
    key_prefix: str,
) -> None:
    """Matplotlib fallback for the live charts."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), facecolor="#0F1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0F1117")
        ax.tick_params(colors="#CCCCCC")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if losses:
        ax1.plot(steps, losses, color="#4C9BE8", linewidth=1.5)
        ax1.set_title("Training Loss", color="#CCCCCC")
        ax1.set_yscale("log")

    if grad_snapshots:
        all_layers = list(grad_snapshots[-1].keys())[:8]
        for name in all_layers:
            series = [snap[name]["mean"] for snap in grad_snapshots if name in snap]
            ax2.plot(steps[-len(series):], series,
                     label=name.split(".")[-1][:15], linewidth=1)
        ax2.set_title("Layer Grad Norms", color="#CCCCCC")
        ax2.set_yscale("log")
        ax2.legend(fontsize=7, labelcolor="#CCCCCC", facecolor="#1A1D24")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_static_summary(report: GradientReport) -> None:
    """Render a compact summary from a static report (no live data)."""
    n        = len(report.layer_stats)
    n_bad    = len(report.get_problematic_layers())
    n_good   = n - n_bad
    health   = n_good / n * 100 if n > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total layers",       n)
    c2.metric("Healthy",            f"{n_good} ({health:.0f}%)")
    c3.metric("Problematic",        n_bad, delta=f"-{n_bad}" if n_bad > 0 else None,
              delta_color="inverse")

    from gradient_pathology.dashboard.expert_panel import render_expert_banner
    render_expert_banner(report)


def _render_setup_guide() -> None:
    """Show an onboarding guide when no data is available yet."""
    st.markdown("""
**Connect your training loop to see live charts.**

```python
from gradient_pathology.monitor import LiveGradientBridge, StreamlitCallback

bridge   = LiveGradientBridge(max_steps=300)
callback = StreamlitCallback(model, bridge=bridge)

for step, (x, y) in enumerate(loader):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    callback.on_batch_end(optimizer, loss=loss.item(), step=step)

callback.on_train_end()
```

Or add to a HuggingFace `Trainer`:

```python
trainer = Trainer(
    model=model,
    callbacks=[StreamlitCallback.as_hf_callback(bridge=bridge)],
)
```
""")
