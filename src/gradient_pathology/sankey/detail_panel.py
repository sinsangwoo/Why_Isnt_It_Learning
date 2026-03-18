"""Phase-3 sub-module: per-layer diagnostic detail panel.

When a user clicks a node in the Sankey diagram they should see a rich
diagnostic breakdown for that layer.  This module provides:

:class:`LayerDetailPanel`
    Builds a Plotly figure (radar + bar + text) that shows all diagnostic
    dimensions for a single :class:`~gradient_pathology.core.LayerGradientStats`
    entry.  It can also return a Streamlit-compatible dict for use in
    ``st.expander`` blocks.

Two rendering paths
-------------------

1. **Plotly** (``build_plotly``): produces a self-contained ``go.Figure``
   with a 2×2 subplot grid:

   * *Top-left*: radar chart (grad_norm, zero_ratio, |mean|, std, depth-score)
   * *Top-right*: bar chart comparing this layer vs. global mean
   * *Bottom-left*: group-peer comparison (how does this layer rank within
     its LayerGroup?)
   * *Bottom-right*: textual diagnosis card

2. **Dict** (``build_dict``): plain Python dict suitable for rendering in a
   Streamlit ``st.columns`` layout without Plotly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from gradient_pathology.core import (
    GradientPathology,
    GradientReport,
    LayerGradientStats,
    LayerGroup,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------

_PATHOLOGY_COLORS = {
    GradientPathology.HEALTHY:      "#2ECC71",
    GradientPathology.VANISHING:    "#E74C3C",
    GradientPathology.EXPLODING:    "#E67E22",
    GradientPathology.DEAD_NEURONS: "#8E44AD",
    GradientPathology.UNSTABLE:     "#F39C12",
}

_PATHOLOGY_ADVICE = {
    GradientPathology.HEALTHY: (
        "✅ Gradient flow is healthy.",
        [],
    ),
    GradientPathology.VANISHING: (
        "🔴 Vanishing gradients detected.",
        [
            "Use ReLU, GELU, or Swish activation instead of Sigmoid/Tanh.",
            "Apply He/Kaiming initialisation for ReLU layers.",
            "Add LayerNorm or BatchNorm before or after this layer.",
            "Increase learning rate specifically for this layer (layer-wise LR).",
            "Consider residual connections to bypass this layer.",
        ],
    ),
    GradientPathology.EXPLODING: (
        "🟠 Exploding gradients detected.",
        [
            "Apply gradient clipping: torch.nn.utils.clip_grad_norm_(params, max_norm=1.0).",
            "Reduce the global learning rate.",
            "Verify weight initialisation scale (use Xavier for tanh layers).",
            "Check for unusually large inputs or labels.",
        ],
    ),
    GradientPathology.DEAD_NEURONS: (
        "🟣 Dead neurons detected (>90% zero gradients).",
        [
            "Switch from ReLU to Leaky ReLU (negative_slope=0.01) or GELU.",
            "Check for very large negative bias values.",
            "Reduce learning rate — dead neurons often result from a too-large LR.",
            "Consider re-initialising this layer's weights.",
        ],
    ),
    GradientPathology.UNSTABLE: (
        "🟡 Unstable gradients detected (high variance).",
        [
            "Apply gradient clipping as a stabiliser.",
            "Add LayerNorm to regularise the activation distribution.",
            "Use a learning rate warm-up schedule.",
            "Reduce batch size to lower gradient noise.",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LayerDetailPanel:
    """Build detailed diagnostic views for a single layer.

    Parameters
    ----------
    report:
        Full :class:`~gradient_pathology.core.GradientReport`.
    """

    def __init__(self, report: GradientReport) -> None:
        self.report    = report
        self._name_map = {s.layer_name: s for s in report.layer_stats}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_stats(self, layer_name: str) -> Optional[LayerGradientStats]:
        """Return :class:`LayerGradientStats` for *layer_name*, or ``None``."""
        return self._name_map.get(layer_name)

    def build_dict(self, layer_name: str) -> Dict[str, Any]:
        """Return a plain dict with all diagnostic data for *layer_name*.

        Keys
        ----
        ``found``          bool — whether the layer was in the report.
        ``layer_name``     str
        ``layer_type``     str
        ``group``          str (LayerGroup.value)
        ``grad_norm``      float
        ``mean``           float
        ``std``            float
        ``zero_ratio``     float
        ``depth``          int
        ``pathology``      str (GradientPathology.value)
        ``pathology_color`` str — hex colour
        ``headline``       str — one-line human summary
        ``recommendations`` list[str]
        ``peer_rank``       int — rank within same LayerGroup (1 = best)
        ``peer_count``      int — number of layers in same group
        ``global_rank``     int — rank across all layers by grad_norm
        ``global_count``    int
        """
        s = self.get_stats(layer_name)
        if s is None:
            return {"found": False, "layer_name": layer_name}

        pathology          = s.diagnose()
        headline, recs     = _PATHOLOGY_ADVICE.get(pathology, ("", []))
        color              = _PATHOLOGY_COLORS.get(pathology, "#95A5A6")
        gn                 = float(getattr(s, "grad_norm", abs(s.mean)))
        peer_rank, peer_n  = self._peer_rank(s)
        global_rank, glob_n = self._global_rank(s)

        return {
            "found":           True,
            "layer_name":      s.layer_name,
            "layer_type":      s.layer_type,
            "group":           s.group.value,
            "grad_norm":       gn,
            "mean":            s.mean,
            "std":             s.std,
            "zero_ratio":      s.zero_ratio,
            "depth":           s.depth,
            "pathology":       pathology.value,
            "pathology_color": color,
            "headline":        headline,
            "recommendations": recs,
            "peer_rank":       peer_rank,
            "peer_count":      peer_n,
            "global_rank":     global_rank,
            "global_count":    glob_n,
        }

    def build_plotly(self, layer_name: str) -> "go.Figure":
        """Build a 2×2 Plotly subplot figure for *layer_name*.

        Raises
        ------
        ImportError
            When Plotly is not installed.
        ValueError
            When *layer_name* is not found in the report.
        """
        if not _PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is required for LayerDetailPanel.build_plotly().\n"
                "Install with: pip install plotly"
            )
        d = self.build_dict(layer_name)
        if not d.get("found"):
            raise ValueError(f"Layer '{layer_name}' not found in report.")
        return self._assemble_plotly(d)

    def all_layer_names(self) -> List[str]:
        """Return all layer names in depth order."""
        return [s.layer_name
                for s in sorted(self.report.layer_stats, key=lambda x: x.depth)]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _peer_rank(self, stats: LayerGradientStats) -> tuple:
        """Return (rank, total) within the same LayerGroup by grad_norm (descending)."""
        peers = [
            s for s in self.report.layer_stats
            if s.group == stats.group
        ]
        peers_sorted = sorted(
            peers,
            key=lambda s: float(getattr(s, "grad_norm", abs(s.mean))),
            reverse=True,
        )
        gn = float(getattr(stats, "grad_norm", abs(stats.mean)))
        for i, p in enumerate(peers_sorted):
            if p.layer_name == stats.layer_name:
                return i + 1, len(peers_sorted)
        return len(peers_sorted), len(peers_sorted)

    def _global_rank(self, stats: LayerGradientStats) -> tuple:
        """Return (rank, total) across all layers by grad_norm (descending)."""
        all_sorted = sorted(
            self.report.layer_stats,
            key=lambda s: float(getattr(s, "grad_norm", abs(s.mean))),
            reverse=True,
        )
        for i, s in enumerate(all_sorted):
            if s.layer_name == stats.layer_name:
                return i + 1, len(all_sorted)
        return len(all_sorted), len(all_sorted)

    def _assemble_plotly(self, d: Dict[str, Any]) -> "go.Figure":
        """Build the 2×2 Plotly subplot figure from pre-computed dict *d*."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Gradient Health Radar",
                "Layer vs Global Mean",
                f"Rank within {d['group'].title()} group",
                "Diagnosis",
            ],
            specs=[
                [{"type": "polar"},    {"type": "bar"}],
                [{"type": "bar"},      {"type": "table"}],
            ],
            vertical_spacing=0.14,
            horizontal_spacing=0.10,
        )

        color = d["pathology_color"]

        # ---- 1. Radar chart -----------------------------------------------
        radar_cats = [
            "grad_norm (log)",
            "1 - zero_ratio",
            "1 - |mean| (norm)",
            "stability (1-std)",
            "depth score",
        ]
        all_norms   = [float(getattr(s, "grad_norm", abs(s.mean)))
                       for s in self.report.layer_stats]
        max_depth   = max((s.depth for s in self.report.layer_stats), default=1)
        gn_log_norm = float(
            np.clip(
                (np.log10(d["grad_norm"] + 1e-12) - np.log10(min(all_norms) + 1e-12))
                / (np.log10(max(all_norms) + 1e-12) - np.log10(min(all_norms) + 1e-12) + 1e-9),
                0, 1,
            )
        )
        zero_health = 1.0 - float(d["zero_ratio"])
        mean_health = float(np.clip(
            1.0 - abs(d["mean"]) / (max(abs(s.mean) for s in self.report.layer_stats) + 1e-12),
            0, 1,
        ))
        std_health  = float(np.clip(
            1.0 - d["std"] / (max(s.std for s in self.report.layer_stats) + 1e-12),
            0, 1,
        ))
        depth_score = 1.0 - float(d["depth"]) / max(max_depth, 1)

        radar_vals = [gn_log_norm, zero_health, mean_health, std_health, depth_score]
        radar_vals_closed = radar_vals + [radar_vals[0]]
        radar_cats_closed = radar_cats + [radar_cats[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=radar_vals_closed,
                theta=radar_cats_closed,
                fill="toself",
                fillcolor=color.replace("#", "rgba(") + ",0.20)" if color.startswith("#") else color,
                line=dict(color=color, width=2),
                name="health",
            ),
            row=1, col=1,
        )

        # ---- 2. Bar: this layer vs global mean ---------------------------
        global_mean_gn = float(np.mean(all_norms))
        fig.add_trace(
            go.Bar(
                x=["This layer", "Global mean"],
                y=[d["grad_norm"], global_mean_gn],
                marker_color=[color, "#4A90D9"],
                text=[f"{d['grad_norm']:.2e}", f"{global_mean_gn:.2e}"],
                textposition="outside",
                showlegend=False,
            ),
            row=1, col=2,
        )
        fig.update_yaxes(type="log", row=1, col=2,
                         gridcolor="#2A2D3A", title_text="grad_norm")
        fig.update_xaxes(row=1, col=2, tickfont=dict(color="#CCCCCC"))

        # ---- 3. Bar: rank within peer group ------------------------------
        peer_layers = [
            s for s in self.report.layer_stats
            if s.group.value == d["group"]
        ]
        peer_norms  = [
            float(getattr(s, "grad_norm", abs(s.mean)))
            for s in peer_layers
        ]
        peer_names  = [_short_peer_label(s.layer_name) for s in peer_layers]
        peer_colors = [
            color if s.layer_name == d["layer_name"] else "#4A90D9"
            for s in peer_layers
        ]
        fig.add_trace(
            go.Bar(
                x=peer_names,
                y=peer_norms,
                marker_color=peer_colors,
                showlegend=False,
            ),
            row=2, col=1,
        )
        fig.update_yaxes(type="log", row=2, col=1,
                         gridcolor="#2A2D3A", title_text="grad_norm")
        fig.update_xaxes(row=2, col=1,
                         tickangle=-30, tickfont=dict(color="#CCCCCC", size=8))

        # ---- 4. Diagnosis table ------------------------------------------
        rec_text = "<br>".join(
            f"• {r}" for r in d["recommendations"]
        ) or "No action required."
        diag_text = (
            f"<b>Layer:</b> {d['layer_name']}<br>"
            f"<b>Type:</b>  {d['layer_type']}<br>"
            f"<b>Group:</b> {d['group']}<br>"
            f"<b>Depth:</b> {d['depth']}<br>"
            f"<b>Pathology:</b> "
            f"<span style='color:{color}'>{d['pathology'].upper()}</span><br>"
            f"<b>grad_norm:</b> {d['grad_norm']:.3e}<br>"
            f"<b>zero_ratio:</b> {d['zero_ratio']:.2%}<br>"
            f"<b>Peer rank:</b> {d['peer_rank']} / {d['peer_count']}<br>"
            f"<b>Global rank:</b> {d['global_rank']} / {d['global_count']}<br><br>"
            f"<b>Recommendations:</b><br>{rec_text}"
        )
        fig.add_trace(
            go.Table(
                header=dict(
                    values=[f"<b>{d['headline']}</b>"],
                    fill_color="#1E2130",
                    font=dict(color=color, size=12),
                    align="left",
                    height=32,
                ),
                cells=dict(
                    values=[[diag_text]],
                    fill_color="#141720",
                    font=dict(color="#CCCCCC", size=11),
                    align="left",
                    height=220,
                ),
            ),
            row=2, col=2,
        )

        # ---- Layout ----------------------------------------------------------
        fig.update_layout(
            title=dict(
                text=f"<b>Layer Diagnostics:</b> {d['layer_name']}",
                font=dict(size=14, color="#FAFAFA"),
                x=0.5,
            ),
            polar=dict(
                bgcolor="#141720",
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="#333", tickfont=dict(color="#888", size=8)),
                angularaxis=dict(tickfont=dict(color="#CCCCCC", size=9),
                                 gridcolor="#333"),
            ),
            plot_bgcolor="#0F1117",
            paper_bgcolor="#0F1117",
            font=dict(color="#FAFAFA", size=11),
            height=620,
            showlegend=False,
            margin=dict(l=20, r=20, t=70, b=20),
        )
        # Dark axis styling for bar subplots
        for row, col in [(1, 2), (2, 1)]:
            fig.update_xaxes(gridcolor="#2A2D3A", row=row, col=col)

        return fig


def _short_peer_label(name: str, max_len: int = 14) -> str:
    parts = name.split(".")
    label = parts[-1] if parts else name
    return label if len(label) <= max_len else label[:max_len - 1] + "\u2026"
