"""Phase-3 core: GradientSankeyRenderer.

Converts a :class:`~gradient_pathology.sankey.flow.SankeyFlow` into a
fully interactive Plotly ``go.Sankey`` figure.

Visual design
-------------
* **Link width** encodes ``grad_norm`` — narrow = information loss.
* **Link colour** encodes :class:`~gradient_pathology.sankey.flow.FlowZone`:
    - Healthy   → semi-transparent green  ``rgba(46,204,113,0.55)``
    - Vanishing → semi-transparent red    ``rgba(231,76,60,0.70)``
    - Exploding → semi-transparent orange ``rgba(230,126,34,0.70)``
    - Bottleneck→ semi-transparent amber  ``rgba(241,196,15,0.65)``
    - Dead      → semi-transparent purple ``rgba(142,68,173,0.60)``
* **Node colour** encodes :class:`~gradient_pathology.core.LayerGroup`
  (matching the Heatmap border-ring palette from Phase 2).
* The figure uses a **dark theme** (``#0F1117``) consistent with the
  Heatmap renderer.

Interactivity
-------------
Plotly Sankey diagrams support built-in hover tooltips.  Each node and link
carries a rich ``customdata`` + ``hovertemplate`` that shows:

* Node: ``layer_name``, ``type``, ``group``, ``grad_norm``, ``pathology``
* Link: source/target names, ``grad_norm`` at both ends, ``loss_fraction``,
  ``zone``

For click-based deep-dive diagnostics the companion
:class:`~gradient_pathology.sankey.detail_panel.LayerDetailPanel` is used
from the Streamlit tab (see :mod:`gradient_pathology.sankey.dashboard_tab`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from gradient_pathology.core import GradientPathology, GradientReport, LayerGroup
from gradient_pathology.sankey.flow import (
    FlowStrategy,
    FlowZone,
    SankeyFlow,
    SankeyFlowBuilder,
    SankeyLink,
)

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Colour tables
# ---------------------------------------------------------------------------

#: Link colour per FlowZone (RGBA, semi-transparent for link overlap legibility)
ZONE_LINK_COLORS: Dict[FlowZone, str] = {
    FlowZone.HEALTHY:    "rgba(46,  204, 113, 0.55)",
    FlowZone.VANISHING:  "rgba(231, 76,  60,  0.72)",
    FlowZone.EXPLODING:  "rgba(230, 126, 34,  0.72)",
    FlowZone.BOTTLENECK: "rgba(241, 196, 15,  0.65)",
    FlowZone.DEAD:       "rgba(142, 68,  173, 0.60)",
}

#: Node fill colour per LayerGroup (matching Phase-2 Heatmap border palette)
GROUP_NODE_COLORS: Dict[LayerGroup, str] = {
    LayerGroup.ATTENTION:  "rgba(76,  155, 232, 0.85)",
    LayerGroup.FFN:        "rgba(245, 166, 35,  0.85)",
    LayerGroup.LAYER_NORM: "rgba(126, 211, 33,  0.85)",
    LayerGroup.EMBEDDING:  "rgba(155, 89,  182, 0.85)",
    LayerGroup.HEAD:       "rgba(231, 76,  60,  0.85)",
    LayerGroup.OTHER:      "rgba(149, 165, 166, 0.85)",
}

# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class GradientSankeyRenderer:
    """Render a gradient information-flow Sankey diagram.

    Parameters
    ----------
    report:
        :class:`~gradient_pathology.core.GradientReport` (must have Phase-1
        fields populated for best results).
    strategy:
        How ``grad_norm`` is mapped to link width.
    vanishing_threshold:
        Layers below this are tagged as vanishing (narrow red link).
    exploding_threshold:
        Layers above this are tagged as exploding.
    bottleneck_drop_ratio:
        Relative drop that qualifies a link as a bottleneck.
    show_group_colours:
        Colour nodes by :class:`~gradient_pathology.core.LayerGroup`.
    group_by_layer:
        Merge weight+bias pairs into single nodes (reduces clutter).
    title:
        Figure title; auto-generated when ``None``.

    Examples
    --------
    ::

        renderer = GradientSankeyRenderer(report)
        fig = renderer.build()
        renderer.show()
        renderer.save_html("gradient_flow.html")
    """

    def __init__(
        self,
        report: GradientReport,
        strategy: FlowStrategy             = FlowStrategy.LOG,
        vanishing_threshold: float         = 1e-7,
        exploding_threshold: float         = 1e3,
        bottleneck_drop_ratio: float       = 0.5,
        show_group_colours: bool           = True,
        group_by_layer: bool               = True,
        title: Optional[str]               = None,
    ) -> None:
        self.report                = report
        self.strategy              = strategy
        self.vanishing_threshold   = vanishing_threshold
        self.exploding_threshold   = exploding_threshold
        self.bottleneck_drop_ratio = bottleneck_drop_ratio
        self.show_group_colours    = show_group_colours
        self.group_by_layer        = group_by_layer
        self.title                 = title or self._auto_title()

        self._flow: Optional[SankeyFlow] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> "go.Figure":
        """Build and return an interactive Plotly ``go.Sankey`` figure.

        Raises
        ------
        ImportError
            When Plotly is not installed.
        """
        if not _PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is required for the Sankey diagram.\n"
                "Install with: pip install plotly\n"
                "or: pip install gradient-pathology[dashboard]"
            )
        self._prepare()
        return self._assemble_figure()

    def show(self) -> None:
        """Build and open the figure in the default browser."""
        self.build().show()

    def save_html(self, path: str, include_plotlyjs: str = "cdn") -> Path:
        """Save a standalone HTML file.

        Parameters
        ----------
        path:
            Output file path.
        include_plotlyjs:
            ``'cdn'`` (default, small) or ``'inline'`` (self-contained).

        Returns
        -------
        Path
        """
        fig = self.build()
        out = Path(path).resolve()
        pio.write_html(fig, str(out), include_plotlyjs=include_plotlyjs)
        return out

    @property
    def flow(self) -> SankeyFlow:
        """Return the pre-built :class:`SankeyFlow` (builds if needed)."""
        self._prepare()
        assert self._flow is not None
        return self._flow

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prepare(self) -> None:
        """Build the SankeyFlow (idempotent)."""
        if self._flow is not None:
            return
        builder = SankeyFlowBuilder(
            self.report,
            strategy=self.strategy,
            vanishing_threshold=self.vanishing_threshold,
            exploding_threshold=self.exploding_threshold,
            bottleneck_drop_ratio=self.bottleneck_drop_ratio,
            group_by_layer=self.group_by_layer,
        )
        self._flow = builder.build()

    def _assemble_figure(self) -> "go.Figure":
        flow = self._flow
        assert flow is not None

        if flow.n_nodes == 0:
            return go.Figure(layout=go.Layout(title="No layer data available"))

        # ---- Node arrays --------------------------------------------------
        node_colors  = self._node_colors(flow)
        node_hovers  = self._node_hovertexts(flow)

        # ---- Link arrays --------------------------------------------------
        sources     = [lk.source_idx for lk in flow.links]
        targets     = [lk.target_idx for lk in flow.links]
        values      = [lk.value      for lk in flow.links]
        link_colors = [ZONE_LINK_COLORS[lk.zone] for lk in flow.links]
        link_hovers = self._link_hovertexts(flow)

        # ---- Counters for subtitle ----------------------------------------
        n_vanishing  = len(flow.vanishing_links)
        n_bottleneck = len(flow.bottleneck_links)
        subtitle = (
            f"Nodes: {flow.n_nodes} | "
            f"Strategy: {flow.strategy.value} | "
            f"Vanishing links: {n_vanishing} | "
            f"Bottlenecks: {n_bottleneck}"
        )

        sankey_trace = go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18,
                thickness=22,
                line=dict(color="rgba(255,255,255,0.15)", width=0.5),
                label=flow.node_labels,
                color=node_colors,
                hovertemplate=node_hovers,
                hoverlabel=dict(
                    bgcolor="#1E2130",
                    font=dict(color="#FAFAFA", size=12),
                ),
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                hovertemplate=link_hovers,
                hoverlabel=dict(
                    bgcolor="#1E2130",
                    font=dict(color="#FAFAFA", size=12),
                ),
            ),
            textfont=dict(color="#CCCCCC", size=10),
        )

        # ---- Loss-zone annotation strip -----------------------------------
        annotations = self._build_loss_annotations(flow)

        fig = go.Figure(
            data=[sankey_trace],
            layout=go.Layout(
                title=dict(
                    text=(
                        f"<b>{self.title}</b><br>"
                        f"<sup>{subtitle}</sup>"
                    ),
                    x=0.5,
                    xanchor="center",
                    font=dict(size=16, color="#FAFAFA"),
                ),
                font=dict(color="#FAFAFA", size=11),
                plot_bgcolor="#0F1117",
                paper_bgcolor="#0F1117",
                annotations=annotations,
                margin=dict(l=20, r=20, t=90, b=20),
                height=max(500, flow.n_nodes * 28 + 140),
            ),
        )
        return fig

    # ---- Colour helpers ---------------------------------------------------

    def _node_colors(self, flow: SankeyFlow) -> List[str]:
        if not self.show_group_colours:
            return ["rgba(149,165,166,0.85)"] * flow.n_nodes
        return [GROUP_NODE_COLORS.get(g, "rgba(149,165,166,0.85)")
                for g in flow.node_groups]

    # ---- Hover text helpers -----------------------------------------------

    def _node_hovertexts(self, flow: SankeyFlow) -> str:
        """Single hovertemplate string for all nodes (Plotly syntax)."""
        # Plotly passes per-node data via customdata; we encode it in the label
        # using <br> enrichment instead since go.Sankey customdata support is
        # limited in older versions.  We build a list-based template.
        parts = []
        for i, name in enumerate(flow.node_layer_names):
            gn   = flow.node_grad_norms[i]
            path = flow.node_pathologies[i]
            grp  = flow.node_groups[i]
            parts.append(
                f"<b>{name}</b><br>"
                f"group: {grp.value}<br>"
                f"grad_norm: {gn:.3e}<br>"
                f"pathology: <b>{path.value.upper()}</b>"
                "<extra></extra>"
            )
        # Plotly Sankey accepts a single template string OR a list.
        return parts  # type: ignore[return-value]

    def _link_hovertexts(self, flow: SankeyFlow) -> List[str]:
        texts = []
        for lk in flow.links:
            src_name = flow.node_layer_names[lk.source_idx]
            dst_name = flow.node_layer_names[lk.target_idx]
            loss_pct = lk.loss_fraction * 100
            texts.append(
                f"<b>{src_name}</b> → <b>{dst_name}</b><br>"
                f"zone: <b>{lk.zone.value.upper()}</b><br>"
                f"src grad_norm: {lk.raw_source_norm:.3e}<br>"
                f"dst grad_norm: {lk.raw_target_norm:.3e}<br>"
                f"info loss: {loss_pct:.1f}%"
                "<extra></extra>"
            )
        return texts

    # ---- Loss-zone annotation strip ---------------------------------------

    def _build_loss_annotations(self, flow: SankeyFlow) -> List[dict]:
        """Build a legend annotation block listing zone colour meanings."""
        legend_items = [
            ("■", ZONE_LINK_COLORS[FlowZone.HEALTHY].replace("0.55", "1"),    "Healthy"),
            ("■", ZONE_LINK_COLORS[FlowZone.VANISHING].replace("0.72", "1"),  "Vanishing"),
            ("■", ZONE_LINK_COLORS[FlowZone.BOTTLENECK].replace("0.65", "1"), "Bottleneck"),
            ("■", ZONE_LINK_COLORS[FlowZone.EXPLODING].replace("0.72", "1"),  "Exploding"),
        ]
        annotations = []
        x_pos = 0.0
        for symbol, color, label in legend_items:
            annotations.append(dict(
                text=f'<span style="color:{color};font-size:16px">{symbol}</span> {label}',
                x=x_pos,
                y=1.06,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="#CCCCCC", size=11),
                align="left",
            ))
            x_pos += 0.18
        return annotations

    def _auto_title(self) -> str:
        n = len(self.report.layer_stats)
        return f"Gradient Information Flow \u2014 {n} layers ({self.report.data_source})"
