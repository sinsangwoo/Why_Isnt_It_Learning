"""Phase-2 core: GradientHeatmapRenderer.

Builds a fully interactive Plotly figure that overlays ``grad_norm`` intensity
(via Viridis/RdYlGn colourmap) on the model's architecture graph.

Architecture
------------
The renderer is composed of three concerns kept in separate helper modules:

* :mod:`~gradient_pathology.heatmap.colormap`  — colour mapping and thresholds
* :mod:`~gradient_pathology.heatmap.layout`    — node (x, y) placement
* this module                                  — Plotly figure assembly

Plotly is an *optional* dependency.  When it is absent,
:meth:`GradientHeatmapRenderer.build` raises :class:`ImportError` with a
helpful installation hint, and :meth:`GradientHeatmapRenderer.build_static`
falls back to a pure-Matplotlib figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from gradient_pathology.core import GradientPathology, GradientReport, LayerGroup
from gradient_pathology.heatmap.colormap import (
    COLOR_SCHEME_DEFAULT,
    GROUP_BORDER_COLORS,
    VANISHING_WARN_COLOR,
    EXPLODING_WARN_COLOR,
    ColorScheme,
    grad_norm_to_color,
    pathology_border_color,
    plotly_colorscale,
)
from gradient_pathology.heatmap.layout import (
    ArchitectureLayout,
    LayoutStrategy,
    NodeLayout,
)

# ---------------------------------------------------------------------------
# Optional imports (Plotly, Matplotlib)
# ---------------------------------------------------------------------------

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as mcm
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

# Sentinel used in colormap module init
COLOR_SCHEME_DEFAULT = ColorScheme.VIRIDIS

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

VANISHING_THRESHOLD: float = 1e-7
EXPLODING_THRESHOLD: float = 1e3


class GradientHeatmapRenderer:
    """Render a gradient-intensity Heatmap on top of the model architecture.

    Parameters
    ----------
    report:
        :class:`~gradient_pathology.core.GradientReport` produced by
        :class:`~gradient_pathology.analyzer.GradientAnalyzer`.  Every
        :class:`~gradient_pathology.core.LayerGradientStats` object **must**
        have Phase-1 fields (``grad_norm``, ``layer_type``, ``group``) set;
        if they are absent the renderer falls back to ``abs(mean)`` for
        intensity and ``LayerGroup.OTHER`` for grouping.
    scheme:
        Colormap to use for ``grad_norm`` fill colour.
        Defaults to :attr:`ColorScheme.VIRIDIS`.
    layout_strategy:
        Node placement algorithm.  Defaults to
        :attr:`LayoutStrategy.SEQUENTIAL`.
    vanishing_threshold:
        ``grad_norm`` below this value triggers a vanishing-gradient warning
        overlay and border highlight.
    exploding_threshold:
        ``grad_norm`` above this value triggers an exploding-gradient warning.
    show_edges:
        Whether to draw gradient-flow arrows between nodes.
    title:
        Figure title.  Auto-generated when ``None``.

    Examples
    --------
    ::

        from gradient_pathology import GradientAnalyzer
        from gradient_pathology.heatmap import GradientHeatmapRenderer

        model = build_my_transformer()
        analyzer = GradientAnalyzer(model)
        report = analyzer.diagnose(num_steps=50)

        renderer = GradientHeatmapRenderer(report)
        fig = renderer.build()        # Plotly Figure
        renderer.show()               # opens in browser
        renderer.save_html("out.html")

        # Matplotlib static fallback
        mpl_fig = renderer.build_static()
    """

    def __init__(
        self,
        report: GradientReport,
        scheme: ColorScheme = ColorScheme.VIRIDIS,
        layout_strategy: LayoutStrategy = LayoutStrategy.SEQUENTIAL,
        vanishing_threshold: float = VANISHING_THRESHOLD,
        exploding_threshold: float = EXPLODING_THRESHOLD,
        show_edges: bool = True,
        title: Optional[str] = None,
    ) -> None:
        self.report = report
        self.scheme = scheme
        self.layout_strategy = layout_strategy
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.show_edges = show_edges
        self.title = title or self._auto_title()

        # Pre-compute layout and colour data once.
        self._layout: Optional[ArchitectureLayout] = None
        self._all_norms: List[float] = []
        self._node_fill_colors: List[str] = []
        self._node_border_colors: List[str] = []

    # ------------------------------------------------------------------
    # Public API — Plotly
    # ------------------------------------------------------------------

    def build(self) -> "go.Figure":
        """Build and return an interactive Plotly :class:`go.Figure`.

        Raises
        ------
        ImportError
            When Plotly is not installed.
        """
        if not _PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is required for the interactive Heatmap.  "
                "Install it with:\n\n    pip install plotly\n\n"
                "or:\n\n    pip install gradient-pathology[dashboard]"
            )
        self._prepare()
        return self._assemble_plotly()

    def show(self) -> None:
        """Build the figure and open it in the default browser."""
        self.build().show()

    def save_html(
        self,
        path: str,
        include_plotlyjs: str = "cdn",
    ) -> Path:
        """Save a standalone HTML file.

        Parameters
        ----------
        path:
            Output file path (will be created / overwritten).
        include_plotlyjs:
            Passed to ``plotly.io.write_html``.  Use ``'cdn'`` (default) for
            a small file that requires internet, or ``'inline'`` for a
            fully self-contained file.

        Returns
        -------
        Path
            Absolute path of the written file.
        """
        fig = self.build()
        out = Path(path).resolve()
        pio.write_html(fig, str(out), include_plotlyjs=include_plotlyjs)
        return out

    # ------------------------------------------------------------------
    # Public API — Matplotlib static fallback
    # ------------------------------------------------------------------

    def build_static(
        self,
        figsize: tuple = (14, 9),
    ) -> "plt.Figure":
        """Build a static Matplotlib figure (no Plotly required).

        This provides a lower-fidelity but zero-extra-dependency view of the
        same Heatmap data.  Useful in environments where Plotly is unavailable
        (e.g. CI, notebooks without Plotly kernel).

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _MPL_AVAILABLE:
            raise ImportError("matplotlib is required for build_static().")

        self._prepare()
        return self._assemble_matplotlib(figsize)

    # ------------------------------------------------------------------
    # Internal helpers — data preparation
    # ------------------------------------------------------------------

    def _prepare(self) -> None:
        """Compute layout + colour arrays (idempotent)."""
        if self._layout is not None:
            return

        self._layout = ArchitectureLayout.from_report(
            self.report, strategy=self.layout_strategy
        )
        stats_map = {s.layer_index: s for s in self.report.layer_stats}

        # Resolve grad_norm — fall back to abs(mean) for pre-Phase-1 reports.
        self._all_norms = [
            _safe_grad_norm(stats_map[n.layer_index])
            for n in self._layout.nodes
        ]

        for i, node in enumerate(self._layout.nodes):
            s = stats_map[node.layer_index]
            fill = grad_norm_to_color(
                self._all_norms[i],
                self._all_norms,
                scheme=self.scheme,
                vanishing_threshold=self.vanishing_threshold,
                exploding_threshold=self.exploding_threshold,
            )
            border = pathology_border_color(s.diagnose())
            self._node_fill_colors.append(fill)
            self._node_border_colors.append(border)

    # ------------------------------------------------------------------
    # Internal helpers — Plotly assembly
    # ------------------------------------------------------------------

    def _assemble_plotly(self) -> "go.Figure":
        layout = self._layout
        assert layout is not None
        stats_map = {s.layer_index: s for s in self.report.layer_stats}

        traces: List["go.BaseTraceType"] = []

        # 1. Warning overlay rectangles for vanishing/exploding layers -------
        warn_shapes = self._build_warning_shapes(layout, stats_map)

        # 2. Edge traces (backward-pass arrows) --------------------------------
        if self.show_edges:
            edge_trace = self._build_edge_trace(layout)
            if edge_trace is not None:
                traces.append(edge_trace)

        # 3. Node scatter trace -----------------------------------------------
        node_trace = self._build_node_trace(layout, stats_map)
        traces.append(node_trace)

        # 4. Colorbar dummy trace (invisible, just for the colorbar) ----------
        colorbar_trace = self._build_colorbar_trace()
        traces.append(colorbar_trace)

        # 5. Group legend annotations -----------------------------------------
        group_annotations = self._build_group_legend()

        # 6. Assemble figure --------------------------------------------------
        vanish_count  = sum(
            1 for n in self._all_norms if n < self.vanishing_threshold
        )
        explode_count = sum(
            1 for n in self._all_norms if n > self.exploding_threshold
        )
        subtitle = (
            f"Layers: {len(self.report.layer_stats)} | "
            f"Vanishing: {vanish_count} | "
            f"Exploding: {explode_count} | "
            f"Colormap: {self.scheme.value}"
        )

        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title=dict(
                    text=(
                        f"<b>{self.title}</b><br>"
                        f"<sup>{subtitle}</sup>"
                    ),
                    x=0.5,
                    xanchor="center",
                    font=dict(size=16),
                ),
                showlegend=False,
                hovermode="closest",
                xaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False,
                    range=[-50, layout.canvas_width + 50],
                ),
                yaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False,
                    range=[layout.canvas_height + 30, -30],  # inverted: top=input
                    scaleanchor="x",
                ),
                shapes=warn_shapes,
                annotations=group_annotations,
                plot_bgcolor="#0F1117",
                paper_bgcolor="#0F1117",
                font=dict(color="#FAFAFA"),
                margin=dict(l=20, r=20, t=80, b=20),
                width=layout.canvas_width + 100,
                height=layout.canvas_height + 120,
            ),
        )
        return fig

    def _build_warning_shapes(
        self,
        layout: ArchitectureLayout,
        stats_map: dict,
    ) -> list:
        """Build semi-transparent warning rectangles behind vanishing/exploding nodes."""
        shapes = []
        node_r = 18   # approximate node radius for bounding box

        for i, node in enumerate(layout.nodes):
            s = stats_map[node.layer_index]
            norm = self._all_norms[i]
            pathology = s.diagnose()

            if pathology == GradientPathology.VANISHING:
                color = VANISHING_WARN_COLOR
            elif pathology == GradientPathology.EXPLODING:
                color = EXPLODING_WARN_COLOR
            else:
                continue

            shapes.append(dict(
                type="rect",
                x0=node.x - node_r * 2.2,
                y0=node.y - node_r * 1.4,
                x1=node.x + node_r * 2.2,
                y1=node.y + node_r * 1.4,
                fillcolor=color,
                line=dict(width=0),
                layer="below",
            ))
        return shapes

    def _build_edge_trace(
        self,
        layout: ArchitectureLayout,
    ) -> Optional["go.Scatter"]:
        """Build a single Scatter trace for all edges (lines only, no markers)."""
        idx_to_node = {n.layer_index: n for n in layout.nodes}
        xs, ys = [], []
        for src_idx, dst_idx in layout.edges:
            src = idx_to_node.get(src_idx)
            dst = idx_to_node.get(dst_idx)
            if src is None or dst is None:
                continue
            xs += [src.x, dst.x, None]
            ys += [src.y, dst.y, None]

        if not xs:
            return None

        return go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="rgba(180, 180, 180, 0.25)", width=1),
            hoverinfo="none",
            name="gradient flow",
        )

    def _build_node_trace(
        self,
        layout: ArchitectureLayout,
        stats_map: dict,
    ) -> "go.Scatter":
        """Build a Scatter trace for all nodes with hover tooltips."""
        xs = [n.x for n in layout.nodes]
        ys = [n.y for n in layout.nodes]
        labels = [n.label for n in layout.nodes]

        hover_texts = []
        for i, node in enumerate(layout.nodes):
            s = stats_map[node.layer_index]
            norm = self._all_norms[i]
            pathology = s.diagnose()
            hover_texts.append(
                f"<b>{s.layer_name}</b><br>"
                f"Type: {s.layer_type}<br>"
                f"Group: {s.group.value}<br>"
                f"grad_norm: {norm:.3e}<br>"
                f"mean: {s.mean:.3e}<br>"
                f"std: {s.std:.3e}<br>"
                f"depth: {s.depth}<br>"
                f"<b>Status: {pathology.value.upper()}</b>"
            )

        marker = dict(
            size=22,
            color=self._node_fill_colors,
            line=dict(
                color=self._node_border_colors,
                width=3,
            ),
            symbol="circle",
        )

        return go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker=marker,
            text=labels,
            textposition="middle right",
            textfont=dict(size=9, color="#CCCCCC"),
            hovertext=hover_texts,
            hoverinfo="text",
            name="layers",
        )

    def _build_colorbar_trace(self) -> "go.Scatter":
        """Invisible trace whose sole purpose is to render the colour bar."""
        norms = self._all_norms
        return go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=plotly_colorscale(self.scheme),
                cmin=float(np.log10(max(min(norms), 1e-12) + 1e-12)),
                cmax=float(np.log10(max(norms) + 1e-12)),
                color=[0],
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="log₁₀(grad_norm)",
                        side="right",
                    ),
                    tickformat=".1f",
                    len=0.6,
                    x=1.02,
                    bgcolor="#1A1D24",
                    bordercolor="#444",
                    tickfont=dict(color="#CCCCCC"),
                    titlefont=dict(color="#CCCCCC"),
                ),
            ),
            hoverinfo="none",
            showlegend=False,
        )

    def _build_group_legend(self) -> list:
        """Build annotation objects that serve as a LayerGroup colour legend."""
        present_groups = {n.group for n in self._layout.nodes}  # type: ignore[union-attr]
        annotations = []
        x_start = 10
        y_pos   = -15

        for g in LayerGroup:
            if g not in present_groups:
                continue
            color = GROUP_BORDER_COLORS[g]
            annotations.append(dict(
                x=x_start,
                y=y_pos,
                xref="x",
                yref="y",
                text=(
                    f'<span style="color:{color};">⬤</span> {g.value}'
                ),
                showarrow=False,
                font=dict(size=11, color="#CCCCCC"),
                align="left",
            ))
            x_start += 130
        return annotations

    # ------------------------------------------------------------------
    # Internal helpers — Matplotlib static
    # ------------------------------------------------------------------

    def _assemble_matplotlib(
        self,
        figsize: tuple,
    ) -> "plt.Figure":
        """Produce a static Matplotlib figure of the Heatmap."""
        layout = self._layout
        assert layout is not None
        stats_map = {s.layer_index: s for s in self.report.layer_stats}

        fig, ax = plt.subplots(figsize=figsize, facecolor="#0F1117")
        ax.set_facecolor("#0F1117")

        # --- Edges ---
        if self.show_edges:
            idx_to_node = {n.layer_index: n for n in layout.nodes}
            for src_idx, dst_idx in layout.edges:
                src = idx_to_node.get(src_idx)
                dst = idx_to_node.get(dst_idx)
                if src and dst:
                    ax.plot(
                        [src.x, dst.x], [src.y, dst.y],
                        color="#444", linewidth=0.7, zorder=1,
                    )

        # --- Nodes ---
        for i, node in enumerate(layout.nodes):
            fill   = self._node_fill_colors[i]
            border = self._node_border_colors[i]
            circle = plt.Circle(
                (node.x, node.y), 14,
                color=fill,
                ec=border, linewidth=2,
                zorder=3,
            )
            ax.add_patch(circle)

            # Label to the right
            ax.text(
                node.x + 18, node.y,
                node.label,
                fontsize=6.5, color="#CCCCCC",
                va="center", ha="left", zorder=4,
            )

            # Vanishing annotation
            s = stats_map[node.layer_index]
            if s.diagnose() == GradientPathology.VANISHING:
                ax.text(
                    node.x, node.y,
                    "✕", fontsize=8, color="white",
                    ha="center", va="center", zorder=5,
                )

        # --- Colorbar via ScalarMappable ---
        cmap = plt.get_cmap(
            "viridis" if self.scheme == ColorScheme.VIRIDIS else "RdYlGn"
        )
        log_norms = np.log10(np.array(self._all_norms) + 1e-12)
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=mcolors.Normalize(vmin=log_norms.min(), vmax=log_norms.max()),
        )
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.01)
        cb.set_label("log₁₀(grad_norm)", color="#CCCCCC")
        cb.ax.yaxis.set_tick_params(color="#CCCCCC")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#CCCCCC")

        ax.set_xlim(-40, layout.canvas_width + 200)
        ax.set_ylim(layout.canvas_height + 20, -20)  # inverted y
        ax.axis("off")
        ax.set_title(self.title, color="white", fontsize=13, pad=10)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _auto_title(self) -> str:
        n = len(self.report.layer_stats)
        return f"Gradient Heatmap — {n} layers ({self.report.data_source})"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe_grad_norm(stats: object) -> float:
    """Return grad_norm if available, otherwise fall back to abs(mean)."""
    gn = getattr(stats, "grad_norm", None)
    if gn is not None and gn > 0:
        return float(gn)
    mean = getattr(stats, "mean", 0.0)
    return float(abs(mean)) + 1e-12
