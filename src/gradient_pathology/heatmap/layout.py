"""Architecture graph layout engine for the Phase-2 Heatmap renderer.

Given a :class:`~gradient_pathology.core.GradientReport`, this module
produces (x, y) pixel coordinates for every layer node so they can be
rendered as a Plotly scatter plot.

Two layout strategies are implemented:

``SEQUENTIAL``
    Layers are stacked vertically in parameter-index order (deepest layer at
    the top, output at the bottom, mirroring the typical "model card" view).
    This is the default and is always available.

``GROUPED``
    Layers are arranged in columns by :class:`~gradient_pathology.core.LayerGroup`
    (Attention, FFN, LayerNorm, …).  Within each column they are stacked
    vertically by depth.  Requires ``networkx`` for the group-aware layout
    but falls back to ``SEQUENTIAL`` automatically when not installed.

``SPRING``
    NetworkX spring / force-directed layout.  Produces organic, graph-style
    positioning.  Requires ``networkx``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from gradient_pathology.core import GradientReport, LayerGroup

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False


class LayoutStrategy(Enum):
    SEQUENTIAL = "sequential"   # vertical stack by depth (always available)
    GROUPED    = "grouped"      # column per LayerGroup
    SPRING     = "spring"       # force-directed (requires networkx)


@dataclass
class NodeLayout:
    """Position and display metadata for a single node."""
    layer_name: str
    layer_index: int
    x: float
    y: float
    group: LayerGroup = LayerGroup.OTHER
    label: str = ""


@dataclass
class ArchitectureLayout:
    """Complete layout for all nodes, plus edge list.

    Attributes
    ----------
    nodes:
        Ordered list of :class:`NodeLayout` objects (one per layer).
    edges:
        List of ``(src_index, dst_index)`` pairs representing the backward
        pass direction (``src`` → ``dst`` means gradient flows from ``src``
        to ``dst``).
    canvas_width:
        Suggested canvas width in pixels.
    canvas_height:
        Suggested canvas height in pixels.
    """
    nodes: List[NodeLayout] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)
    canvas_width: int = 900
    canvas_height: int = 600

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_report(
        cls,
        report: GradientReport,
        strategy: LayoutStrategy = LayoutStrategy.SEQUENTIAL,
        node_spacing_y: float = 48.0,
        node_spacing_x: float = 160.0,
    ) -> "ArchitectureLayout":
        """Build a layout from a :class:`~gradient_pathology.core.GradientReport`.

        Parameters
        ----------
        report:
            The report produced by
            :class:`~gradient_pathology.analyzer.GradientAnalyzer`.
        strategy:
            Which layout algorithm to use.  Falls back to
            :attr:`LayoutStrategy.SEQUENTIAL` when ``networkx`` is absent.
        node_spacing_y:
            Vertical distance (px) between nodes in the same column.
        node_spacing_x:
            Horizontal distance (px) between group columns.

        Returns
        -------
        ArchitectureLayout
        """
        if not report.layer_stats:
            return cls()

        if strategy in (LayoutStrategy.GROUPED, LayoutStrategy.SPRING) and not _NX_AVAILABLE:
            strategy = LayoutStrategy.SEQUENTIAL

        if strategy == LayoutStrategy.SEQUENTIAL:
            return cls._sequential(report, node_spacing_y)
        if strategy == LayoutStrategy.GROUPED:
            return cls._grouped(report, node_spacing_x, node_spacing_y)
        # SPRING
        return cls._spring(report)

    # ------------------------------------------------------------------
    # Layout algorithms
    # ------------------------------------------------------------------

    @classmethod
    def _sequential(cls, report: GradientReport, spacing: float) -> "ArchitectureLayout":
        """Stack all layers vertically, ordered by depth."""
        stats = sorted(report.layer_stats, key=lambda s: s.depth)
        n = len(stats)
        canvas_h = int(max(400, n * spacing + 80))
        canvas_w = 900

        nodes: List[NodeLayout] = []
        x_center = canvas_w / 2
        y_start  = 40.0

        for i, s in enumerate(stats):
            nodes.append(NodeLayout(
                layer_name=s.layer_name,
                layer_index=s.layer_index,
                x=x_center,
                y=y_start + i * spacing,
                group=s.group,
                label=_short_label(s.layer_name),
            ))

        edges = [(i + 1, i) for i in range(n - 1)]  # backward direction
        return cls(
            nodes=nodes,
            edges=edges,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
        )

    @classmethod
    def _grouped(
        cls,
        report: GradientReport,
        x_spacing: float,
        y_spacing: float,
    ) -> "ArchitectureLayout":
        """One column per LayerGroup, nodes stacked by depth within column."""
        # Determine column order: preserve natural group order from the enum.
        group_order = list(LayerGroup)
        present_groups = {
            s.group for s in report.layer_stats
        }
        ordered_groups = [g for g in group_order if g in present_groups]

        # Assign x per group
        group_x: Dict[LayerGroup, float] = {
            g: 80.0 + i * x_spacing
            for i, g in enumerate(ordered_groups)
        }

        # Group layers and sort by depth within each group
        grouped: Dict[LayerGroup, list] = {g: [] for g in ordered_groups}
        for s in report.layer_stats:
            grouped[s.group].append(s)
        for g in ordered_groups:
            grouped[g].sort(key=lambda s: s.depth)

        nodes: List[NodeLayout] = []
        y_start = 40.0
        idx_map: Dict[int, int] = {}   # layer_index → node list index

        for g in ordered_groups:
            for row, s in enumerate(grouped[g]):
                nl = NodeLayout(
                    layer_name=s.layer_name,
                    layer_index=s.layer_index,
                    x=group_x[g],
                    y=y_start + row * y_spacing,
                    group=g,
                    label=_short_label(s.layer_name),
                )
                idx_map[s.layer_index] = len(nodes)
                nodes.append(nl)

        # Edges: depth-ordered consecutive pairs across all layers
        all_sorted = sorted(report.layer_stats, key=lambda s: s.depth)
        edges = [
            (all_sorted[i + 1].layer_index, all_sorted[i].layer_index)
            for i in range(len(all_sorted) - 1)
        ]

        max_col_nodes = max((len(v) for v in grouped.values()), default=1)
        canvas_h = int(max(400, max_col_nodes * y_spacing + 80))
        canvas_w = int(max(600, 80 + len(ordered_groups) * x_spacing + 80))

        return cls(
            nodes=nodes,
            edges=edges,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
        )

    @classmethod
    def _spring(
        cls,
        report: GradientReport,
    ) -> "ArchitectureLayout":
        """NetworkX spring layout — force-directed organic positioning."""
        import networkx as nx  # guaranteed available (checked earlier)

        G = nx.DiGraph()
        stats = sorted(report.layer_stats, key=lambda s: s.depth)
        for s in stats:
            G.add_node(s.layer_index)
        for i in range(len(stats) - 1):
            G.add_edge(stats[i + 1].layer_index, stats[i].layer_index)

        raw_pos = nx.spring_layout(G, k=2.0, iterations=80, seed=42)

        # Scale to canvas
        xs = np.array([raw_pos[n][0] for n in G.nodes()])
        ys = np.array([raw_pos[n][1] for n in G.nodes()])
        canvas_w, canvas_h = 1000, 700
        margin = 60

        def _scale(vals: np.ndarray, lo: int, hi: int) -> np.ndarray:
            mn, mx = vals.min(), vals.max()
            if mx == mn:
                return np.full_like(vals, (lo + hi) / 2)
            return lo + (vals - mn) / (mx - mn) * (hi - lo)

        xs_scaled = _scale(xs, margin, canvas_w - margin)
        ys_scaled = _scale(ys, margin, canvas_h - margin)

        idx_to_scaled = {
            list(G.nodes())[i]: (float(xs_scaled[i]), float(ys_scaled[i]))
            for i in range(len(G.nodes()))
        }

        nodes: List[NodeLayout] = []
        for s in stats:
            x, y = idx_to_scaled[s.layer_index]
            nodes.append(NodeLayout(
                layer_name=s.layer_name,
                layer_index=s.layer_index,
                x=x,
                y=y,
                group=s.group,
                label=_short_label(s.layer_name),
            ))

        edges = [
            (stats[i + 1].layer_index, stats[i].layer_index)
            for i in range(len(stats) - 1)
        ]

        return cls(
            nodes=nodes,
            edges=edges,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_label(layer_name: str, max_len: int = 22) -> str:
    """Shorten a fully-qualified parameter name for display."""
    # Keep last two dot-separated segments, truncate if still too long.
    parts = layer_name.split(".")
    label = ".".join(parts[-2:]) if len(parts) >= 2 else layer_name
    if len(label) > max_len:
        label = "…" + label[-(max_len - 1):]
    return label
