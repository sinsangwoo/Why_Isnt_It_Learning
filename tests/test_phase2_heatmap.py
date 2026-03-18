"""Tests for Phase-2: GradientHeatmapRenderer, colormap, layout.

All Plotly-dependent tests are automatically skipped when Plotly is absent,
so the suite stays green in minimal CI environments.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn

from gradient_pathology.core import (
    GradientPathology,
    GradientReport,
    LayerGradientStats,
    LayerGroup,
)
from gradient_pathology.heatmap.colormap import (
    ColorScheme,
    GROUP_BORDER_COLORS,
    grad_norm_to_color,
    pathology_border_color,
    plotly_colorscale,
    _hex_to_rgb,
    _interpolate_colorscale,
    _viridis_stops,
    _rdylgn_stops,
)
from gradient_pathology.heatmap.layout import (
    ArchitectureLayout,
    LayoutStrategy,
    NodeLayout,
    _short_label,
)
from gradient_pathology.heatmap.renderer import (
    GradientHeatmapRenderer,
    VANISHING_THRESHOLD,
    EXPLODING_THRESHOLD,
    _safe_grad_norm,
)

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

requires_plotly = pytest.mark.skipif(
    not _PLOTLY_AVAILABLE,
    reason="plotly not installed",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_stats(
    n: int = 6,
    vanishing: bool = False,
    exploding: bool = False,
) -> List[LayerGradientStats]:
    """Create a list of LayerGradientStats with controlled pathology."""
    groups = [
        LayerGroup.EMBEDDING, LayerGroup.ATTENTION, LayerGroup.LAYER_NORM,
        LayerGroup.FFN, LayerGroup.LAYER_NORM, LayerGroup.HEAD,
    ]
    stats = []
    for i in range(n):
        if vanishing and i == 0:
            mean = 1e-9
        elif exploding and i == n - 1:
            mean = 1e4
        else:
            mean = 1e-3 * (i + 1)

        gn = abs(mean) * 10
        s = LayerGradientStats(
            layer_name=f"model.layer_{i}.weight",
            layer_index=i,
            mean=mean,
            std=abs(mean) * 0.1,
            min=-abs(mean),
            max=abs(mean),
            median=mean,
            num_zeros=0,
            total_params=64,
            layer_type="Linear",
            depth=i,
            group=groups[i % len(groups)],
            grad_norm=gn,
        )
        stats.append(s)
    return stats


def _make_report(
    n: int = 6,
    vanishing: bool = False,
    exploding: bool = False,
) -> GradientReport:
    stats = _make_stats(n=n, vanishing=vanishing, exploding=exploding)
    all_means = np.array([abs(s.mean) for s in stats])
    return GradientReport(
        layer_stats=stats,
        global_mean=float(all_means.mean()),
        global_std=float(all_means.std()),
        num_steps=10,
        data_source="synthetic",
    )


# ---------------------------------------------------------------------------
# Colormap tests
# ---------------------------------------------------------------------------

class TestColormapHelpers:
    def test_hex_to_rgb_known_value(self):
        assert _hex_to_rgb("#FF0000") == (255, 0, 0)
        assert _hex_to_rgb("#00FF00") == (0, 255, 0)
        assert _hex_to_rgb("#0000FF") == (0, 0, 255)
        assert _hex_to_rgb("#FFFFFF") == (255, 255, 255)

    def test_viridis_stops_range(self):
        stops = _viridis_stops()
        positions = [s[0] for s in stops]
        assert positions[0] == 0.0
        assert positions[-1] == 1.0
        assert all(0.0 <= p <= 1.0 for p in positions)

    def test_rdylgn_stops_range(self):
        stops = _rdylgn_stops()
        assert stops[0][0] == 0.0
        assert stops[-1][0] == 1.0

    def test_interpolate_colorscale_endpoints(self):
        stops = _viridis_stops()
        assert _interpolate_colorscale(stops, 0.0) == stops[0][1]
        assert _interpolate_colorscale(stops, 1.0) == stops[-1][1]

    def test_interpolate_colorscale_clamp(self):
        stops = _viridis_stops()
        # Beyond range should not raise
        c_low  = _interpolate_colorscale(stops, -0.5)
        c_high = _interpolate_colorscale(stops, 1.5)
        assert c_low.startswith("#")
        assert c_high.startswith("#")

    def test_interpolate_midpoint_is_intermediate(self):
        stops = _viridis_stops()
        c_start = _interpolate_colorscale(stops, 0.0)
        c_mid   = _interpolate_colorscale(stops, 0.5)
        c_end   = _interpolate_colorscale(stops, 1.0)
        # All three should be different hex strings
        assert c_start != c_mid != c_end


class TestGradNormToColor:
    def test_returns_hex_string(self):
        norms = [1e-3, 2e-3, 5e-3]
        color = grad_norm_to_color(1e-3, norms)
        assert color.startswith("#")
        assert len(color) == 7

    def test_vanishing_maps_to_darkest(self):
        norms = [1e-10, 1e-3, 5e-3]
        dark = grad_norm_to_color(1e-10, norms, vanishing_threshold=1e-7)
        normal = grad_norm_to_color(1e-3, norms)
        # Darkest Viridis stop is #440154 — vanishing should be at or near it
        assert dark == "#440154"

    def test_exploding_maps_to_brightest(self):
        norms = [1e-3, 1e4]
        bright = grad_norm_to_color(1e4, norms, exploding_threshold=1e3)
        assert bright == "#FDE725"  # brightest Viridis stop

    def test_rdylgn_scheme(self):
        norms = [1e-3, 2e-3, 5e-3]
        color = grad_norm_to_color(1e-3, norms, scheme=ColorScheme.RDYLGN)
        assert color.startswith("#")

    def test_all_same_norms_does_not_crash(self):
        norms = [1e-3, 1e-3, 1e-3]
        color = grad_norm_to_color(1e-3, norms)
        assert color.startswith("#")

    def test_different_norms_produce_different_colors(self):
        norms = [1e-5, 1e-3, 1e-1]
        colors = [grad_norm_to_color(n, norms) for n in norms]
        # At least some colours should differ
        assert len(set(colors)) > 1


class TestPathologyBorderColor:
    @pytest.mark.parametrize("pathology,expected_hex", [
        (GradientPathology.HEALTHY,      "#2ECC71"),
        (GradientPathology.VANISHING,    "#E74C3C"),
        (GradientPathology.EXPLODING,    "#E67E22"),
        (GradientPathology.DEAD_NEURONS, "#8E44AD"),
        (GradientPathology.UNSTABLE,     "#F39C12"),
    ])
    def test_returns_correct_hex(self, pathology, expected_hex):
        assert pathology_border_color(pathology) == expected_hex


class TestGroupBorderColors:
    def test_all_groups_have_a_color(self):
        for group in LayerGroup:
            assert group in GROUP_BORDER_COLORS
            assert GROUP_BORDER_COLORS[group].startswith("#")


class TestPlotlyColorscale:
    def test_returns_nested_list(self):
        cs = plotly_colorscale(ColorScheme.VIRIDIS)
        assert isinstance(cs, list)
        assert all(isinstance(entry, list) and len(entry) == 2 for entry in cs)

    def test_positions_start_at_zero_end_at_one(self):
        cs = plotly_colorscale(ColorScheme.RDYLGN)
        assert cs[0][0] == 0.0
        assert cs[-1][0] == 1.0


# ---------------------------------------------------------------------------
# Layout tests
# ---------------------------------------------------------------------------

class TestShortLabel:
    def test_short_name_unchanged(self):
        assert _short_label("weight") == "weight"

    def test_two_segment_name(self):
        assert _short_label("fc1.weight") == "fc1.weight"

    def test_deep_name_uses_last_two_segments(self):
        result = _short_label("transformer.h.0.attn.c_attn.weight")
        assert result == "c_attn.weight"

    def test_long_label_truncated(self):
        long = "a" * 30 + ".weight"
        result = _short_label(long, max_len=22)
        assert len(result) <= 22
        assert result.startswith("…")


class TestArchitectureLayoutSequential:
    def test_node_count_matches_layers(self):
        report = _make_report(n=8)
        layout = ArchitectureLayout.from_report(report, strategy=LayoutStrategy.SEQUENTIAL)
        assert len(layout.nodes) == 8

    def test_all_nodes_have_same_x_in_sequential(self):
        report = _make_report(n=6)
        layout = ArchitectureLayout.from_report(report, strategy=LayoutStrategy.SEQUENTIAL)
        xs = [n.x for n in layout.nodes]
        assert len(set(xs)) == 1, "Sequential layout: all nodes should share x"

    def test_y_values_are_strictly_increasing(self):
        report = _make_report(n=6)
        layout = ArchitectureLayout.from_report(report, strategy=LayoutStrategy.SEQUENTIAL)
        ys = [n.y for n in layout.nodes]
        assert ys == sorted(ys)

    def test_edges_connect_consecutive_nodes(self):
        report = _make_report(n=4)
        layout = ArchitectureLayout.from_report(report, strategy=LayoutStrategy.SEQUENTIAL)
        assert len(layout.edges) == 3  # n-1 edges

    def test_empty_report_returns_empty_layout(self):
        empty_report = GradientReport(
            layer_stats=[], global_mean=0.0, global_std=0.0, num_steps=0
        )
        layout = ArchitectureLayout.from_report(empty_report)
        assert layout.nodes == []
        assert layout.edges == []

    def test_canvas_height_scales_with_layers(self):
        small = ArchitectureLayout.from_report(_make_report(n=3))
        large = ArchitectureLayout.from_report(_make_report(n=20))
        assert large.canvas_height > small.canvas_height


class TestArchitectureLayoutGrouped:
    def test_grouped_produces_multiple_x_values(self):
        report = _make_report(n=6)
        layout = ArchitectureLayout.from_report(report, strategy=LayoutStrategy.GROUPED)
        xs = {n.x for n in layout.nodes}
        # Should have more than one unique x (one per group)
        assert len(xs) > 1

    def test_grouped_node_count_unchanged(self):
        report = _make_report(n=6)
        layout = ArchitectureLayout.from_report(report, strategy=LayoutStrategy.GROUPED)
        assert len(layout.nodes) == 6

    def test_grouped_falls_back_to_sequential_without_networkx(self, monkeypatch):
        """When networkx is absent, GROUPED silently falls back to SEQUENTIAL."""
        import gradient_pathology.heatmap.layout as lay_mod
        monkeypatch.setattr(lay_mod, "_NX_AVAILABLE", False)
        report = _make_report(n=4)
        layout = ArchitectureLayout.from_report(report, strategy=LayoutStrategy.GROUPED)
        # Falls back to sequential → all x values equal
        xs = {n.x for n in layout.nodes}
        assert len(xs) == 1


# ---------------------------------------------------------------------------
# Renderer tests
# ---------------------------------------------------------------------------

class TestSafeGradNorm:
    def test_returns_grad_norm_when_positive(self):
        s = _make_stats(n=1)[0]
        s.grad_norm = 0.5
        assert _safe_grad_norm(s) == pytest.approx(0.5)

    def test_falls_back_to_abs_mean(self):
        s = _make_stats(n=1)[0]
        s.grad_norm = 0.0
        s.mean = -0.7
        result = _safe_grad_norm(s)
        assert result == pytest.approx(0.7 + 1e-12)


class TestGradientHeatmapRendererInit:
    def test_default_thresholds(self):
        report = _make_report()
        renderer = GradientHeatmapRenderer(report)
        assert renderer.vanishing_threshold == VANISHING_THRESHOLD
        assert renderer.exploding_threshold == EXPLODING_THRESHOLD

    def test_custom_scheme(self):
        report = _make_report()
        renderer = GradientHeatmapRenderer(report, scheme=ColorScheme.RDYLGN)
        assert renderer.scheme == ColorScheme.RDYLGN

    def test_auto_title_contains_layer_count(self):
        report = _make_report(n=6)
        renderer = GradientHeatmapRenderer(report)
        assert "6" in renderer.title

    def test_custom_title(self):
        report = _make_report()
        renderer = GradientHeatmapRenderer(report, title="My Custom Title")
        assert renderer.title == "My Custom Title"


class TestGradientHeatmapRendererPrepare:
    def test_prepare_populates_norms(self):
        report = _make_report(n=4)
        renderer = GradientHeatmapRenderer(report)
        renderer._prepare()
        assert len(renderer._all_norms) == 4
        assert all(n >= 0 for n in renderer._all_norms)

    def test_prepare_populates_fill_colors(self):
        report = _make_report(n=4)
        renderer = GradientHeatmapRenderer(report)
        renderer._prepare()
        assert len(renderer._node_fill_colors) == 4
        assert all(c.startswith("#") for c in renderer._node_fill_colors)

    def test_prepare_populates_border_colors(self):
        report = _make_report(n=4)
        renderer = GradientHeatmapRenderer(report)
        renderer._prepare()
        assert len(renderer._node_border_colors) == 4

    def test_prepare_is_idempotent(self):
        report = _make_report(n=4)
        renderer = GradientHeatmapRenderer(report)
        renderer._prepare()
        colors_first = list(renderer._node_fill_colors)
        renderer._prepare()  # second call
        assert renderer._node_fill_colors == colors_first

    def test_vanishing_node_gets_darkest_viridis(self):
        report = _make_report(n=4, vanishing=True)
        renderer = GradientHeatmapRenderer(report, scheme=ColorScheme.VIRIDIS)
        renderer._prepare()
        # First node is the vanishing one
        assert renderer._node_fill_colors[0] == "#440154"

    def test_vanishing_node_gets_red_border(self):
        report = _make_report(n=4, vanishing=True)
        renderer = GradientHeatmapRenderer(report)
        renderer._prepare()
        assert renderer._node_border_colors[0] == "#E74C3C"


@requires_plotly
class TestGradientHeatmapRendererBuildPlotly:
    def test_build_returns_figure(self):
        report = _make_report(n=4)
        renderer = GradientHeatmapRenderer(report)
        fig = renderer.build()
        assert isinstance(fig, go.Figure)

    def test_figure_has_traces(self):
        report = _make_report(n=4)
        renderer = GradientHeatmapRenderer(report)
        fig = renderer.build()
        assert len(fig.data) > 0

    def test_figure_has_title(self):
        report = _make_report(n=4)
        renderer = GradientHeatmapRenderer(report, title="Test Title")
        fig = renderer.build()
        assert "Test Title" in fig.layout.title.text

    def test_build_with_vanishing_layers_has_warning_shapes(self):
        report = _make_report(n=4, vanishing=True)
        renderer = GradientHeatmapRenderer(report)
        fig = renderer.build()
        # Should have at least one warning shape
        assert len(fig.layout.shapes) >= 1

    def test_build_no_edges_omits_edge_trace(self):
        report = _make_report(n=4)
        renderer = GradientHeatmapRenderer(report, show_edges=False)
        fig = renderer.build()
        # With show_edges=False, no edge line trace should exist
        line_traces = [t for t in fig.data if t.mode == "lines"]
        assert len(line_traces) == 0

    def test_rdylgn_scheme_figure_builds(self):
        report = _make_report(n=4)
        renderer = GradientHeatmapRenderer(report, scheme=ColorScheme.RDYLGN)
        fig = renderer.build()
        assert isinstance(fig, go.Figure)

    def test_grouped_layout_builds(self):
        report = _make_report(n=6)
        renderer = GradientHeatmapRenderer(
            report, layout_strategy=LayoutStrategy.GROUPED
        )
        fig = renderer.build()
        assert isinstance(fig, go.Figure)


class TestGradientHeatmapRendererStaticBuild:
    def test_build_static_returns_figure(self):
        report = _make_report(n=4)
        renderer = GradientHeatmapRenderer(report)
        mpl_fig = renderer.build_static()
        import matplotlib.figure
        assert isinstance(mpl_fig, matplotlib.figure.Figure)
        plt_mod = pytest.importorskip("matplotlib.pyplot")
        plt_mod.close(mpl_fig)


class TestGradientFlowGraphHeatmapShim:
    """Verify the GradientFlowGraph.plot_heatmap() integration shim."""

    def test_build_report_returns_gradient_report(self):
        from gradient_pathology.auto.gradient_flow_graph import GradientFlowGraph
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        gfg = GradientFlowGraph(model)
        report = gfg.build_report(num_steps=3, input_shape=(8,))
        from gradient_pathology.core import GradientReport
        assert isinstance(report, GradientReport)
        assert len(report.layer_stats) > 0

    @requires_plotly
    def test_plot_heatmap_returns_figure(self):
        from gradient_pathology.auto.gradient_flow_graph import GradientFlowGraph
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        gfg = GradientFlowGraph(model)
        fig = gfg.plot_heatmap(num_steps=3, input_shape=(8,))
        assert isinstance(fig, go.Figure)
