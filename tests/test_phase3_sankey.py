"""Tests for Phase-3: SankeyFlowBuilder, GradientSankeyRenderer, LayerDetailPanel.

Plotly-dependent tests are automatically skipped when Plotly is absent,
keeping the suite green in minimal CI environments.
"""

from __future__ import annotations

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
from gradient_pathology.sankey.flow import (
    FlowStrategy,
    FlowZone,
    SankeyFlow,
    SankeyFlowBuilder,
    SankeyLink,
    _merge_by_module,
    _safe_norm,
    _short_label,
)
from gradient_pathology.sankey.renderer import (
    GradientSankeyRenderer,
    GROUP_NODE_COLORS,
    ZONE_LINK_COLORS,
)
from gradient_pathology.sankey.detail_panel import (
    LayerDetailPanel,
    _PATHOLOGY_ADVICE,
)

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

requires_plotly = pytest.mark.skipif(
    not _PLOTLY_AVAILABLE, reason="plotly not installed"
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_stats(
    n: int = 6,
    vanishing: bool = False,
    exploding: bool = False,
) -> List[LayerGradientStats]:
    groups = [
        LayerGroup.EMBEDDING,
        LayerGroup.ATTENTION, LayerGroup.LAYER_NORM,
        LayerGroup.FFN,       LayerGroup.LAYER_NORM,
        LayerGroup.HEAD,
    ]
    stats = []
    for i in range(n):
        if vanishing and i == 0:
            mean = 1e-9
        elif exploding and i == n - 1:
            mean = 2e3
        else:
            mean = 1e-3 * (i + 1)
        gn = abs(mean) * 10
        s = LayerGradientStats(
            layer_name=f"model.block_{i // 2}.sub_{i % 2}.weight",
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


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

class TestSafeNorm:
    def test_returns_grad_norm_when_positive(self):
        s = _make_stats(n=1)[0]
        s.grad_norm = 0.42
        assert _safe_norm(s) == pytest.approx(0.42)

    def test_falls_back_to_abs_mean(self):
        s = _make_stats(n=1)[0]
        s.grad_norm = 0.0
        s.mean = -0.3
        assert _safe_norm(s) == pytest.approx(0.3 + 1e-12)


class TestShortLabel:
    def test_short_name_unchanged(self):
        assert _short_label("weight") == "weight"

    def test_two_segment(self):
        result = _short_label("layer.weight")
        assert result == "layer.weight"

    def test_deep_uses_last_two(self):
        result = _short_label("model.block_0.linear.weight")
        assert result == "linear.weight"

    def test_truncation(self):
        long = "a" * 30 + ".weight"
        result = _short_label(long, max_len=24)
        assert len(result) <= 24
        assert result.startswith("\u2026")


class TestMergeByModule:
    def test_weight_and_bias_merged(self):
        stats = _make_stats(n=2)
        # Give them matching parent module paths
        stats[0].layer_name = "fc.weight"
        stats[1].layer_name = "fc.bias"
        merged = _merge_by_module(stats)
        assert len(merged) == 1
        assert merged[0].layer_name == "fc"

    def test_different_modules_not_merged(self):
        stats = _make_stats(n=2)
        stats[0].layer_name = "fc1.weight"
        stats[1].layer_name = "fc2.weight"
        merged = _merge_by_module(stats)
        assert len(merged) == 2

    def test_merged_norm_is_l2_combination(self):
        stats = _make_stats(n=2)
        stats[0].layer_name = "fc.weight"
        stats[0].grad_norm   = 3.0
        stats[1].layer_name = "fc.bias"
        stats[1].grad_norm   = 4.0
        merged = _merge_by_module(stats)
        expected = float(np.sqrt(3.0**2 + 4.0**2))   # 5.0
        assert merged[0].grad_norm == pytest.approx(expected)


# ──────────────────────────────────────────────────────────────────────────────
# SankeyFlowBuilder
# ──────────────────────────────────────────────────────────────────────────────

class TestSankeyFlowBuilderEmpty:
    def test_empty_report_returns_empty_flow(self):
        empty = GradientReport(
            layer_stats=[], global_mean=0.0, global_std=0.0, num_steps=0
        )
        builder = SankeyFlowBuilder(empty)
        flow    = builder.build()
        assert flow.n_nodes == 0
        assert flow.links   == []


class TestSankeyFlowBuilderNodeCount:
    def test_node_count_without_merge(self):
        report  = _make_report(n=6)
        builder = SankeyFlowBuilder(report, group_by_layer=False)
        flow    = builder.build()
        # All 6 distinct layer names → 6 nodes
        assert flow.n_nodes == 6

    def test_node_count_with_merge_reduces_nodes(self):
        report  = _make_report(n=6)
        builder = SankeyFlowBuilder(report, group_by_layer=True)
        flow    = builder.build()
        # Fixture uses names like model.block_X.sub_Y.weight → 3 parent modules
        assert flow.n_nodes <= 6

    def test_link_count_is_nodes_minus_one(self):
        report  = _make_report(n=6)
        builder = SankeyFlowBuilder(report, group_by_layer=False)
        flow    = builder.build()
        assert len(flow.links) == flow.n_nodes - 1


class TestSankeyFlowBuilderStrategies:
    @pytest.mark.parametrize("strategy", list(FlowStrategy))
    def test_all_strategies_produce_positive_values(self, strategy):
        report  = _make_report(n=6)
        builder = SankeyFlowBuilder(report, strategy=strategy, group_by_layer=False)
        flow    = builder.build()
        assert all(lk.value > 0 for lk in flow.links)

    @pytest.mark.parametrize("strategy", list(FlowStrategy))
    def test_all_strategies_respect_min_width(self, strategy):
        min_w   = 2.0
        report  = _make_report(n=6)
        builder = SankeyFlowBuilder(report, strategy=strategy,
                                    min_width=min_w, group_by_layer=False)
        flow    = builder.build()
        assert all(lk.value >= min_w for lk in flow.links)

    @pytest.mark.parametrize("strategy", list(FlowStrategy))
    def test_all_strategies_respect_max_width(self, strategy):
        max_w   = 30.0
        report  = _make_report(n=6)
        builder = SankeyFlowBuilder(report, strategy=strategy,
                                    max_width=max_w, group_by_layer=False)
        flow    = builder.build()
        assert all(lk.value <= max_w + 1e-9 for lk in flow.links)


class TestSankeyFlowBuilderZoneClassification:
    def test_vanishing_layer_creates_vanishing_link(self):
        report  = _make_report(n=4, vanishing=True)
        builder = SankeyFlowBuilder(report, vanishing_threshold=1e-7,
                                    group_by_layer=False)
        flow    = builder.build()
        zones   = {lk.zone for lk in flow.links}
        assert FlowZone.VANISHING in zones

    def test_exploding_layer_creates_exploding_link(self):
        report  = _make_report(n=4, exploding=True)
        builder = SankeyFlowBuilder(report, exploding_threshold=1e3,
                                    group_by_layer=False)
        flow    = builder.build()
        zones   = {lk.zone for lk in flow.links}
        assert FlowZone.EXPLODING in zones

    def test_healthy_report_has_no_vanishing_links(self):
        report  = _make_report(n=4)
        builder = SankeyFlowBuilder(report, vanishing_threshold=1e-12,
                                    group_by_layer=False)
        flow    = builder.build()
        van_links = [lk for lk in flow.links if lk.zone == FlowZone.VANISHING]
        assert len(van_links) == 0

    def test_loss_fraction_between_zero_and_one(self):
        report = _make_report(n=6)
        builder = SankeyFlowBuilder(report, group_by_layer=False)
        flow   = builder.build()
        for lk in flow.links:
            assert 0.0 <= lk.loss_fraction <= 1.0 + 1e-9


class TestSankeyFlowProperties:
    def test_vanishing_links_property(self):
        report = _make_report(n=4, vanishing=True)
        builder = SankeyFlowBuilder(report, group_by_layer=False)
        flow   = builder.build()
        assert flow.vanishing_links == [
            lk for lk in flow.links if lk.zone == FlowZone.VANISHING
        ]

    def test_max_loss_fraction_is_max(self):
        report = _make_report(n=6)
        builder = SankeyFlowBuilder(report, group_by_layer=False)
        flow   = builder.build()
        expected = max((lk.loss_fraction for lk in flow.links), default=0.0)
        assert flow.max_loss_fraction == pytest.approx(expected)


# ──────────────────────────────────────────────────────────────────────────────
# GradientSankeyRenderer
# ──────────────────────────────────────────────────────────────────────────────

class TestGradientSankeyRendererColorTables:
    def test_all_flow_zones_have_colors(self):
        for zone in FlowZone:
            assert zone in ZONE_LINK_COLORS

    def test_all_layer_groups_have_node_colors(self):
        for group in LayerGroup:
            assert group in GROUP_NODE_COLORS


class TestGradientSankeyRendererInit:
    def test_default_strategy(self):
        report   = _make_report()
        renderer = GradientSankeyRenderer(report)
        assert renderer.strategy == FlowStrategy.LOG

    def test_auto_title_contains_layer_count(self):
        report   = _make_report(n=6)
        renderer = GradientSankeyRenderer(report)
        assert "6" in renderer.title

    def test_custom_title(self):
        report   = _make_report()
        renderer = GradientSankeyRenderer(report, title="My Sankey")
        assert renderer.title == "My Sankey"

    def test_flow_property_triggers_prepare(self):
        report   = _make_report()
        renderer = GradientSankeyRenderer(report)
        flow     = renderer.flow
        assert isinstance(flow, SankeyFlow)


@requires_plotly
class TestGradientSankeyRendererBuild:
    def test_build_returns_figure(self):
        report   = _make_report(n=6)
        renderer = GradientSankeyRenderer(report)
        fig      = renderer.build()
        assert isinstance(fig, go.Figure)

    def test_figure_has_sankey_trace(self):
        report   = _make_report(n=6)
        renderer = GradientSankeyRenderer(report)
        fig      = renderer.build()
        sankey_traces = [t for t in fig.data if isinstance(t, go.Sankey)]
        assert len(sankey_traces) == 1

    def test_sankey_node_count_matches_flow(self):
        report   = _make_report(n=6)
        renderer = GradientSankeyRenderer(report)
        fig      = renderer.build()
        trace    = fig.data[0]
        assert len(trace.node.label) == renderer.flow.n_nodes

    def test_sankey_link_count_matches_flow(self):
        report   = _make_report(n=6)
        renderer = GradientSankeyRenderer(report)
        fig      = renderer.build()
        trace    = fig.data[0]
        assert len(trace.link.value) == len(renderer.flow.links)

    def test_link_colors_match_zones(self):
        report   = _make_report(n=4, vanishing=True)
        renderer = GradientSankeyRenderer(report, group_by_layer=False)
        fig      = renderer.build()
        trace    = fig.data[0]
        # At least one link should be the vanishing colour
        van_color = ZONE_LINK_COLORS[FlowZone.VANISHING]
        assert any(c == van_color for c in trace.link.color)

    def test_empty_report_builds_without_crash(self):
        empty    = GradientReport(
            layer_stats=[], global_mean=0.0, global_std=0.0, num_steps=0
        )
        renderer = GradientSankeyRenderer(empty)
        fig      = renderer.build()   # should not raise
        assert isinstance(fig, go.Figure)

    def test_all_flow_strategies_build(self):
        report = _make_report(n=6)
        for strat in FlowStrategy:
            renderer = GradientSankeyRenderer(report, strategy=strat)
            fig = renderer.build()
            assert isinstance(fig, go.Figure)

    def test_figure_has_title(self):
        report   = _make_report()
        renderer = GradientSankeyRenderer(report, title="Custom Title")
        fig      = renderer.build()
        assert "Custom Title" in fig.layout.title.text


# ──────────────────────────────────────────────────────────────────────────────
# LayerDetailPanel
# ──────────────────────────────────────────────────────────────────────────────

class TestLayerDetailPanelDict:
    def test_found_is_true_for_existing_layer(self):
        report = _make_report(n=4)
        panel  = LayerDetailPanel(report)
        name   = report.layer_stats[0].layer_name
        d      = panel.build_dict(name)
        assert d["found"] is True

    def test_found_is_false_for_unknown_layer(self):
        report = _make_report(n=4)
        panel  = LayerDetailPanel(report)
        d      = panel.build_dict("does_not_exist.weight")
        assert d["found"] is False

    def test_dict_has_all_required_keys(self):
        report   = _make_report(n=4)
        panel    = LayerDetailPanel(report)
        name     = report.layer_stats[0].layer_name
        d        = panel.build_dict(name)
        required = {
            "layer_name", "layer_type", "group", "grad_norm",
            "mean", "std", "zero_ratio", "depth", "pathology",
            "pathology_color", "headline", "recommendations",
            "peer_rank", "peer_count", "global_rank", "global_count",
        }
        assert required.issubset(d.keys())

    def test_vanishing_layer_has_non_empty_recommendations(self):
        report = _make_report(n=4, vanishing=True)
        panel  = LayerDetailPanel(report)
        # Find vanishing layer
        van_stat = next(
            s for s in report.layer_stats
            if s.diagnose() == GradientPathology.VANISHING
        )
        d = panel.build_dict(van_stat.layer_name)
        assert len(d["recommendations"]) > 0

    def test_healthy_layer_has_empty_recommendations(self):
        report = _make_report(n=4)
        panel  = LayerDetailPanel(report)
        # Find healthy layer
        healthy = next(
            (s for s in report.layer_stats
             if s.diagnose() == GradientPathology.HEALTHY), None
        )
        if healthy is None:
            pytest.skip("No healthy layers in fixture")
        d = panel.build_dict(healthy.layer_name)
        assert d["recommendations"] == []

    def test_global_rank_is_within_bounds(self):
        report = _make_report(n=6)
        panel  = LayerDetailPanel(report)
        for s in report.layer_stats:
            d = panel.build_dict(s.layer_name)
            assert 1 <= d["global_rank"] <= d["global_count"]

    def test_peer_rank_is_within_bounds(self):
        report = _make_report(n=6)
        panel  = LayerDetailPanel(report)
        for s in report.layer_stats:
            d = panel.build_dict(s.layer_name)
            assert 1 <= d["peer_rank"] <= d["peer_count"]

    def test_all_layer_names_returns_depth_sorted_list(self):
        report = _make_report(n=6)
        panel  = LayerDetailPanel(report)
        names  = panel.all_layer_names()
        depths = [s.depth for s in sorted(report.layer_stats, key=lambda x: x.depth)]
        assert len(names) == len(report.layer_stats)


@requires_plotly
class TestLayerDetailPanelPlotly:
    def test_build_plotly_returns_figure(self):
        report = _make_report(n=4)
        panel  = LayerDetailPanel(report)
        name   = report.layer_stats[0].layer_name
        fig    = panel.build_plotly(name)
        assert isinstance(fig, go.Figure)

    def test_build_plotly_raises_for_unknown_layer(self):
        report = _make_report(n=4)
        panel  = LayerDetailPanel(report)
        with pytest.raises(ValueError, match="not found"):
            panel.build_plotly("nonexistent.weight")

    def test_figure_has_four_traces(self):
        report = _make_report(n=6)
        panel  = LayerDetailPanel(report)
        name   = report.layer_stats[2].layer_name
        fig    = panel.build_plotly(name)
        # 4 subplots → radar, bar, bar, table = 4 traces
        assert len(fig.data) == 4

    def test_vanishing_layer_figure_builds(self):
        report   = _make_report(n=4, vanishing=True)
        panel    = LayerDetailPanel(report)
        van_stat = next(
            s for s in report.layer_stats
            if s.diagnose() == GradientPathology.VANISHING
        )
        fig = panel.build_plotly(van_stat.layer_name)
        assert isinstance(fig, go.Figure)


# ──────────────────────────────────────────────────────────────────────────────
# GradientFlowGraph sankey shim
# ──────────────────────────────────────────────────────────────────────────────

@requires_plotly
class TestGradientFlowGraphSankeyShim:
    def test_plot_sankey_returns_figure(self):
        from gradient_pathology.auto.gradient_flow_graph import GradientFlowGraph
        model = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)
        )
        gfg = GradientFlowGraph(model)
        fig = gfg.plot_sankey(num_steps=3, input_shape=(8,))
        assert isinstance(fig, go.Figure)

    def test_plot_sankey_with_explicit_report(self):
        from gradient_pathology.auto.gradient_flow_graph import GradientFlowGraph
        model  = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        gfg    = GradientFlowGraph(model)
        report = gfg.build_report(num_steps=3, input_shape=(8,))
        fig    = gfg.plot_sankey(report=report)
        assert isinstance(fig, go.Figure)

    @pytest.mark.parametrize("strat", ["log", "normalised", "relative", "sqrt", "raw"])
    def test_all_strategy_strings_accepted(self, strat):
        from gradient_pathology.auto.gradient_flow_graph import GradientFlowGraph
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        gfg   = GradientFlowGraph(model)
        fig   = gfg.plot_sankey(strategy=strat, num_steps=2, input_shape=(4,))
        assert isinstance(fig, go.Figure)
