"""Tests for Phase-4: LiveGradientBridge, StreamlitCallback,
ExpertEngine, and integration.
"""

from __future__ import annotations

import math
import threading
import time
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
from gradient_pathology.monitor.bridge import GradientSnapshot, LiveGradientBridge
from gradient_pathology.monitor.callback import StreamlitCallback
from gradient_pathology.expert.engine import (
    ExpertEngine,
    ExpertFinding,
    inject_layer_norms,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def _make_stats(
    n: int = 6,
    vanishing: bool = False,
    exploding: bool = False,
    dead: bool = False,
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
            mean, gn = 1e-9, 1e-9
        elif exploding and i == n - 1:
            mean, gn = 2e4, 2e4
        elif dead and i == 1:
            mean, gn, zero_ratio = 1e-3, 1e-3, 0.95
        else:
            mean, gn = 1e-3 * (i + 1), 1e-2 * (i + 1)
        s = LayerGradientStats(
            layer_name=f"model.layer_{i}.weight",
            layer_index=i,
            mean=mean if not (dead and i == 1) else 1e-3,
            std=abs(mean if not (dead and i == 1) else 1e-3) * 0.1,
            min=-abs(mean if not (dead and i == 1) else 1e-3),
            max=abs(mean if not (dead and i == 1) else 1e-3),
            median=mean if not (dead and i == 1) else 1e-3,
            num_zeros=int((zero_ratio if dead and i == 1 else 0) * 64),
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
    dead: bool = False,
) -> GradientReport:
    s = _make_stats(n=n, vanishing=vanishing, exploding=exploding, dead=dead)
    return GradientReport(
        layer_stats=s,
        global_mean=float(np.mean([abs(x.mean) for x in s])),
        global_std=float(np.std([abs(x.mean)  for x in s])),
        num_steps=10,
        data_source="synthetic",
    )


@pytest.fixture()
def simple_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(8, 16), nn.ReLU(),
        nn.Linear(16, 8), nn.ReLU(),
        nn.Linear(8, 2),
    )


# ===========================================================================
# LiveGradientBridge
# ===========================================================================

class TestLiveGradientBridgeBasic:
    def test_initial_state_is_empty(self):
        bridge = LiveGradientBridge()
        assert bridge.is_empty
        assert bridge.latest_snapshot() is None
        assert bridge.all_snapshots() == []

    def test_push_increments_total(self):
        bridge = LiveGradientBridge()
        bridge.push(step=0, loss=1.0)
        assert bridge.total_pushed == 1
        bridge.push(step=1, loss=0.9)
        assert bridge.total_pushed == 2

    def test_push_returns_snapshot(self):
        bridge = LiveGradientBridge()
        snap = bridge.push(step=0, loss=2.0)
        assert isinstance(snap, GradientSnapshot)
        assert snap.step == 0
        assert snap.loss == pytest.approx(2.0)

    def test_latest_snapshot_after_push(self):
        bridge = LiveGradientBridge()
        bridge.push(step=0, loss=1.5)
        bridge.push(step=1, loss=1.2)
        latest = bridge.latest_snapshot()
        assert latest is not None
        assert latest.step == 1

    def test_ring_buffer_maxlen_respected(self):
        bridge = LiveGradientBridge(max_steps=3)
        for i in range(7):
            bridge.push(step=i, loss=float(i))
        snaps = bridge.all_snapshots()
        assert len(snaps) == 3
        assert snaps[0].step == 4  # oldest retained is step 4

    def test_clear_resets_state(self):
        bridge = LiveGradientBridge()
        bridge.push(step=0, loss=1.0)
        bridge.clear()
        assert bridge.is_empty
        assert bridge.total_pushed == 0

    def test_metrics_series_loss(self):
        bridge = LiveGradientBridge()
        for i in range(4):
            bridge.push(step=i, loss=float(i) * 0.5)
        steps, values = bridge.metrics_series("loss")
        assert steps == [0, 1, 2, 3]
        assert values == pytest.approx([0.0, 0.5, 1.0, 1.5])

    def test_drain_alerts_clears_pending(self):
        bridge = LiveGradientBridge(alert_threshold=1e-7)
        bridge.push(step=0, loss=0.5, layer_norms={"fc.weight": 1e-9})
        alerts = bridge.drain_alerts()
        assert len(alerts) >= 1
        assert bridge.drain_alerts() == []

    def test_push_with_layer_norms(self):
        bridge = LiveGradientBridge()
        snap = bridge.push(
            step=0, loss=1.0,
            layer_norms={"a.weight": 0.1, "b.weight": 0.2},
        )
        assert snap.layer_norms == {"a.weight": 0.1, "b.weight": 0.2}
        assert math.isfinite(snap.global_mean)


class TestLiveGradientBridgeModelPush:
    def test_push_with_model_collects_norms(self, simple_model):
        bridge = LiveGradientBridge()
        x = torch.randn(4, 8)
        y = torch.randn(4, 2)
        loss = nn.MSELoss()(simple_model(x), y)
        loss.backward()
        snap = bridge.push(step=0, loss=loss.item(), model=simple_model)
        assert len(snap.layer_norms) > 0
        assert all(v >= 0 for v in snap.layer_norms.values())

    def test_vanishing_gradient_triggers_alert(self):
        bridge = LiveGradientBridge(alert_threshold=1.0)  # very high threshold
        bridge.push(step=0, loss=1.0, layer_norms={"fc.w": 0.01})
        alerts = bridge.drain_alerts()
        # 0.01 < 1.0 threshold → vanishing alert
        assert any("Vanishing" in a for a in alerts)

    def test_exploding_gradient_triggers_alert(self):
        bridge = LiveGradientBridge(explode_threshold=1.0)
        bridge.push(step=0, loss=1.0, layer_norms={"fc.w": 100.0})
        alerts = bridge.drain_alerts()
        assert any("Exploding" in a for a in alerts)


class TestLiveGradientBridgeThreadSafety:
    def test_concurrent_pushes_do_not_corrupt(self):
        """100 threads each push 10 snapshots; buffer must be internally consistent."""
        bridge = LiveGradientBridge(max_steps=2000)
        errors: List[Exception] = []

        def worker(tid: int) -> None:
            try:
                for j in range(10):
                    bridge.push(step=tid * 10 + j, loss=float(j))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert bridge.total_pushed == 1000
        snaps = bridge.all_snapshots()
        assert len(snaps) <= 2000

    def test_concurrent_read_write_no_deadlock(self):
        """Writer and reader threads must not deadlock or raise."""
        bridge = LiveGradientBridge(max_steps=100)
        stop_event = threading.Event()
        errors: List[Exception] = []

        def writer() -> None:
            step = 0
            while not stop_event.is_set():
                try:
                    bridge.push(step=step, loss=1.0)
                    step += 1
                except Exception as e:
                    errors.append(e)

        def reader() -> None:
            while not stop_event.is_set():
                try:
                    bridge.all_snapshots()
                    bridge.latest_snapshot()
                except Exception as e:
                    errors.append(e)

        wt = threading.Thread(target=writer)
        rt = threading.Thread(target=reader)
        wt.start(); rt.start()
        time.sleep(0.15)
        stop_event.set()
        wt.join(timeout=2); rt.join(timeout=2)

        assert errors == []


# ===========================================================================
# StreamlitCallback
# ===========================================================================

class TestStreamlitCallback:
    def test_on_batch_end_pushes_snapshot(self, simple_model):
        bridge   = LiveGradientBridge()
        callback = StreamlitCallback(simple_model, bridge)
        x = torch.randn(4, 8)
        y = torch.randn(4, 2)
        loss = nn.MSELoss()(simple_model(x), y)
        loss.backward()
        callback.on_batch_end(step=0, loss=loss.item())
        assert bridge.total_pushed == 1

    def test_push_every_n_steps(self, simple_model):
        bridge   = LiveGradientBridge()
        callback = StreamlitCallback(simple_model, bridge, push_every_n_steps=3)
        x = torch.randn(4, 8)
        for step in range(6):
            nn.MSELoss()(simple_model(x), torch.randn(4, 2)).backward()
            callback.on_batch_end(step=step, loss=0.5)
        # Steps 3 and 6 pushed (internal counter ticks: 1,2,3→push,4,5,6→push)
        assert bridge.total_pushed == 2

    def test_reset_clears_counter(self, simple_model):
        bridge   = LiveGradientBridge()
        callback = StreamlitCallback(simple_model, bridge, push_every_n_steps=5)
        for step in range(4):
            callback.on_batch_end(step=step, loss=0.1)
        callback.reset()
        callback.on_batch_end(step=10, loss=0.1)
        # After reset, counter is 1 — only the first push after reset (at step 5) counts
        assert bridge.total_pushed == 0  # never hit multiple of 5


# ===========================================================================
# ExpertEngine
# ===========================================================================

class TestExpertEngineFindings:
    def test_healthy_report_returns_no_critical_findings(self):
        report  = _make_report(n=4)
        engine  = ExpertEngine()
        findings = engine.analyse(report)
        crit = [f for f in findings if f.severity == "critical"]
        assert len(crit) == 0

    def test_vanishing_rule_fires(self):
        report   = _make_report(n=4, vanishing=True)
        engine   = ExpertEngine(vanishing_threshold=1e-7)
        findings = engine.analyse(report)
        ids = [f.rule_id for f in findings]
        assert "vanishing_layers" in ids

    def test_exploding_rule_fires(self):
        report   = _make_report(n=4, exploding=True)
        engine   = ExpertEngine(exploding_threshold=1e3)
        findings = engine.analyse(report)
        ids = [f.rule_id for f in findings]
        assert "exploding_layers" in ids

    def test_dead_neuron_rule_fires(self):
        report   = _make_report(n=4, dead=True)
        engine   = ExpertEngine()
        findings = engine.analyse(report)
        # layer_index=1 has zero_ratio=0.95 > 0.9 threshold
        ids = [f.rule_id for f in findings]
        assert "dead_neurons" in ids

    def test_findings_sorted_by_severity(self):
        report   = _make_report(n=6, vanishing=True, dead=True)
        engine   = ExpertEngine()
        findings = engine.analyse(report)
        if len(findings) >= 2:
            for i in range(len(findings) - 1):
                assert findings[i].severity_rank <= findings[i + 1].severity_rank

    def test_quick_summary_format(self):
        report  = _make_report(n=4, vanishing=True)
        engine  = ExpertEngine()
        summary = engine.quick_summary(report)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_quick_summary_healthy(self):
        report  = _make_report(n=4)
        engine  = ExpertEngine()
        summary = engine.quick_summary(report)
        assert "✅" in summary

    def test_custom_rule_registered(self):
        report = _make_report(n=4)
        engine = ExpertEngine()

        @engine.register_rule
        def my_rule(r: GradientReport) -> list:
            return [ExpertFinding(
                rule_id="my_custom",
                severity="info",
                title="Custom rule fired",
                detail="detail",
            )]

        findings = engine.analyse(report)
        ids = [f.rule_id for f in findings]
        assert "my_custom" in ids

    def test_finding_has_code_hint_for_vanishing(self):
        report   = _make_report(n=4, vanishing=True)
        engine   = ExpertEngine()
        findings = engine.analyse(report)
        van = next(f for f in findings if f.rule_id == "vanishing_layers")
        assert len(van.code_hint) > 0

    def test_finding_lists_affected_layers(self):
        report   = _make_report(n=4, vanishing=True)
        engine   = ExpertEngine()
        findings = engine.analyse(report)
        van = next(f for f in findings if f.rule_id == "vanishing_layers")
        assert len(van.layers) > 0

    def test_no_layernorm_rule_fires_for_deep_network(self):
        """Build a report where every layer is OTHER group (no LN) and depth >= threshold."""
        stats = []
        for i in range(15):
            s = LayerGradientStats(
                layer_name=f"fc_{i}.weight",
                layer_index=i,
                mean=1e-3,
                std=1e-4,
                min=-1e-3, max=1e-3, median=1e-3,
                num_zeros=0,
                total_params=64,
                layer_type="Linear",
                depth=i,
                group=LayerGroup.OTHER,  # no LAYER_NORM
                grad_norm=1e-2,
            )
            stats.append(s)
        report = GradientReport(
            layer_stats=stats,
            global_mean=1e-3, global_std=1e-4,
            num_steps=10, data_source="synthetic",
        )
        engine   = ExpertEngine()
        findings = engine.analyse(report)
        ids = [f.rule_id for f in findings]
        assert "no_layernorm" in ids or "no_layernorm_vanishing" in ids

    def test_bottleneck_cascade_rule_fires(self):
        """Create stats where norm drops by >50% from depth 0 to depth 1."""
        stats = []
        norms = [1.0, 0.1, 0.08, 0.07, 0.06]
        for i, gn in enumerate(norms):
            s = LayerGradientStats(
                layer_name=f"layer_{i}.w",
                layer_index=i,
                mean=gn * 0.1, std=gn * 0.01,
                min=-gn, max=gn, median=gn * 0.1,
                num_zeros=0, total_params=64,
                layer_type="Linear", depth=i,
                group=LayerGroup.OTHER,
                grad_norm=gn,
            )
            stats.append(s)
        report = GradientReport(
            layer_stats=stats,
            global_mean=0.1, global_std=0.1,
            num_steps=5, data_source="synthetic",
        )
        engine   = ExpertEngine(bottleneck_drop_ratio=0.5)
        findings = engine.analyse(report)
        ids = [f.rule_id for f in findings]
        assert "bottleneck_cascade" in ids


class TestExpertEngineFindingDataclass:
    def test_finding_emoji_critical(self):
        f = ExpertFinding(
            rule_id="x", severity="critical", title="t", detail="d"
        )
        assert f.emoji == "🚨"

    def test_finding_emoji_warning(self):
        f = ExpertFinding(
            rule_id="x", severity="warning", title="t", detail="d"
        )
        assert f.emoji == "⚠️"

    def test_finding_emoji_info(self):
        f = ExpertFinding(
            rule_id="x", severity="info", title="t", detail="d"
        )
        assert f.emoji == "ℹ️"

    def test_severity_rank_ordering(self):
        c = ExpertFinding(rule_id="c", severity="critical", title="", detail="")
        w = ExpertFinding(rule_id="w", severity="warning",  title="", detail="")
        i = ExpertFinding(rule_id="i", severity="info",     title="", detail="")
        assert c.severity_rank < w.severity_rank < i.severity_rank


# ===========================================================================
# inject_layer_norms utility
# ===========================================================================

class TestInjectLayerNorms:
    def test_injects_layer_norm_after_linear(self):
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        new_model = inject_layer_norms(model)
        has_ln = any(isinstance(m, nn.LayerNorm) for m in new_model.modules())
        assert has_ln

    def test_non_sequential_returned_unchanged(self):
        model = nn.Linear(4, 4)
        result = inject_layer_norms(model)
        assert result is model


# ===========================================================================
# Integration: bridge + callback + engine
# ===========================================================================

class TestPhase4Integration:
    def test_full_training_loop_integration(self, simple_model):
        """Simulate 10 training steps and verify bridge + engine work end-to-end."""
        bridge   = LiveGradientBridge(max_steps=50)
        callback = StreamlitCallback(simple_model, bridge)
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)

        for step in range(10):
            optimizer.zero_grad()
            x = torch.randn(4, 8)
            y = torch.randn(4, 2)
            loss = nn.MSELoss()(simple_model(x), y)
            loss.backward()
            optimizer.step()
            callback.on_batch_end(step=step, loss=loss.item())

        assert bridge.total_pushed == 10
        snaps = bridge.all_snapshots()
        assert all(math.isfinite(s.loss) for s in snaps)

        # Verify metrics_series is consistent
        steps_l, losses = bridge.metrics_series("loss")
        assert len(steps_l) == 10
        assert all(math.isfinite(v) for v in losses)

    def test_engine_on_analyzer_report(self, simple_model):
        """Verify ExpertEngine accepts a real GradientAnalyzer report."""
        from gradient_pathology.analyzer import GradientAnalyzer
        analyzer = GradientAnalyzer(simple_model)
        report   = analyzer.diagnose(num_steps=5, input_shape=(8,))

        engine   = ExpertEngine()
        findings = engine.analyse(report)
        # Should not crash and return a list (may be empty for small healthy model)
        assert isinstance(findings, list)

    def test_hf_adapter_does_not_crash_without_transformers(self, simple_model):
        """HuggingFaceCallbackAdapter must not raise ImportError on import."""
        from gradient_pathology.monitor.callback import HuggingFaceCallbackAdapter
        bridge  = LiveGradientBridge()
        adapter = HuggingFaceCallbackAdapter(simple_model, bridge)
        assert adapter is not None
