"""Tests for Phase 1 fine-tuning failure detectors.

All tests use a lightweight synthetic LoRA-style model so the suite
runs quickly on CPU without requiring the PEFT library.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gradient_pathology.finetuning import (
    AdapterMonitor,
    ForgettingDetector,
    LoRARankTracker,
)


# ---------------------------------------------------------------------------
# Minimal LoRA-style model fixture
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Toy LoRA layer: out = base(x) + (x @ lora_A.T) @ lora_B.T."""

    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.base.weight.requires_grad_(False)  # frozen
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T


class TinyLoRAModel(nn.Module):
    def __init__(self, dim: int = 32, rank: int = 4):
        super().__init__()
        self.fc1 = LoRALinear(dim, dim, rank=rank)
        self.act = nn.ReLU()
        self.fc2 = LoRALinear(dim, 1, rank=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _run_steps(model: nn.Module, *callbacks: object, steps: int = 20, dim: int = 32) -> None:
    """Run *steps* synthetic forward/backward passes calling each callback's step()."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(steps):
        x = torch.randn(8, dim)
        y = torch.randn(8, 1)
        loss = nn.MSELoss()(model(x), y)
        opt.zero_grad()
        loss.backward()
        for cb in callbacks:
            cb.step()  # type: ignore[attr-defined]
        opt.step()


# ---------------------------------------------------------------------------
# LoRARankTracker
# ---------------------------------------------------------------------------


def test_lora_rank_tracker_discovers_params() -> None:
    model = TinyLoRAModel()
    tracker = LoRARankTracker(model)
    assert len(tracker._lora_pairs) > 0


def test_lora_rank_tracker_step_and_report() -> None:
    model = TinyLoRAModel(rank=4)
    tracker = LoRARankTracker(model)
    _run_steps(model, tracker, steps=15)
    report = tracker.report()
    assert len(report) > 0
    for name, info in report.items():
        assert "rank" in info
        assert "effective_rank" in info
        assert "rank_utilisation" in info
        assert info["status"] in ("healthy", "rank_collapse", "no_data")
        assert info["trend"] in ("stable", "improving", "degrading", "unknown")


def test_lora_rank_tracker_utilisation_range() -> None:
    model = TinyLoRAModel(rank=4)
    tracker = LoRARankTracker(model)
    _run_steps(model, tracker, steps=30)
    for info in tracker.report().values():
        if info["status"] == "no_data":
            continue
        assert 0.0 <= info["rank_utilisation"] <= 1.5  # allow slight overshoot


def test_lora_rank_tracker_collapsed_layers_is_list() -> None:
    model = TinyLoRAModel(rank=4)
    tracker = LoRARankTracker(model, min_rank_utilisation=0.9)  # strict threshold
    _run_steps(model, tracker, steps=10)
    assert isinstance(tracker.collapsed_layers(), list)


# ---------------------------------------------------------------------------
# AdapterMonitor
# ---------------------------------------------------------------------------


def test_adapter_monitor_classifies_params() -> None:
    model = TinyLoRAModel()
    monitor = AdapterMonitor(model)
    assert len(monitor._adapter_params) > 0
    assert len(monitor._frozen_params) > 0


def test_adapter_monitor_step_and_ratio() -> None:
    model = TinyLoRAModel()
    monitor = AdapterMonitor(model)
    _run_steps(model, monitor, steps=10)
    ratio = monitor.adapter_ratio
    assert ratio is not None
    assert ratio >= 0.0


def test_adapter_monitor_summary_str() -> None:
    model = TinyLoRAModel()
    monitor = AdapterMonitor(model)
    _run_steps(model, monitor, steps=5)
    s = monitor.summary()
    assert isinstance(s, str)
    assert "ADAPTER MONITOR" in s


def test_adapter_monitor_no_false_leakage() -> None:
    """Properly frozen params should NOT appear in leaking_layers."""
    model = TinyLoRAModel()
    monitor = AdapterMonitor(model)
    _run_steps(model, monitor, steps=10)
    leaking = monitor.leaking_layers
    base_weight_leaks = [l for l in leaking if "base.weight" in l]
    assert base_weight_leaks == []


# ---------------------------------------------------------------------------
# ForgettingDetector
# ---------------------------------------------------------------------------


def test_forgetting_detector_step_runs() -> None:
    model = TinyLoRAModel()
    detector = ForgettingDetector(model)
    _run_steps(model, detector, steps=15)
    assert len(detector._drift_history) > 0


def test_forgetting_detector_risk_in_range() -> None:
    model = TinyLoRAModel()
    detector = ForgettingDetector(model)
    _run_steps(model, detector, steps=20)
    assert 0.0 <= detector.forgetting_risk <= 1.0


def test_forgetting_detector_warning_is_bool() -> None:
    model = TinyLoRAModel()
    detector = ForgettingDetector(model, sensitivity=0.5)
    _run_steps(model, detector, steps=15)
    assert isinstance(detector.warning_triggered, bool)


def test_forgetting_detector_summary_str() -> None:
    model = TinyLoRAModel()
    detector = ForgettingDetector(model)
    _run_steps(model, detector, steps=10)
    s = detector.summary()
    assert isinstance(s, str)
    assert "FORGETTING" in s


def test_forgetting_detector_reset_clears_history() -> None:
    model = TinyLoRAModel()
    detector = ForgettingDetector(model)
    _run_steps(model, detector, steps=10)
    detector.reset()
    assert len(detector._drift_history) == 0
    assert len(detector._conflict_history) == 0
    assert detector._prev_base_grad is None


# ---------------------------------------------------------------------------
# Integration: all three running together
# ---------------------------------------------------------------------------


def test_all_three_detectors_together() -> None:
    """Smoke test: all three detectors run together without errors."""
    model = TinyLoRAModel(rank=4)
    tracker = LoRARankTracker(model)
    monitor = AdapterMonitor(model)
    detector = ForgettingDetector(model)
    _run_steps(model, tracker, monitor, detector, steps=25)
    assert len(tracker.report()) > 0
    assert monitor.adapter_ratio is not None
    assert 0.0 <= detector.forgetting_risk <= 1.0
