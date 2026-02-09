"""Tests for gradient analyzer."""

import pytest
import torch.nn as nn

from gradient_pathology.analyzer import GradientAnalyzer
from gradient_pathology.core import GradientPathology


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_analyzer_basic() -> None:
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    analyzer = GradientAnalyzer(model, device="cpu")
    report = analyzer.diagnose(num_steps=10, input_shape=(10,))

    assert len(report.layer_stats) > 0
    assert report.global_mean is not None
    assert report.global_std is not None


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_vanishing_gradient_detection() -> None:
    model = nn.Sequential(
        *[nn.Linear(64, 64), nn.Sigmoid()] * 100,
        nn.Linear(64, 1),
    )

    analyzer = GradientAnalyzer(model, device="cpu")
    report = analyzer.diagnose(num_steps=50, input_shape=(64,))

    pathological_count = sum(
        1 for stats in report.layer_stats
        if stats.diagnose() != GradientPathology.HEALTHY
    )
    assert pathological_count > 0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_healthy_gradient_flow() -> None:
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.LayerNorm(64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.LayerNorm(64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

    analyzer = GradientAnalyzer(model, device="cpu")
    report = analyzer.diagnose(num_steps=10, input_shape=(32,))

    healthy_count = sum(
        1 for stats in report.layer_stats
        if stats.diagnose() == GradientPathology.HEALTHY
    )
    assert healthy_count >= 1


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_report_summary() -> None:
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    analyzer = GradientAnalyzer(model, device="cpu")
    report = analyzer.diagnose(num_steps=5, input_shape=(10,))

    summary = report.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
