"""Tests for GradientAnalyzer."""

import pytest
import torch
import torch.nn as nn

from gradient_pathology import GradientAnalyzer
from gradient_pathology.core import GradientPathology


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_analyzer_basic() -> None:
    """Test basic analyzer functionality."""
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

    analyzer = GradientAnalyzer(model, device="cpu")
    report = analyzer.diagnose(num_steps=10)

    assert report.num_steps == 10
    assert len(report.layer_stats) > 0
    assert report.global_mean is not None
    assert report.global_std >= 0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_vanishing_gradient_detection() -> None:
    """Test detection of vanishing or unstable gradients in deep networks."""
    # Deep sigmoid network should trigger gradient pathologies
    model = nn.Sequential(
        *[nn.Linear(64, 64), nn.Sigmoid()] * 30,
        nn.Linear(64, 1),
    )

    analyzer = GradientAnalyzer(model, device="cpu")
    report = analyzer.diagnose(num_steps=10, input_shape=(64,))

    # Deep sigmoid networks show vanishing OR unstable gradients
    pathologies = [stats.diagnose() for stats in report.layer_stats]
    problematic = [
        p for p in pathologies 
        if p in (GradientPathology.VANISHING, GradientPathology.UNSTABLE)
    ]
    assert len(problematic) > 0, "Deep sigmoid network should show gradient pathologies"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_healthy_gradient_flow() -> None:
    """Test that modern architectures show healthy gradients."""
    # Shallow ReLU network should be healthy
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

    analyzer = GradientAnalyzer(model, device="cpu")
    report = analyzer.diagnose(num_steps=10)

    # Most layers should be healthy
    healthy_count = sum(
        1 for stats in report.layer_stats if stats.diagnose() == GradientPathology.HEALTHY
    )
    assert healthy_count >= len(report.layer_stats) // 2


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_report_summary() -> None:
    """Test report summary generation."""
    model = nn.Sequential(nn.Linear(10, 1))
    analyzer = GradientAnalyzer(model, device="cpu")
    report = analyzer.diagnose(num_steps=5)

    summary = report.summary()
    assert "GRADIENT PATHOLOGY REPORT" in summary
    assert "Analysis over 5 steps" in summary
    assert len(summary) > 0
