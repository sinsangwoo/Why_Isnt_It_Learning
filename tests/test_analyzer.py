"""Tests for GradientAnalyzer."""

import torch
import torch.nn as nn

from gradient_pathology import GradientAnalyzer
from gradient_pathology.core import GradientPathology


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


def test_vanishing_gradient_detection() -> None:
    """Test detection of vanishing gradients."""
    # Deep sigmoid network should trigger vanishing gradients
    model = nn.Sequential(
        *[nn.Linear(64, 64), nn.Sigmoid()] * 30,
        nn.Linear(64, 1),
    )

    analyzer = GradientAnalyzer(model, device="cpu")
    report = analyzer.diagnose(num_steps=10, input_shape=(64,))

    # At least some layers should show vanishing
    pathologies = [stats.diagnose() for stats in report.layer_stats]
    assert GradientPathology.VANISHING in pathologies


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


def test_report_summary() -> None:
    """Test report summary generation."""
    model = nn.Sequential(nn.Linear(10, 1))
    analyzer = GradientAnalyzer(model, device="cpu")
    report = analyzer.diagnose(num_steps=5)

    summary = report.summary()
    assert "GRADIENT PATHOLOGY REPORT" in summary
    assert "Analysis over 5 steps" in summary
    assert len(summary) > 0
