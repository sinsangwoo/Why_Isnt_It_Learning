"""Tests for gradient analyzer — both synthetic and real-data modes."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gradient_pathology.analyzer import GradientAnalyzer
from gradient_pathology.core import GradientPathology


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader(
    n_samples: int = 128,
    input_dim: int = 10,
    batch_size: int = 32,
) -> DataLoader:  # type: ignore[type-arg]
    """Create a minimal regression DataLoader for testing."""
    X = torch.randn(n_samples, input_dim)
    y = torch.randn(n_samples, 1)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Synthetic-mode tests (backward-compatible)
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_analyzer_synthetic_basic() -> None:
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    analyzer = GradientAnalyzer(model, device="cpu")
    report = analyzer.diagnose(num_steps=10, input_shape=(10,))

    assert len(report.layer_stats) > 0
    assert report.data_source == "synthetic"
    assert report.global_mean is not None
    assert report.global_std is not None


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_vanishing_gradient_detection_synthetic() -> None:
    """Deep sigmoid stack should trigger at least one pathology."""
    model = nn.Sequential(
        *[nn.Linear(64, 64), nn.Sigmoid()] * 100,
        nn.Linear(64, 1),
    )
    analyzer = GradientAnalyzer(model, device="cpu")
    report = analyzer.diagnose(num_steps=50, input_shape=(64,))

    pathological_count = sum(
        1 for s in report.layer_stats
        if s.diagnose() != GradientPathology.HEALTHY
    )
    assert pathological_count > 0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_healthy_gradient_flow_synthetic() -> None:
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
        1 for s in report.layer_stats
        if s.diagnose() == GradientPathology.HEALTHY
    )
    assert healthy_count >= 1


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_report_summary_synthetic() -> None:
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
    analyzer = GradientAnalyzer(model, device="cpu")
    report = analyzer.diagnose(num_steps=5, input_shape=(10,))

    summary = report.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
    # Synthetic mode should surface a warning in the summary
    assert "synthetic" in summary.lower()


# ---------------------------------------------------------------------------
# Real-data (DataLoader) mode tests
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_analyzer_with_real_dataloader() -> None:
    """Diagnose using an actual DataLoader — data_source must be 'dataloader'."""
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
    loader = _make_loader(input_dim=10)

    analyzer = GradientAnalyzer(model, device="cpu")
    report = analyzer.diagnose(dataloader=loader, loss_fn=nn.MSELoss())

    assert report.data_source == "dataloader"
    assert len(report.layer_stats) > 0
    assert report.num_steps == len(loader)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dataloader_mode_no_synthetic_warning() -> None:
    """Summary for real-data mode must not show the synthetic warning."""
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))
    loader = _make_loader(input_dim=10)

    analyzer = GradientAnalyzer(model, device="cpu")
    report = analyzer.diagnose(dataloader=loader, loss_fn=nn.MSELoss())

    summary = report.summary()
    assert "SYNTHETIC" not in summary


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_problematic_layers_populated() -> None:
    """get_problematic_layers() should return a list (empty or non-empty)."""
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
    loader = _make_loader(input_dim=10)

    analyzer = GradientAnalyzer(model, device="cpu")
    report = analyzer.diagnose(dataloader=loader, loss_fn=nn.MSELoss())

    problematic = report.get_problematic_layers()
    assert isinstance(problematic, list)
