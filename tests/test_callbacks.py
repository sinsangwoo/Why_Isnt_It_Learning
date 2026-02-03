"""Tests for gradient monitoring callbacks."""

import pytest
import torch
import torch.nn as nn

from gradient_pathology.callbacks import GradientMonitor


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_gradient_monitor_basic() -> None:
    """Test basic gradient monitoring."""
    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
    monitor = GradientMonitor(model)

    # Simulate training step
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()

    monitor.record_step()
    stats = monitor.get_statistics()

    assert len(stats) > 0
    assert not monitor.should_alert()  # Healthy model


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_gradient_monitor_alerts() -> None:
    """Test alert system for gradient pathologies."""
    # Deep sigmoid network
    model = nn.Sequential(*[nn.Linear(64, 64), nn.Sigmoid()] * 20, nn.Linear(64, 1))
    monitor = GradientMonitor(model, alert_threshold=1e-6)

    # Simulate training
    for _ in range(5):
        x = torch.randn(32, 64)
        y = torch.randn(32, 1)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        monitor.record_step()
        model.zero_grad()

    # Should detect vanishing gradients
    assert monitor.should_alert()
    message = monitor.get_alert_message()
    assert "Vanishing" in message or "gradient" in message.lower()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_gradient_monitor_statistics() -> None:
    """Test statistical aggregation."""
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
    monitor = GradientMonitor(model, window_size=10)

    # Record multiple steps
    for _ in range(15):
        x = torch.randn(16, 10)
        y = torch.randn(16, 1)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        monitor.record_step()
        model.zero_grad()

    stats = monitor.get_statistics()
    assert len(stats) > 0

    # Check window size
    assert len(monitor.history) == 10  # Should keep only last 10

    # Check trend analysis
    trends = monitor.diagnose_trends()
    assert isinstance(trends, list)
