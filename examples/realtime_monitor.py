#!/usr/bin/env python3
"""Example: Real-time gradient monitoring during training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from gradient_pathology.callbacks import GradientMonitor


def train_with_monitoring():
    """Train a model with real-time gradient monitoring."""
    # Create synthetic data
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create problematic deep model
    model = nn.Sequential(
        *[nn.Linear(64 if i > 0 else 10, 64), nn.Sigmoid()] * 30,
        nn.Linear(64, 1),
    )

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Initialize monitor
    monitor = GradientMonitor(model, alert_threshold=1e-7)

    print("🔬 Training with gradient monitoring...\n")

    for epoch in range(5):
        epoch_loss = 0.0

        for batch_x, batch_y in loader:
            # Forward pass
            output = model(batch_x)
            loss = criterion(output, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Monitor gradients BEFORE optimizer step
            monitor.record_step()

            # Check for alerts
            if monitor.should_alert():
                print(f"⚠️  Epoch {epoch + 1} Alert:")
                print(monitor.get_alert_message())
                print()

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1}/5 - Loss: {avg_loss:.4f}")

        # Periodic trend diagnosis
        if (epoch + 1) % 2 == 0:
            issues = monitor.diagnose_trends()
            if issues:
                print("\n📉 Trend Analysis:")
                for issue in issues[:5]:  # Top 5 issues
                    print(f"  - {issue}")
                print()

    print("\n✅ Training complete")
    print("\n📊 Final Statistics:")
    stats = monitor.get_statistics()
    for layer_name, layer_stats in list(stats.items())[:3]:  # First 3 layers
        print(f"  {layer_name}:")
        print(f"    Mean: {layer_stats['mean_of_means']:.2e}")
        print(f"    Trend: {layer_stats['trend']:.2e}")


if __name__ == "__main__":
    train_with_monitoring()
