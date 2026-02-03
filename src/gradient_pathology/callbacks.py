"""Training callbacks for real-time gradient monitoring."""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


class GradientMonitor:
    """Real-time gradient monitor for training loops.

    Example:
        >>> monitor = GradientMonitor(model, alert_threshold=1e-7)
        >>> for epoch in range(epochs):
        ...     loss.backward()
        ...     monitor.record_step()
        ...     if monitor.should_alert():
        ...         print(f"Alert: {monitor.get_alert_message()}")
    """

    def __init__(
        self,
        model: nn.Module,
        alert_threshold: float = 1e-8,
        window_size: int = 100,
    ):
        """Initialize gradient monitor.

        Args:
            model: PyTorch model to monitor
            alert_threshold: Gradient magnitude threshold for alerts
            window_size: Number of steps to keep in history
        """
        self.model = model
        self.alert_threshold = alert_threshold
        self.window_size = window_size
        self.history: List[Dict[str, Dict[str, float]]] = []
        self._alerts: List[str] = []

    def record_step(self) -> None:
        """Record gradient statistics for current step."""
        stats: Dict[str, Dict[str, float]] = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().cpu().numpy()
                stats[name] = {
                    "mean": float(np.mean(np.abs(grad))),
                    "std": float(np.std(grad)),
                    "max": float(np.max(np.abs(grad))),
                }

                # Check for pathologies
                if stats[name]["mean"] < self.alert_threshold:
                    self._alerts.append(f"Vanishing gradient in {name}: {stats[name]['mean']:.2e}")
                if stats[name]["mean"] > 1e3:
                    self._alerts.append(f"Exploding gradient in {name}: {stats[name]['mean']:.2e}")

        self.history.append(stats)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def should_alert(self) -> bool:
        """Check if there are any alerts."""
        return len(self._alerts) > 0

    def get_alert_message(self) -> str:
        """Get formatted alert message."""
        msg = "\n".join(self._alerts[-5:])  # Last 5 alerts
        self._alerts.clear()
        return msg

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated statistics over window."""
        if not self.history:
            return {}

        aggregated: Dict[str, Dict[str, float]] = {}
        for layer_name in self.history[0].keys():
            means = [step[layer_name]["mean"] for step in self.history]
            stds = [step[layer_name]["std"] for step in self.history]

            # Compute trend only if enough samples
            trend = 0.0
            if len(means) >= 2:
                try:
                    trend = float(np.polyfit(range(len(means)), means, 1)[0])
                except (np.linalg.LinAlgError, ValueError):
                    trend = 0.0

            aggregated[layer_name] = {
                "mean_of_means": float(np.mean(means)),
                "std_of_means": float(np.std(means)),
                "trend": trend,
            }

        return aggregated

    def diagnose_trends(self) -> List[str]:
        """Diagnose gradient trends over time."""
        stats = self.get_statistics()
        issues = []

        for layer_name, layer_stats in stats.items():
            # Degrading gradients
            if layer_stats["trend"] < -1e-5:
                issues.append(f"{layer_name}: Gradients degrading over time")
            # Unstable training (balanced threshold)
            if (
                layer_stats["std_of_means"] > 30 * layer_stats["mean_of_means"]
                and layer_stats["mean_of_means"] > 1e-6
            ):
                issues.append(f"{layer_name}: High variance (unstable training)")

        return issues
