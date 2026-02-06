"""Ray Tune integration."""

from typing import Any, Dict, Optional

import torch.nn as nn

try:
    from ray import tune
    from ray.tune import Trainable

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    tune = None
    Trainable = object

from gradient_pathology.analyzer import GradientAnalyzer


class GradientPathologyReporter:
    """Report gradient metrics to Ray Tune."""

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        if not RAY_AVAILABLE:
            raise ImportError("Ray Tune not installed")
        self.analyzer = GradientAnalyzer(model, device=device)

    def report_metrics(self, step: int) -> Dict[str, float]:
        """Compute and report gradient metrics."""
        report = self.analyzer.diagnose(num_steps=10)
        metrics = {
            "gradient_mean": float(report.global_mean),
            "gradient_std": float(report.global_std),
        }
        if RAY_AVAILABLE and tune:
            tune.report(**metrics, training_iteration=step)
        return metrics
