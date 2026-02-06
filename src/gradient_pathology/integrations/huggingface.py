"""HuggingFace Transformers integration."""

from typing import Any, Dict, Optional

import torch.nn as nn

from gradient_pathology.analyzer import GradientAnalyzer


class HuggingFacePlugin:
    """Integration with HuggingFace Transformers."""

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.analyzer = GradientAnalyzer(model, device=device)
        self.reports: list = []

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Called at training start."""
        pass

    def on_step_end(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> Dict[str, float]:
        """Called after each training step."""
        if state.global_step % 100 == 0:
            model = kwargs.get("model")
            if model is not None:
                self.analyzer.model = model
                report = self.analyzer.diagnose(num_steps=10)
                self.reports.append(report)
                return {
                    "gradient_mean": float(report.global_mean),
                    "gradient_std": float(report.global_std),
                }
        return {}

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Called at training end."""
        pass
