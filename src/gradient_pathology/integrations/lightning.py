"""PyTorch Lightning integration."""

from typing import Any, Optional

import torch
import torch.nn as nn

try:
    from pytorch_lightning import Callback
    from pytorch_lightning import Trainer

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    Callback = object
    Trainer = object

from gradient_pathology.analyzer import GradientAnalyzer


class GradientPathologyCallback(Callback):  # type: ignore
    """Lightning callback for gradient monitoring."""

    def __init__(
        self,
        check_every_n_steps: int = 100,
        num_diagnostic_steps: int = 10,
    ) -> None:
        if not LIGHTNING_AVAILABLE:
            raise ImportError("PyTorch Lightning not installed")
        self.check_every_n_steps = check_every_n_steps
        self.num_diagnostic_steps = num_diagnostic_steps
        self.analyzer: Optional[GradientAnalyzer] = None

    def on_train_start(self, trainer: Any, pl_module: nn.Module) -> None:
        """Initialize analyzer."""
        device = str(pl_module.device)
        self.analyzer = GradientAnalyzer(pl_module, device=device)

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: nn.Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Check gradients periodically."""
        if trainer.global_step % self.check_every_n_steps == 0 and self.analyzer:
            report = self.analyzer.diagnose(num_steps=self.num_diagnostic_steps)
            pl_module.log("gradient/mean", report.global_mean)
            pl_module.log("gradient/std", report.global_std)
