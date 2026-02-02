"""Main gradient analysis engine."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from gradient_pathology.core import GradientReport, LayerGradientStats


class GradientAnalyzer:
    """Analyzes gradient flow in PyTorch models.

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
        >>> analyzer = GradientAnalyzer(model)
        >>> report = analyzer.diagnose(num_steps=100)
        >>> print(report.summary())
    """

    def __init__(self, model: nn.Module, device: Optional[str] = None):
        """Initialize analyzer.

        Args:
            model: PyTorch model to analyze
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def diagnose(
        self,
        num_steps: int = 100,
        batch_size: int = 32,
        input_shape: tuple = (10,),
        loss_fn: Optional[nn.Module] = None,
    ) -> GradientReport:
        """Run gradient diagnosis.

        Args:
            num_steps: Number of forward/backward passes
            batch_size: Batch size for synthetic data
            input_shape: Shape of input data (excluding batch dimension)
            loss_fn: Loss function (defaults to MSE for regression)

        Returns:
            GradientReport with detailed analysis
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        # Collect gradients over multiple steps
        gradient_history = {name: [] for name, _ in self.model.named_parameters()}

        self.model.train()
        for _ in tqdm(range(num_steps), desc="Analyzing gradients"):
            # Generate synthetic data
            x = torch.randn(batch_size, *input_shape, device=self.device)
            target = torch.randn(batch_size, 1, device=self.device)

            # Forward pass
            self.model.zero_grad()
            output = self.model(x)
            loss = loss_fn(output, target)

            # Backward pass
            loss.backward()

            # Record gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradient_history[name].append(param.grad.detach().cpu().numpy().copy())

        # Compute statistics
        layer_stats = []
        all_gradients = []

        for idx, (name, grads) in enumerate(gradient_history.items()):
            if not grads:
                continue

            # Aggregate across steps
            grads_array = np.concatenate([g.flatten() for g in grads])
            all_gradients.append(grads_array)

            stats = LayerGradientStats(
                layer_name=name,
                layer_index=idx,
                mean=float(np.mean(grads_array)),
                std=float(np.std(grads_array)),
                min=float(np.min(grads_array)),
                max=float(np.max(grads_array)),
                median=float(np.median(grads_array)),
                num_zeros=int(np.sum(grads_array == 0)),
                total_params=len(grads_array),
            )
            layer_stats.append(stats)

        # Global statistics
        all_gradients_flat = np.concatenate(all_gradients)
        global_mean = float(np.mean(all_gradients_flat))
        global_std = float(np.std(all_gradients_flat))

        return GradientReport(
            layer_stats=layer_stats,
            global_mean=global_mean,
            global_std=global_std,
            num_steps=num_steps,
        )
