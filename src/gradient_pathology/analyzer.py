"""Main gradient analysis engine."""

from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from gradient_pathology.core import GradientReport, LayerGradientStats


class GradientAnalyzer:
    """Analyzes gradient flow in PyTorch models.

    Supports two modes:
    1. Real data mode (recommended): pass a ``dataloader`` and ``loss_fn``.
       Gradients are computed on your actual training distribution, making
       the diagnostics actionable.
    2. Synthetic mode (legacy / quick-check): omit ``dataloader``.  A random
       Gaussian input is used.  Results indicate architectural issues but
       cannot reflect data-specific pathologies.

    Example — real data::

        from torch.utils.data import DataLoader, TensorDataset
        import torch

        X = torch.randn(256, 10)
        y = torch.randn(256, 1)
        loader = DataLoader(TensorDataset(X, y), batch_size=32)

        model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
        analyzer = GradientAnalyzer(model)
        report = analyzer.diagnose(dataloader=loader, loss_fn=nn.MSELoss())
        print(report.summary())

    Example — synthetic (backward-compatible)::

        analyzer = GradientAnalyzer(model)
        report = analyzer.diagnose(num_steps=100, input_shape=(10,))
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        """Initialize analyzer.

        Args:
            model: PyTorch model to analyze.
            device: Device to run on (``'cuda'`` or ``'cpu'``).
        """
        self.model = model
        self.device = device
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diagnose(
        self,
        dataloader: Optional[object] = None,
        loss_fn: Optional[nn.Module] = None,
        # Legacy / synthetic-mode parameters
        num_steps: int = 100,
        batch_size: int = 32,
        input_shape: Tuple[int, ...] = (10,),
    ) -> GradientReport:
        """Run gradient diagnosis.

        Args:
            dataloader: A PyTorch ``DataLoader`` (or any iterable yielding
                ``(inputs, targets)`` pairs).  When provided, real data is
                used and ``num_steps`` / ``batch_size`` / ``input_shape``
                are ignored.
            loss_fn: Loss function.  Defaults to ``nn.MSELoss()``.
            num_steps: *Synthetic mode only.* Number of forward/backward
                passes with random inputs.
            batch_size: *Synthetic mode only.* Batch size.
            input_shape: *Synthetic mode only.* Input shape excluding the
                batch dimension.

        Returns:
            :class:`~gradient_pathology.core.GradientReport` with per-layer
            statistics and a ``data_source`` field indicating which mode was
            used.
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        if dataloader is not None:
            return self._diagnose_with_dataloader(dataloader, loss_fn)
        else:
            return self._diagnose_synthetic(
                num_steps=num_steps,
                batch_size=batch_size,
                input_shape=input_shape,
                loss_fn=loss_fn,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_gradients(
        self,
        batch_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        loss_fn: nn.Module,
        num_steps: int,
        desc: str,
    ) -> Dict[str, List[np.ndarray]]:
        """Run forward/backward passes and collect per-parameter gradients."""
        gradient_history: Dict[str, List[np.ndarray]] = {
            name: [] for name, _ in self.model.named_parameters()
        }

        self.model.train()
        for _ in tqdm(range(num_steps), desc=desc):
            try:
                inputs, targets = next(batch_iter)
            except StopIteration:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.zero_grad()
            output = self.model(inputs)

            # Handle shape mismatch between output and target gracefully
            if output.shape != targets.shape:
                targets = targets.view_as(output)

            loss = loss_fn(output, targets)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradient_history[name].append(
                        param.grad.detach().cpu().numpy().copy()
                    )

        return gradient_history

    def _build_report(
        self,
        gradient_history: Dict[str, List[np.ndarray]],
        num_steps: int,
        data_source: str,
    ) -> GradientReport:
        """Compute statistics from collected gradients and build a report."""
        layer_stats: List[LayerGradientStats] = []
        all_gradients: List[np.ndarray] = []

        for idx, (name, grads) in enumerate(gradient_history.items()):
            if not grads:
                continue

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

        if not all_gradients:
            return GradientReport(
                layer_stats=[],
                global_mean=0.0,
                global_std=0.0,
                num_steps=num_steps,
                data_source=data_source,
            )

        all_flat = np.concatenate(all_gradients)
        return GradientReport(
            layer_stats=layer_stats,
            global_mean=float(np.mean(all_flat)),
            global_std=float(np.std(all_flat)),
            num_steps=num_steps,
            data_source=data_source,
        )

    def _diagnose_with_dataloader(
        self,
        dataloader: object,
        loss_fn: nn.Module,
    ) -> GradientReport:
        """Diagnose using a real DataLoader."""
        # Wrap in an infinite iterator so we can use next() uniformly
        def _cycle(loader: object) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
            while True:
                for batch in loader:  # type: ignore[attr-defined]
                    yield batch

        # Determine num_steps from dataset length when possible
        num_steps: int
        try:
            num_steps = len(dataloader)  # type: ignore[arg-type]
        except TypeError:
            num_steps = 100

        batch_iter = _cycle(dataloader)
        gradient_history = self._collect_gradients(
            batch_iter=batch_iter,
            loss_fn=loss_fn,
            num_steps=num_steps,
            desc="Analyzing gradients (real data)",
        )
        return self._build_report(
            gradient_history, num_steps=num_steps, data_source="dataloader"
        )

    def _diagnose_synthetic(
        self,
        num_steps: int,
        batch_size: int,
        input_shape: Tuple[int, ...],
        loss_fn: nn.Module,
    ) -> GradientReport:
        """Diagnose using synthetic random inputs (legacy mode)."""

        def _synthetic_iter() -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
            while True:
                x = torch.randn(batch_size, *input_shape)
                y = torch.randn(batch_size, 1)
                yield x, y

        gradient_history = self._collect_gradients(
            batch_iter=_synthetic_iter(),
            loss_fn=loss_fn,
            num_steps=num_steps,
            desc="Analyzing gradients (synthetic)",
        )
        return self._build_report(
            gradient_history, num_steps=num_steps, data_source="synthetic"
        )
