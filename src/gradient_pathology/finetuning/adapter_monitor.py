"""Adapter vs. frozen-layer gradient magnitude monitor.

Two common LoRA fine-tuning failure modes:

1. **Adapter dominance loss** -- adapter gradients collapse relative to base
   gradients; the adapter is not driving learning.
2. **Frozen-layer leakage** -- frozen parameters accumulate non-zero gradients
   (requires_grad accidentally True, or optimizer misconfigured), causing
   unintended base-model drift.

Usage::

    from gradient_pathology.finetuning import AdapterMonitor

    monitor = AdapterMonitor(model)
    for batch in dataloader:
        loss = compute_loss(model, batch)
        loss.backward()
        monitor.step()
        optimizer.step()
        optimizer.zero_grad()

    print(monitor.summary())
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import torch.nn as nn


class AdapterMonitor:
    """Monitors gradient magnitude ratio between adapter and frozen parameters.

    Parameters
    ----------
    model:
        PyTorch model.
    adapter_pattern:
        Regex matching adapter (trainable) parameter names.
        Defaults to ``"lora_"``.
    warn_ratio_low:
        Adapter-to-base ratio below this value triggers ``'adapter_weak'``.
        Default ``0.1``.
    warn_frozen_leak:
        Mean absolute gradient above this value on a *frozen* parameter
        triggers a leakage warning.  Default ``1e-6``.
    history_window:
        Steps to retain for rolling statistics.  Default ``50``.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter_pattern: str = "lora_",
        warn_ratio_low: float = 0.1,
        warn_frozen_leak: float = 1e-6,
        history_window: int = 50,
    ) -> None:
        self.model = model
        self.adapter_pattern = re.compile(adapter_pattern)
        self.warn_ratio_low = warn_ratio_low
        self.warn_frozen_leak = warn_frozen_leak
        self.history_window = history_window
        self._adapter_params: Dict[str, nn.Parameter] = {}
        self._frozen_params: Dict[str, nn.Parameter] = {}
        self._ratio_history: List[float] = []
        self._leaking_layers: Dict[str, List[float]] = {}
        self._classify_params()

    def step(self) -> None:
        """Record gradient statistics after a backward pass."""
        adapter_mag = self._mean_abs_grad(self._adapter_params)
        frozen_mag = self._mean_abs_grad(self._frozen_params)

        if frozen_mag is not None and adapter_mag is not None and frozen_mag > 1e-12:
            ratio = adapter_mag / frozen_mag
        elif adapter_mag is not None:
            ratio = float(self.history_window)  # no frozen grad -- ideal
        else:
            ratio = 0.0

        self._ratio_history.append(ratio)
        if len(self._ratio_history) > self.history_window:
            self._ratio_history.pop(0)

        for name, param in self._frozen_params.items():
            if param.grad is None:
                continue
            leak = float(param.grad.detach().abs().mean().item())
            if leak > self.warn_frozen_leak:
                bucket = self._leaking_layers.setdefault(name, [])
                bucket.append(leak)
                if len(bucket) > self.history_window:
                    bucket.pop(0)

    @property
    def adapter_ratio(self) -> Optional[float]:
        """Rolling-average adapter-to-frozen gradient magnitude ratio."""
        if not self._ratio_history:
            return None
        window = self._ratio_history[-min(10, len(self._ratio_history)):]
        return float(np.mean(window))

    @property
    def is_adapter_weak(self) -> bool:
        """True when adapter gradients are suspiciously small vs. base."""
        r = self.adapter_ratio
        return r is not None and r < self.warn_ratio_low

    @property
    def leaking_layers(self) -> List[str]:
        """Names of frozen parameters that have non-trivial gradients."""
        return list(self._leaking_layers.keys())

    def summary(self) -> str:
        """Return a human-readable diagnostic string."""
        lines = ["=" * 60, "ADAPTER MONITOR REPORT", "=" * 60]
        lines.append(f"Adapter params : {len(self._adapter_params)}")
        lines.append(f"Frozen params  : {len(self._frozen_params)}")
        ratio = self.adapter_ratio
        if ratio is not None:
            lines.append(f"Adapter/frozen ratio : {ratio:.4f}")
            if self.is_adapter_weak:
                lines.append(
                    "  Warning: adapter gradients are weak.\n"
                    "  Consider a higher LoRA learning rate or larger rank."
                )
        else:
            lines.append("Adapter/frozen ratio : no data yet")
        if self._leaking_layers:
            lines.append(f"\nFrozen-layer leakage detected ({len(self._leaking_layers)} layers):")
            for name in list(self._leaking_layers)[:5]:
                avg = np.mean(self._leaking_layers[name])
                lines.append(f"  {name}  mean|grad|={avg:.2e}")
            lines.append("  Verify requires_grad=False and exclude from optimizer.")
        else:
            lines.append("\nNo frozen-layer gradient leakage detected.")
        return "\n".join(lines)

    def _classify_params(self) -> None:
        for name, param in self.model.named_parameters():
            if self.adapter_pattern.search(name):
                self._adapter_params[name] = param
            else:
                self._frozen_params[name] = param

    @staticmethod
    def _mean_abs_grad(params: Dict[str, nn.Parameter]) -> Optional[float]:
        values = [
            float(p.grad.detach().abs().mean().item())
            for p in params.values()
            if p.grad is not None
        ]
        return float(np.mean(values)) if values else None
