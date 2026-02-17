"""Catastrophic forgetting early-warning detector.

Detects *precursor signals* before forgetting manifests in validation loss:

1. **Gradient direction drift** -- if cosine similarity between consecutive
   base-layer gradients falls consistently, the optimiser is being pulled in
   rapidly-changing directions, a sign that new-task gradients are overwriting
   old knowledge.

2. **Gradient conflict** -- when adapter (new-task) and base-layer gradients
   point in opposite directions (cosine similarity < 0), the fine-tuning step
   is actively destructive to pretrained representations.

Usage::

    from gradient_pathology.finetuning import ForgettingDetector

    detector = ForgettingDetector(model, sensitivity=0.7)
    for step, batch in enumerate(dataloader):
        loss = compute_loss(model, batch)
        loss.backward()
        detector.step()
        optimizer.step()
        optimizer.zero_grad()

        if detector.warning_triggered:
            print(f"Step {step}: risk={detector.forgetting_risk:.2f}")
            print(detector.summary())
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import torch.nn as nn


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class ForgettingDetector:
    """Detects catastrophic forgetting precursors before validation loss rises.

    Parameters
    ----------
    model:
        PyTorch model.
    adapter_pattern:
        Regex matching adapter parameter names.  Base parameters are everything
        else.
    sensitivity:
        Cosine-similarity threshold below which gradient drift is flagged.
        Higher -> more sensitive.  Default ``0.7``.
    conflict_threshold:
        Adapter-vs-base cosine similarity below this value triggers a conflict
        warning.  Default ``0.0``.
    history_window:
        Steps to retain.  Default ``100``.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter_pattern: str = "lora_",
        sensitivity: float = 0.7,
        conflict_threshold: float = 0.0,
        history_window: int = 100,
    ) -> None:
        self.model = model
        self.adapter_pattern = re.compile(adapter_pattern)
        self.sensitivity = sensitivity
        self.conflict_threshold = conflict_threshold
        self.history_window = history_window
        self._adapter_params: Dict[str, nn.Parameter] = {}
        self._base_params: Dict[str, nn.Parameter] = {}
        self._prev_base_grad: Optional[np.ndarray] = None
        self._drift_history: List[float] = []
        self._conflict_history: List[float] = []
        self._classify_params()

    def step(self) -> None:
        """Record gradient signals after a backward pass."""
        base_grad = self._flat_grad(self._base_params)
        adapter_grad = self._flat_grad(self._adapter_params)

        if self._prev_base_grad is not None and base_grad is not None:
            sim = _cosine_sim(base_grad, self._prev_base_grad)
            self._drift_history.append(sim)
            if len(self._drift_history) > self.history_window:
                self._drift_history.pop(0)

        if base_grad is not None:
            self._prev_base_grad = base_grad.copy()

        if adapter_grad is not None and base_grad is not None:
            sim = _cosine_sim(adapter_grad, base_grad)
            self._conflict_history.append(sim)
            if len(self._conflict_history) > self.history_window:
                self._conflict_history.pop(0)

    @property
    def forgetting_risk(self) -> float:
        """Estimated forgetting risk in [0, 1].  Higher is worse."""
        return min(1.0, 0.6 * self._drift_score() + 0.4 * self._conflict_score())

    @property
    def warning_triggered(self) -> bool:
        """True when forgetting_risk exceeds sensitivity threshold."""
        return self.forgetting_risk >= self.sensitivity

    def summary(self) -> str:
        """Return a human-readable diagnostic string."""
        risk = self.forgetting_risk
        status = "HIGH" if risk > 0.7 else "MEDIUM" if risk > 0.4 else "LOW"
        dw = min(10, max(1, len(self._drift_history)))
        cw = min(10, max(1, len(self._conflict_history)))
        drift = float(np.mean(self._drift_history[-dw:])) if self._drift_history else None
        conflict = float(np.mean(self._conflict_history[-cw:])) if self._conflict_history else None

        lines = [
            "=" * 60,
            "FORGETTING DETECTOR REPORT",
            "=" * 60,
            f"Forgetting risk : {risk:.3f}  [{status}]",
        ]
        if drift is not None:
            lines.append(f"Gradient drift  : mean cosine sim = {drift:.3f}  "
                         f"(target >= {self.sensitivity:.2f})")
        if conflict is not None:
            lines.append(f"Grad conflict   : adapter.base sim = {conflict:.3f}  "
                         f"(threshold {self.conflict_threshold:.2f})")

        if self.warning_triggered:
            lines.append("\nWARNING: Forgetting precursors detected.")
            if drift is not None and drift < self.sensitivity:
                lines.append("  - Reduce LoRA learning rate (base gradient direction unstable)")
                lines.append("  - Add learning-rate warmup / cosine schedule")
            if conflict is not None and conflict < self.conflict_threshold:
                lines.append("  - Consider gradient projection (GradOrth or AdaLoRA)")
                lines.append("  - Try a smaller LoRA rank to reduce interference")
        else:
            lines.append("\nNo forgetting precursors detected.")
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all history (e.g., between training phases)."""
        self._drift_history.clear()
        self._conflict_history.clear()
        self._prev_base_grad = None

    def _classify_params(self) -> None:
        for name, param in self.model.named_parameters():
            if self.adapter_pattern.search(name):
                self._adapter_params[name] = param
            else:
                self._base_params[name] = param

    @staticmethod
    def _flat_grad(params: Dict[str, nn.Parameter]) -> Optional[np.ndarray]:
        parts = [
            p.grad.detach().cpu().float().numpy().ravel()
            for p in params.values()
            if p.grad is not None
        ]
        return np.concatenate(parts) if parts else None

    def _drift_score(self) -> float:
        if not self._drift_history:
            return 0.0
        window = min(20, len(self._drift_history))
        mean_sim = float(np.mean(self._drift_history[-window:]))
        return float(np.clip(1.0 - mean_sim, 0.0, 1.0))

    def _conflict_score(self) -> float:
        if not self._conflict_history:
            return 0.0
        window = min(20, len(self._conflict_history))
        mean_sim = float(np.mean(self._conflict_history[-window:]))
        return float(np.clip(-mean_sim, 0.0, 1.0))
