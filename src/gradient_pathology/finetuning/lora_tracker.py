"""LoRA gradient effective-rank tracker.

Tracks the gradient effective rank of every LoRA projection pair.

Effective rank (Roy & Vetterli, 2007)::

    effective_rank(M) = exp(-sum(p_i * log(p_i)))
    where p_i = sigma_i / sum(sigma_j)   (normalised singular values)

Usage::

    from gradient_pathology.finetuning import LoRARankTracker

    tracker = LoRARankTracker(model)
    for batch in dataloader:
        loss = compute_loss(model, batch)
        loss.backward()
        tracker.step()
        optimizer.step()
        optimizer.zero_grad()

    for layer, info in tracker.report().items():
        print(f"{layer}: eff={info['effective_rank']:.2f}/{info['rank']} "
              f"({info['rank_utilisation']:.1%}) [{info['status']}]")
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch.nn as nn


def _effective_rank(matrix: np.ndarray) -> float:
    """Compute effective rank of a 2-D matrix via singular-value entropy."""
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    s = s[s > 1e-10]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    entropy = -float(np.sum(p * np.log(p + 1e-12)))
    return float(np.exp(entropy))


class LoRARankTracker:
    """Tracks gradient effective rank for LoRA adapter matrices.

    Parameters
    ----------
    model:
        PyTorch model (typically loaded via HuggingFace PEFT).
    lora_pattern:
        Regex identifying LoRA parameter names.  Defaults to ``"lora_"``.
    min_rank_utilisation:
        Threshold below which a layer is flagged as ``'rank_collapse'``.
        Default ``0.4``.
    history_window:
        Number of recent steps for rolling averages.  Default ``50``.
    """

    def __init__(
        self,
        model: nn.Module,
        lora_pattern: str = "lora_",
        min_rank_utilisation: float = 0.4,
        history_window: int = 50,
    ) -> None:
        self.model = model
        self.lora_pattern = re.compile(lora_pattern)
        self.min_rank_utilisation = min_rank_utilisation
        self.history_window = history_window
        self._lora_pairs: Dict[str, Dict[str, nn.Parameter]] = {}
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._discover_lora_params()

    def step(self) -> None:
        """Record gradient effective rank after a backward pass."""
        for base_name, pair in self._lora_pairs.items():
            grad = self._get_combined_grad(pair)
            if grad is None:
                continue
            eff_rank = _effective_rank(grad)
            history = self._history[base_name]
            history.append(eff_rank)
            if len(history) > self.history_window:
                history.pop(0)

    def report(self) -> Dict[str, Dict]:
        """Return a summary dict keyed by LoRA layer base name.

        Each entry contains: ``rank``, ``effective_rank``,
        ``rank_utilisation``, ``status``, ``trend``.
        """
        result: Dict[str, Dict] = {}
        for base_name, pair in self._lora_pairs.items():
            rank = self._infer_rank(pair)
            history = self._history[base_name]
            if not history:
                result[base_name] = {
                    "rank": rank, "effective_rank": 0.0,
                    "rank_utilisation": 0.0, "status": "no_data", "trend": "unknown",
                }
                continue
            window = history[-min(10, len(history)):]
            eff_rank = float(np.mean(window))
            utilisation = eff_rank / max(rank, 1)
            result[base_name] = {
                "rank": rank,
                "effective_rank": round(eff_rank, 3),
                "rank_utilisation": round(utilisation, 4),
                "status": "healthy" if utilisation >= self.min_rank_utilisation else "rank_collapse",
                "trend": self._compute_trend(history),
            }
        return result

    def collapsed_layers(self) -> List[str]:
        """Return base names of layers currently in rank collapse."""
        return [n for n, i in self.report().items() if i["status"] == "rank_collapse"]

    def _discover_lora_params(self) -> None:
        for name, param in self.model.named_parameters():
            if not self.lora_pattern.search(name):
                continue
            base = re.sub(r"lora_[AB](\.weight)?", "", name).rstrip(".")
            key = "A" if ("lora_A" in name or name.endswith("_A")) else "B"
            self._lora_pairs.setdefault(base, {})[key] = param

    def _get_combined_grad(self, pair: Dict[str, nn.Parameter]) -> Optional[np.ndarray]:
        for key in ("A", "B"):
            p = pair.get(key)
            if p is not None and p.grad is not None:
                return p.grad.detach().cpu().float().numpy()
        return None

    @staticmethod
    def _infer_rank(pair: Dict[str, nn.Parameter]) -> int:
        a = pair.get("A")
        b = pair.get("B")
        if a is not None and a.ndim >= 2:
            return a.shape[0]
        if b is not None and b.ndim >= 2:
            return b.shape[1]
        return 1

    @staticmethod
    def _compute_trend(history: List[float]) -> str:
        if len(history) < 6:
            return "unknown"
        recent = np.mean(history[-3:])
        older = np.mean(history[-6:-3])
        delta = recent - older
        if abs(delta) < 0.05 * max(float(older), 1e-6):
            return "stable"
        return "improving" if delta > 0 else "degrading"
