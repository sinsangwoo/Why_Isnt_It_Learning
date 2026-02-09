"""Distributed training support (FSDP/DeepSpeed)."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    FSDP = None

from gradient_pathology.analyzer import GradientAnalyzer


class FSDPAnalyzer:
    """Gradient analysis for FSDP models."""

    def __init__(self, model: nn.Module, device: str = "cuda") -> None:
        if not FSDP_AVAILABLE:
            raise ImportError("FSDP not available")
        self.model = model
        self.device = device
        self.is_fsdp = isinstance(model, FSDP)

    def analyze_shard_gradients(self) -> Dict[str, Dict[str, float]]:
        """Analyze gradients across FSDP shards.

        Returns:
            Per-shard gradient statistics
        """
        if not self.is_fsdp:
            raise ValueError("Model is not FSDP")

        shard_stats = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                shard_stats[name] = {
                    "mean": float(grad.mean()),
                    "std": float(grad.std()),
                    "max": float(grad.abs().max()),
                    "norm": float(grad.norm()),
                }

        return shard_stats

    def check_shard_balance(self) -> Dict[str, float]:
        """Check if gradients are balanced across shards.

        Returns:
            Balance metrics
        """
        shard_stats = self.analyze_shard_gradients()

        norms = [s["norm"] for s in shard_stats.values()]
        return {
            "mean_norm": float(sum(norms) / len(norms)),
            "std_norm": float(torch.tensor(norms).std()),
            "imbalance_ratio": float(max(norms) / (min(norms) + 1e-10)),
        }

    def diagnose(self, num_steps: int = 10) -> str:
        """Run diagnostic on FSDP model.

        Args:
            num_steps: Number of diagnostic steps

        Returns:
            Diagnostic report
        """
        lines = ["=" * 70]
        lines.append("FSDP GRADIENT ANALYSIS")
        lines.append("=" * 70)

        balance = self.check_shard_balance()
        lines.append(f"\nShard Balance:")
        lines.append(f"  Imbalance ratio: {balance['imbalance_ratio']:.2f}")

        if balance["imbalance_ratio"] > 10.0:
            lines.append("  ⚠️  HIGH IMBALANCE")
            lines.append("  Recommendation: Check data distribution across ranks")

        return "\n".join(lines)
