"""Advanced transformer-specific diagnostics."""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class TransformerDiagnostics:
    """Advanced diagnostics for transformer models."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.attention_stats: Dict[str, Dict[str, float]] = {}
        self.ffn_stats: Dict[str, Dict[str, float]] = {}

    def analyze_attention_entropy(
        self, attention_weights: torch.Tensor, layer_name: str
    ) -> Dict[str, float]:
        """Analyze attention pattern entropy.

        Args:
            attention_weights: Shape (batch, heads, seq, seq)
            layer_name: Layer identifier

        Returns:
            Entropy statistics
        """
        attn = attention_weights.detach().cpu().numpy()
        entropies = []

        for b in range(attn.shape[0]):
            for h in range(attn.shape[1]):
                probs = attn[b, h] + 1e-10
                probs = probs / probs.sum(axis=-1, keepdims=True)
                entropy = -np.sum(probs * np.log(probs), axis=-1).mean()
                entropies.append(entropy)

        stats = {
            "mean_entropy": float(np.mean(entropies)),
            "std_entropy": float(np.std(entropies)),
            "min_entropy": float(np.min(entropies)),
            "max_entropy": float(np.max(entropies)),
        }

        self.attention_stats[layer_name] = stats
        return stats

    def detect_attention_collapse(
        self, attention_weights: torch.Tensor, threshold: float = 0.1
    ) -> bool:
        """Detect if attention has collapsed.

        Args:
            attention_weights: Attention matrix
            threshold: Entropy threshold for collapse

        Returns:
            True if collapsed
        """
        stats = self.analyze_attention_entropy(attention_weights, "temp")
        return stats["mean_entropy"] < threshold

    def analyze_ffn_saturation(
        self, ffn_activations: torch.Tensor, layer_name: str
    ) -> Dict[str, float]:
        """Analyze FFN activation saturation.

        Args:
            ffn_activations: FFN intermediate activations
            layer_name: Layer identifier

        Returns:
            Saturation statistics
        """
        acts = ffn_activations.detach().cpu().numpy()

        near_zero = np.abs(acts) < 0.01
        near_max = np.abs(acts) > 0.99 * np.abs(acts).max()

        stats = {
            "zero_fraction": float(near_zero.mean()),
            "saturated_fraction": float(near_max.mean()),
            "mean_activation": float(np.abs(acts).mean()),
            "std_activation": float(np.abs(acts).std()),
        }

        self.ffn_stats[layer_name] = stats
        return stats

    def detect_ffn_saturation(self, ffn_activations: torch.Tensor) -> bool:
        """Detect if FFN is saturated.

        Args:
            ffn_activations: FFN activations

        Returns:
            True if saturated
        """
        stats = self.analyze_ffn_saturation(ffn_activations, "temp")
        return stats["saturated_fraction"] > 0.5 or stats["zero_fraction"] > 0.8

    def generate_report(self) -> str:
        """Generate diagnostic report."""
        lines = ["=" * 70]
        lines.append("TRANSFORMER DIAGNOSTICS")
        lines.append("=" * 70)

        if self.attention_stats:
            lines.append("\nAttention Analysis:")
            for layer, stats in self.attention_stats.items():
                lines.append(f"  {layer}:")
                lines.append(f"    Entropy: {stats['mean_entropy']:.3f}")
                if stats["mean_entropy"] < 0.1:
                    lines.append("    ⚠️  COLLAPSED")

        if self.ffn_stats:
            lines.append("\nFFN Analysis:")
            for layer, stats in self.ffn_stats.items():
                lines.append(f"  {layer}:")
                lines.append(f"    Saturation: {stats['saturated_fraction']:.3f}")
                lines.append(f"    Dead neurons: {stats['zero_fraction']:.3f}")
                if stats["saturated_fraction"] > 0.5:
                    lines.append("    ⚠️  SATURATED")
                if stats["zero_fraction"] > 0.8:
                    lines.append("    ⚠️  DEAD NEURONS")

        return "\n".join(lines)
