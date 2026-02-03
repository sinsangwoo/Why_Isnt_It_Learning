"""Transformer-specific gradient diagnostics."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional


class TransformerDiagnostics:
    """Specialized diagnostics for Transformer architectures.
    
    Detects:
    - Attention entropy collapse
    - FFN saturation
    - Layer norm instability
    - Gradient flow through residual connections
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.attention_entropies: List[float] = []
        self.ffn_saturations: List[float] = []

    def analyze_attention_entropy(
        self,
        attention_weights: torch.Tensor,
    ) -> float:
        """Compute entropy of attention distribution.
        
        Low entropy = Attention is focused (few tokens attended)
        High entropy = Attention is diffuse (many tokens attended)
        
        Args:
            attention_weights: [batch, heads, seq_len, seq_len]
            
        Returns:
            Mean entropy across batch and heads
        """
        # Avoid log(0)
        attention_weights = attention_weights + 1e-10
        
        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights),
            dim=-1
        )
        
        mean_entropy = entropy.mean().item()
        self.attention_entropies.append(mean_entropy)
        
        return mean_entropy
    
    def detect_attention_collapse(self, threshold: float = 0.1) -> bool:
        """Detect if attention has collapsed.
        
        Attention collapse = Model attends to very few tokens.
        """
        if not self.attention_entropies:
            return False
        
        recent_entropy = np.mean(self.attention_entropies[-10:])
        
        return recent_entropy < threshold
    
    def analyze_ffn_saturation(
        self,
        ffn_activations: torch.Tensor,
        activation_fn: str = "gelu",
    ) -> float:
        """Measure FFN saturation.
        
        High saturation = Many neurons at activation extremes
        
        Args:
            ffn_activations: FFN layer outputs
            activation_fn: Type of activation (gelu, relu, etc.)
            
        Returns:
            Saturation ratio (0-1)
        """
        if activation_fn.lower() == "relu":
            # ReLU saturates at 0
            saturated = (ffn_activations == 0).float().mean()
        elif activation_fn.lower() == "gelu":
            # GELU saturates at extremes
            saturated = (
                (ffn_activations > 5) | (ffn_activations < -5)
            ).float().mean()
        else:
            # Generic: check if values are at extremes
            mean = ffn_activations.mean()
            std = ffn_activations.std()
            saturated = (
                (ffn_activations > mean + 3 * std) | 
                (ffn_activations < mean - 3 * std)
            ).float().mean()
        
        saturation_ratio = saturated.item()
        self.ffn_saturations.append(saturation_ratio)
        
        return saturation_ratio
    
    def diagnose_layernorm_instability(
        self,
        layernorm_outputs: torch.Tensor,
    ) -> Dict[str, float]:
        """Check for LayerNorm instability.
        
        Returns:
            Dict with mean, std, and stability score
        """
        mean_val = layernorm_outputs.mean().item()
        std_val = layernorm_outputs.std().item()
        
        # LayerNorm should produce outputs with ~0 mean, ~1 std
        mean_deviation = abs(mean_val)
        std_deviation = abs(std_val - 1.0)
        
        stability_score = 1.0 / (1.0 + mean_deviation + std_deviation)
        
        return {
            "mean": mean_val,
            "std": std_val,
            "stability_score": stability_score,
        }
    
    def generate_report(self) -> str:
        """Generate diagnostic report for Transformer."""
        lines = ["=" * 60]
        lines.append("TRANSFORMER DIAGNOSTICS REPORT")
        lines.append("=" * 60)
        
        if self.attention_entropies:
            avg_entropy = np.mean(self.attention_entropies)
            lines.append(f"\nAttention Entropy: {avg_entropy:.3f}")
            
            if self.detect_attention_collapse():
                lines.append("⚠️  WARNING: Attention collapse detected!")
                lines.append("  - Increase dropout in attention")
                lines.append("  - Check learning rate")
        
        if self.ffn_saturations:
            avg_saturation = np.mean(self.ffn_saturations)
            lines.append(f"\nFFN Saturation: {avg_saturation:.1%}")
            
            if avg_saturation > 0.5:
                lines.append("⚠️  WARNING: High FFN saturation!")
                lines.append("  - Consider different activation function")
                lines.append("  - Reduce learning rate")
        
        return "\n".join(lines)
