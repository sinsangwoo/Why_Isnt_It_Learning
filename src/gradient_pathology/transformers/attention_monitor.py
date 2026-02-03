"""Real-time attention pattern monitoring for Transformers."""

import torch
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


class AttentionMonitor:
    """Monitor attention patterns during training.
    
    Detects:
    - Attention collapse (all queries attend to one key)
    - Attention dispersion (uniform attention)
    - Head specialization (different heads learn different patterns)
    - Layer-wise attention evolution
    """

    def __init__(self):
        self.attention_history: List[Dict[str, torch.Tensor]] = []
        self.entropy_history: List[float] = []
        self.head_specialization: Dict[int, List[float]] = {}

    def record_attention(
        self,
        attention_weights: torch.Tensor,
        layer_name: str = "layer_0",
    ) -> Dict[str, float]:
        """Record attention weights and compute diagnostics.
        
        Args:
            attention_weights: [batch, heads, seq_len, seq_len]
            layer_name: Name of transformer layer
            
        Returns:
            Dict with entropy, max_attention, head_variance
        """
        # Compute entropy per head
        attn = attention_weights + 1e-10
        entropy = -torch.sum(attn * torch.log(attn), dim=-1)  # [B, H, Q]
        mean_entropy = entropy.mean().item()
        
        # Max attention (detect collapse)
        max_attn = attention_weights.max(dim=-1)[0].mean().item()
        
        # Head specialization (variance across heads)
        head_means = attention_weights.mean(dim=(0, 2, 3))  # [H]
        head_variance = head_means.std().item()
        
        stats = {
            "entropy": mean_entropy,
            "max_attention": max_attn,
            "head_variance": head_variance,
            "layer": layer_name,
        }
        
        self.attention_history.append({
            "weights": attention_weights.detach().cpu(),
            "stats": stats,
        })
        self.entropy_history.append(mean_entropy)
        
        return stats

    def detect_collapse(self, threshold: float = 0.1) -> bool:
        """Detect attention collapse.
        
        Returns True if recent entropy is below threshold.
        """
        if len(self.entropy_history) < 10:
            return False
        
        recent_entropy = np.mean(self.entropy_history[-10:])
        return recent_entropy < threshold

    def detect_dispersion(self, threshold: float = 0.9) -> bool:
        """Detect overly uniform attention (high entropy)."""
        if len(self.entropy_history) < 10:
            return False
        
        recent_entropy = np.mean(self.entropy_history[-10:])
        max_possible = np.log(self.attention_history[-1]["weights"].shape[-1])
        
        return recent_entropy > threshold * max_possible

    def plot_entropy_timeline(self, save_path: Optional[str] = None):
        """Plot attention entropy over training."""
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(self.entropy_history, linewidth=2)
        ax.axhline(y=0.1, color='r', linestyle='--', label='Collapse threshold')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Attention Entropy')
        ax.set_title('Attention Pattern Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()

    def visualize_attention_pattern(
        self,
        step_idx: int = -1,
        head_idx: int = 0,
        save_path: Optional[str] = None,
    ):
        """Visualize attention pattern for specific step and head."""
        if not self.attention_history:
            raise ValueError("No attention history recorded")
        
        attn = self.attention_history[step_idx]["weights"]
        # Get first batch, specific head
        pattern = attn[0, head_idx].numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pattern, cmap='viridis', aspect='auto')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'Attention Pattern (Step {step_idx}, Head {head_idx})')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()

    def generate_report(self) -> str:
        """Generate diagnostic report."""
        if not self.entropy_history:
            return "No attention data recorded."
        
        lines = ["=" * 60]
        lines.append("ATTENTION DIAGNOSTICS REPORT")
        lines.append("=" * 60)
        
        avg_entropy = np.mean(self.entropy_history)
        lines.append(f"\nAverage Attention Entropy: {avg_entropy:.3f}")
        
        if self.detect_collapse():
            lines.append("\n⚠️  WARNING: Attention Collapse Detected!")
            lines.append("   Recommendations:")
            lines.append("   - Increase attention dropout (0.1 → 0.2)")
            lines.append("   - Reduce learning rate by 2-5x")
            lines.append("   - Add auxiliary losses for attention diversity")
        
        if self.detect_dispersion():
            lines.append("\n⚠️  WARNING: Overly Dispersed Attention!")
            lines.append("   Recommendations:")
            lines.append("   - Model may not be learning patterns")
            lines.append("   - Check if task requires focused attention")
            lines.append("   - Consider positional encodings")
        
        if not self.detect_collapse() and not self.detect_dispersion():
            lines.append("\n✅ Attention patterns appear healthy")
        
        return "\n".join(lines)
