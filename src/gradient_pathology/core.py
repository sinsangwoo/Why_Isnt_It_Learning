"""Core data structures for gradient analysis."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import numpy as np


class GradientPathology(Enum):
    """Types of gradient pathologies that can be detected."""

    HEALTHY = "healthy"
    VANISHING = "vanishing"
    EXPLODING = "exploding"
    DEAD_NEURONS = "dead_neurons"
    UNSTABLE = "unstable"


@dataclass
class LayerGradientStats:
    """Statistics for gradients in a single layer."""

    layer_name: str
    layer_index: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    num_zeros: int
    total_params: int

    @property
    def zero_ratio(self) -> float:
        """Ratio of zero gradients (indicator of dead neurons)."""
        return self.num_zeros / max(self.total_params, 1)

    def diagnose(self) -> GradientPathology:
        """Diagnose gradient health for this layer.

        Thresholds are calibrated against empirical observations on common
        architectures.  They should be treated as heuristics, not ground
        truth.
        """
        abs_mean = abs(self.mean)

        if abs_mean < 1e-8:
            return GradientPathology.VANISHING

        if abs_mean > 1e3:
            return GradientPathology.EXPLODING

        if self.zero_ratio > 0.9:
            return GradientPathology.DEAD_NEURONS

        # Unstable: high variance relative to mean
        if self.std > 30 * abs_mean and abs_mean > 1e-6:
            return GradientPathology.UNSTABLE

        return GradientPathology.HEALTHY


@dataclass
class GradientReport:
    """Complete gradient analysis report for a model."""

    layer_stats: List[LayerGradientStats]
    global_mean: float
    global_std: float
    num_steps: int
    #: ``'dataloader'`` when real data was used; ``'synthetic'`` otherwise.
    data_source: str = field(default="synthetic")

    def summary(self) -> str:
        """Generate human-readable summary."""
        source_warning = (
            "  ⚠ NOTE: Gradients computed on SYNTHETIC data.\n"
            "          Pass a real DataLoader for actionable diagnostics.\n"
            if self.data_source == "synthetic"
            else ""
        )

        lines = ["=" * 64]
        lines.append("GRADIENT PATHOLOGY REPORT")
        lines.append("=" * 64)
        lines.append(f"Data source : {self.data_source}")
        lines.append(f"Steps       : {self.num_steps}")
        lines.append(f"Global mean : {self.global_mean:.2e}")
        lines.append(f"Global std  : {self.global_std:.2e}")
        if source_warning:
            lines.append("")
            lines.append(source_warning.rstrip())
        lines.append("\nPer-Layer Diagnostics:")
        lines.append("-" * 64)

        for stats in self.layer_stats:
            pathology = stats.diagnose()
            symbol = "✓" if pathology == GradientPathology.HEALTHY else "✗"
            lines.append(
                f"{symbol} {stats.layer_name} (#{stats.layer_index}): "
                f"mean={stats.mean:.2e}, status={pathology.value}"
            )

        issues = [
            s for s in self.layer_stats
            if s.diagnose() != GradientPathology.HEALTHY
        ]
        if issues:
            lines.append("\n" + "=" * 64)
            lines.append("RECOMMENDATIONS:")
            lines.append("=" * 64)
            if any(s.diagnose() == GradientPathology.VANISHING for s in issues):
                lines.append("• Vanishing gradients detected:")
                lines.append("  - Use ReLU/GELU instead of Sigmoid/Tanh")
                lines.append("  - Apply He/Xavier initialisation")
                lines.append("  - Add LayerNorm or BatchNorm")
            if any(s.diagnose() == GradientPathology.EXPLODING for s in issues):
                lines.append("• Exploding gradients detected:")
                lines.append("  - Clip gradients (torch.nn.utils.clip_grad_norm_)")
                lines.append("  - Reduce learning rate")
                lines.append("  - Verify weight initialisation scale")
            if any(s.diagnose() == GradientPathology.DEAD_NEURONS for s in issues):
                lines.append("• Dead neurons detected:")
                lines.append("  - Switch from ReLU to Leaky ReLU or GELU")
                lines.append("  - Check for very large negative biases")

        return "\n".join(lines)

    def get_problematic_layers(self) -> List[LayerGradientStats]:
        """Return layers with gradient pathologies."""
        return [
            stats
            for stats in self.layer_stats
            if stats.diagnose() != GradientPathology.HEALTHY
        ]
