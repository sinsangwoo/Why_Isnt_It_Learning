"""Core data structures for gradient analysis."""

from dataclasses import dataclass
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

        Balanced thresholds: 20x std catches sigmoid issues without false positives.
        """
        abs_mean = abs(self.mean)

        # Vanishing: extremely small gradients
        if abs_mean < 1e-8:
            return GradientPathology.VANISHING

        # Exploding: very large gradients
        if abs_mean > 1e3:
            return GradientPathology.EXPLODING

        # Dead neurons: vast majority are zero
        if self.zero_ratio > 0.9:
            return GradientPathology.DEAD_NEURONS

        # Unstable: 20x threshold - balanced for production
        if self.std > 20 * abs_mean and abs_mean > 1e-6:
            return GradientPathology.UNSTABLE

        return GradientPathology.HEALTHY


@dataclass
class GradientReport:
    """Complete gradient analysis report for a model."""

    layer_stats: List[LayerGradientStats]
    global_mean: float
    global_std: float
    num_steps: int

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["=" * 60]
        lines.append("GRADIENT PATHOLOGY REPORT")
        lines.append("=" * 60)
        lines.append(f"Analysis over {self.num_steps} steps")
        lines.append(f"Global mean gradient: {self.global_mean:.2e}")
        lines.append(f"Global std gradient: {self.global_std:.2e}")
        lines.append("\nPer-Layer Diagnostics:")
        lines.append("-" * 60)

        for stats in self.layer_stats:
            pathology = stats.diagnose()
            status_symbol = "✓" if pathology == GradientPathology.HEALTHY else "✗"
            lines.append(
                f"{status_symbol} {stats.layer_name} (#{stats.layer_index}): "
                f"mean={stats.mean:.2e}, pathology={pathology.value}"
            )

        # Recommendations
        issues = [s for s in self.layer_stats if s.diagnose() != GradientPathology.HEALTHY]
        if issues:
            lines.append("\n" + "=" * 60)
            lines.append("RECOMMENDATIONS:")
            lines.append("=" * 60)
            if any(s.diagnose() == GradientPathology.VANISHING for s in issues):
                lines.append("• Vanishing gradients detected:")
                lines.append("  - Consider: ReLU/GELU activation instead of Sigmoid/Tanh")
                lines.append("  - Consider: He/Xavier initialization")
                lines.append("  - Consider: LayerNorm or BatchNorm")
            if any(s.diagnose() == GradientPathology.EXPLODING for s in issues):
                lines.append("• Exploding gradients detected:")
                lines.append("  - Use gradient clipping (clip_grad_norm)")
                lines.append("  - Reduce learning rate")
                lines.append("  - Check weight initialization scale")

        return "\n".join(lines)

    def get_problematic_layers(self) -> List[LayerGradientStats]:
        """Return layers with gradient pathologies."""
        return [
            stats
            for stats in self.layer_stats
            if stats.diagnose() != GradientPathology.HEALTHY
        ]
