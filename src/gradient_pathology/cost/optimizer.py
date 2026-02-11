"""Training optimization recommendations."""

from typing import Any, Dict, List, Optional

import torch.nn as nn

from gradient_pathology.analyzer import GradientAnalyzer
from gradient_pathology.cost.calculator import CostCalculator


class TrainingOptimizer:
    """Suggest cost-optimized training configurations."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.analyzer = GradientAnalyzer(model)
        self.calculator = CostCalculator()

    def estimate_speedup_from_diagnosis(
        self, gradient_health: str
    ) -> Dict[str, float]:
        """Estimate speedup potential from gradient diagnosis.

        Args:
            gradient_health: HEALTHY, VANISHING, EXPLODING, UNSTABLE

        Returns:
            Speedup estimates
        """
        if gradient_health == "HEALTHY":
            return {
                "convergence_speedup": 1.0,
                "gpu_downgrade_possible": True,
                "suggested_gpu_tier": "lower",
            }
        elif gradient_health == "VANISHING":
            return {
                "convergence_speedup": 3.0,
                "gpu_downgrade_possible": False,
                "suggested_gpu_tier": "same",
            }
        elif gradient_health == "EXPLODING":
            return {
                "convergence_speedup": 2.5,
                "gpu_downgrade_possible": False,
                "suggested_gpu_tier": "same",
            }
        else:
            return {
                "convergence_speedup": 4.0,
                "gpu_downgrade_possible": False,
                "suggested_gpu_tier": "same",
            }

    def suggest_optimization(
        self,
        current_gpu: str,
        estimated_hours: float,
        gradient_health: str,
    ) -> Dict[str, Any]:
        """Suggest cost-optimized configuration.

        Args:
            current_gpu: Current GPU type
            estimated_hours: Current training time estimate
            gradient_health: Gradient pathology status

        Returns:
            Optimization suggestions
        """
        speedup = self.estimate_speedup_from_diagnosis(gradient_health)

        if gradient_health == "HEALTHY" and speedup["gpu_downgrade_possible"]:
            gpu_map = {
                "A100": "V100",
                "A100-40GB": "A10G",
                "V100": "T4",
                "A10G": "T4",
            }
            suggested_gpu = gpu_map.get(current_gpu, current_gpu)
            suggested_hours = estimated_hours * 1.2
        else:
            suggested_gpu = current_gpu
            suggested_hours = estimated_hours / speedup["convergence_speedup"]

        savings = self.calculator.calculate_savings(
            current_gpu,
            estimated_hours,
            suggested_gpu,
            suggested_hours,
        )

        return {
            "current": {
                "gpu": current_gpu,
                "hours": estimated_hours,
                "cost": savings["baseline_cost"],
            },
            "suggested": {
                "gpu": suggested_gpu,
                "hours": suggested_hours,
                "cost": savings["optimized_cost"],
            },
            "savings": {
                "amount": savings["savings"],
                "percent": savings["savings_percent"],
            },
            "reason": self._get_optimization_reason(gradient_health, speedup),
        }

    def _get_optimization_reason(self, health: str, speedup: Dict) -> str:
        """Generate optimization reasoning."""
        if health == "HEALTHY" and speedup["gpu_downgrade_possible"]:
            return "Gradients healthy. Can use cheaper GPU with minor slowdown."
        elif health == "VANISHING":
            return "Fix vanishing gradients to converge 3x faster on same GPU."
        elif health == "EXPLODING":
            return "Fix exploding gradients to converge 2.5x faster on same GPU."
        else:
            return "Fix gradient instability to converge 4x faster on same GPU."

    def generate_report(
        self,
        current_gpu: str,
        estimated_hours: float,
        gradient_health: str,
    ) -> str:
        """Generate cost optimization report."""
        suggestion = self.suggest_optimization(
            current_gpu, estimated_hours, gradient_health
        )

        lines = ["=" * 70]
        lines.append("COST OPTIMIZATION REPORT")
        lines.append("=" * 70)

        lines.append("\nCurrent Configuration:")
        lines.append(f"  GPU: {suggestion['current']['gpu']}")
        lines.append(f"  Training time: {suggestion['current']['hours']:.1f}h")
        lines.append(f"  Cost: ${suggestion['current']['cost']:.2f}")

        lines.append("\nOptimized Configuration:")
        lines.append(f"  GPU: {suggestion['suggested']['gpu']}")
        lines.append(f"  Training time: {suggestion['suggested']['hours']:.1f}h")
        lines.append(f"  Cost: ${suggestion['suggested']['cost']:.2f}")

        lines.append("\nSavings:")
        lines.append(f"  Amount: ${suggestion['savings']['amount']:.2f}")
        lines.append(f"  Percent: {suggestion['savings']['percent']:.1f}%")

        lines.append(f"\nReason:\n  {suggestion['reason']}")

        return "\n".join(lines)
