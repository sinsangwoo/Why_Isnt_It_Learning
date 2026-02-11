"""Cloud cost calculator."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class GPUPrice:
    name: str
    price_per_hour: float
    memory_gb: int
    compute_capability: float


GPU_PRICES = {
    "A100": GPUPrice("A100", 3.67, 80, 8.0),
    "A100-40GB": GPUPrice("A100-40GB", 2.93, 40, 8.0),
    "V100": GPUPrice("V100", 2.48, 16, 7.0),
    "A10G": GPUPrice("A10G", 1.01, 24, 8.6),
    "T4": GPUPrice("T4", 0.526, 16, 7.5),
    "L4": GPUPrice("L4", 0.80, 24, 8.9),
}


class CostCalculator:
    """Calculate training costs across GPU types."""

    def __init__(self) -> None:
        self.gpu_prices = GPU_PRICES

    def estimate_cost(
        self,
        gpu_type: str,
        training_hours: float,
        num_gpus: int = 1,
    ) -> Dict[str, float]:
        """Estimate training cost.

        Args:
            gpu_type: GPU type (A100, V100, etc.)
            training_hours: Expected training time
            num_gpus: Number of GPUs

        Returns:
            Cost breakdown
        """
        if gpu_type not in self.gpu_prices:
            raise ValueError(f"Unknown GPU: {gpu_type}")

        price = self.gpu_prices[gpu_type]
        total_cost = price.price_per_hour * training_hours * num_gpus

        return {
            "gpu_type": gpu_type,
            "price_per_hour": price.price_per_hour,
            "training_hours": training_hours,
            "num_gpus": num_gpus,
            "total_cost": total_cost,
        }

    def compare_gpus(
        self,
        training_hours_by_gpu: Dict[str, float],
        num_gpus: int = 1,
    ) -> Dict[str, Dict[str, float]]:
        """Compare costs across GPU types.

        Args:
            training_hours_by_gpu: {gpu_type: hours} mapping
            num_gpus: Number of GPUs per type

        Returns:
            Cost comparison
        """
        results = {}

        for gpu_type, hours in training_hours_by_gpu.items():
            results[gpu_type] = self.estimate_cost(gpu_type, hours, num_gpus)

        return results

    def find_cheapest(
        self,
        training_hours_by_gpu: Dict[str, float],
        num_gpus: int = 1,
    ) -> Dict[str, float]:
        """Find cheapest GPU option.

        Args:
            training_hours_by_gpu: {gpu_type: hours} mapping
            num_gpus: Number of GPUs

        Returns:
            Cheapest option details
        """
        comparison = self.compare_gpus(training_hours_by_gpu, num_gpus)
        cheapest = min(comparison.values(), key=lambda x: x["total_cost"])
        return cheapest

    def calculate_savings(
        self,
        baseline_gpu: str,
        baseline_hours: float,
        optimized_gpu: str,
        optimized_hours: float,
        num_gpus: int = 1,
    ) -> Dict[str, float]:
        """Calculate cost savings from optimization.

        Args:
            baseline_gpu: Original GPU type
            baseline_hours: Original training time
            optimized_gpu: New GPU type
            optimized_hours: New training time
            num_gpus: Number of GPUs

        Returns:
            Savings breakdown
        """
        baseline_cost = self.estimate_cost(baseline_gpu, baseline_hours, num_gpus)
        optimized_cost = self.estimate_cost(optimized_gpu, optimized_hours, num_gpus)

        savings = baseline_cost["total_cost"] - optimized_cost["total_cost"]
        savings_pct = (savings / baseline_cost["total_cost"]) * 100

        return {
            "baseline_cost": baseline_cost["total_cost"],
            "optimized_cost": optimized_cost["total_cost"],
            "savings": savings,
            "savings_percent": savings_pct,
            "time_reduction": baseline_hours - optimized_hours,
        }
