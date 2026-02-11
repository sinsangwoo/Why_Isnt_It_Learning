"""Cost optimization demo."""

import torch.nn as nn

from gradient_pathology.cost import CostCalculator, TrainingOptimizer


def demo_cost_calculator() -> None:
    """Demo cost calculator."""
    print("=" * 70)
    print("COST CALCULATOR DEMO")
    print("=" * 70)

    calc = CostCalculator()

    training_times = {
        "A100": 32.0,
        "V100": 48.0,
        "A10G": 72.0,
        "T4": 96.0,
    }

    print("\nCost Comparison:")
    comparison = calc.compare_gpus(training_times)
    for gpu, details in comparison.items():
        print(f"\n{gpu}:")
        print(f"  Hours: {details['training_hours']:.1f}h")
        print(f"  Cost: ${details['total_cost']:.2f}")

    cheapest = calc.find_cheapest(training_times)
    print(f"\nCheapest option: {cheapest['gpu_type']}")
    print(f"Total cost: ${cheapest['total_cost']:.2f}")


def demo_training_optimizer() -> None:
    """Demo training optimizer."""
    print("\n" + "=" * 70)
    print("TRAINING OPTIMIZER DEMO")
    print("=" * 70)

    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
    )

    optimizer = TrainingOptimizer(model)

    print("\nScenario 1: Healthy gradients")
    print(optimizer.generate_report("A100", 32.0, "HEALTHY"))

    print("\nScenario 2: Vanishing gradients")
    print(optimizer.generate_report("A100", 32.0, "VANISHING"))

    print("\nScenario 3: Exploding gradients")
    print(optimizer.generate_report("V100", 48.0, "EXPLODING"))


if __name__ == "__main__":
    demo_cost_calculator()
    demo_training_optimizer()
