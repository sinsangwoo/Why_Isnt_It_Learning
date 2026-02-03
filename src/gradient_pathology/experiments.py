"""Experimental utilities for benchmarking gradient behavior."""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from gradient_pathology.analyzer import GradientAnalyzer
from gradient_pathology.core import GradientReport


def create_deep_network(
    depth: int,
    activation: str = "relu",
    hidden_size: int = 64,
    input_size: int = 10,
    use_norm: bool = False,
) -> nn.Module:
    """Create a deep feedforward network for testing.

    Args:
        depth: Number of hidden layers
        activation: Activation function name
        hidden_size: Hidden layer width
        input_size: Input dimension
        use_norm: Whether to use LayerNorm

    Returns:
        PyTorch Sequential model
    """
    activation_map = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }

    if activation not in activation_map:
        raise ValueError(f"Unknown activation: {activation}")

    layers = [nn.Linear(input_size, hidden_size)]
    if use_norm:
        layers.append(nn.LayerNorm(hidden_size))
    layers.append(activation_map[activation]())

    for _ in range(depth - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        if use_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(activation_map[activation]())

    layers.append(nn.Linear(hidden_size, 1))

    return nn.Sequential(*layers)


def compare_activations(
    depth: int = 20,
    activations: list = None,
    samples: int = 100,
) -> Dict[str, GradientReport]:
    """Compare gradient behavior across activation functions.

    Args:
        depth: Network depth
        activations: List of activation functions to test
        samples: Number of gradient samples

    Returns:
        Dictionary mapping activation names to their reports
    """
    if activations is None:
        activations = ["sigmoid", "tanh", "relu"]

    results: Dict[str, GradientReport] = {}

    for act in activations:
        model = create_deep_network(depth=depth, activation=act)
        analyzer = GradientAnalyzer(model)
        report = analyzer.diagnose(num_steps=samples)
        results[act] = report
        print(f"\n{act.upper()} Results:")
        print(report.summary())

    return results


def plot_gradient_comparison(results: Dict[str, GradientReport]) -> None:
    """Plot gradient statistics across different configurations.

    Args:
        results: Dictionary of experiment results from compare_activations
    """
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), sharey=True)

    if len(results) == 1:
        axes = [axes]

    for ax, (name, report) in zip(axes, results.items()):
        layer_means = [stats.mean for stats in report.layer_stats]
        layer_indices = [stats.layer_index for stats in report.layer_stats]

        ax.plot(layer_indices, layer_means, marker="o", label="Mean gradient")
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.3)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Mean Gradient")
        ax.set_title(f"Activation: {name}")
        ax.set_yscale("symlog", linthresh=1e-10)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()
