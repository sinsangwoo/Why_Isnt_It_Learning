"""Advanced visualization utilities for gradient analysis."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from gradient_pathology.core import GradientPathology, GradientReport


def plot_gradient_heatmap(report: GradientReport, save_path: Optional[str] = None) -> plt.Figure:
    """Create heatmap of gradient statistics across layers.

    Args:
        report: Gradient analysis report
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Prepare data matrix
    metrics = ["mean", "std", "min", "max", "median"]
    data = np.zeros((len(report.layer_stats), len(metrics)))

    for i, stats in enumerate(report.layer_stats):
        data[i, 0] = abs(stats.mean)
        data[i, 1] = stats.std
        data[i, 2] = abs(stats.min)
        data[i, 3] = abs(stats.max)
        data[i, 4] = abs(stats.median)

    # Log scale for better visualization
    data = np.log10(data + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto")

    # Labels
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(report.layer_stats)))
    ax.set_yticklabels([s.layer_name for s in report.layer_stats], fontsize=8)

    ax.set_xlabel("Gradient Statistics")
    ax.set_ylabel("Layer")
    ax.set_title("Gradient Heatmap (log10 scale)")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("log10(Gradient Magnitude)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_layer_comparison(
    reports: Dict[str, GradientReport],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare gradient flow across multiple configurations.

    Args:
        reports: Dict mapping config names to reports
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for config_name, report in reports.items():
        layer_indices = [s.layer_index for s in report.layer_stats]
        layer_means = [abs(s.mean) for s in report.layer_stats]
        ax.plot(layer_indices, layer_means, marker="o", label=config_name, alpha=0.7)

    ax.axhline(y=1e-7, color="red", linestyle="--", alpha=0.5, label="Vanishing threshold")
    ax.axhline(y=1e2, color="orange", linestyle="--", alpha=0.5, label="Exploding threshold")

    ax.set_yscale("log")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Gradient (log scale)")
    ax.set_title("Gradient Flow Comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_timeline(
    gradient_history: List[Dict[str, Dict[str, float]]],
    layer_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot gradient evolution over training steps.

    Args:
        gradient_history: List of dicts with gradient stats per step
        layer_names: Subset of layers to plot (default: all)
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if not gradient_history:
        raise ValueError("gradient_history is empty")

    all_layers = list(gradient_history[0].keys())
    if layer_names is None:
        # Plot up to 10 layers
        layer_names = all_layers[::max(1, len(all_layers) // 10)]

    fig, ax = plt.subplots(figsize=(14, 6))

    for layer in layer_names:
        values = [
            step.get(layer, {}).get("mean", np.nan) 
            if isinstance(step.get(layer), dict) else np.nan
            for step in gradient_history
        ]
        ax.plot(values, label=layer, alpha=0.7)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Gradient")
    ax.set_title("Gradient Evolution During Training")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
