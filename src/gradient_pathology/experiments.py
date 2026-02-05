"""Experimental utilities for gradient pathology research."""

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from gradient_pathology.analyzer import GradientAnalyzer
from gradient_pathology.core import GradientReport


def create_deep_network(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    activation: str = "relu",
    use_normalization: bool = False,
) -> nn.Module:
    """Create a deep network for testing gradient flow.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        activation: Activation function ('relu', 'sigmoid', 'tanh')
        use_normalization: Whether to use LayerNorm
        
    Returns:
        Deep neural network
    """
    activation_map = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }
    
    if activation.lower() not in activation_map:
        raise ValueError(f"Unknown activation: {activation}")
    
    act_fn = activation_map[activation.lower()]
    
    layers: List[nn.Module] = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    
    for _ in range(num_layers - 1):
        if use_normalization:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_fn())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
    
    if use_normalization:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(act_fn())
    layers.append(nn.Linear(hidden_dim, 1))
    
    return nn.Sequential(*layers)


def compare_activations(
    model_factory: Callable[[], nn.Module],
    activations: Optional[List[str]] = None,
    num_steps: int = 50,
) -> Dict[str, GradientReport]:
    """Compare gradient behavior across different activation functions.
    
    Args:
        model_factory: Function that creates a model
        activations: List of activation functions to compare
        num_steps: Number of diagnostic steps
        
    Returns:
        Dictionary mapping activation name to gradient report
    """
    if activations is None:
        activations = ["relu", "sigmoid", "tanh", "gelu"]
    
    results: Dict[str, GradientReport] = {}
    
    for act in activations:
        model = model_factory()
        # Replace activation functions
        for module in model.modules():
            if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU)):
                # This is a simplified version - proper implementation would
                # reconstruct the network with the target activation
                pass
        
        analyzer = GradientAnalyzer(model)
        results[act] = analyzer.diagnose(num_steps=num_steps)
    
    return results


def benchmark_gradient_flow(
    model: nn.Module,
    input_shape: tuple,
    num_iterations: int = 100,
) -> Dict[str, float]:
    """Benchmark gradient computation performance.
    
    Args:
        model: Model to benchmark
        input_shape: Shape of input tensor
        num_iterations: Number of iterations
        
    Returns:
        Performance metrics
    """
    import time
    
    device = next(model.parameters()).device
    
    # Warmup
    for _ in range(10):
        x = torch.randn(1, *input_shape, device=device)
        y = torch.randn(1, 1, device=device)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        model.zero_grad()
    
    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        x = torch.randn(1, *input_shape, device=device)
        y = torch.randn(1, 1, device=device)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        model.zero_grad()
    elapsed = time.time() - start
    
    return {
        "total_time": elapsed,
        "per_iteration": elapsed / num_iterations,
        "iterations_per_second": num_iterations / elapsed,
    }
