"""Tests for experimental utilities."""

import pytest
import torch.nn as nn

from gradient_pathology.experiments import compare_activations, create_deep_network


def test_create_deep_network() -> None:
    """Test network creation utility."""
    model = create_deep_network(depth=5, activation="relu")
    assert isinstance(model, nn.Sequential)
    assert len(model) > 5  # depth * (linear + activation) + output


def test_create_deep_network_with_norm() -> None:
    """Test network creation with normalization."""
    model = create_deep_network(depth=3, activation="relu", use_norm=True)
    # Should contain LayerNorm layers
    has_norm = any(isinstance(layer, nn.LayerNorm) for layer in model)
    assert has_norm


def test_create_deep_network_invalid_activation() -> None:
    """Test error handling for invalid activation."""
    with pytest.raises(ValueError, match="Unknown activation"):
        create_deep_network(depth=2, activation="invalid_activation")


def test_compare_activations() -> None:
    """Test activation comparison utility."""
    results = compare_activations(depth=5, activations=["relu", "sigmoid"], samples=5)

    assert "relu" in results
    assert "sigmoid" in results
    assert results["relu"].num_steps == 5
    assert results["sigmoid"].num_steps == 5
