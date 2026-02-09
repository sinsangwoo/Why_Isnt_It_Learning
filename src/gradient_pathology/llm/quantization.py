"""Quantization impact analysis."""

from typing import Dict, Optional

import torch
import torch.nn as nn


class QuantizationAnalyzer:
    """Analyze gradient flow in quantized models."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.quantized_layers: Dict[str, bool] = {}

    def detect_quantized_layers(self) -> Dict[str, str]:
        """Detect which layers are quantized.

        Returns:
            Layer name -> quantization type mapping
        """
        quantized = {}

        for name, module in self.model.named_modules():
            if hasattr(module, "weight") and hasattr(module.weight, "dtype"):
                dtype = str(module.weight.dtype)
                if "int8" in dtype or "uint8" in dtype:
                    quantized[name] = "8-bit"
                elif "int4" in dtype:
                    quantized[name] = "4-bit"

        return quantized

    def analyze_quantization_error(
        self, original_weight: torch.Tensor, quantized_weight: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze error from quantization.

        Args:
            original_weight: Original FP32/FP16 weights
            quantized_weight: Quantized weights

        Returns:
            Error metrics
        """
        error = (original_weight - quantized_weight).abs()

        return {
            "mean_error": float(error.mean()),
            "max_error": float(error.max()),
            "relative_error": float(
                error.mean() / (original_weight.abs().mean() + 1e-10)
            ),
        }

    def check_gradient_quantization_impact(
        self, layer_name: str
    ) -> Optional[Dict[str, float]]:
        """Check if quantization affects gradient flow.

        Args:
            layer_name: Name of layer to check

        Returns:
            Impact metrics if quantized, None otherwise
        """
        for name, module in self.model.named_modules():
            if name == layer_name:
                if hasattr(module, "weight") and module.weight.grad is not None:
                    grad = module.weight.grad.detach()
                    return {
                        "grad_mean": float(grad.mean()),
                        "grad_std": float(grad.std()),
                        "grad_sparsity": float((grad.abs() < 1e-7).float().mean()),
                    }
        return None

    def generate_report(self) -> str:
        """Generate quantization analysis report."""
        lines = ["=" * 70]
        lines.append("QUANTIZATION ANALYSIS")
        lines.append("=" * 70)

        quantized = self.detect_quantized_layers()

        if not quantized:
            lines.append("\nNo quantized layers detected")
        else:
            lines.append(f"\nQuantized layers: {len(quantized)}")
            for layer, qtype in list(quantized.items())[:10]:
                lines.append(f"  {layer}: {qtype}")

            lines.append("\nRecommendations:")
            lines.append("  • Monitor gradient sparsity in quantized layers")
            lines.append("  • Use gradient accumulation if gradients are noisy")
            lines.append("  • Consider QLoRA for large models")

        return "\n".join(lines)
