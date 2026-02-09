"""Demo of LLM-specific diagnostics."""

import torch
import torch.nn as nn

from gradient_pathology.llm import QuantizationAnalyzer, TransformerDiagnostics


def demo_transformer_diagnostics() -> None:
    """Demo advanced transformer diagnostics."""
    print("\n" + "=" * 70)
    print("TRANSFORMER DIAGNOSTICS DEMO")
    print("=" * 70)

    model = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048)

    diagnostics = TransformerDiagnostics(model)

    dummy_attn = torch.randn(2, 8, 64, 64).softmax(dim=-1)
    stats = diagnostics.analyze_attention_entropy(dummy_attn, "layer_0")
    print(f"\nAttention entropy: {stats['mean_entropy']:.3f}")

    collapsed = diagnostics.detect_attention_collapse(dummy_attn)
    print(f"Attention collapsed: {collapsed}")

    dummy_ffn = torch.randn(2, 64, 2048)
    ffn_stats = diagnostics.analyze_ffn_saturation(dummy_ffn, "layer_0")
    print(f"\nFFN saturation: {ffn_stats['saturated_fraction']:.3f}")
    print(f"Dead neurons: {ffn_stats['zero_fraction']:.3f}")

    print(diagnostics.generate_report())


def demo_quantization_analysis() -> None:
    """Demo quantization impact analysis."""
    print("\n" + "=" * 70)
    print("QUANTIZATION ANALYSIS DEMO")
    print("=" * 70)

    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
    )

    analyzer = QuantizationAnalyzer(model)

    original_weight = model[0].weight.data.clone()
    quantized_weight = original_weight.to(torch.int8).to(torch.float32)

    error_stats = analyzer.analyze_quantization_error(original_weight, quantized_weight)
    print(f"\nQuantization error:")
    print(f"  Mean error: {error_stats['mean_error']:.6f}")
    print(f"  Relative error: {error_stats['relative_error']:.3%}")

    print(analyzer.generate_report())


if __name__ == "__main__":
    demo_transformer_diagnostics()
    demo_quantization_analysis()
