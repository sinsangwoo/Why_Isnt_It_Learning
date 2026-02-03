#!/usr/bin/env python3
"""Demo: Advanced gradient analysis features."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gradient_pathology.advanced import HessianAnalyzer, LRFinder, TransformerDiagnostics


def demo_lr_finder():
    """Demonstrate learning rate finder."""
    print("\n" + "="*60)
    print("LR FINDER DEMO")
    print("="*60)
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    
    # Create synthetic data
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # Run LR finder
    lr_finder = LRFinder(model, optimizer)
    lrs, losses = lr_finder.range_test(dataloader, loss_fn, num_iter=50)
    
    # Get suggestion
    suggested_lr = lr_finder.suggest_lr(lrs, losses)
    print(f"\n✅ Suggested Learning Rate: {suggested_lr:.2e}")
    print(f"   Range tested: {min(lrs):.2e} to {max(lrs):.2e}")
    
    # Plot results
    print("\n📊 Plotting LR range test...")
    lr_finder.plot(lrs, losses)


def demo_hessian_analyzer():
    """Demonstrate Hessian analysis."""
    print("\n" + "="*60)
    print("HESSIAN ANALYZER DEMO")
    print("="*60)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    
    # Create data
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16)
    
    # Analyze Hessian
    analyzer = HessianAnalyzer(model)
    results = analyzer.compute_hessian_eigenvalues(
        dataloader,
        nn.MSELoss(),
        top_k=5
    )
    
    print(f"\nMax Eigenvalue: {results['max_eigenvalue']:.2e}")
    print(f"Effective Rank: {results['effective_rank']}")
    
    diagnosis = analyzer.diagnose_sharpness(results['eigenvalues'])
    print(f"\nSharpness Diagnosis: {diagnosis}")


def demo_transformer_diagnostics():
    """Demonstrate Transformer diagnostics."""
    print("\n" + "="*60)
    print("TRANSFORMER DIAGNOSTICS DEMO")
    print("="*60)
    
    # Simulate attention weights
    batch_size, num_heads, seq_len = 8, 4, 32
    attention_weights = torch.softmax(
        torch.randn(batch_size, num_heads, seq_len, seq_len),
        dim=-1
    )
    
    # Analyze
    diagnostics = TransformerDiagnostics(nn.Identity())
    
    entropy = diagnostics.analyze_attention_entropy(attention_weights)
    print(f"\nAttention Entropy: {entropy:.3f}")
    
    # Check for collapse
    if diagnostics.detect_attention_collapse():
        print("⚠️  Attention collapse detected!")
    else:
        print("✅ Attention distribution healthy")
    
    # Simulate FFN activations
    ffn_activations = torch.randn(batch_size, seq_len, 512)
    saturation = diagnostics.analyze_ffn_saturation(ffn_activations, "gelu")
    print(f"\nFFN Saturation: {saturation:.1%}")
    
    # Generate report
    print(diagnostics.generate_report())


if __name__ == "__main__":
    print("🚀 Advanced Gradient Analysis Demos")
    print("="*60)
    
    demo_lr_finder()
    demo_hessian_analyzer()
    demo_transformer_diagnostics()
    
    print("\n✅ All demos completed!")
