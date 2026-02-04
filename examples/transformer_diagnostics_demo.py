#!/usr/bin/env python3
"""Demo: Transformer-specific diagnostics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn

from gradient_pathology.transformers import AttentionMonitor, TransformerHooks


def demo_attention_monitor():
    """Demonstrate attention monitoring."""
    print("\n" + "="*60)
    print("ATTENTION MONITOR DEMO")
    print("="*60)
    
    # Create monitor
    monitor = AttentionMonitor()
    
    # Simulate attention weights over training
    batch_size, num_heads, seq_len = 8, 4, 32
    
    print("\nSimulating 50 training steps...")
    for step in range(50):
        # Simulate attention weights
        # Early training: Random (high entropy)
        # Later training: Focused (lower entropy)
        if step < 25:
            attn = torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)
        else:
            # Simulate learning: more focused attention
            logits = torch.randn(batch_size, num_heads, seq_len, seq_len) * 2
            attn = torch.softmax(logits, dim=-1)
        
        stats = monitor.record_attention(attn, layer_name=f"layer_0")
        
        if step % 10 == 0:
            print(f"Step {step}: Entropy={stats['entropy']:.3f}, "
                  f"Max Attn={stats['max_attention']:.3f}")
    
    # Check for issues
    print("\n" + "-"*60)
    if monitor.detect_collapse():
        print("⚠️  Attention collapse detected!")
    else:
        print("✅ Attention patterns healthy")
    
    # Generate report
    print(monitor.generate_report())
    
    # Visualize
    print("\n📊 Plotting attention entropy timeline...")
    monitor.plot_entropy_timeline()
    
    print("\n📊 Visualizing attention pattern...")
    monitor.visualize_attention_pattern(step_idx=-1, head_idx=0)


def demo_transformer_hooks():
    """Demonstrate automatic hook injection."""
    print("\n" + "="*60)
    print("TRANSFORMER HOOKS DEMO")
    print("="*60)
    
    # Create simple transformer layer
    class SimpleTransformer(nn.Module):
        def __init__(self, d_model=64, nhead=4):
            super().__init__()
            self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
        
        def forward(self, x):
            # Self-attention
            attn_out, attn_weights = self.attention(x, x, x, need_weights=True)
            x = self.ln1(x + attn_out)
            
            # FFN
            ffn_out = self.ffn(x)
            x = self.ln2(x + ffn_out)
            
            return x
    
    model = SimpleTransformer()
    
    # Register hooks
    hooks = TransformerHooks(model)
    hooks.register_attention_hooks(['attention'])
    hooks.register_ffn_hooks(['ffn'])
    hooks.register_layernorm_hooks()
    
    # Forward pass
    x = torch.randn(2, 10, 64)  # [batch, seq, features]
    
    print("\nRunning forward pass with hooks...")
    with torch.no_grad():
        output = model(x)
    
    # Check captured data
    print("\n📊 Captured Data:")
    print(f"  Attention weights: {len(hooks.get_attention_weights())} modules")
    print(f"  FFN activations: {len(hooks.get_ffn_activations())} modules")
    print(f"  LayerNorm stats: {len(hooks.get_layernorm_stats())} modules")
    
    # Show LayerNorm stats
    print("\n📊 LayerNorm Statistics:")
    for name, stats in hooks.get_layernorm_stats().items():
        print(f"  {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    # Cleanup
    hooks.remove_hooks()
    print("\n✅ Hooks removed")


if __name__ == "__main__":
    print("🚀 Transformer Diagnostics Demos")
    print("="*60)
    
    demo_attention_monitor()
    demo_transformer_hooks()
    
    print("\n✅ All demos completed!")
