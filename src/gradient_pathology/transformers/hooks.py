"""PyTorch hooks for automatic Transformer monitoring."""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional


class TransformerHooks:
    """Automatic hook injection for Transformer monitoring.
    
    Automatically captures:
    - Attention weights from all layers
    - FFN activations
    - Residual stream values
    - Layer norm statistics
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: List = []
        self.attention_weights: Dict[str, torch.Tensor] = {}
        self.ffn_activations: Dict[str, torch.Tensor] = {}
        self.layernorm_stats: Dict[str, Dict[str, float]] = {}

    def register_attention_hooks(self, attention_modules: List[str]):
        """Register hooks on attention modules.
        
        Args:
            attention_modules: List of module names containing attention
        """
        def attention_hook(name: str):
            def hook(module, input, output):
                # Assuming output contains attention weights
                # Format depends on implementation (MultiheadAttention, custom, etc.)
                if isinstance(output, tuple) and len(output) > 1:
                    attn_weights = output[1]  # Standard PyTorch format
                    self.attention_weights[name] = attn_weights.detach().cpu()
            return hook
        
        for name, module in self.model.named_modules():
            if any(attn_name in name for attn_name in attention_modules):
                handle = module.register_forward_hook(attention_hook(name))
                self.handles.append(handle)

    def register_ffn_hooks(self, ffn_modules: List[str]):
        """Register hooks on FFN modules."""
        def ffn_hook(name: str):
            def hook(module, input, output):
                self.ffn_activations[name] = output.detach().cpu()
            return hook
        
        for name, module in self.model.named_modules():
            if any(ffn_name in name for ffn_name in ffn_modules):
                handle = module.register_forward_hook(ffn_hook(name))
                self.handles.append(handle)

    def register_layernorm_hooks(self):
        """Register hooks on LayerNorm modules."""
        def ln_hook(name: str):
            def hook(module, input, output):
                self.layernorm_stats[name] = {
                    "mean": output.mean().item(),
                    "std": output.std().item(),
                }
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.LayerNorm):
                handle = module.register_forward_hook(ln_hook(name))
                self.handles.append(handle)

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get captured attention weights."""
        return self.attention_weights

    def get_ffn_activations(self) -> Dict[str, torch.Tensor]:
        """Get captured FFN activations."""
        return self.ffn_activations

    def get_layernorm_stats(self) -> Dict[str, Dict[str, float]]:
        """Get LayerNorm statistics."""
        return self.layernorm_stats

    def clear(self):
        """Clear captured data."""
        self.attention_weights.clear()
        self.ffn_activations.clear()
        self.layernorm_stats.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
