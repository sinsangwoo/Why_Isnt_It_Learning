# Transformer Diagnostics Guide

> **Specialized tools for debugging Transformer architectures in production**

---

## 🎯 Why Transformer-Specific Diagnostics?

Generic gradient analysis misses critical Transformer pathologies:

- ❌ **Generic tools**: Check gradient magnitudes
- ✅ **Transformer diagnostics**: Detect attention collapse, head specialization, FFN saturation

---

## 🔍 What We Detect

### 1. Attention Collapse

**Symptom**: All queries attend to same key (e.g., `[CLS]` token)  
**Impact**: Model ignores most of input sequence  
**Detection**: Entropy < 0.1

```python
from gradient_pathology.transformers import AttentionMonitor

monitor = AttentionMonitor()
stats = monitor.record_attention(attention_weights)

if monitor.detect_collapse():
    print("⚠️ Attention collapse!")
    # Fix: Increase attention dropout, reduce LR
```

### 2. Attention Dispersion

**Symptom**: Uniform attention across all keys  
**Impact**: Model not learning patterns  
**Detection**: Entropy > 0.9 * max_entropy

### 3. FFN Saturation

**Symptom**: Most neurons at activation extremes  
**Impact**: Gradient flow blocked  
**Detection**: >50% neurons saturated

### 4. LayerNorm Instability

**Symptom**: Mean ≠ 0 or Std ≠ 1  
**Impact**: Training instability  
**Detection**: Deviation from expected values

---

## 📊 Usage Examples

### Real-time Monitoring

```python
from gradient_pathology.transformers import AttentionMonitor

monitor = AttentionMonitor()

# In training loop:
for batch in dataloader:
    outputs = model(batch)
    
    # Capture attention from model.attention_weights
    stats = monitor.record_attention(
        model.attention_weights,
        layer_name="encoder_layer_0"
    )
    
    if monitor.detect_collapse():
        print(f"⚠️ Step {step}: Attention collapsed!")
        # Trigger early stopping or LR adjustment
```

### Automatic Hook Injection

```python
from gradient_pathology.transformers import TransformerHooks

hooks = TransformerHooks(model)

# Auto-capture attention, FFN, LayerNorm
hooks.register_attention_hooks(['attention'])
hooks.register_ffn_hooks(['ffn'])
hooks.register_layernorm_hooks()

# Forward pass (hooks auto-capture)
output = model(input)

# Analyze
attn_weights = hooks.get_attention_weights()
for layer, weights in attn_weights.items():
    entropy = compute_entropy(weights)
    print(f"{layer}: entropy={entropy:.3f}")

# Cleanup
hooks.remove_hooks()
```

### Post-Training Analysis

```python
monitor = AttentionMonitor()

# Load saved attention weights
for step, attn in enumerate(saved_attention):
    monitor.record_attention(attn)

# Visualize evolution
monitor.plot_entropy_timeline(save_path="attention_evolution.png")
monitor.visualize_attention_pattern(step_idx=100, head_idx=0)

# Get report
print(monitor.generate_report())
```

---

## 🚨 Common Issues & Fixes

### Issue: Attention Collapse

**Symptoms**:
- Entropy < 0.1
- Max attention weight > 0.8
- All queries attend to `[CLS]` or `[SEP]`

**Fixes**:
```python
# 1. Increase attention dropout
model.attention.dropout = 0.2  # was: 0.1

# 2. Reduce learning rate
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.5

# 3. Add auxiliary loss
loss += 0.01 * attention_entropy_loss(attention_weights)
```

### Issue: FFN Saturation

**Symptoms**:
- >50% neurons at activation extremes
- Gradient magnitudes very small

**Fixes**:
```python
# 1. Switch activation
model.ffn.activation = nn.GELU()  # was: nn.ReLU()

# 2. Reduce learning rate
optimizer = Adam(model.parameters(), lr=1e-5)  # was: 1e-4

# 3. Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 📈 Integration with Existing Tools

### HuggingFace Transformers

```python
from transformers import BertModel
from gradient_pathology.transformers import TransformerHooks

model = BertModel.from_pretrained('bert-base-uncased')

hooks = TransformerHooks(model)
hooks.register_attention_hooks(['attention.self'])

outputs = model(**inputs, output_attentions=True)
attn_weights = hooks.get_attention_weights()

# Diagnose each layer
for layer_name, weights in attn_weights.items():
    monitor.record_attention(weights, layer_name=layer_name)
```

### PyTorch Lightning

```python
import pytorch_lightning as pl
from gradient_pathology.transformers import AttentionMonitor

class TransformerModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ...
        self.attention_monitor = AttentionMonitor()
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        
        # Monitor attention
        stats = self.attention_monitor.record_attention(
            outputs.attentions[0]
        )
        
        # Log metrics
        self.log('attention_entropy', stats['entropy'])
        
        return loss
```

---

## 🎓 When to Use

| Scenario | Tool | Why |
|----------|------|-----|
| **Training BERT/GPT** | AttentionMonitor | Detect collapse early |
| **Fine-tuning Transformers** | TransformerHooks | Auto-capture all layers |
| **Research on attention** | Both | Comprehensive analysis |
| **Production debugging** | AttentionMonitor | Lightweight, real-time |

---

## 🔬 Technical Details

### Entropy Calculation

```python
# Attention entropy measures distribution uniformity
entropy = -sum(p * log(p)) for p in attention_weights

# Low entropy → Focused (possibly collapsed)
# High entropy → Dispersed (possibly not learning)
```

### Head Specialization

```python
# Different heads should learn different patterns
head_variance = std(mean_attention_per_head)

# Low variance → Heads not specializing
# High variance → Good head diversity
```

---

**Built for 2025 ML reality** 🤖 | **Where Transformers rule** 👑
