# Basic Gradient Diagnosis

## Why Your Model Isn't Learning: A 5-Minute Guide

### The Problem

You've set up your model, started training, and... nothing happens. The loss barely moves. What's wrong?

### The Solution

Let's diagnose it in 5 minutes.

## Step-by-Step Walkthrough

### 1. The Problematic Model

Here's a common mistake - deep network with sigmoid activations:

```python
import torch.nn as nn

model = nn.Sequential(
    *[nn.Linear(64, 64), nn.Sigmoid()] * 20,  # 20 layers!
    nn.Linear(64, 1),
)
```

### 2. Run Diagnosis

```python
from gradient_pathology import GradientAnalyzer

analyzer = GradientAnalyzer(model)
report = analyzer.diagnose(num_steps=50, input_shape=(64,))
```

### 3. Check Results

```python
print(report.summary())
```

Output:
```
=================================================================
GRADIENT ANALYSIS REPORT
=================================================================
Global Statistics:
  Mean: 2.34e-09  ⚠️ VANISHING
  Std:  1.12e-08
  
Detected Issues:
  ⚠️ 18 layers with vanishing gradients
  
Recommendations:
  1. Replace Sigmoid with ReLU or GELU
  2. Add LayerNorm after each layer
  3. Consider residual connections
```

### 4. Fix It

```python
# Better architecture
model = nn.Sequential(
    *[
        nn.Linear(64, 64),
        nn.LayerNorm(64),
        nn.GELU(),
    ] * 20,
    nn.Linear(64, 1),
)

# Verify the fix
report = analyzer.diagnose(num_steps=50, input_shape=(64,))
print(report.summary())
```

Output:
```
Global Statistics:
  Mean: 3.45e-03  ✅ HEALTHY
  Std:  2.11e-02
  
No critical issues detected!
```

## Common Patterns

### Pattern 1: Vanishing in Deep Networks

**Symptoms**:
- Gradients < 1e-7
- Early layers don't update
- Loss plateaus immediately

**Quick Fix**:
```python
# Add normalization
for i in range(num_layers):
    layers.append(nn.Linear(hidden, hidden))
    layers.append(nn.LayerNorm(hidden))  # <-- This!
    layers.append(nn.ReLU())
```

### Pattern 2: Exploding with High Learning Rate

**Symptoms**:
- Gradients > 100
- Loss becomes NaN
- Training crashes

**Quick Fix**:
```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Or reduce learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Lower!
```

### Pattern 3: Unstable Variance

**Symptoms**:
- High gradient std (std > 10 * mean)
- Erratic loss curves
- Needs many epochs to converge

**Quick Fix**:
```python
# Use layer-wise learning rates
from gradient_pathology.auto import LayerLRFinder

finder = LayerLRFinder(model)
optimal_lrs = finder.find_layer_lrs(dataloader, loss_fn)
optimizer = Adam(finder.suggest_optimizer_groups(optimal_lrs))
```

## Next Steps

- [Training Integration](training_integration.md): Use during actual training
- [Dashboard Usage](dashboard_usage.md): Visual monitoring
- [Expert System](expert_system.md): Automated recommendations
