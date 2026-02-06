# Quick Start Guide

## Installation

```bash
pip install gradient-pathology
```

## 5-Minute Diagnosis

### Step 1: Import

```python
import torch
import torch.nn as nn
from gradient_pathology import GradientAnalyzer
```

### Step 2: Create Your Model

```python
model = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

### Step 3: Run Diagnosis

```python
analyzer = GradientAnalyzer(model)
report = analyzer.diagnose(
    num_steps=50,
    input_shape=(64,),
)

print(report.summary())
```

### Step 4: Interpret Results

The report will show:
- **Global statistics**: Overall gradient health
- **Per-layer analysis**: Which layers have issues
- **Detected pathologies**: Vanishing, exploding, or unstable

## Common Issues

### Problem: Vanishing Gradients

**Symptoms**: Very small gradients (< 1e-7)

**Solutions**:
- Replace Sigmoid/Tanh with ReLU or GELU
- Add LayerNorm or BatchNorm
- Use residual connections
- Try Xavier/He initialization

### Problem: Exploding Gradients

**Symptoms**: Very large gradients (> 100)

**Solutions**:
- Implement gradient clipping
- Reduce learning rate
- Add normalization layers
- Check weight initialization

### Problem: Unstable Training

**Symptoms**: High variance in gradients

**Solutions**:
- Use layer-wise learning rates
- Add warmup schedule
- Increase batch size
- Apply gradient smoothing

## Next Steps

- [Tutorials](tutorials/index.md): Detailed walkthroughs
- [API Reference](api/index.md): Complete documentation
- [Case Studies](casestudies/index.md): Real-world examples
