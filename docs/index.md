# Gradient Pathology

Automated gradient analysis and diagnostics for deep learning.

## Quick Start

```python
import torch.nn as nn
from gradient_pathology import GradientAnalyzer

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

analyzer = GradientAnalyzer(model)
report = analyzer.diagnose(num_steps=50)
print(report.summary())
```

## Features

- **Automatic Detection**: Vanishing, exploding, and unstable gradients
- **Real-time Monitoring**: Dashboard and training callbacks
- **Advanced Analysis**: LR finder, Hessian analysis, effective rank
- **Transformer Support**: Attention monitoring and specialized diagnostics
- **Expert System**: Automated recommendations based on detected issues
- **Reproducibility**: Docker, MLflow, and benchmark suite

## Contents

```{toctree}
:maxdepth: 2

quickstart
tutorials/index
api/index
casestudies/index
```
