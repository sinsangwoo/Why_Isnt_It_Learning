# Gradient Pathology

Production-grade gradient analysis for PyTorch models.

[![CI](https://github.com/sinsangwoo/Why_Isnt_It_Learning/workflows/CI/badge.svg)](https://github.com/sinsangwoo/Why_Isnt_It_Learning/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)

## Overview

Automated detection and diagnosis of gradient pathologies in deep learning models. Supports vanilla networks, transformers, FSDP, and quantized models.

## Installation

```bash
pip install gradient-pathology
```

## Quick Start

```python
import torch.nn as nn
from gradient_pathology import GradientAnalyzer

model = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

analyzer = GradientAnalyzer(model)
report = analyzer.diagnose(num_steps=50, input_shape=(64,))
print(report.summary())
```

## Features

### Core Analysis
- Automatic pathology detection (vanishing, exploding, unstable)
- Per-layer gradient statistics
- Real-time training callbacks

### Advanced Tools
- Learning rate finder
- Hessian eigenvalue analysis
- Effective rank computation
- Gradient flow visualization

### Transformer Support
- Attention entropy monitoring
- FFN saturation detection
- Layer-wise diagnostics

### LLM Features
- FSDP shard balance analysis
- Quantization impact measurement (8-bit/4-bit)
- Distributed training support

### Integration
- HuggingFace Transformers
- PyTorch Lightning
- Ray Tune
- Streamlit dashboard

### Reproducibility
- Docker containers
- MLflow tracking
- Benchmark suite

## Usage

### Training Integration

```python
from gradient_pathology.callbacks import GradientMonitorCallback

callback = GradientMonitorCallback(model, check_every_n_steps=100)

for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(batch)
        callback.on_batch_end(optimizer)
```

### Real-time Dashboard

```bash
streamlit run gradient_pathology/dashboard.py
```

### LR Finder

```python
from gradient_pathology.advanced import LRFinder

finder = LRFinder(model, optimizer)
suggested_lr = finder.find(dataloader, loss_fn)
```

### Expert System

```python
from gradient_pathology.expert import ExpertSystem

expert = ExpertSystem()
diagnoses = expert.diagnose_architecture(model, gradient_stats)
print(expert.generate_report())
```

### FSDP Analysis

```python
from gradient_pathology.llm import FSDPAnalyzer

analyzer = FSDPAnalyzer(fsdp_model)
balance = analyzer.check_shard_balance()
```

## Documentation

- [Quick Start Guide](docs/quickstart.md)
- [API Reference](docs/api/)
- [Tutorials](docs/tutorials/)
- [Case Studies](docs/casestudies/)

## Benchmarks

```bash
python -m gradient_pathology.benchmark --device cpu
```

## Development

```bash
git clone https://github.com/sinsangwoo/Why_Isnt_It_Learning.git
cd Why_Isnt_It_Learning
pip install -e ".[dev]"
pytest tests/
```

## Docker

```bash
docker build -t gradient-pathology .
docker run gradient-pathology
```

## Citation

```bibtex
@software{gradient_pathology,
  title={Gradient Pathology: Automated Gradient Analysis for PyTorch},
  author={Sin, Sangwoo},
  year={2025},
  url={https://github.com/sinsangwoo/Why_Isnt_It_Learning}
}
```

## License

MIT
