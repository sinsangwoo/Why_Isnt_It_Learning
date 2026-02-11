# Gradient Pathology

**Production-grade gradient diagnostics for PyTorch models.**

[![CI](https://github.com/sinsangwoo/Why_Isnt_It_Learning/workflows/CI/badge.svg)](https://github.com/sinsangwoo/Why_Isnt_It_Learning/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Automated detection and resolution of gradient pathologies in neural networks. From vanilla MLPs to large language models with FSDP and quantization.

**Key capabilities:**
- Automatic pathology detection (vanishing, exploding, unstable gradients)
- Transformer-specific diagnostics (attention entropy, FFN saturation)
- Distributed training support (FSDP shard analysis)
- Quantization impact measurement (8-bit/4-bit)
- Cost optimization (GPU selection, training time estimation)
- Production integrations (HuggingFace, Lightning, Ray Tune)

## Installation

```bash
pip install gradient-pathology
```

Optional dependencies:
```bash
pip install gradient-pathology[dashboard]  # Streamlit UI
pip install gradient-pathology[dev]        # Development tools
```

## Quick Start

### Basic Diagnosis

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

**Output:**
```
================================================================
GRADIENT ANALYSIS REPORT
================================================================
Global Statistics:
  Mean: 3.45e-03  ✓ HEALTHY
  Std:  2.11e-02

No critical issues detected.
================================================================
```

## Core Features

### 1. Automatic Pathology Detection

Detects and diagnoses gradient issues across all layers:

```python
analyzer = GradientAnalyzer(model)
report = analyzer.diagnose(num_steps=100)

for layer_name, stats in zip(model_layers, report.layer_stats):
    diagnosis = stats.diagnose()  # HEALTHY, VANISHING, EXPLODING, UNSTABLE
    if diagnosis != "HEALTHY":
        print(f"{layer_name}: {diagnosis}")
```

### 2. Real-time Monitoring

**Training Callbacks:**
```python
from gradient_pathology.callbacks import GradientMonitorCallback

callback = GradientMonitorCallback(model, check_every_n_steps=100)

for batch in dataloader:
    loss = train_step(batch)
    callback.on_batch_end(optimizer)
```

**Dashboard:**
```bash
streamlit run gradient_pathology/dashboard.py
```

### 3. Advanced Analysis

**Learning Rate Finder:**
```python
from gradient_pathology.advanced import LRFinder

finder = LRFinder(model, optimizer)
suggested_lr = finder.find(dataloader, loss_fn)
```

**Hessian Analysis:**
```python
from gradient_pathology.advanced import HessianAnalyzer

analyzer = HessianAnalyzer(model)
eigenvalues = analyzer.compute_top_eigenvalues(dataloader, loss_fn)
```

**Effective Rank:**
```python
from gradient_pathology.auto import EffectiveRankAnalyzer

analyzer = EffectiveRankAnalyzer(model)
rank = analyzer.compute_effective_rank()  # Parameter efficiency
```

### 4. Expert System

Automated recommendations based on detected issues:

```python
from gradient_pathology.expert import ExpertSystem

expert = ExpertSystem()
diagnoses = expert.diagnose_architecture(model, gradient_stats)
print(expert.generate_report())
```

**Example output:**
```
🚨 CRITICAL ISSUES:

Deep network without proper normalization (confidence: 95%)
Recommendations:
  • Add LayerNorm after each layer
  • Use PreLN (Pre-LayerNorm) architecture
  • Implement gradient checkpointing
```

## LLM-Specific Features

### Transformer Diagnostics

```python
from gradient_pathology.llm import TransformerDiagnostics

diag = TransformerDiagnostics(model)

# Attention analysis
stats = diag.analyze_attention_entropy(attn_weights, "layer_0")
if diag.detect_attention_collapse(attn_weights):
    print("⚠️ Attention collapsed")

# FFN analysis
ffn_stats = diag.analyze_ffn_saturation(ffn_activations, "layer_0")
if ffn_stats["saturated_fraction"] > 0.5:
    print("⚠️ FFN saturated")
```

### FSDP Support

```python
from gradient_pathology.llm import FSDPAnalyzer

analyzer = FSDPAnalyzer(fsdp_model)
balance = analyzer.check_shard_balance()
if balance["imbalance_ratio"] > 10.0:
    print("⚠️ High shard imbalance")
```

### Quantization Analysis

```python
from gradient_pathology.llm import QuantizationAnalyzer

analyzer = QuantizationAnalyzer(model)
quantized_layers = analyzer.detect_quantized_layers()
error_stats = analyzer.analyze_quantization_error(original, quantized)
```

## Cost Optimization

### GPU Cost Calculator

```python
from gradient_pathology.cost import CostCalculator

calc = CostCalculator()

# Compare GPU options
costs = calc.compare_gpus({
    "A100": 32.0,
    "V100": 48.0,
    "T4": 96.0,
})

cheapest = calc.find_cheapest(costs)
print(f"Best option: {cheapest['gpu_type']} - ${cheapest['total_cost']:.2f}")
```

### Training Optimizer

```python
from gradient_pathology.cost import TrainingOptimizer

optimizer = TrainingOptimizer(model)
suggestion = optimizer.suggest_optimization(
    current_gpu="A100",
    estimated_hours=32.0,
    gradient_health="VANISHING"
)

print(optimizer.generate_report("A100", 32.0, "VANISHING"))
```

**Example output:**
```
======================================================================
COST OPTIMIZATION REPORT
======================================================================
Current Configuration:
  GPU: A100
  Training time: 32.0h
  Cost: $117.44

Optimized Configuration:
  GPU: A100
  Training time: 10.7h
  Cost: $39.15

Savings:
  Amount: $78.29
  Percent: 66.7%

Reason:
  Fix vanishing gradients to converge 3x faster on same GPU.
```

## Integrations

### HuggingFace Transformers

```python
from gradient_pathology.integrations import HuggingFacePlugin

plugin = HuggingFacePlugin(model)
trainer = Trainer(model=model, callbacks=[plugin])
```

### PyTorch Lightning

```python
from gradient_pathology.integrations import GradientPathologyCallback

callback = GradientPathologyCallback(check_every_n_steps=100)
trainer = pl.Trainer(callbacks=[callback])
```

### Ray Tune

```python
from gradient_pathology.integrations import GradientPathologyReporter

reporter = GradientPathologyReporter(model)
reporter.report_metrics(step=iteration)
```

## Reproducibility

### Docker

```bash
docker build -t gradient-pathology .
docker run gradient-pathology
```

### Benchmarks

```bash
python -m gradient_pathology.benchmark --device cpu --suite standard
```

### MLflow Tracking

```python
from gradient_pathology.benchmark import ExperimentTracker

with ExperimentTracker(experiment_name="my_experiments") as tracker:
    tracker.log_params({"layers": 20, "activation": "gelu"})
    report = runner.run_benchmark(config)
    tracker.log_metrics({"gradient_mean": report.global_mean})
```

## Documentation

- **[Quick Start Guide](docs/quickstart.md)** - Get started in 5 minutes
- **[API Reference](docs/api/)** - Complete API documentation
- **[Tutorials](docs/tutorials/)** - Step-by-step guides
- **[Case Studies](docs/casestudies/)** - Real-world examples
  - [Stable Diffusion Training Stabilization](docs/casestudies/stable_diffusion.md)
  - BERT Fine-tuning (coming soon)
  - GPT Training (coming soon)

## Architecture

```
gradient_pathology/
├── analyzer.py              # Core gradient analysis
├── core.py                  # Data structures
├── callbacks.py             # Training integration
├── dashboard.py             # Streamlit UI
├── visualize.py             # Plotting utilities
├── advanced/
│   ├── lr_finder.py        # Learning rate search
│   ├── hessian.py          # Second-order analysis
│   └── transformer_diagnostics.py
├── auto/
│   ├── effective_rank.py   # Parameter efficiency
│   ├── layer_lr_finder.py  # Layer-wise LR
│   └── gradient_flow_graph.py
├── expert/
│   └── rules.py            # Diagnostic expert system
├── transformers/
│   ├── attention_monitor.py
│   └── hooks.py
├── llm/
│   ├── transformer_advanced.py
│   ├── distributed.py      # FSDP support
│   └── quantization.py
├── cost/
│   ├── calculator.py       # GPU cost estimation
│   └── optimizer.py        # Training optimization
├── integrations/
│   ├── huggingface.py
│   ├── lightning.py
│   └── raytune.py
└── benchmark/
    ├── runner.py
    └── tracker.py          # MLflow integration
```

## Development

### Setup

```bash
git clone https://github.com/sinsangwoo/Why_Isnt_It_Learning.git
cd Why_Isnt_It_Learning
pip install -e ".[dev]"
```

### Testing

```bash
pytest tests/ -v --cov=src/gradient_pathology
```

### Code Quality

```bash
ruff check src/ tests/
mypy src/
```

## Performance

**Benchmark results** (Intel i7, CPU mode):

| Model | Layers | Diagnostic Time | Overhead |
|-------|--------|-----------------|----------|
| Small MLP | 5 | 0.12s | <1% |
| Deep Network | 50 | 0.89s | ~2% |
| Transformer | 12 | 1.34s | ~3% |

## Citation

If you use Gradient Pathology in your research:

```bibtex
@software{gradient_pathology,
  title={Gradient Pathology: Automated Gradient Analysis for PyTorch},
  author={Sin, Sangwoo},
  year={2025},
  url={https://github.com/sinsangwoo/Why_Isnt_It_Learning}
}
```

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

### Areas for Contribution

- Additional case studies
- New diagnostic rules
- Framework integrations
- Performance optimizations
- Documentation improvements

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Based on research from:
- Glorot & Bengio (2010) - Understanding the difficulty of training deep feedforward neural networks
- He et al. (2015) - Delving Deep into Rectifiers
- Ba et al. (2016) - Layer Normalization
- Xiong et al. (2020) - On Layer Normalization in the Transformer Architecture

## Support

- **Issues**: [GitHub Issues](https://github.com/sinsangwoo/Why_Isnt_It_Learning/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sinsangwoo/Why_Isnt_It_Learning/discussions)
- **Email**: Contact via GitHub profile

---

**Built with ❤️ for the deep learning community**
