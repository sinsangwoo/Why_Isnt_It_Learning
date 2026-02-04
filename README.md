# Gradient Pathology

[![CI](https://github.com/sinsangwoo/Why_Isnt_It_Learning/actions/workflows/ci.yml/badge.svg)](https://github.com/sinsangwoo/Why_Isnt_It_Learning/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **From high school curiosity to production ML diagnostics**
>
> *"왜 학습이 안 될까?"라는 고3 시절의 순수한 질문에서 시작해,*  
> *실전 모델 디버깅에 즉시 쓸 수 있는 자동화 도구로 진화한 프로젝트*

---

## 🎯 What is this?

**Gradient Pathology** is a PyTorch-based framework for diagnosing training instabilities in deep neural networks. From basic gradient analysis to advanced Hessian insights and Transformer-specific diagnostics.

### The Origin Story

In 2023, as a high school senior, I wondered:
> *"Can I actually reproduce gradient vanishing and exploding on my own computer?"*

What started as a simple script has evolved into:
- **Phase 1**: Static gradient analysis
- **Phase 2**: Real-time monitoring dashboard
- **Phase 3**: Research-grade advanced analysis (NEW!)

---

## ⚡ Quick Start

### Installation

```bash
pip install gradient-pathology
```

### Basic Usage

```python
from gradient_pathology import GradientAnalyzer
import torch.nn as nn

# Your model
model = nn.Sequential(*[nn.Linear(64, 64), nn.ReLU()] * 20)

# Diagnose
analyzer = GradientAnalyzer(model)
report = analyzer.diagnose(num_steps=100)
print(report.summary())
```

### Advanced Analysis (Phase 3)

```python
from gradient_pathology.advanced import LRFinder, HessianAnalyzer

# Find optimal learning rate
lr_finder = LRFinder(model, optimizer)
lrs, losses = lr_finder.range_test(dataloader, loss_fn)
suggested_lr = lr_finder.suggest_lr(lrs, losses)

# Analyze loss landscape
hessian = HessianAnalyzer(model)
results = hessian.compute_hessian_eigenvalues(dataloader, loss_fn)
print(hessian.diagnose_sharpness(results['eigenvalues']))
```

---

## 🚀 Features

### Phase 1: Core Analysis
- ✅ Automatic gradient pathology detection
- ✅ Layer-wise statistics and visualization
- ✅ Actionable recommendations

### Phase 2: Real-time Monitoring
- ✅ Streamlit dashboard
- ✅ Training loop callbacks
- ✅ Live gradient flow tracking

### Phase 3: Advanced Analysis (NEW)
- ✅ **Learning Rate Finder** - Automatic LR discovery
- ✅ **Hessian Analysis** - Loss landscape insights
- ✅ **Transformer Diagnostics** - Attention entropy, FFN saturation
- 🚧 HuggingFace integration (coming)
- 🚧 PyTorch Lightning callbacks (coming)

---

## 📊 Example: LR Finder

```python
from gradient_pathology.advanced import LRFinder

lr_finder = LRFinder(model, optimizer)
lrs, losses = lr_finder.range_test(
    dataloader,
    loss_fn,
    start_lr=1e-7,
    end_lr=10,
    num_iter=100
)

# Visualize
lr_finder.plot(lrs, losses)

# Get recommendation
optimal_lr = lr_finder.suggest_lr(lrs, losses)
print(f"Suggested LR: {optimal_lr:.2e}")
```

---

## 🎓 Use Cases

### For ML Engineers
- **Debug training failures** in production models
- **Optimize hyperparameters** with LR finder
- **Monitor gradient health** during long training runs

### For Researchers
- **Analyze loss landscapes** with Hessian eigenvalues
- **Diagnose Transformer issues** (attention collapse, FFN saturation)
- **Generate publication-quality** gradient flow visualizations

### For Students
- **Visualize gradient problems** in real-time
- **Understand optimization** through interactive demos
- **Build intuition** for deep learning pathologies

---

## 🏗️ Project Evolution

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** (2023) | High school exploration | ✅ Complete |
| **Phase 1** (2025) | PyTorch foundation | ✅ Complete |
| **Phase 2** | Real-time dashboard | ✅ Complete |
| **Phase 3** | Advanced analysis | 🚧 **In Progress** |
| Phase 4 | Ecosystem integration | 📅 Planned |
| Phase 5 | LLM-era specialization | 📅 Planned |

---

## 🚀 Advanced Features

### 1. Hessian Analyzer

Understand your loss landscape:

```python
from gradient_pathology.advanced import HessianAnalyzer

analyzer = HessianAnalyzer(model)
results = analyzer.compute_hessian_eigenvalues(dataloader, loss_fn)

print(f"Max eigenvalue: {results['max_eigenvalue']:.2e}")
print(f"Effective rank: {results['effective_rank']}")
print(analyzer.diagnose_sharpness(results['eigenvalues']))
# Output: "FLAT_MINIMUM (Good generalization expected)"
```

### 2. Transformer Diagnostics

Specialized analysis for Transformers:

```python
from gradient_pathology.advanced import TransformerDiagnostics

diag = TransformerDiagnostics(transformer_model)

# Analyze attention
entropy = diag.analyze_attention_entropy(attention_weights)
if diag.detect_attention_collapse():
    print("⚠️ Attention collapse detected!")

# Check FFN saturation
saturation = diag.analyze_ffn_saturation(ffn_activations)
print(f"FFN saturation: {saturation:.1%}")
```

---

## 📚 Documentation

See `/examples` for:
- `advanced_analysis_demo.py` - Full demo of Phase 3 features
- `dashboard_demo.py` - Interactive dashboard
- `realtime_monitor.py` - Training loop integration

---

## 🤝 Contributing

Welcome contributions in:
- HuggingFace Transformers integration
- Additional Transformer diagnostics
- PyTorch Lightning callbacks
- Documentation improvements

---

## 📖 Citation

```bibtex
@software{gradient_pathology2025,
  author = {Sin, Sangwoo},
  title = {Gradient Pathology: From High School Curiosity to Production ML Diagnostics},
  year = {2025},
  url = {https://github.com/sinsangwoo/Why_Isnt_It_Learning}
}
```

---

## 📄 License

MIT License - Free for research, education, and production.

---

**Built with curiosity 🔬 | Maintained with pragmatism 🛠️**
