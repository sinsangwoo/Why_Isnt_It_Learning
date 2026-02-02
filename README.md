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

**Gradient Pathology** is a PyTorch-based framework that automatically diagnoses training instabilities in deep neural networks. Born from a high school student's curiosity about gradient vanishing/exploding, this tool now helps ML engineers debug real-world models in production.

### The Origin Story

In 2023, as a high school senior, I wondered:
> *"Can I actually reproduce gradient vanishing and exploding on my own computer?"*

What started as a simple TensorFlow script to visualize gradients across deep networks has evolved into a practical diagnostic framework used by researchers and engineers.

---

## ⚡ Quick Start

### Installation

```bash
pip install gradient-pathology
```

### 5-Line Usage

```python
from gradient_pathology import GradientAnalyzer
import torch.nn as nn

# Your PyTorch model
model = nn.Sequential(*[nn.Linear(64, 64), nn.ReLU()] * 20)

# Automatic diagnosis
analyzer = GradientAnalyzer(model)
report = analyzer.diagnose(num_steps=100)
print(report.summary())  # "Layer 15: Gradient vanishing detected (mean: 1e-8)"
```

---

## 🔬 What Problems Does It Solve?

### For ML Engineers
- **"Why won't my model train?"** → Pinpoint gradient pathologies in seconds
- **"Which layer is broken?"** → Layer-wise gradient flow visualization
- **"What hyperparameters should I try?"** → Automatic recommendations

### For Researchers
- Reproduce classic gradient problems (vanishing/exploding) with controlled experiments
- Benchmark initialization schemes, activation functions, and normalization techniques
- Generate publication-ready visualizations

---

## 🛠️ Features

### Current (v0.1.0)
- ✅ PyTorch-native gradient tracking
- ✅ Automatic detection of vanishing/exploding gradients
- ✅ Multi-layer histogram visualization
- ✅ Comparison across activation functions (Sigmoid, Tanh, ReLU, etc.)
- ✅ Initialization scheme benchmarking

### Coming Soon (Phase 2-5)
- 🚧 Real-time training dashboard (Streamlit/TensorBoard)
- 🚧 Transformer-specific diagnostics (attention entropy, FFN saturation)
- 🚧 Distributed training support (FSDP, DeepSpeed)
- 🚧 Automatic hyperparameter recommendations
- 🚧 Cost optimization analysis ("Save 20 GPU-hours with these settings")

---

## 📊 Example: Reproducing Gradient Vanishing

```python
from gradient_pathology.experiments import compare_activations

# Compare sigmoid vs ReLU in a 50-layer network
results = compare_activations(
    depth=50,
    activations=['sigmoid', 'relu'],
    samples=1000
)

results.plot()  # Generates publication-ready figures
```

**Output:**
- Sigmoid: Mean gradient @ layer 50 = `1.2e-9` ❌ (vanishing)
- ReLU: Mean gradient @ layer 50 = `0.32` ✅ (healthy)

---

## 🎓 Educational Use Cases

Perfect for:
- **ML course projects**: Demonstrate gradient flow concepts visually
- **Research onboarding**: Quickly understand why classic architectures (pre-ResNet) struggled with depth
- **Interview prep**: "Can you explain gradient vanishing?" → Show working code

---

## 🏗️ Project Evolution

| Stage | Description | Status |
|-------|-------------|--------|
| **Phase 0** (2023) | High school exploration: TensorFlow script | ✅ Complete |
| **Phase 1** (2025) | Modern foundation: PyTorch, packaging, CI/CD | 🚧 In Progress |
| **Phase 2** | Real-time diagnostics: Dashboard, monitoring | 📅 Planned |
| **Phase 3** | Research-grade: Advanced analysis, reproducibility | 📅 Planned |
| **Phase 4** | Ecosystem integration: HuggingFace, Lightning | 📅 Planned |
| **Phase 5** | LLM-era specialization: Transformer diagnostics | 📅 Planned |

---

## 🤝 Contributing

This project welcomes contributions from students, researchers, and engineers!

### Development Setup

```bash
git clone https://github.com/sinsangwoo/Why_Isnt_It_Learning.git
cd Why_Isnt_It_Learning
pip install -e ".[dev]"
pytest  # Run tests
```

### Roadmap Priorities
1. Transformer-specific diagnostics (help wanted!)
2. Web dashboard (Streamlit contributors welcome)
3. Additional language support (currently English/Korean)

---

## 📚 Technical Background

### Why Gradients Matter

In deep learning, backpropagation computes gradients layer-by-layer. When networks are too deep or use problematic activation functions:
- **Vanishing gradients**: Gradients → 0, early layers don't learn
- **Exploding gradients**: Gradients → ∞, training diverges

This tool makes these invisible problems visible.

### Classic Solutions We Benchmark
- Initialization: Xavier/He
- Activations: ReLU family, GELU
- Normalization: BatchNorm, LayerNorm
- Architecture: Skip connections (ResNet), attention (Transformers)

---

## 📖 Citation

If you use this tool in research, please cite:

```bibtex
@software{gradient_pathology2025,
  author = {Sin, Sangwoo},
  title = {Gradient Pathology: From High School Curiosity to Production ML Diagnostics},
  year = {2025},
  url = {https://github.com/sinsangwoo/Why_Isnt_It_Learning}
}
```

---

## 🙏 Acknowledgments

- Original inspiration: CS231n (Stanford), Coursera Deep Learning Specialization
- Built with: PyTorch, NumPy, Matplotlib
- Community: Thanks to everyone who asked "왜 학습이 안 돼요?" on forums

---

## 📄 License

MIT License - Feel free to use in research, education, or production.

---

## 🔗 Links

- [Documentation](https://github.com/sinsangwoo/Why_Isnt_It_Learning/wiki) (coming soon)
- [Issue Tracker](https://github.com/sinsangwoo/Why_Isnt_It_Learning/issues)
- [Changelog](CHANGELOG.md)

---

**Built with curiosity 🔬 | Maintained with pragmatism 🛠️**
