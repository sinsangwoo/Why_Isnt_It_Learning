# Gradient Pathology

Gradient 문제를 자동으로 찾아주는 도구.

## 설치

```bash
pip install gradient-pathology
```

## 사용법

### 기본 진단

```python
from gradient_pathology import GradientAnalyzer
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

analyzer = GradientAnalyzer(model)
report = analyzer.diagnose(num_steps=50)
print(report.summary())
```

### 대시보드

```bash
streamlit run src/gradient_pathology/dashboard.py
```

### Transformer 진단

```python
from gradient_pathology.llm import TransformerDiagnostics

diag = TransformerDiagnostics(model)
stats = diag.analyze_attention_entropy(attn_weights, "layer_0")
```

### FSDP 분석

```python
from gradient_pathology.llm import FSDPAnalyzer

analyzer = FSDPAnalyzer(fsdp_model)
balance = analyzer.check_shard_balance()
```

### 양자화 분석

```python
from gradient_pathology.llm import QuantizationAnalyzer

analyzer = QuantizationAnalyzer(model)
quantized = analyzer.detect_quantized_layers()
```

## 주요 기능

- Vanishing/Exploding gradient 자동 감지
- 실시간 모니터링 대시보드
- Transformer attention entropy 분석
- FSDP shard balance 체크
- 양자화 영향 측정
- HuggingFace/Lightning/Ray Tune 통합
- MLflow 실험 추적
- Docker 재현성

## 예제

```python
# 문제 있는 모델
model = nn.Sequential(
    *[nn.Linear(64, 64), nn.Sigmoid()] * 20,
)

analyzer = GradientAnalyzer(model)
report = analyzer.diagnose()
# → Vanishing gradients 감지

# 수정
model = nn.Sequential(
    *[nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU()] * 20,
)

report = analyzer.diagnose()
# → Healthy
```

## 문서

- [Quick Start](docs/quickstart.md)
- [Tutorials](docs/tutorials/)
- [Case Studies](docs/casestudies/)

## 라이선스

MIT
