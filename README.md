# Gradient Pathology

**Production-grade gradient diagnostics for PyTorch — diagnosis, visualisation, real-time monitoring, and automated expert advice in one library.**

[![CI](https://github.com/sinsangwoo/Why_Isnt_It_Learning/workflows/CI/badge.svg)](https://github.com/sinsangwoo/Why_Isnt_It_Learning/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.8.0-green.svg)](https://github.com/sinsangwoo/Why_Isnt_It_Learning)

---

## Overview

Gradient Pathology automates the full diagnostic workflow for deep learning training:

| Phase | What it delivers |
|---|---|
| **Phase 1** — Data pipeline | `grad_norm`, `LayerGroup`, snapshot store for every parameter |
| **Phase 2** — Heatmap | Interactive Plotly architecture graph: node fill = grad intensity, border = health |
| **Phase 3** — Sankey | Information-flow diagram: narrow bands = information-loss zones, layer deep-dive panel |
| **Phase 4** — Live + Expert | Real-time training monitor + 7-rule `ExpertEngine` with code-snippet recommendations |

Works with vanilla MLPs, Transformers, FSDP, 8/4-bit quantization, HuggingFace Trainer, and PyTorch Lightning.

---

## Installation

```bash
pip install gradient-pathology
```

```bash
# With interactive dashboard (Streamlit + Plotly)
pip install "gradient-pathology[dashboard]"

# With snapshot storage (Pandas + Parquet)
pip install "gradient-pathology[storage]"

# Full development environment
pip install "gradient-pathology[dev]"
```

---

## Quick Start

### One-line diagnosis

```python
from gradient_pathology import GradientAnalyzer

analyzer = GradientAnalyzer(model)
report   = analyzer.diagnose(num_steps=50, input_shape=(64,))
print(report.summary())
```

```
================================================================
GRADIENT ANALYSIS REPORT  —  synthetic  —  50 steps
================================================================
Global Statistics:
  Mean: 3.45e-03  HEALTHY
  Std:  2.11e-02

Layer breakdown:
  [  0] transformer.h.0.attn.weight  group=attention  grad_norm=4.12e-03  HEALTHY
  [  1] transformer.h.0.attn.bias    group=attention  grad_norm=2.08e-04  HEALTHY
  ...

No critical issues detected.
================================================================
```

### Launch the dashboard

```bash
streamlit run src/gradient_pathology/dashboard.py
```

The dashboard opens with four tabs:

```
Live Monitor  |  Sankey Flow  |  Architecture Heatmap  |  Classic Report
```

---

## Feature Guide

### Phase 1 — Data Pipeline

#### Layer metadata and grad_norm

`GradientAnalyzer` enriches every `LayerGradientStats` with Phase-1 fields:

```python
report = analyzer.diagnose(num_steps=100)

for s in report.layer_stats:
    print(s.layer_name, s.layer_type, s.group, f"grad_norm={s.grad_norm:.2e}")
```

`LayerGroup` values: `ATTENTION`, `FFN`, `LAYER_NORM`, `EMBEDDING`, `HEAD`, `OTHER`

#### Snapshot store

Record step-by-step gradient statistics to JSON or Parquet:

```python
from gradient_pathology import GradientSnapshotStore

store = GradientSnapshotStore(output_dir="runs/exp1", fmt="parquet")

for step, (x, y) in enumerate(loader):
    loss.backward()
    store.record_from_stats(step, report.layer_stats)

store.flush()
df = store.load()   # pandas DataFrame: step x layer
```

#### TransformerLayerClassifier

Classify any model's parameters without a forward pass:

```python
from gradient_pathology import TransformerLayerClassifier

clf  = TransformerLayerClassifier(model)
meta = clf.build_param_metadata()
# {'transformer.h.0.attn.c_attn.weight': ('Linear', LayerGroup.ATTENTION), ...}
```

---

### Phase 2 — Architecture Heatmap

An interactive Plotly node graph where **node fill** encodes grad_norm intensity and **node border** encodes pathology health status.

```python
from gradient_pathology import GradientHeatmapRenderer

renderer = GradientHeatmapRenderer(report)
renderer.show()                          # browser
renderer.save_html("heatmap.html")       # standalone file
```

```python
# Inside the Streamlit dashboard
from gradient_pathology.heatmap.dashboard_tab import render_heatmap_tab

with tab_heatmap:
    render_heatmap_tab(report)
```

**Visual encoding:**

| Channel | Meaning |
|---|---|
| Node fill colour | `grad_norm` intensity (Viridis or RdYlGn) |
| Node border ring | Pathology health (green = healthy, red = vanishing, ...) |
| Edge opacity | Gradient flow connection |

**Hard-pin rule:** vanishing layers always map to the darkest colour stop regardless of relative normalisation — they are never hidden by a healthy neighbour.

---

### Phase 3 — Sankey Information-Flow Diagram

Converts `grad_norm` values to link widths. **Narrow bands are information-loss zones.**

```python
from gradient_pathology import GradientSankeyRenderer

renderer = GradientSankeyRenderer(report, strategy="log")
renderer.show()
renderer.save_html("sankey.html")
```

```python
# From GradientFlowGraph (one-liner shim)
from gradient_pathology.auto.gradient_flow_graph import GradientFlowGraph

gfg = GradientFlowGraph(model)
fig = gfg.plot_sankey(strategy="log", vanishing_threshold=1e-7)
fig.show()
```

**Five flow strategies:**

| Strategy | When to use |
|---|---|
| `LOG` (default) | Wide dynamic range — most Transformer models |
| `NORMALISED` | Linear spread when norms are on similar scales |
| `RELATIVE` | Emphasise fraction of peak, not absolute magnitude |
| `SQRT` | Middle ground between LOG and NORMALISED |
| `RAW` | When norms are already comparable |

**Link colour = FlowZone:**

| Zone | Colour | Condition |
|---|---|---|
| `HEALTHY` | Semi-transparent green | Both endpoints healthy |
| `VANISHING` | Semi-transparent red | Downstream node vanishing |
| `BOTTLENECK` | Semi-transparent amber | Relative drop > threshold |
| `EXPLODING` | Semi-transparent orange | Either endpoint exploding |

#### Layer deep-dive panel

```python
from gradient_pathology.sankey import LayerDetailPanel

panel = LayerDetailPanel(report)

# Plain dict (for any UI)
d = panel.build_dict("transformer.h.0.attn.c_attn.weight")
print(d["pathology"], d["peer_rank"], d["recommendations"])

# 2x2 Plotly subplot (Radar / Bar / Peer-rank / Diagnosis table)
fig = panel.build_plotly("transformer.h.0.attn.c_attn.weight")
fig.show()
```

---

### Phase 4 — Real-time Monitoring + Expert System

#### LiveGradientBridge

Connect your training loop to the Streamlit dashboard with zero blocking:

```python
from gradient_pathology import LiveGradientBridge, StreamlitCallback

# Shared object (thread-safe ring-buffer)
bridge   = LiveGradientBridge.from_session_state(max_steps=500)
callback = StreamlitCallback(model, bridge, push_every_n_steps=5)

for step, (x, y) in enumerate(loader):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    callback.on_batch_end(step=step, loss=loss.item())
```

```python
# Dashboard side — reads on every Streamlit rerun
snap  = bridge.latest_snapshot()         # GradientSnapshot
steps, losses = bridge.metrics_series("loss")
alerts = bridge.drain_alerts()           # vanishing/exploding auto-alerts
```

#### HuggingFace Trainer drop-in

```python
from gradient_pathology import LiveGradientBridge, HuggingFaceCallbackAdapter

bridge  = LiveGradientBridge()
trainer = Trainer(
    model=model,
    callbacks=[HuggingFaceCallbackAdapter(model, bridge)],
)
trainer.train()
```

#### ExpertEngine

Layer-aware rule engine with 7 built-in diagnostic rules:

```python
from gradient_pathology import ExpertEngine

engine   = ExpertEngine()
findings = engine.analyse(report)

for f in findings:
    print(f.emoji, f.title)
    print(f.detail)
    if f.code_hint:
        print("--- fix ---")
        print(f.code_hint)
```

**Example output:**

```
Vanishing gradients in 4 layer(s) (20%)
Layers below grad_norm < 1e-07 cannot propagate gradients back to early layers.
Likely causes: sigmoid/tanh saturation, missing normalisation.
--- fix ---
# Option 1: Replace saturating activations
model = replace_activations(model, nn.Sigmoid, nn.GELU)

# Option 2: Add LayerNorm after each Linear
# Linear -> LayerNorm -> GELU  (PreLN pattern)
```

**Built-in rules:**

| Rule | Severity | What it catches |
|---|---|---|
| `vanishing_layers` | Critical | `grad_norm < vanishing_threshold` |
| `exploding_layers` | Critical | `grad_norm > exploding_threshold` |
| `dead_neurons` | Warning | `zero_ratio > 90%` |
| `bottleneck_cascade` | Warning | Abrupt consecutive-depth norm drop |
| `no_layernorm` | Info / Critical | Deep network missing normalisation |
| `attention_health` | Warning | Attention-group near-zero gradients |
| `layernorm_explosion` | Warning | LayerNorm parameter explosion |

**Custom rules:**

```python
from gradient_pathology import ExpertEngine, ExpertFinding

engine = ExpertEngine()

@engine.register_rule
def my_rule(report):
    if some_condition(report):
        return [ExpertFinding(
            rule_id="my_rule",
            severity="warning",
            title="Custom issue found",
            detail="Explanation in **markdown**.",
            code_hint="# Fix:\nmodel.apply(fix_fn)",
        )]
    return []
```

---

## Streamlit Dashboard — Full Walkthrough

### Starting

```bash
pip install "gradient-pathology[dashboard]"
streamlit run src/gradient_pathology/dashboard.py
```

### Tab 1 — Live Monitor

Displays live training data pushed through `LiveGradientBridge`:

- Status row: last step, loss, global grad-mean, alert count
- Training loss curve (Plotly, log scale)
- Grad-norm trend with vanishing/exploding threshold bands
- Per-layer bar chart (top-N layers, colour = health status)
- Alert feed (latest 10 auto-detected vanishing/exploding events)
- Refresh button

### Tab 2 — Sankey Flow

- Settings: flow strategy, vanishing threshold, bottleneck ratio, merge toggle, group-colour toggle
- Full-width Sankey diagram
- Info-loss summary: vanishing links, bottleneck links, max info loss %
- Critical zones table: top-8 worst links
- Layer deep-dive: selectbox (default = worst layer) -> 2x2 Plotly diagnostic figure
- Expert panel: filtered findings for the selected layer

### Tab 3 — Architecture Heatmap

- Settings: colormap (Viridis / RdYlGn), layout (Sequential / Grouped / Spring), vanishing threshold, edge toggle
- Full interactive Plotly architecture graph
- Vanishing / Exploding warning panels

### Tab 4 — Classic Report

- Gradient distribution bar chart + pathology pie chart
- Full layer-by-layer text report
- Expert System full report (collapsible)
- Actionable recommendations per problematic layer

### Expert System banner

A global coloured banner appears above all tabs when findings exist:

- Green: all healthy
- Orange: warnings only
- Red: critical issues (auto-expanded with detail + code hints)

---

## Other Capabilities

### LLM / Transformer Diagnostics

```python
from gradient_pathology.llm import TransformerDiagnostics

diag = TransformerDiagnostics(model)
if diag.detect_attention_collapse(attn_weights):
    print("Attention collapsed — check temperature / dropout")

ffn_stats = diag.analyze_ffn_saturation(ffn_activations, "layer_0")
```

### FSDP Support

```python
from gradient_pathology.llm import FSDPAnalyzer

analyzer = FSDPAnalyzer(fsdp_model)
balance  = analyzer.check_shard_balance()
```

### Quantization Analysis

```python
from gradient_pathology.llm import QuantizationAnalyzer

analyzer = QuantizationAnalyzer(model)
error    = analyzer.analyze_quantization_error(original, quantized)
```

### Fine-tuning Monitors

```python
from gradient_pathology import LoRARankTracker, AdapterMonitor, ForgettingDetector

tracker   = LoRARankTracker(model)
monitor   = AdapterMonitor(model)
forgetter = ForgettingDetector(model)
```

### Cost Optimisation

```python
from gradient_pathology.cost import CostCalculator, TrainingOptimizer

calc      = CostCalculator()
optimizer = TrainingOptimizer(model)
print(optimizer.generate_report("A100", hours=32.0, health="VANISHING"))
```

---

## Project Structure

```
src/gradient_pathology/
|-- analyzer.py              # GradientAnalyzer -- report generation
|-- core.py                  # LayerGradientStats, GradientReport, LayerGroup
|-- callbacks.py             # GradientMonitor (legacy training monitor)
|
|-- pipeline/                # Phase 1
|   |-- classifier.py        # TransformerLayerClassifier
|   `-- snapshot.py          # GradientSnapshotStore (JSON / Parquet)
|
|-- heatmap/                 # Phase 2
|   |-- colormap.py          # ColorScheme, grad_norm_to_color
|   |-- layout.py            # LayoutStrategy, ArchitectureLayout
|   |-- renderer.py          # GradientHeatmapRenderer (Plotly)
|   `-- dashboard_tab.py     # render_heatmap_tab()
|
|-- sankey/                  # Phase 3
|   |-- flow.py              # SankeyFlowBuilder, FlowStrategy, FlowZone
|   |-- renderer.py          # GradientSankeyRenderer (go.Sankey)
|   |-- detail_panel.py      # LayerDetailPanel (2x2 diagnostic subplot)
|   `-- dashboard_tab.py     # render_sankey_tab()
|
|-- monitor/                 # Phase 4
|   |-- bridge.py            # LiveGradientBridge (thread-safe ring-buffer)
|   `-- callback.py          # StreamlitCallback, HuggingFaceCallbackAdapter
|
|-- expert/                  # Phase 4
|   |-- rules.py             # ExpertSystem (global-scalar rules, legacy)
|   `-- engine.py            # ExpertEngine -- 7 layer-aware rules + ExpertFinding
|
|-- dashboard/               # Phase 4
|   |-- expert_panel.py      # render_expert_banner/popup/layer_panel()
|   |-- realtime_tab.py      # render_realtime_tab()
|   `-- layout.py            # run_dashboard() -- 4-tab orchestrator
|
|-- dashboard.py             # Backward-compat shim (streamlit run target)
|
|-- auto/
|   |-- gradient_flow_graph.py  # GradientFlowGraph (plot_heatmap, plot_sankey)
|   |-- effective_rank.py
|   `-- layer_lr_finder.py
|
|-- llm/                     # LLM-specific diagnostics
|-- cost/                    # GPU cost calculator
|-- integrations/            # HuggingFace / Lightning / Ray Tune
|-- benchmark/               # MLflow tracking
`-- finetuning/              # LoRA, Adapter, Forgetting monitors
```

---

## Testing

```bash
# All tests
pytest tests/ -v --cov=src/gradient_pathology

# Per-phase
pytest tests/test_phase1_pipeline.py          # 25 tests
pytest tests/test_phase2_heatmap.py           # 30 tests
pytest tests/test_phase3_sankey.py            # 35 tests
pytest tests/test_phase4_realtime_expert.py   # 30 tests
```

Tests are self-contained. Plotly-dependent cases auto-skip when Plotly is absent.

---

## Development Setup

```bash
git clone https://github.com/sinsangwoo/Why_Isnt_It_Learning.git
cd Why_Isnt_It_Learning
pip install -e ".[dev]"
```

```bash
ruff check src/ tests/   # lint
mypy src/                # type check
```

---

## Benchmark

```bash
python -m gradient_pathology.benchmark --device cpu --suite standard
```

| Model | Layers | Diagnostic time | Overhead |
|---|---|---|---|
| Small MLP | 5 | 0.12 s | < 1% |
| Deep network | 50 | 0.89 s | ~ 2% |
| Transformer | 12 | 1.34 s | ~ 3% |

---

## Citation

```bibtex
@software{gradient_pathology,
  title  = {Gradient Pathology: Automated Gradient Analysis for PyTorch},
  author = {Sin, Sangwoo},
  year   = {2025},
  url    = {https://github.com/sinsangwoo/Why_Isnt_It_Learning}
}
```

---

## References

- Glorot & Bengio (2010) -- Understanding the difficulty of training deep feedforward neural networks
- He et al. (2015) -- Delving Deep into Rectifiers
- Ba et al. (2016) -- Layer Normalization
- Xiong et al. (2020) -- On Layer Normalization in the Transformer Architecture

---

## Contributing

Contributions are welcome. Please open an issue first to discuss your idea.

Focus areas: additional diagnostic rules, new framework integrations, performance optimisations, case studies, documentation.

## License

MIT License -- see [LICENSE](LICENSE) for details.

---

**Built with love for the deep learning community**
