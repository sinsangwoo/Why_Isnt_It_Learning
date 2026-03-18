# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — feature/phase3-sankey-diagram

### Added (Phase 3 — Sankey Diagram)

#### Data transformer (`sankey/flow.py`)
- `FlowStrategy` enum: `LOG` (default), `NORMALISED`, `RELATIVE`, `SQRT`, `RAW` — five strategies for mapping `grad_norm` to link width.
- `FlowZone` enum: `HEALTHY`, `VANISHING`, `BOTTLENECK`, `EXPLODING`, `DEAD` — semantic health classification per link.
- `SankeyLink` dataclass: `source_idx`, `target_idx`, `value`, `raw_source_norm`, `raw_target_norm`, `zone`, `loss_fraction`.
- `SankeyFlow` dataclass: full Sankey-ready data (node labels, groups, pathologies, links) with convenience properties `vanishing_links`, `bottleneck_links`, `max_loss_fraction`.
- `SankeyFlowBuilder`: transforms a `GradientReport` into a `SankeyFlow`:
  - Log-space min-max normalisation (default) for link-width scaling.
  - Vanishing/Exploding/Bottleneck zone classification with configurable thresholds.
  - `_merge_by_module()`: merges weight+bias pairs into single nodes (L2-combined `grad_norm`); dramatically reduces node count for large Transformers.
  - Nodes ordered in **reverse depth** (output→input = left→right backprop direction).

#### Renderer (`sankey/renderer.py`)
- `GradientSankeyRenderer`: builds a Plotly `go.Sankey` figure:
  - Link width = `grad_norm`-derived value; **narrow links = information loss zones**.
  - Link colour per `FlowZone`: green (healthy), red (vanishing), amber (bottleneck), orange (exploding).
  - Node colour per `LayerGroup` (matching Phase-2 palette).
  - Rich hover tooltips on nodes (name, type, group, grad_norm, pathology) and links (zone, loss %, src/dst norms).
  - Zone colour legend annotations at figure top.
  - Dark theme (`#0F1117`) consistent with Phase-2 Heatmap.
  - `.build()`, `.show()`, `.save_html()` public API.
  - `ZONE_LINK_COLORS` and `GROUP_NODE_COLORS` colour tables.

#### Layer detail panel (`sankey/detail_panel.py`)
- `LayerDetailPanel`: per-layer deep-dive diagnostics:
  - `build_dict(layer_name)` — returns plain Python dict with full stats, pathology, peer/global rank, headline, and actionable recommendations.
  - `build_plotly(layer_name)` — 2×2 Plotly subplot figure:
    - *Top-left*: health radar chart (5 axes: grad_norm, zero_ratio, mean, std, depth).
    - *Top-right*: bar comparing this layer vs. global mean grad_norm.
    - *Bottom-left*: peer-group bar showing this layer's rank within its `LayerGroup`.
    - *Bottom-right*: diagnosis table (headline + itemised recommendations).
  - `_PATHOLOGY_ADVICE` dict: 5 pathologies × (headline, recommendations) with concrete fix suggestions.

#### Streamlit tab (`sankey/dashboard_tab.py`)
- `render_sankey_tab(report)` — full Streamlit tab with:
  - Flow strategy selector, vanishing threshold slider, bottleneck ratio slider, merge toggle, group-colour toggle.
  - Full-width Sankey figure.
  - Info-loss summary metrics (vanishing links, bottleneck links, max info loss %).
  - Sorted critical-zone table (top 8 worst links).
  - Layer deep-dive section: selectbox pre-seeded with worst layer → 2×2 `LayerDetailPanel` figure.

#### `gradient_flow_graph.py` shim
- `GradientFlowGraph.plot_sankey()` — one-liner entry point; accepts all major renderer parameters as strings.

#### `dashboard.py` upgrade
- Three-tab layout: **🌊 Sankey Flow** | 🌡️ Architecture Heatmap | 📊 Classic Report.

#### Infrastructure
- `pyproject.toml`: version bumped to `0.7.0`.
- `__init__.py`: exports `GradientSankeyRenderer`.
- `tests/test_phase3_sankey.py`: 35 tests across all sub-modules.

---

## [Unreleased] — feature/phase2-heatmap-visualization

### Added (Phase 2 — Heatmap Visualisation)
- `GradientHeatmapRenderer` with Plotly interactive architecture graph.
- Colormap utilities, Layout engine, Streamlit tab.

---

## [Unreleased] — feature/phase1-data-pipeline

### Added (Phase 1 — Data Pipeline Foundation)
- `LayerGroup` enum, `grad_norm`/`layer_type`/`depth`/`group` metadata.
- `GradientSnapshotStore`, `TransformerLayerClassifier`.
