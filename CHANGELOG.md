# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — feature/phase2-heatmap-visualization

### Added (Phase 2 — Heatmap Visualisation)

#### Core renderer (`heatmap/renderer.py`)
- New `GradientHeatmapRenderer` class:
  - `build()` → interactive Plotly `Figure` with architecture graph + `grad_norm` colour overlay.
  - `build_static()` → Matplotlib fallback (zero extra dependencies beyond the base install).
  - `show()` → opens figure in browser.
  - `save_html(path)` → writes self-contained HTML.
- Dark-mode canvas (`#0F1117` background) for high-contrast viewing.
- Semi-transparent warning overlays (red = vanishing, orange = exploding) behind flagged nodes.
- Gradient-flow edge traces (backward-pass direction arrows, toggleable via `show_edges`).
- Per-node hover tooltip: `layer_name`, `layer_type`, `group`, `grad_norm`, `mean`, `std`, `depth`, `pathology`.
- Colour-bar trace with `log₁₀(grad_norm)` axis.
- Group-membership legend (inline annotations with colour-coded bullets).

#### Colormap utilities (`heatmap/colormap.py`)
- `ColorScheme` enum: `VIRIDIS` (perceptually uniform intensity) and `RDYLGN` (diverging health signal).
- `grad_norm_to_color()` — log-normalised mapping of a single `grad_norm` to a hex colour; hard-pins vanishing/exploding layers to the extreme colormap stops.
- `pathology_border_color()` — returns the node border hex for each `GradientPathology` value.
- `GROUP_BORDER_COLORS` dict — maps every `LayerGroup` to a distinct border hex (sky-blue = Attention, amber = FFN, lime = LayerNorm, purple = Embedding, red = Head, grey = Other).
- `plotly_colorscale()` — converts internal stop lists to Plotly's `[[pos, color]]` format.

#### Layout engine (`heatmap/layout.py`)
- `LayoutStrategy` enum: `SEQUENTIAL` (vertical stack), `GROUPED` (columns per `LayerGroup`), `SPRING` (NetworkX force-directed).
- `ArchitectureLayout.from_report()` factory — dispatches to the chosen algorithm and returns a list of `NodeLayout` objects with `(x, y)` canvas coordinates + edge list.
- Graceful fallback: `GROUPED` and `SPRING` silently downgrade to `SEQUENTIAL` when NetworkX is absent.

#### Streamlit dashboard tab (`heatmap/dashboard_tab.py`)
- `render_heatmap_tab(report)` — drop-in Streamlit component:
  - Colormap selector, Layout selector, Vanishing-threshold slider, Show-edges toggle — all in a collapsible settings expander.
  - Falls back to `build_static()` (Matplotlib) when Plotly is not installed.
  - Expandable warning panels listing vanishing and exploding layers with remediation hints.

#### `gradient_flow_graph.py` integration shim
- `GradientFlowGraph.build_report()` — runs `GradientAnalyzer` and returns a `GradientReport`.
- `GradientFlowGraph.plot_heatmap()` — one-liner entry point to the Phase-2 renderer.

#### `dashboard.py` upgrade
- Added **Architecture Heatmap** tab alongside the existing Classic Report tab.
- Imports `render_heatmap_tab` from `heatmap.dashboard_tab`.

#### Infrastructure
- `pyproject.toml`: `[dashboard]` extra now includes `plotly>=5.14.0`; `networkx>=3.0` added to `[dev]`; version bumped to `0.6.0`.
- `__init__.py`: exports `GradientHeatmapRenderer`.
- `tests/test_phase2_heatmap.py`: 30 tests across colormap, layout, renderer (Plotly & Matplotlib), and `GradientFlowGraph` shim.

---

## [Unreleased] — feature/phase1-data-pipeline

### Added (Phase 1 — Data Pipeline Foundation)

#### 1-A: Layer Metadata Enrichment (`core.py`, `analyzer.py`)
- Added `LayerGroup` enum with six semantic groups: `ATTENTION`, `FFN`, `LAYER_NORM`, `EMBEDDING`, `HEAD`, `OTHER`.
- Extended `LayerGradientStats` with four new fields:
  - `layer_type` — PyTorch module class name (e.g. `"Linear"`, `"LayerNorm"`).
  - `depth` — 0-based index of the layer in the parameter list.
  - `group` — inferred `LayerGroup` from `TransformerLayerClassifier`.
  - `grad_norm` — L2 norm of per-step mean gradient; primary Heatmap intensity scalar.
- Updated `GradientReport.summary()` to include `group`, `layer_type`, and `grad_norm` in per-layer output.
- Updated `GradientAnalyzer` to instantiate `TransformerLayerClassifier` at init time and inject metadata into every `LayerGradientStats`.

#### 1-B: Snapshot Storage (`pipeline/snapshot.py`)
- New class `GradientSnapshotStore`.

#### 1-C: Transformer Layer Classifier (`pipeline/classifier.py`)
- New class `TransformerLayerClassifier`.
