# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — feature/phase4-realtime-expert

### Added (Phase 4 — Real-time Monitoring + Expert System Integration)

#### `monitor/bridge.py` — `LiveGradientBridge`
- Thread-safe ring-buffer connecting the training loop to the Streamlit dashboard.
- `push(step, loss, model, layer_norms, extra_alerts)` — O(1) write, acquires lock, never blocks.
- `latest_snapshot()`, `all_snapshots()`, `metrics_series()` — thread-safe read API.
- `drain_alerts()` — returns and clears pending alert strings.
- `from_session_state(key)` — class method that creates or retrieves the bridge from `st.session_state`, enabling persistence across Streamlit reruns.
- Auto-detects vanishing (`< alert_threshold`) and exploding (`> explode_threshold`) gradients on each push.

#### `monitor/callback.py` — `StreamlitCallback`, `HuggingFaceCallbackAdapter`
- `StreamlitCallback.on_batch_end(step, loss)` — drop-in hook for vanilla PyTorch loops.
- `push_every_n_steps` parameter to reduce overhead on large models.
- `HuggingFaceCallbackAdapter` — wraps `StreamlitCallback` as a `transformers.TrainerCallback`; lazily resolves the HF base class so the package installs without HuggingFace.

#### `expert/engine.py` — `ExpertEngine` + `ExpertFinding`
- Layer-aware diagnostic engine operating on `GradientReport` (not just global scalars).
- 7 built-in rules:
  1. `vanishing_layers` — flags `grad_norm < vanishing_threshold` layers with code hints.
  2. `exploding_layers` — flags `grad_norm > exploding_threshold` with gradient-clipping snippet.
  3. `dead_neurons` — flags layers with `zero_ratio > 0.9`.
  4. `bottleneck_cascade` — detects abrupt consecutive-depth norm drops.
  5. `no_layernorm` / `no_layernorm_vanishing` — checks for missing normalisation in deep networks.
  6. `attention_health` — Attention-group specific near-zero gradient check.
  7. `layernorm_explosion` — LayerNorm parameter gradient explosion.
- `register_rule(func)` decorator for user-defined rules.
- `quick_summary(report)` — one-line health string.
- `inject_layer_norms(model)` utility — adds `LayerNorm` after each `Linear` in a `Sequential`.
- `ExpertFinding` dataclass: `rule_id`, `severity`, `title`, `detail`, `layers`, `code_hint`, `confidence`, `emoji`, `severity_rank`.

#### `dashboard/expert_panel.py`
- `render_expert_banner(report)` — compact coloured status banner (green/orange/red) with expand-on-click.
- `render_expert_popup(report, findings)` — full findings panel grouped by severity with detail + affected layers + copy-paste code hints.
- `render_layer_expert_panel(layer_name, report)` — layer-filtered findings for the Sankey deep-dive section.

#### `dashboard/realtime_tab.py`
- `render_realtime_tab(bridge, report)` — 4-section Live Monitor tab:
  - Status row (step, loss, grad mean, alert count).
  - Loss curve (Plotly line + fill; Matplotlib fallback).
  - Grad-norm trend with vanishing/exploding threshold bands.
  - Per-layer bar chart (top-N layers, colour-coded by health).
  - Alert feed (most recent 10 alerts).
  - Refresh button + buffer status line.

#### `dashboard/layout.py` — `run_dashboard()`
- 4-tab layout: **📡 Live Monitor** | 🌊 Sankey Flow | 🌡️ Architecture Heatmap | 📊 Classic Report.
- Global Expert System banner above all tabs (toggled from sidebar).
- Sidebar Expert System toggle checkbox.
- Seeds the `LiveGradientBridge` from the post-analysis report so Live Monitor shows initial data immediately.
- Stores bridge in `st.session_state["_gp_bridge"]` for cross-rerun persistence.

#### `dashboard.py` (backward-compat shim)
- Thin shim: imports `run_dashboard` from the new package so `streamlit run dashboard.py` still works.

#### Infrastructure
- `pyproject.toml`: version bumped to `0.8.0`; `streamlit>=1.28.0` in `[dashboard]`.
- `__init__.py`: exports `ExpertEngine`, `ExpertFinding`, `LiveGradientBridge`, `StreamlitCallback`, `HuggingFaceCallbackAdapter`.
- `tests/test_phase4_realtime_expert.py`: 30 tests covering bridge, callback, engine rules, integration.

---

## [Unreleased] — feature/phase3-sankey-diagram
### Added (Phase 3 — Sankey Diagram)
- `GradientSankeyRenderer`, `SankeyFlowBuilder`, `LayerDetailPanel`.

## [Unreleased] — feature/phase2-heatmap-visualization
### Added (Phase 2 — Heatmap Visualisation)
- `GradientHeatmapRenderer`.

## [Unreleased] — feature/phase1-data-pipeline
### Added (Phase 1 — Data Pipeline Foundation)
- `LayerGroup`, `GradientSnapshotStore`, `TransformerLayerClassifier`.
