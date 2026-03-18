# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — feature/phase4-realtime-expert

### Added (Phase 4 — Real-time Monitoring + Expert System)

#### Real-time monitoring bridge (`monitor/`)
- `LiveGradientBridge` — thread-safe in-memory ring-buffer store:
  - `push_step(step, loss, grad_snapshot)` — called from training loop.
  - `push_report(report)` — store latest `GradientReport`.
  - `push_alert(message)` — enqueue a pathology alert string.
  - `pop_alerts()` — consume + clear alert queue.
  - `snapshot()` — consistent point-in-time read (returns copies).
  - `inject_session_state(st_session)` — populate Streamlit session_state.
  - `signal_done()` — mark training as complete.
  - `clear()` — reset all buffers.
  - `max_steps` ring buffer (default 500) — bounded memory.
  - `get_global_bridge()` / `reset_global_bridge()` — module-level singleton.
- `StreamlitCallback` — connects the training loop to `LiveGradientBridge`:
  - `on_batch_end(optimizer, loss, step)` — collect gradients, push to bridge.
  - `on_train_end()` — final report + `signal_done()`.
  - `_check_alerts()` — vanishing/exploding alerts after every step.
  - `_rebuild_report()` — full `GradientReport` every `report_every_n_steps`.
  - `as_hf_callback(**kwargs)` — HuggingFace `TrainerCallback` adapter.

#### ExpertEngine (`expert/engine.py`)
- `ExpertFinding` dataclass: `rule_id`, `severity`, `headline`, `detail`, `recommendations`, `code_snippets`, `affected_layers`, `confidence`.
- `ExpertEngine` — 7 diagnostic rules:
  - `vanishing_layers` — layers below `vanishing_threshold`.
  - `exploding_layers` — layers above `exploding_threshold`.
  - `dead_neurons` — layers with > 90% zero-gradient ratio.
  - `structural_bottleneck` — sharp relative grad-norm drops between consecutive layers.
  - `attention_collapse` — vanishing specifically in Attention group layers.
  - `norm_layer_overload` — LayerNorm carrying 10× above-average gradients.
  - `gradient_imbalance` — high CV (std / |mean| > threshold) instability.
  - `global_health_ok` — positive info finding when all layers are healthy.
- `analyze(report)` — run all rules, return findings sorted by severity.
- `analyze_layer(layer_name, report)` — filter findings for a single layer.
- `top_finding(report)` — return most severe finding.
- `expert/__init__.py`: exports `ExpertSystem`, `ExpertEngine`, `ExpertFinding`.

#### Dashboard components (`dashboard/`)
- `expert_panel.py`:
  - `render_expert_banner(report)` — compact top-of-dashboard notification strip.
  - `render_expert_panel(report)` — full expandable expert diagnostics panel.
  - `render_layer_expert_panel(layer_name, report)` — popup panel for a specific layer.
  - `render_realtime_alerts(alerts)` — live alert feed from `LiveGradientBridge`.
- `realtime_tab.py`:
  - `render_realtime_tab(bridge, static_report)` — live loss curve + grad-norm trend chart (Plotly with Matplotlib fallback), status bar, alert feed, auto-refresh, setup guide.
- `layout.py` — master 4-tab orchestrator:
  - Tabs: 📊 Live Monitor | 🌊 Sankey Flow | 🌡️ Heatmap | 📝 Classic Report.
  - Expert banner above all tabs.
  - Detection threshold sliders in sidebar.
  - Live bridge status in sidebar.
  - `_render_metrics_strip`, `_render_sidebar`, `_render_classic_tab`.
- `dashboard/__init__.py` — exports `run_dashboard`.
- `dashboard.py` (root) — thin backward-compat shim over `dashboard.layout`.

#### Infrastructure
- `pyproject.toml`: version bumped to `0.8.0`.
- `__init__.py`: exports `LiveGradientBridge`, `StreamlitCallback`, `ExpertEngine`, `ExpertFinding`.
- `tests/test_phase4_realtime_expert.py`: 30+ tests covering bridge, callback, expert engine rules, and full integration pipeline.

---

## [Unreleased] — feature/phase3-sankey-diagram
### Added (Phase 3 — Sankey Diagram)
- `SankeyFlowBuilder`, `GradientSankeyRenderer`, `LayerDetailPanel`, `render_sankey_tab`.

## [Unreleased] — feature/phase2-heatmap-visualization
### Added (Phase 2 — Heatmap Visualisation)
- `GradientHeatmapRenderer`, colormap, layout, dashboard tab.

## [Unreleased] — feature/phase1-data-pipeline
### Added (Phase 1 — Data Pipeline Foundation)
- `LayerGroup`, `grad_norm`/`layer_type`/`depth`, `GradientSnapshotStore`, `TransformerLayerClassifier`.
