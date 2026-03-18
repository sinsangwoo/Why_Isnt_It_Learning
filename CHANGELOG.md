# Changelog

All notable changes to this project will be documented in this file.

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
- New class `GradientSnapshotStore`:
  - `record_from_stats(step, layer_stats)` — primary entry point from `GradientReport`.
  - `record_from_monitor(step, monitor_history_entry)` — lightweight path from `GradientMonitor`.
  - `flush()` — writes buffered rows to disk; auto-flushed when buffer reaches `buffer_size`.
  - `load()` — returns a `pandas.DataFrame` (requires `[storage]` extra).
  - `load_json_raw()` — zero-dependency JSON loader returning `list[dict]`.
  - Supports `json` (default) and `parquet` output formats.
  - Schema: `step`, `layer_name`, `layer_type`, `depth`, `group`, `grad_norm`, `mean`, `std`, `min`, `max`, `zero_ratio`, `pathology`.

#### 1-C: Transformer Layer Classifier (`pipeline/classifier.py`)
- New class `TransformerLayerClassifier`:
  - `build_param_metadata()` — returns `{param_name: (layer_type, LayerGroup)}` for every parameter.
  - `classify_param(param_name)` — classify a single parameter name.
  - `group_summary()` — group-to-param-list mapping for quick sanity-checking.
- Name-based heuristic covers GPT-2, LLaMA, Mistral, BERT-style, and hand-rolled Transformer variants.
- `pipeline/__init__.py` exports `TransformerLayerClassifier` and `GradientSnapshotStore`.

#### Infrastructure
- `pyproject.toml`: added `[storage]` optional extra (`pandas>=1.5.0`, `pyarrow>=12.0.0`); bumped version to `0.5.0`.
- `__init__.py`: exports `LayerGroup`, `TransformerLayerClassifier`, `GradientSnapshotStore`.
- `tests/test_phase1_pipeline.py`: 25 new tests covering all three sub-tasks.
