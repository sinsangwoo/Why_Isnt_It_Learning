# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed — **Breaking** (Phase 0 refactor)

- `GradientAnalyzer.diagnose()` now accepts an optional `dataloader` parameter.
  When provided, gradients are computed on **real training data** rather than
  synthetic random tensors.  This is the recommended usage going forward.
- `GradientReport` gains a `data_source` field (`"dataloader"` | `"synthetic"`)
  so callers can tell at a glance which mode was used.
- The summary string now includes a visible warning when `data_source == "synthetic"`.

### Backward Compatibility

Existing code that calls `diagnose(num_steps=..., input_shape=...)` continues to
work unchanged.  The new `dataloader` parameter is optional and defaults to
`None`, which triggers the legacy synthetic path.

### Migration

```python
# Before (still works, but limited diagnostic value)
report = analyzer.diagnose(num_steps=100, input_shape=(64,))

# After (recommended — use your actual training loader)
report = analyzer.diagnose(dataloader=train_loader, loss_fn=criterion)
```

---

## [0.3.0] — 2026-02-11

### Added
- Phase 5.2: GPU cost optimisation (CostCalculator, TrainingOptimizer).
- Complete README overhaul with professional structure.

## [0.2.0] — 2026-02-09

### Added
- Phase 5.1: LLM-specific features (TransformerDiagnostics, FSDPAnalyzer,
  QuantizationAnalyzer).
- Phase 4.2: Ecosystem integrations (HuggingFace, PyTorch Lightning, Ray Tune).

## [0.1.0] — 2026-02-05

### Added
- Phase 4.1: Sphinx documentation, tutorials, case studies.
- Phase 3.4: Docker, MLflow experiment tracking, benchmark suite.
- Phase 3.3: Rule-based ExpertSystem for automated diagnosis.
- Phase 3.2: EffectiveRankAnalyzer, LayerLRFinder, GradientFlowGraph.
- Phase 3.1: AttentionMonitor, TransformerHooks.
- Phase 3.0: HessianAnalyzer, LRFinder, TransformerDiagnostics.
- Phase 2: Real-time monitoring callbacks.
- Phase 1: Core gradient analysis engine and report structures.
