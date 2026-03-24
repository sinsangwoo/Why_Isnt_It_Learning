"""Gradient Pathology: Automated diagnostics for deep learning training instabilities."""

from gradient_pathology.analyzer import GradientAnalyzer
from gradient_pathology.core import GradientReport, LayerGroup
from gradient_pathology.finetuning import (
    AdapterMonitor,
    ForgettingDetector,
    LoRARankTracker,
)
from gradient_pathology.pipeline import (
    GradientSnapshotStore,
    TransformerLayerClassifier,
)
from gradient_pathology.heatmap import GradientHeatmapRenderer
from gradient_pathology.sankey import GradientSankeyRenderer
from gradient_pathology.expert import ExpertEngine, ExpertFinding
from gradient_pathology.monitor import (
    LiveGradientBridge,
    StreamlitCallback,
    HuggingFaceCallbackAdapter,
)
# Phase 2 — non-invasive watch API
from gradient_pathology.watch import ModelWatcher, watch

__version__ = "1.0.0"
__all__ = [
    "GradientAnalyzer",
    "GradientReport",
    "LayerGroup",
    "LoRARankTracker",
    "AdapterMonitor",
    "ForgettingDetector",
    # Phase-1 pipeline
    "TransformerLayerClassifier",
    "GradientSnapshotStore",
    # Phase-2 heatmap
    "GradientHeatmapRenderer",
    # Phase-3 sankey
    "GradientSankeyRenderer",
    # Phase-4 expert + monitor
    "ExpertEngine",
    "ExpertFinding",
    "LiveGradientBridge",
    "StreamlitCallback",
    "HuggingFaceCallbackAdapter",
    # Phase-5 (this PR) — non-invasive watch
    "ModelWatcher",
    "watch",
]
