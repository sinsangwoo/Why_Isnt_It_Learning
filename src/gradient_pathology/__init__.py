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

__version__ = "0.5.0"
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
]
