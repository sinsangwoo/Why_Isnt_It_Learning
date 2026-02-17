"""Gradient Pathology: Automated diagnostics for deep learning training instabilities."""

from gradient_pathology.analyzer import GradientAnalyzer
from gradient_pathology.core import GradientReport
from gradient_pathology.finetuning import (
    AdapterMonitor,
    ForgettingDetector,
    LoRARankTracker,
)

__version__ = "0.4.0"
__all__ = [
    "GradientAnalyzer",
    "GradientReport",
    "LoRARankTracker",
    "AdapterMonitor",
    "ForgettingDetector",
]
