"""Phase-4 dashboard package — 4-tab Streamlit orchestrator.

All rendering logic lives in sub-modules; ``dashboard.py`` at the package
root is a thin backward-compatible shim that calls :func:`run_dashboard`.
"""

from gradient_pathology.dashboard.layout import run_dashboard

__all__ = ["run_dashboard"]
