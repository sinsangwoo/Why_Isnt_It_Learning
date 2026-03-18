"""Backward-compatible shim: ``streamlit run dashboard.py``.

Phase-4 re-routes all rendering through the new
:mod:`gradient_pathology.dashboard` package.  This file is kept so
existing ``streamlit run`` commands continue to work unchanged.
"""

from gradient_pathology.dashboard.layout import run_dashboard  # noqa: F401

if __name__ == "__main__":
    run_dashboard()
