"""Phase-4 dashboard package.

Orchestrates all tabs and components into the final unified dashboard.

Public surface::

    from gradient_pathology.dashboard import run_dashboard
    run_dashboard()   # launches Streamlit
"""

from gradient_pathology.dashboard.layout import run_dashboard

__all__ = ["run_dashboard"]
