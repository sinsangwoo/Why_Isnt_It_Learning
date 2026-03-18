"""Phase-4 dashboard entry point — thin shim over the new dashboard package.

Keeps backward compatibility: existing code that does::

    from gradient_pathology.dashboard import run_dashboard
    run_dashboard()

...continues to work unchanged.  The implementation now lives in
:mod:`gradient_pathology.dashboard.layout`.
"""

from gradient_pathology.dashboard.layout import run_dashboard  # noqa: F401
from gradient_pathology.dashboard.layout import _plot_gradient_distribution as plot_gradient_distribution  # noqa: F401

__all__ = ["run_dashboard", "plot_gradient_distribution"]


if __name__ == "__main__":
    run_dashboard()
