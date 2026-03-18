"""Phase-2 Heatmap visualisation package.

Public surface::

    from gradient_pathology.heatmap import GradientHeatmapRenderer

    renderer = GradientHeatmapRenderer(report)
    fig = renderer.build()          # Plotly Figure
    renderer.show()                 # opens browser
    renderer.save_html("out.html")  # standalone HTML

    # Matplotlib static fallback (no Plotly required)
    mpl_fig = renderer.build_static()
"""

from gradient_pathology.heatmap.renderer import GradientHeatmapRenderer
from gradient_pathology.heatmap.colormap import (
    ColorScheme,
    grad_norm_to_color,
    pathology_border_color,
)
from gradient_pathology.heatmap.layout import ArchitectureLayout

__all__ = [
    "GradientHeatmapRenderer",
    "ColorScheme",
    "grad_norm_to_color",
    "pathology_border_color",
    "ArchitectureLayout",
]
