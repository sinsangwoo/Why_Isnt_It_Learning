"""Phase-3 Sankey diagram package.

Visualises gradient information flow from the output layer back to the
input, encoding ``grad_norm`` as link width so that **narrow bands reveal
exact information-loss zones**.

Public surface::

    from gradient_pathology.sankey import GradientSankeyRenderer

    renderer = GradientSankeyRenderer(report)
    fig = renderer.build()               # Plotly Figure (go.Sankey)
    renderer.show()                      # opens browser
    renderer.save_html("sankey.html")    # standalone HTML

    # Low-level building blocks
    from gradient_pathology.sankey import SankeyFlowBuilder, LayerDetailPanel

    builder = SankeyFlowBuilder(report)
    flow    = builder.build()            # SankeyFlow dataclass

    panel = LayerDetailPanel(report)
    fig2  = panel.build_plotly(layer_name="transformer.h.0.attn.weight")
"""

from gradient_pathology.sankey.flow import SankeyFlowBuilder, SankeyFlow, SankeyLink
from gradient_pathology.sankey.renderer import GradientSankeyRenderer
from gradient_pathology.sankey.detail_panel import LayerDetailPanel

__all__ = [
    "SankeyFlowBuilder",
    "SankeyFlow",
    "SankeyLink",
    "GradientSankeyRenderer",
    "LayerDetailPanel",
]
