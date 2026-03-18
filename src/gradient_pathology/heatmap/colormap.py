"""Colormap utilities for the Phase-2 Heatmap renderer.

Two mapping strategies are provided:

1. **Viridis** — perceptually uniform, sequential.  Best for showing raw
   ``grad_norm`` intensity: dark-purple = near-zero, yellow = high.
2. **RdYlGn** — diverging, intuitive health signal.  Red = bad
   (vanishing/exploding), green = healthy.

Additionally, :data:`GROUP_BORDER_COLORS` maps each
:class:`~gradient_pathology.core.LayerGroup` to a distinct hex colour used as
the node border ring, so group membership is visible at a glance.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Tuple

import numpy as np

from gradient_pathology.core import GradientPathology, LayerGroup

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Hex colour for each LayerGroup border ring.
GROUP_BORDER_COLORS: dict[LayerGroup, str] = {
    LayerGroup.ATTENTION:   "#4C9BE8",   # sky-blue
    LayerGroup.FFN:         "#F5A623",   # amber
    LayerGroup.LAYER_NORM:  "#7ED321",   # lime-green
    LayerGroup.EMBEDDING:   "#9B59B6",   # purple
    LayerGroup.HEAD:        "#E74C3C",   # red-orange
    LayerGroup.OTHER:       "#95A5A6",   # muted grey
}

#: Warning overlay colour for vanishing layers.
VANISHING_WARN_COLOR = "rgba(231, 76, 60, 0.18)"
#: Warning overlay colour for exploding layers.
EXPLODING_WARN_COLOR = "rgba(230, 126, 34, 0.18)"


class ColorScheme(Enum):
    """Available colormaps for ``grad_norm`` intensity mapping."""

    VIRIDIS = "Viridis"       # perceptually uniform, intensity
    RDYLGN  = "RdYlGn"       # diverging, health-oriented
    PLASMA  = "Plasma"        # high-contrast alternative


# ---------------------------------------------------------------------------
# Core mapping helpers
# ---------------------------------------------------------------------------

def _viridis_stops() -> List[Tuple[float, str]]:
    """Return a 10-stop Viridis colorscale as (position, hex) pairs."""
    # Manually encoded from matplotlib Viridis so we have no extra dep.
    return [
        (0.00, "#440154"),
        (0.11, "#482878"),
        (0.22, "#3E4A89"),
        (0.33, "#31688E"),
        (0.44, "#26828E"),
        (0.56, "#1F9E89"),
        (0.67, "#35B779"),
        (0.78, "#6ECE58"),
        (0.89, "#B5DE2B"),
        (1.00, "#FDE725"),
    ]


def _rdylgn_stops() -> List[Tuple[float, str]]:
    """Return a 10-stop RdYlGn colorscale."""
    return [
        (0.00, "#A50026"),
        (0.11, "#D73027"),
        (0.22, "#F46D43"),
        (0.33, "#FDAE61"),
        (0.44, "#FEE08B"),
        (0.56, "#D9EF8B"),
        (0.67, "#A6D96A"),
        (0.78, "#66BD63"),
        (0.89, "#1A9850"),
        (1.00, "#006837"),
    ]


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _interpolate_colorscale(
    stops: List[Tuple[float, str]],
    t: float,
) -> str:
    """Linearly interpolate *stops* at normalised position *t* ∈ [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            r0, g0, b0 = _hex_to_rgb(c0)
            r1, g1, b1 = _hex_to_rgb(c1)
            r = int(r0 + alpha * (r1 - r0))
            g = int(g0 + alpha * (g1 - g0))
            b = int(b0 + alpha * (b1 - b0))
            return f"#{r:02X}{g:02X}{b:02X}"
    return stops[-1][1]


def grad_norm_to_color(
    grad_norm: float,
    all_norms: List[float],
    scheme: ColorScheme = ColorScheme.VIRIDIS,
    vanishing_threshold: float = 1e-7,
    exploding_threshold: float = 1e3,
) -> str:
    """Map a single ``grad_norm`` value to a hex colour string.

    The mapping is normalised across *all_norms* so the full colormap range is
    always used regardless of the absolute magnitude of the gradients.

    Vanishing layers (``grad_norm < vanishing_threshold``) are **always**
    mapped to the darkest stop of the chosen colormap, regardless of the
    relative normalisation, to make them visually stand out.

    Parameters
    ----------
    grad_norm:
        L2 norm of the gradient for this layer.
    all_norms:
        All ``grad_norm`` values in the report (used for min-max normalisation).
    scheme:
        Which colormap to use.
    vanishing_threshold:
        Layers below this threshold are always mapped to the lowest color stop.
    exploding_threshold:
        Layers above this threshold are always mapped to the highest color stop.

    Returns
    -------
    str
        Hex colour string, e.g. ``"#FDE725"``.
    """
    stops = _viridis_stops() if scheme == ColorScheme.VIRIDIS else (
        _rdylgn_stops() if scheme == ColorScheme.RDYLGN else _viridis_stops()
    )

    # Hard-pin pathological layers to extreme ends of the colormap.
    if grad_norm < vanishing_threshold:
        return stops[0][1]   # darkest
    if grad_norm > exploding_threshold:
        return stops[-1][1]  # brightest

    # Log-space normalisation for better perceptual spread.
    log_norms = np.log10(np.array(all_norms, dtype=float) + 1e-12)
    log_val   = np.log10(float(grad_norm) + 1e-12)
    lo, hi = log_norms.min(), log_norms.max()

    t = (log_val - lo) / (hi - lo) if hi > lo else 0.5
    return _interpolate_colorscale(stops, float(t))


def pathology_border_color(pathology: GradientPathology) -> str:
    """Return the border colour for a node given its :class:`GradientPathology`.

    The border colour encodes the *health* status independent of the fill
    colour (which encodes ``grad_norm`` intensity).

    Returns
    -------
    str
        CSS/Plotly hex colour.
    """
    return {
        GradientPathology.HEALTHY:      "#2ECC71",  # green
        GradientPathology.VANISHING:    "#E74C3C",  # red
        GradientPathology.EXPLODING:    "#E67E22",  # orange
        GradientPathology.DEAD_NEURONS: "#8E44AD",  # purple
        GradientPathology.UNSTABLE:     "#F39C12",  # yellow
    }.get(pathology, "#95A5A6")


def plotly_colorscale(scheme: ColorScheme) -> List[List]:
    """Return a Plotly-format colorscale list for *scheme*.

    Plotly expects ``[[position, color], ...]`` where position ∈ [0, 1].

    Returns
    -------
    list[list]
        e.g. ``[[0.0, "#440154"], [0.11, "#482878"], ...]``
    """
    stops = _viridis_stops() if scheme == ColorScheme.VIRIDIS else _rdylgn_stops()
    return [[pos, color] for pos, color in stops]
