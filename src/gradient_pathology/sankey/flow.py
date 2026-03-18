"""Phase-3 sub-module: grad_norm → Sankey source/target/value transformer.

The core design problem
-----------------------
A Sankey diagram requires three parallel arrays:

* ``source`` — index of the upstream node in the node list
* ``target`` — index of the downstream node in the node list
* ``value``  — link width (the "amount" flowing through)

For gradient flow we want *narrow links to coincide with vanishing zones*,
so ``value`` must be a monotone function of ``grad_norm`` where higher
``grad_norm`` = wider link.  Because ``grad_norm`` spans many decades (1e-9
to 1e2 is common), we work in **log-space** and then re-scale to a
user-visible range.

Five flow strategies are supported (see :class:`FlowStrategy`).

Node ordering
-------------
Nodes are listed in *reverse depth order* (output layer first, input last)
so the Sankey reads left→right in the direction of the backward pass, which
is how engineers think about gradient flow.

Loss-zone classification
------------------------
Each link is tagged with a :class:`FlowZone` that drives its colour in the
renderer:

* ``HEALTHY``   — both endpoints have healthy gradients
* ``VANISHING`` — the downstream (earlier) node has a vanishing gradient
* ``EXPLODING`` — either endpoint is exploding
* ``BOTTLENECK``— relative drop > *bottleneck_drop_ratio* compared with peak
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from gradient_pathology.core import GradientPathology, GradientReport, LayerGroup


# ---------------------------------------------------------------------------
# Public enums
# ---------------------------------------------------------------------------

class FlowStrategy(Enum):
    """How ``grad_norm`` values are converted to link widths.

    Attributes
    ----------
    RAW
        Use raw ``grad_norm`` values directly (only sensible when norms are
        already on a similar scale).
    LOG
        Map values to log₁₀ space, then linearly re-scale to
        [*min_width*, *max_width*].  Best general-purpose choice.
    NORMALISED
        Min-max normalise in linear space to [*min_width*, *max_width*].
    RELATIVE
        Each link value is the ratio ``grad_norm[i] / max(grad_norm)``
        multiplied by *max_width*.  Emphasises relative, not absolute, loss.
    SQRT
        Square-root of ``grad_norm``, then re-scaled.  Compresses the
        dynamic range less aggressively than LOG.
    """

    RAW        = "raw"
    LOG        = "log"
    NORMALISED = "normalised"
    RELATIVE   = "relative"
    SQRT       = "sqrt"


class FlowZone(Enum):
    """Semantic zone of a Sankey link — drives link colour."""

    HEALTHY     = "healthy"     # both nodes healthy
    VANISHING   = "vanishing"   # downstream node vanishing
    EXPLODING   = "exploding"   # either node exploding
    BOTTLENECK  = "bottleneck"  # relative drop exceeds threshold
    DEAD        = "dead"        # dead neurons detected


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SankeyLink:
    """A single directed link in the Sankey diagram.

    Attributes
    ----------
    source_idx:
        Index into the *nodes* list of the upstream node.
    target_idx:
        Index into the *nodes* list of the downstream node.
    value:
        Link width (post-strategy transformation).
    raw_source_norm:
        Original ``grad_norm`` of the source layer (for tooltip display).
    raw_target_norm:
        Original ``grad_norm`` of the target layer.
    zone:
        Semantic health zone that drives the link's colour.
    loss_fraction:
        Fraction of flow *lost* at this link relative to the peak link in the
        diagram.  ``1.0`` means all flow is lost; ``0.0`` means no loss.
    """

    source_idx:       int
    target_idx:       int
    value:            float
    raw_source_norm:  float
    raw_target_norm:  float
    zone:             FlowZone = FlowZone.HEALTHY
    loss_fraction:    float    = 0.0


@dataclass
class SankeyFlow:
    """Complete Sankey flow data ready for rendering.

    Attributes
    ----------
    node_labels:
        Display labels for every node, in source→target order (output first).
    node_layer_names:
        Fully-qualified parameter names, parallel to *node_labels*.
    node_groups:
        :class:`~gradient_pathology.core.LayerGroup` for each node.
    node_grad_norms:
        Raw ``grad_norm`` for each node.
    node_pathologies:
        :class:`~gradient_pathology.core.GradientPathology` for each node.
    links:
        List of :class:`SankeyLink` objects.
    strategy:
        The :class:`FlowStrategy` used to compute link widths.
    vanishing_threshold:
        Threshold that was used for vanishing detection.
    bottleneck_drop_ratio:
        Relative drop threshold used for bottleneck detection.
    """

    node_labels:          List[str]
    node_layer_names:     List[str]
    node_groups:          List[LayerGroup]
    node_grad_norms:      List[float]
    node_pathologies:     List[GradientPathology]
    links:                List[SankeyLink]
    strategy:             FlowStrategy       = FlowStrategy.LOG
    vanishing_threshold:  float              = 1e-7
    bottleneck_drop_ratio: float             = 0.5

    # Derived convenience properties ------------------------------------------

    @property
    def n_nodes(self) -> int:
        return len(self.node_labels)

    @property
    def vanishing_links(self) -> List[SankeyLink]:
        return [lk for lk in self.links if lk.zone == FlowZone.VANISHING]

    @property
    def bottleneck_links(self) -> List[SankeyLink]:
        return [lk for lk in self.links if lk.zone == FlowZone.BOTTLENECK]

    @property
    def max_loss_fraction(self) -> float:
        if not self.links:
            return 0.0
        return max(lk.loss_fraction for lk in self.links)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class SankeyFlowBuilder:
    """Transform a :class:`~gradient_pathology.core.GradientReport` into a
    :class:`SankeyFlow` ready for :class:`~gradient_pathology.sankey.renderer.GradientSankeyRenderer`.

    Parameters
    ----------
    report:
        The gradient analysis report (Phase-1 fields ``grad_norm``, ``group``,
        ``depth`` are required for best results; falls back gracefully when
        absent).
    strategy:
        How ``grad_norm`` values are mapped to link widths.
    vanishing_threshold:
        Layers with ``grad_norm`` below this value are tagged
        :attr:`FlowZone.VANISHING`.
    exploding_threshold:
        Layers above this value are tagged :attr:`FlowZone.EXPLODING`.
    bottleneck_drop_ratio:
        A link is tagged :attr:`FlowZone.BOTTLENECK` when its value drops
        more than ``bottleneck_drop_ratio * peak_value`` compared with the
        previous link.
    min_width:
        Minimum rendered link width (prevents zero-width invisible links).
    max_width:
        Maximum rendered link width.
    group_by_layer:
        When ``True`` (default), consecutive parameters that belong to the
        same named module (e.g. ``attn.weight`` and ``attn.bias``) are merged
        into a single node whose ``grad_norm`` is the L2-combination of the
        individual norms.  This dramatically reduces node count for large
        models.

    Examples
    --------
    ::

        from gradient_pathology.sankey import SankeyFlowBuilder

        builder = SankeyFlowBuilder(report, strategy=FlowStrategy.LOG)
        flow    = builder.build()
        print(f"Nodes: {flow.n_nodes}, Bottlenecks: {len(flow.bottleneck_links)}")
    """

    def __init__(
        self,
        report: GradientReport,
        strategy: FlowStrategy             = FlowStrategy.LOG,
        vanishing_threshold: float         = 1e-7,
        exploding_threshold: float         = 1e3,
        bottleneck_drop_ratio: float       = 0.5,
        min_width: float                   = 1.0,
        max_width: float                   = 40.0,
        group_by_layer: bool               = True,
    ) -> None:
        self.report               = report
        self.strategy             = strategy
        self.vanishing_threshold  = vanishing_threshold
        self.exploding_threshold  = exploding_threshold
        self.bottleneck_drop_ratio = bottleneck_drop_ratio
        self.min_width            = min_width
        self.max_width            = max_width
        self.group_by_layer       = group_by_layer

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(self) -> SankeyFlow:
        """Build and return the :class:`SankeyFlow`."""
        if not self.report.layer_stats:
            return SankeyFlow(
                node_labels=[], node_layer_names=[], node_groups=[],
                node_grad_norms=[], node_pathologies=[], links=[],
                strategy=self.strategy,
                vanishing_threshold=self.vanishing_threshold,
                bottleneck_drop_ratio=self.bottleneck_drop_ratio,
            )

        # 1. Sort layers by depth (shallowest = closest to input first).
        sorted_stats = sorted(self.report.layer_stats, key=lambda s: s.depth)

        # 2. Optionally merge parameters that share a module path.
        if self.group_by_layer:
            sorted_stats = _merge_by_module(sorted_stats)

        # 3. Reverse so the diagram flows output→input (left→right = backprop).
        sorted_stats = list(reversed(sorted_stats))

        # 4. Extract raw norms and build node metadata.
        raw_norms = [_safe_norm(s) for s in sorted_stats]
        node_labels         = [_short_label(s.layer_name) for s in sorted_stats]
        node_layer_names    = [s.layer_name                for s in sorted_stats]
        node_groups         = [s.group                     for s in sorted_stats]
        node_pathologies    = [s.diagnose()                for s in sorted_stats]

        # 5. Compute scaled link values using the chosen strategy.
        scaled = self._scale(raw_norms)

        # 6. Build links between consecutive nodes.
        links = self._build_links(raw_norms, scaled)

        return SankeyFlow(
            node_labels=node_labels,
            node_layer_names=node_layer_names,
            node_groups=node_groups,
            node_grad_norms=raw_norms,
            node_pathologies=node_pathologies,
            links=links,
            strategy=self.strategy,
            vanishing_threshold=self.vanishing_threshold,
            bottleneck_drop_ratio=self.bottleneck_drop_ratio,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scale(self, raw_norms: List[float]) -> List[float]:
        """Map raw_norms → scaled widths using the chosen strategy."""
        arr = np.array(raw_norms, dtype=float)
        mn, mx = self.min_width, self.max_width

        if self.strategy == FlowStrategy.RAW:
            vals = arr

        elif self.strategy == FlowStrategy.LOG:
            log_arr = np.log10(arr + 1e-12)
            lo, hi  = log_arr.min(), log_arr.max()
            vals    = (log_arr - lo) / (hi - lo + 1e-12) * (mx - mn) + mn

        elif self.strategy == FlowStrategy.NORMALISED:
            lo, hi = arr.min(), arr.max()
            vals   = (arr - lo) / (hi - lo + 1e-12) * (mx - mn) + mn

        elif self.strategy == FlowStrategy.RELATIVE:
            peak = arr.max() if arr.max() > 0 else 1.0
            vals = arr / peak * mx
            vals = np.clip(vals, mn, mx)

        elif self.strategy == FlowStrategy.SQRT:
            sqrt_arr = np.sqrt(arr + 1e-12)
            lo, hi   = sqrt_arr.min(), sqrt_arr.max()
            vals     = (sqrt_arr - lo) / (hi - lo + 1e-12) * (mx - mn) + mn

        else:
            vals = arr

        return list(np.clip(vals, mn, mx).astype(float))

    def _build_links(
        self,
        raw_norms: List[float],
        scaled:    List[float],
    ) -> List[SankeyLink]:
        """Build one link per consecutive node pair and classify each link."""
        links: List[SankeyLink] = []
        n = len(raw_norms)
        peak_val = max(scaled) if scaled else 1.0

        for i in range(n - 1):
            # In the reversed ordering:
            # node i   = shallower (closer to output)
            # node i+1 = deeper   (closer to input)
            # Link direction: i → i+1  (backward-pass direction)
            src_norm = raw_norms[i]
            dst_norm = raw_norms[i + 1]
            val      = scaled[i + 1]   # link width = capacity of the receiving end

            zone          = self._classify_zone(src_norm, dst_norm, val, peak_val)
            loss_fraction = max(0.0, (peak_val - val) / (peak_val + 1e-12))

            links.append(SankeyLink(
                source_idx=i,
                target_idx=i + 1,
                value=max(val, self.min_width),
                raw_source_norm=src_norm,
                raw_target_norm=dst_norm,
                zone=zone,
                loss_fraction=loss_fraction,
            ))

        return links

    def _classify_zone(
        self,
        src_norm:  float,
        dst_norm:  float,
        val:       float,
        peak_val:  float,
    ) -> FlowZone:
        """Assign a :class:`FlowZone` to a link."""
        if src_norm > self.exploding_threshold or dst_norm > self.exploding_threshold:
            return FlowZone.EXPLODING
        if dst_norm < self.vanishing_threshold:
            return FlowZone.VANISHING
        drop_ratio = (peak_val - val) / (peak_val + 1e-12)
        if drop_ratio > self.bottleneck_drop_ratio:
            return FlowZone.BOTTLENECK
        return FlowZone.HEALTHY


# ---------------------------------------------------------------------------
# Module-merging helper
# ---------------------------------------------------------------------------

def _merge_by_module(
    stats_list: List,
) -> List:
    """Merge weight+bias pairs (and similar) that share the same module path.

    E.g.  ``transformer.h.0.attn.weight``  and
          ``transformer.h.0.attn.bias``
    become a single synthetic entry with ``grad_norm = sqrt(w² + b²)``.

    The merge key is everything *before* the last dot-segment
    (``transformer.h.0.attn`` in the example).
    """
    from collections import OrderedDict
    import copy

    buckets: OrderedDict = OrderedDict()
    for s in stats_list:
        parts = s.layer_name.split(".")
        key   = ".".join(parts[:-1]) if len(parts) > 1 else s.layer_name
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(s)

    merged = []
    for key, group in buckets.items():
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Combine: use L2-combination of individual grad_norms.
            base = copy.copy(group[0])
            norms_sq = sum(_safe_norm(s) ** 2 for s in group)
            combined_norm = float(np.sqrt(norms_sq))
            # We can't mutate a frozen dataclass so we rebuild:
            from dataclasses import replace
            try:
                base = replace(base,
                               layer_name=key,
                               grad_norm=combined_norm)
            except TypeError:
                # Fallback if replace isn't supported
                base.layer_name = key
                base.grad_norm  = combined_norm
            merged.append(base)

    return merged


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _safe_norm(stats: object) -> float:
    """Return grad_norm if positive, else fall back to abs(mean)."""
    gn = getattr(stats, "grad_norm", None)
    if gn is not None and float(gn) > 0:
        return float(gn)
    mean = getattr(stats, "mean", 0.0)
    return float(abs(mean)) + 1e-12


def _short_label(layer_name: str, max_len: int = 24) -> str:
    """Return a shortened display label."""
    parts = layer_name.split(".")
    # Use last two segments, e.g. "attn.weight" or "lm_head"
    label = ".".join(parts[-2:]) if len(parts) >= 2 else layer_name
    if len(label) > max_len:
        label = "\u2026" + label[-(max_len - 1):]
    return label
