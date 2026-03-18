"""Phase-4 ExpertEngine — deep, layer-aware diagnostic system.

The original :class:`~gradient_pathology.expert.rules.ExpertSystem` worked
at the *model level* (architecture shape, global gradient stats).  The
``ExpertEngine`` introduced here works at the *layer level*: it takes the
full :class:`~gradient_pathology.core.GradientReport` produced by
:class:`~gradient_pathology.analyzer.GradientAnalyzer` and returns structured
:class:`ExpertFinding` objects that the dashboard can render as interactive
popup panels.

Design
------
Each ``ExpertFinding`` bundles:

* A **severity** badge (``critical`` / ``warning`` / ``info``).
* A short **headline** (shown in the notification banner).
* A long-form **detail** block (shown in the expanded popup).
* A list of **code snippets** — copy-pasteable PyTorch fixes.
* The **affected_layers** list for cross-linking with the Heatmap/Sankey.
* A **confidence** float and a **rule_id** for filtering.

Firing order
------------
Rules are fired in priority order (``critical`` first).  Each rule is a
method prefixed ``_rule_``.  New rules can be added simply by adding
more ``_rule_`` methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from gradient_pathology.core import (
    GradientPathology,
    GradientReport,
    LayerGradientStats,
    LayerGroup,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExpertFinding:
    """A single structured diagnostic finding from the expert engine.

    Attributes
    ----------
    rule_id:
        Short snake_case identifier for this rule (e.g. ``vanishing_deep``).
    severity:
        ``'critical'``, ``'warning'``, or ``'info'``.
    headline:
        One-line summary shown in notification banners.
    detail:
        Multi-line Markdown explanation shown in the expanded popup.
    recommendations:
        Ordered list of actionable fix strings.
    code_snippets:
        Dict mapping a label to a PyTorch code string.  E.g.
        ``{"Add gradient clipping": "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)"}``.
    affected_layers:
        Layer names that triggered this finding.
    confidence:
        Float 0-1 indicating rule confidence.
    """

    rule_id:         str
    severity:        str   # 'critical' | 'warning' | 'info'
    headline:        str
    detail:          str
    recommendations: List[str]  = field(default_factory=list)
    code_snippets:   Dict[str, str] = field(default_factory=dict)
    affected_layers: List[str]  = field(default_factory=list)
    confidence:      float      = 1.0

    # Convenience -------------------------------------------------------

    @property
    def severity_emoji(self) -> str:
        return {
            "critical": "🚨",
            "warning":  "⚠️",
            "info":     "ℹ️",
        }.get(self.severity, "🔵")

    @property
    def severity_color(self) -> str:
        return {
            "critical": "#E74C3C",
            "warning":  "#F39C12",
            "info":     "#3498DB",
        }.get(self.severity, "#95A5A6")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ExpertEngine:
    """Layer-aware expert system that converts a :class:`GradientReport`
    into a list of :class:`ExpertFinding` objects.

    Parameters
    ----------
    vanishing_threshold:
        ``grad_norm`` below this value → vanishing diagnosis.
    exploding_threshold:
        ``grad_norm`` above this value → exploding diagnosis.
    bottleneck_drop_ratio:
        Relative grad_norm drop between consecutive layers that qualifies
        as a structural bottleneck.
    unstable_cv_threshold:
        Coefficient of variation (``std / |mean|``) threshold above which
        a layer is considered unstable.

    Examples
    --------
    ::

        engine   = ExpertEngine()
        findings = engine.analyze(report)
        for f in findings:
            print(f.severity_emoji, f.headline)
    """

    def __init__(
        self,
        vanishing_threshold:  float = 1e-7,
        exploding_threshold:  float = 1e3,
        bottleneck_drop_ratio: float = 0.6,
        unstable_cv_threshold: float = 30.0,
    ) -> None:
        self.vanishing_threshold   = vanishing_threshold
        self.exploding_threshold   = exploding_threshold
        self.bottleneck_drop_ratio = bottleneck_drop_ratio
        self.unstable_cv_threshold = unstable_cv_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, report: GradientReport) -> List[ExpertFinding]:
        """Run all diagnostic rules and return findings sorted by severity.

        Parameters
        ----------
        report:
            Full gradient analysis report.

        Returns
        -------
        list[ExpertFinding]
            Sorted: critical first, then warning, then info.
        """
        if not report.layer_stats:
            return []

        findings: List[ExpertFinding] = []

        findings += self._rule_vanishing_layers(report)
        findings += self._rule_exploding_layers(report)
        findings += self._rule_dead_neurons(report)
        findings += self._rule_structural_bottleneck(report)
        findings += self._rule_attention_collapse(report)
        findings += self._rule_norm_layer_bypass(report)
        findings += self._rule_gradient_imbalance(report)
        findings += self._rule_global_health(report)

        # Sort: critical < warning < info, break ties by confidence desc.
        order = {"critical": 0, "warning": 1, "info": 2}
        findings.sort(key=lambda f: (order.get(f.severity, 3), -f.confidence))
        return findings

    def analyze_layer(
        self,
        layer_name: str,
        report: GradientReport,
    ) -> List[ExpertFinding]:
        """Return findings that directly implicate *layer_name*."""
        return [
            f for f in self.analyze(report)
            if layer_name in f.affected_layers
        ]

    def top_finding(self, report: GradientReport) -> Optional[ExpertFinding]:
        """Return the single most severe finding, or ``None``."""
        findings = self.analyze(report)
        return findings[0] if findings else None

    # ------------------------------------------------------------------
    # Diagnostic rules
    # ------------------------------------------------------------------

    def _rule_vanishing_layers(
        self, report: GradientReport
    ) -> List[ExpertFinding]:
        """Detect layers with grad_norm below the vanishing threshold."""
        affected = [
            s.layer_name
            for s in report.layer_stats
            if _safe_norm(s) < self.vanishing_threshold
        ]
        if not affected:
            return []

        pct = len(affected) / len(report.layer_stats) * 100
        return [ExpertFinding(
            rule_id="vanishing_layers",
            severity="critical",
            headline=(
                f"🔴 Vanishing gradients in {len(affected)} / {len(report.layer_stats)} layers "
                f"({pct:.0f}%)"
            ),
            detail=(
                "Gradients are effectively zero in the listed layers, meaning those layers "
                "receive **no learning signal**. Common causes: saturating activations "
                "(Sigmoid / Tanh in deep stacks), missing normalisation, or a learning rate "
                "that is too small for the initialisation scale."
            ),
            recommendations=[
                "Replace Sigmoid/Tanh activations with ReLU, GELU, or SiLU.",
                "Add LayerNorm (or RMSNorm) before or after each affected layer.",
                "Apply He/Kaiming initialisation for ReLU layers.",
                "Add residual (skip) connections to bypass deep stacks.",
                "Consider a higher base learning rate for earlier layers.",
            ],
            code_snippets={
                "Replace activation": "nn.GELU()  # instead of nn.Sigmoid()",
                "Add LayerNorm": "nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())",
                "He init": "nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')",
            },
            affected_layers=affected,
            confidence=0.95,
        )]

    def _rule_exploding_layers(
        self, report: GradientReport
    ) -> List[ExpertFinding]:
        """Detect layers with grad_norm above the exploding threshold."""
        affected = [
            s.layer_name
            for s in report.layer_stats
            if _safe_norm(s) > self.exploding_threshold
        ]
        if not affected:
            return []

        return [ExpertFinding(
            rule_id="exploding_layers",
            severity="critical",
            headline=f"🟠 Exploding gradients in {len(affected)} layers",
            detail=(
                "Gradient norms are orders of magnitude above normal in the listed layers. "
                "This typically causes NaN losses within a few steps and is often caused by "
                "a learning rate that is too high, missing gradient clipping, or a poor "
                "weight initialisation."
            ),
            recommendations=[
                "Apply gradient clipping immediately.",
                "Reduce the global learning rate by 5-10x.",
                "Verify weight initialisation (use Xavier for Tanh, He for ReLU).",
                "Add LayerNorm or BatchNorm to regularise activation scale.",
            ],
            code_snippets={
                "Gradient clipping": "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
                "Reduce LR": "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)",
            },
            affected_layers=affected,
            confidence=0.93,
        )]

    def _rule_dead_neurons(
        self, report: GradientReport
    ) -> List[ExpertFinding]:
        """Detect layers where the zero-gradient ratio is above 90%."""
        affected = [
            s.layer_name
            for s in report.layer_stats
            if s.zero_ratio > 0.9
        ]
        if not affected:
            return []

        return [ExpertFinding(
            rule_id="dead_neurons",
            severity="critical",
            headline=f"🟣 Dead neurons detected in {len(affected)} layers",
            detail=(
                "More than 90% of gradient values are exactly zero in the listed layers. "
                "With ReLU activations this is the \"dying ReLU\" phenomenon: neurons that "
                "output zero for all inputs and never recover."
            ),
            recommendations=[
                "Switch from ReLU to Leaky ReLU (negative_slope=0.01) or GELU.",
                "Check for large negative biases — initialise biases to zero or small positive values.",
                "Reduce the learning rate to prevent large weight updates that kill ReLUs.",
                "Re-initialise affected layer weights.",
            ],
            code_snippets={
                "Leaky ReLU": "nn.LeakyReLU(negative_slope=0.01)",
                "Zero-init bias": "nn.init.zeros_(layer.bias)",
            },
            affected_layers=affected,
            confidence=0.90,
        )]

    def _rule_structural_bottleneck(
        self, report: GradientReport
    ) -> List[ExpertFinding]:
        """Detect sudden gradient-norm drops between consecutive layers."""
        sorted_stats = sorted(report.layer_stats, key=lambda s: s.depth)
        norms        = [_safe_norm(s) for s in sorted_stats]
        peak         = max(norms) if norms else 1.0

        bottleneck_pairs: List[str] = []
        for i in range(len(sorted_stats) - 1):
            drop = (norms[i] - norms[i + 1]) / (peak + 1e-12)
            if drop > self.bottleneck_drop_ratio:
                bottleneck_pairs.append(
                    f"{sorted_stats[i].layer_name} → {sorted_stats[i + 1].layer_name} "
                    f"(drop {drop * 100:.0f}%)"
                )

        if not bottleneck_pairs:
            return []

        affected = [
            sorted_stats[i].layer_name
            for i in range(len(sorted_stats) - 1)
            if (norms[i] - norms[i + 1]) / (peak + 1e-12) > self.bottleneck_drop_ratio
        ] + [
            sorted_stats[i + 1].layer_name
            for i in range(len(sorted_stats) - 1)
            if (norms[i] - norms[i + 1]) / (peak + 1e-12) > self.bottleneck_drop_ratio
        ]

        pair_list = "\n".join(f"- {p}" for p in bottleneck_pairs[:5])
        return [ExpertFinding(
            rule_id="structural_bottleneck",
            severity="warning",
            headline=f"⚠️ Structural gradient bottleneck(s) at {len(bottleneck_pairs)} transition(s)",
            detail=(
                f"The gradient norm drops sharply (>{self.bottleneck_drop_ratio * 100:.0f}% "
                f"of peak) at the following layer boundaries, indicating that information is "
                f"not flowing smoothly through the backward pass:\n\n{pair_list}\n\n"
                "This is often caused by a missing residual connection, a very deep block "
                "without normalisation, or a sudden change in layer width."
            ),
            recommendations=[
                "Add a residual (skip) connection across the bottleneck block.",
                "Insert a LayerNorm between the two flagged layers.",
                "Smooth the layer-width transition with gradual bottleneck sizing.",
                "Apply layer-wise learning rate scaling to compensate.",
            ],
            code_snippets={
                "Residual block": (
                    "class ResBlock(nn.Module):\n"
                    "    def forward(self, x):\n"
                    "        return x + self.layers(x)  # skip connection"
                ),
            },
            affected_layers=list(dict.fromkeys(affected)),  # deduplicate, preserve order
            confidence=0.80,
        )]

    def _rule_attention_collapse(
        self, report: GradientReport
    ) -> List[ExpertFinding]:
        """Detect vanishing / near-zero gradients specifically in Attention layers."""
        attn_stats = [
            s for s in report.layer_stats
            if s.group == LayerGroup.ATTENTION
        ]
        if not attn_stats:
            return []

        collapsed = [
            s.layer_name for s in attn_stats
            if _safe_norm(s) < self.vanishing_threshold * 100
        ]
        if not collapsed:
            return []

        return [ExpertFinding(
            rule_id="attention_collapse",
            severity="critical",
            headline=f"🔵 Attention gradient collapse in {len(collapsed)} head(s)",
            detail=(
                "Gradient norms in the listed attention sub-layers are extremely small. "
                "This is the \"attention collapse\" pattern: attention weights converge to a "
                "near-uniform distribution, providing no useful signal. Commonly caused by "
                "missing QK normalisation, incorrect masking, or an initialisation scale that "
                "is too large relative to the hidden dimension."
            ),
            recommendations=[
                "Scale QK dot-products by 1 / sqrt(d_head) — verify this scaling is in place.",
                "Apply QK-LayerNorm (as used in PaLM / Gemini) to stabilise attention.",
                "Reduce the initial weight scale for Q/K projections (e.g. std = 0.02 / sqrt(depth)).",
                "Check for incorrect causal masking that forces softmax to near-uniform.",
            ],
            code_snippets={
                "QK scaling": "attn_weight = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)",
                "QK LayerNorm": (
                    "self.q_norm = nn.LayerNorm(d_head)\n"
                    "self.k_norm = nn.LayerNorm(d_head)"
                ),
            },
            affected_layers=collapsed,
            confidence=0.85,
        )]

    def _rule_norm_layer_bypass(
        self, report: GradientReport
    ) -> List[ExpertFinding]:
        """Warn when LayerNorm layers carry disproportionately large gradients."""
        norm_stats = [
            s for s in report.layer_stats
            if s.group == LayerGroup.LAYER_NORM
        ]
        if not norm_stats:
            return []

        all_norms    = [_safe_norm(s) for s in report.layer_stats]
        global_mean  = float(np.mean(all_norms)) if all_norms else 1.0
        overloaded   = [
            s.layer_name for s in norm_stats
            if _safe_norm(s) > global_mean * 10
        ]
        if not overloaded:
            return []

        return [ExpertFinding(
            rule_id="norm_layer_overload",
            severity="warning",
            headline=f"⚠️ LayerNorm carrying 10× above-average gradients ({len(overloaded)} layers)",
            detail=(
                "The listed LayerNorm parameters are receiving unusually large gradient updates. "
                "This usually means the main computation pathway has vanishing gradients and the "
                "normalisation parameters are trying to compensate — a sign that the upstream "
                "layers are not contributing useful signal."
            ),
            recommendations=[
                "Investigate the layers immediately upstream of the flagged LayerNorm.",
                "If upstream layers are vanishing, address those first.",
                "Consider reducing the LayerNorm learning rate or applying weight decay.",
            ],
            code_snippets={
                "LayerNorm weight decay": (
                    "# In AdamW: exclude norm params from weight decay\n"
                    "decay_params  = [p for n, p in model.named_parameters() if 'norm' not in n]\n"
                    "nodecay_params = [p for n, p in model.named_parameters() if 'norm' in n]"
                ),
            },
            affected_layers=overloaded,
            confidence=0.75,
        )]

    def _rule_gradient_imbalance(
        self, report: GradientReport
    ) -> List[ExpertFinding]:
        """Detect high coefficient of variation in a layer's gradients."""
        unstable = [
            s for s in report.layer_stats
            if abs(s.mean) > 1e-9
            and (s.std / (abs(s.mean) + 1e-12)) > self.unstable_cv_threshold
        ]
        if not unstable:
            return []

        affected = [s.layer_name for s in unstable]
        return [ExpertFinding(
            rule_id="gradient_imbalance",
            severity="warning",
            headline=f"🟡 Highly unstable gradients (high CV) in {len(unstable)} layers",
            detail=(
                "The coefficient of variation (std / |mean|) exceeds "
                f"{self.unstable_cv_threshold:.0f} in the listed layers, indicating that "
                "gradient values vary wildly from batch to batch. This leads to noisy "
                "parameter updates and unstable training loss curves."
            ),
            recommendations=[
                "Apply gradient clipping to cap extreme outlier gradients.",
                "Increase batch size or use gradient accumulation to reduce gradient noise.",
                "Add LayerNorm to regularise the activation distribution entering affected layers.",
                "Use a learning rate schedule with warm-up.",
            ],
            code_snippets={
                "Gradient clipping": "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
                "LR warm-up (cosine)": (
                    "scheduler = torch.optim.lr_scheduler.OneCycleLR(\n"
                    "    optimizer, max_lr=1e-3, steps_per_epoch=len(loader), epochs=epochs)"
                ),
            },
            affected_layers=affected,
            confidence=0.82,
        )]

    def _rule_global_health(
        self, report: GradientReport
    ) -> List[ExpertFinding]:
        """Emit a positive 'all healthy' info finding when no issues found."""
        n_total = len(report.layer_stats)
        n_healthy = sum(
            1 for s in report.layer_stats
            if s.diagnose() == GradientPathology.HEALTHY
        )
        if n_healthy < n_total:
            return []
        return [ExpertFinding(
            rule_id="global_health_ok",
            severity="info",
            headline=f"✅ All {n_total} layers show healthy gradient flow",
            detail=(
                "No critical pathologies detected.  Gradient norms are within normal ranges "
                "across all layers.  Continue monitoring — pathologies can appear after "
                "further training or when switching to a new dataset."
            ),
            confidence=1.0,
        )]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe_norm(stats: Any) -> float:
    gn = getattr(stats, "grad_norm", None)
    if gn is not None and float(gn) > 0:
        return float(gn)
    return float(abs(getattr(stats, "mean", 0.0))) + 1e-12
