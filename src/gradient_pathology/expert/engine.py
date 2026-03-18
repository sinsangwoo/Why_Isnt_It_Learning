"""Phase-4 ExpertEngine: a layer-aware, rule-based diagnostic engine.

This module *extends* the original :class:`~gradient_pathology.expert.rules.ExpertSystem`
(which operates on scalar global statistics) with a second engine that
operates on the full :class:`~gradient_pathology.core.GradientReport`,
checking every layer individually and cross-layer patterns.

Design
------
The engine is composed of *rules* — pure functions that each receive the
:class:`~gradient_pathology.core.GradientReport` and return a list of
:class:`ExpertFinding` objects.  Rules are registered via
:meth:`ExpertEngine.register_rule` and executed in definition order.

Each :class:`ExpertFinding` carries:

* ``rule_id``     — unique identifier (e.g. ``"vanishing_cascade"``)
* ``severity``    — ``"critical"`` | ``"warning"`` | ``"info"``
* ``title``       — one-line headline shown in the popup banner
* ``detail``      — multi-line markdown explanation
* ``layers``      — list of affected layer names
* ``code_hint``   — ready-to-paste Python snippet (optional)
* ``confidence``  — 0–1 float
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from gradient_pathology.core import GradientPathology, GradientReport, LayerGroup


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExpertFinding:
    """A single diagnostic finding produced by one rule."""

    rule_id:    str
    severity:   str        # "critical" | "warning" | "info"
    title:      str
    detail:     str        # markdown
    layers:     List[str]  = field(default_factory=list)
    code_hint:  str        = ""
    confidence: float      = 1.0

    # Canonical severity ordering for sorting
    _SEVERITY_ORDER: Dict[str, int] = field(
        default_factory=lambda: {"critical": 0, "warning": 1, "info": 2},
        repr=False, compare=False,
    )

    @property
    def severity_rank(self) -> int:
        return self._SEVERITY_ORDER.get(self.severity, 3)

    @property
    def emoji(self) -> str:
        return {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(self.severity, "")


# Rule type alias
RuleFunc = Callable[[GradientReport], List[ExpertFinding]]


# ---------------------------------------------------------------------------
# ExpertEngine
# ---------------------------------------------------------------------------

class ExpertEngine:
    """Layer-aware, rule-based expert diagnostic engine.

    Usage
    -----
    ::

        from gradient_pathology.expert.engine import ExpertEngine

        engine   = ExpertEngine()
        findings = engine.analyse(report)
        for f in findings:
            print(f.emoji, f.title)
            print(f.detail)
            if f.code_hint:
                print(f.code_hint)

    Custom rules
    ------------
    ::

        @engine.register_rule
        def my_rule(report: GradientReport) -> list[ExpertFinding]:
            # inspect report.layer_stats…
            return [ExpertFinding(rule_id="my", severity="info", title="…", detail="…")]
    """

    # Built-in rule thresholds (overridable via constructor)
    VANISHING_THRESHOLD   = 1e-7
    EXPLODING_THRESHOLD   = 1e3
    DEAD_NEURON_RATIO     = 0.9
    BOTTLENECK_DROP_RATIO = 0.5   # cascade drop threshold
    ATTENTION_NORM_FLOOR  = 1e-6  # specific to Attention group
    LN_NORM_CEIL          = 1e2   # suspiciously large LayerNorm grad
    MIN_LAYERS_FOR_NORM_CHECK = 10

    def __init__(
        self,
        vanishing_threshold: float   = VANISHING_THRESHOLD,
        exploding_threshold: float   = EXPLODING_THRESHOLD,
        bottleneck_drop_ratio: float = BOTTLENECK_DROP_RATIO,
    ) -> None:
        self.vanishing_threshold   = vanishing_threshold
        self.exploding_threshold   = exploding_threshold
        self.bottleneck_drop_ratio = bottleneck_drop_ratio
        self._rules: List[RuleFunc] = []
        self._register_builtin_rules()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, report: GradientReport) -> List[ExpertFinding]:
        """Run all registered rules against *report* and return sorted findings."""
        findings: List[ExpertFinding] = []
        for rule in self._rules:
            try:
                findings.extend(rule(report))
            except Exception:  # pragma: no cover — rule bugs must not crash the UI
                pass
        findings.sort(key=lambda f: (f.severity_rank, -f.confidence))
        return findings

    def register_rule(self, func: RuleFunc) -> RuleFunc:
        """Decorator / callable to add a custom rule."""
        self._rules.append(func)
        return func

    def quick_summary(self, report: GradientReport) -> str:
        """Return a one-line health summary string."""
        findings = self.analyse(report)
        if not findings:
            return "✅ All layers healthy — no issues detected."
        crit = sum(1 for f in findings if f.severity == "critical")
        warn = sum(1 for f in findings if f.severity == "warning")
        parts = []
        if crit:
            parts.append(f"{crit} critical")
        if warn:
            parts.append(f"{warn} warning")
        return "🚨 " + ", ".join(parts) + f" (×{len(findings)} findings total)"

    # ------------------------------------------------------------------
    # Built-in rules
    # ------------------------------------------------------------------

    def _register_builtin_rules(self) -> None:
        """Register the 7 built-in diagnostic rules."""
        self._rules.extend([
            self._rule_vanishing_layers,
            self._rule_exploding_layers,
            self._rule_dead_neurons,
            self._rule_bottleneck_cascade,
            self._rule_missing_layer_norm,
            self._rule_attention_health,
            self._rule_layernorm_explosion,
        ])

    # Rule 1 — Vanishing layers -----------------------------------------------

    def _rule_vanishing_layers(self, report: GradientReport) -> List[ExpertFinding]:
        affected = [
            s for s in report.layer_stats
            if _safe_norm(s) < self.vanishing_threshold
        ]
        if not affected:
            return []
        names = [s.layer_name for s in affected]
        pct   = len(affected) / max(len(report.layer_stats), 1) * 100
        return [ExpertFinding(
            rule_id="vanishing_layers",
            severity="critical",
            title=f"Vanishing gradients in {len(affected)} layer(s) ({pct:.0f}%)",
            detail=(
                f"**{len(affected)}** layer(s) have ``grad_norm < {self.vanishing_threshold:.0e}``.\n\n"
                "This typically means gradients cannot propagate back to early layers, "
                "preventing those layers from learning.\n\n"
                "**Likely causes:** sigmoid/tanh saturation, missing normalisation, "
                "very deep network without residual connections."
            ),
            layers=names,
            code_hint=(
                "# Option 1: Replace saturating activations\n"
                "model = replace_activations(model, nn.Sigmoid, nn.GELU)\n\n"
                "# Option 2: Add LayerNorm after each Linear\n"
                "# Linear → LayerNorm → GELU  (PreLN pattern)\n\n"
                "# Option 3: Increase learning rate for early layers\n"
                "from gradient_pathology.auto import LayerLRFinder\n"
                "finder = LayerLRFinder(model, optimizer)\n"
                "lrs    = finder.suggest_layer_lrs()"
            ),
            confidence=0.97,
        )]

    # Rule 2 — Exploding layers -----------------------------------------------

    def _rule_exploding_layers(self, report: GradientReport) -> List[ExpertFinding]:
        affected = [
            s for s in report.layer_stats
            if _safe_norm(s) > self.exploding_threshold
        ]
        if not affected:
            return []
        names = [s.layer_name for s in affected]
        return [ExpertFinding(
            rule_id="exploding_layers",
            severity="critical",
            title=f"Exploding gradients in {len(affected)} layer(s)",
            detail=(
                f"**{len(affected)}** layer(s) exceed ``grad_norm > {self.exploding_threshold:.0e}``.\n\n"
                "Unchecked, exploding gradients will corrupt weights and cause NaN loss.\n\n"
                "**Likely causes:** bad weight initialisation, high learning rate, "
                "missing gradient clipping."
            ),
            layers=names,
            code_hint=(
                "# Clip gradients before optimizer.step()\n"
                "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n\n"
                "# Or use gradient clipping in the Trainer:\n"
                "TrainingArguments(max_grad_norm=1.0, ...)"
            ),
            confidence=0.99,
        )]

    # Rule 3 — Dead neurons ---------------------------------------------------

    def _rule_dead_neurons(self, report: GradientReport) -> List[ExpertFinding]:
        affected = [
            s for s in report.layer_stats
            if s.zero_ratio > self.DEAD_NEURON_RATIO
        ]
        if not affected:
            return []
        names = [s.layer_name for s in affected]
        worst = max(affected, key=lambda s: s.zero_ratio)
        return [ExpertFinding(
            rule_id="dead_neurons",
            severity="warning",
            title=f"Dead neurons: {len(affected)} layer(s) with >90% zero gradients",
            detail=(
                f"Worst layer: **{worst.layer_name}** ({worst.zero_ratio:.0%} zeros).\n\n"
                "Dead neurons never update and permanently reduce model capacity.\n\n"
                "**Likely causes:** ReLU with large negative biases, very high LR in early training."
            ),
            layers=names,
            code_hint=(
                "# Switch from ReLU to Leaky ReLU or GELU:\n"
                "nn.LeakyReLU(negative_slope=0.01)  # avoids dying ReLU\n"
                "nn.GELU()                           # smooth gradient everywhere\n\n"
                "# Check for large negative biases:\n"
                "for name, p in model.named_parameters():\n"
                "    if 'bias' in name and p.data.min() < -5:\n"
                "        print(f'Large neg bias: {name}: {p.data.min():.2f}')"
            ),
            confidence=0.92,
        )]

    # Rule 4 — Bottleneck cascade --------------------------------------------

    def _rule_bottleneck_cascade(self, report: GradientReport) -> List[ExpertFinding]:
        """Detect consecutive depth-ordered layers where norm drops sharply."""
        sorted_stats = sorted(report.layer_stats, key=lambda s: s.depth)
        if len(sorted_stats) < 3:
            return []

        norms   = [_safe_norm(s) for s in sorted_stats]
        peak    = max(norms) if norms else 1.0
        bottlenecks: List[str] = []

        for i in range(1, len(norms)):
            drop = (norms[i - 1] - norms[i]) / (norms[i - 1] + 1e-12)
            if drop > self.bottleneck_drop_ratio and norms[i - 1] > self.vanishing_threshold:
                bottlenecks.append(sorted_stats[i].layer_name)

        if not bottlenecks:
            return []

        return [ExpertFinding(
            rule_id="bottleneck_cascade",
            severity="warning",
            title=f"Bottleneck cascade: {len(bottlenecks)} abrupt gradient drop(s)",
            detail=(
                f"Gradient norm drops by >{self.bottleneck_drop_ratio:.0%} at "
                f"**{len(bottlenecks)}** transition(s).\n\n"
                "These are the points where information is most likely being lost.\n"
                "Hover over the Sankey diagram to see exact loss fractions."
            ),
            layers=bottlenecks,
            code_hint=(
                "# Add residual connections around bottleneck layers:\n"
                "class ResBlock(nn.Module):\n"
                "    def forward(self, x):\n"
                "        return x + self.sublayer(x)  # skip connection\n\n"
                "# Or reduce the depth-to-width ratio at the bottleneck."
            ),
            confidence=0.85,
        )]

    # Rule 5 — Missing LayerNorm for deep networks ----------------------------

    def _rule_missing_layer_norm(self, report: GradientReport) -> List[ExpertFinding]:
        n_total = len(report.layer_stats)
        if n_total < self.MIN_LAYERS_FOR_NORM_CHECK:
            return []

        n_ln = sum(
            1 for s in report.layer_stats
            if s.group == LayerGroup.LAYER_NORM
        )
        if n_ln > 0:
            return []

        # Only flag if we also have vanishing gradients
        has_vanishing = any(
            _safe_norm(s) < self.vanishing_threshold
            for s in report.layer_stats
        )
        if not has_vanishing:
            return [ExpertFinding(
                rule_id="no_layernorm",
                severity="info",
                title=f"No LayerNorm detected in {n_total}-layer network",
                detail=(
                    f"The model has **{n_total}** layers but no LayerNorm/BatchNorm parameters.\n\n"
                    "For networks with ≥10 layers, normalisation layers typically improve "
                    "training stability and convergence speed."
                ),
                code_hint=(
                    "# Add LayerNorm to an existing Sequential:\n"
                    "from gradient_pathology.expert.engine import inject_layer_norms\n"
                    "model = inject_layer_norms(model)  # wraps each Linear with LN"
                ),
                confidence=0.72,
            )]

        return [ExpertFinding(
            rule_id="no_layernorm_vanishing",
            severity="critical",
            title=f"Deep network ({n_total} layers) lacks normalisation AND has vanishing gradients",
            detail=(
                "Combining a deep architecture with no normalisation and vanishing gradients "
                "is a high-risk configuration.\n\n"
                "Adding LayerNorm is the single most impactful change you can make."
            ),
            code_hint=(
                "# Minimal fix: insert LayerNorm after every Linear layer\n"
                "layers = []\n"
                "for m in model.modules():\n"
                "    if isinstance(m, nn.Linear):\n"
                "        layers += [m, nn.LayerNorm(m.out_features), nn.GELU()]\n"
                "model = nn.Sequential(*layers)"
            ),
            confidence=0.95,
        )]

    # Rule 6 — Attention-specific health check --------------------------------

    def _rule_attention_health(self, report: GradientReport) -> List[ExpertFinding]:
        attn_layers = [
            s for s in report.layer_stats
            if s.group == LayerGroup.ATTENTION
        ]
        if len(attn_layers) < 2:
            return []

        sick = [
            s for s in attn_layers
            if _safe_norm(s) < self.ATTENTION_NORM_FLOOR
        ]
        if not sick:
            return []

        return [ExpertFinding(
            rule_id="attention_low_grad",
            severity="warning",
            title=f"Attention layers with near-zero gradients: {len(sick)}/{len(attn_layers)}",
            detail=(
                f"**{len(sick)}** attention projection layer(s) have "
                f"``grad_norm < {self.ATTENTION_NORM_FLOOR:.0e}``.\n\n"
                "Possible causes: attention collapse (all heads attending to the same token), "
                "QK scaling issue, or excessively small learning rate for the attention block."
            ),
            layers=[s.layer_name for s in sick],
            code_hint=(
                "# Check attention entropy (requires llm diagnostics):\n"
                "from gradient_pathology.llm import TransformerDiagnostics\n"
                "diag = TransformerDiagnostics(model)\n"
                "if diag.detect_attention_collapse(attn_weights):\n"
                "    print('Attention collapsed — try reducing dropout or temp.')\n\n"
                "# Ensure Q/K scaling:\n"
                "scale = head_dim ** -0.5\n"
                "attn_scores = (q @ k.T) * scale"
            ),
            confidence=0.80,
        )]

    # Rule 7 — LayerNorm explosion check ------------------------------------

    def _rule_layernorm_explosion(self, report: GradientReport) -> List[ExpertFinding]:
        ln_layers = [
            s for s in report.layer_stats
            if s.group == LayerGroup.LAYER_NORM
        ]
        if not ln_layers:
            return []

        exploding_ln = [
            s for s in ln_layers
            if _safe_norm(s) > self.LN_NORM_CEIL
        ]
        if not exploding_ln:
            return []

        return [ExpertFinding(
            rule_id="layernorm_explosion",
            severity="warning",
            title=f"LayerNorm parameters have very large gradients: {len(exploding_ln)} layer(s)",
            detail=(
                "LayerNorm scale (gamma) and shift (beta) parameters should have modest gradients.\n\n"
                "Large gradients here often indicate that the network is trying to compensate "
                "for poorly initialised weights or that the learning rate is too high for norm layers."
            ),
            layers=[s.layer_name for s in exploding_ln],
            code_hint=(
                "# Use a lower LR for LayerNorm parameters:\n"
                "param_groups = [\n"
                "    {'params': [p for n, p in model.named_parameters() if 'norm' not in n]},\n"
                "    {'params': [p for n, p in model.named_parameters() if 'norm' in n], 'lr': 1e-5},\n"
                "]\n"
                "optimizer = torch.optim.AdamW(param_groups, lr=3e-4)"
            ),
            confidence=0.78,
        )]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_norm(stats: Any) -> float:
    gn = getattr(stats, "grad_norm", None)
    if gn is not None and float(gn) > 0:
        return float(gn)
    return float(abs(getattr(stats, "mean", 0.0))) + 1e-12


def inject_layer_norms(model: Any) -> Any:
    """Utility: wrap each ``nn.Linear`` in a new ``nn.Sequential`` that adds
    a ``nn.LayerNorm`` after it.  Returns the modified model.

    This is the simplest possible LayerNorm injection and is meant as a
    quick diagnostic tool, not a production recipe.  It works only for
    flat ``nn.Sequential`` models.
    """
    try:
        import torch.nn as nn
    except ImportError:
        return model

    if not isinstance(model, nn.Sequential):
        return model

    new_layers = []
    for m in model.children():
        new_layers.append(m)
        if isinstance(m, nn.Linear):
            new_layers.append(nn.LayerNorm(m.out_features))
    return nn.Sequential(*new_layers)
