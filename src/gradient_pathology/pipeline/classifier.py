"""1-C: Automatic Transformer layer group classifier.

Given a ``torch.nn.Module``, this classifier inspects every *named parameter*
and maps it to a semantic :class:`~gradient_pathology.core.LayerGroup`
(Attention / FFN / LayerNorm / Embedding / Head / Other).

The classification is purely name-based — no forward pass is required — so it
works for any HuggingFace-style Transformer as well as hand-rolled GPT / BERT
variants, as long as the standard naming conventions are followed.

Naming heuristics (case-insensitive substring match, evaluated in order)::

    embed           → EMBEDDING
    lm_head / head  → HEAD
    ln / norm       → LAYER_NORM
    attn / attention / query / key / value / q_proj / k_proj / v_proj
    / out_proj / self_attn / cross_attn
                    → ATTENTION
    mlp / ffn / fc / feed_forward / intermediate / dense
                    → FFN
    (everything else)
                    → OTHER
"""

from typing import Dict, List, Set, Tuple

import torch.nn as nn

from gradient_pathology.core import LayerGroup


# ---------------------------------------------------------------------------
# Keyword tables (evaluated in listed order — first match wins)
# ---------------------------------------------------------------------------

_EMBEDDING_KEYWORDS: Tuple[str, ...] = (
    "embed",
    "wte",
    "wpe",
    "token_emb",
    "position_emb",
)

_HEAD_KEYWORDS: Tuple[str, ...] = (
    "lm_head",
    "cls_head",
    "output_projection",
    ".head.",
)

_NORM_KEYWORDS: Tuple[str, ...] = (
    "layernorm",
    "layer_norm",
    "rmsnorm",
    "rms_norm",
    ".ln",
    "_ln",
    ".norm",
    "_norm",
)

_ATTENTION_KEYWORDS: Tuple[str, ...] = (
    "attn",
    "attention",
    "query",
    "key",
    "value",
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "self_attn",
    "cross_attn",
    "c_attn",
    "c_proj",
)

_FFN_KEYWORDS: Tuple[str, ...] = (
    "mlp",
    "ffn",
    "feed_forward",
    "feedforward",
    "intermediate",
    ".fc",
    "_fc",
    "dense",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _classify_name(param_name: str) -> LayerGroup:
    """Classify a single parameter name into a :class:`LayerGroup`.

    Parameters
    ----------
    param_name:
        The fully-qualified parameter name as returned by
        ``model.named_parameters()`` (e.g.
        ``'transformer.h.0.attn.c_attn.weight'``).

    Returns
    -------
    LayerGroup
        The inferred semantic group.
    """
    name_lower = param_name.lower()

    for kw in _EMBEDDING_KEYWORDS:
        if kw in name_lower:
            return LayerGroup.EMBEDDING

    for kw in _HEAD_KEYWORDS:
        if kw in name_lower:
            return LayerGroup.HEAD

    for kw in _NORM_KEYWORDS:
        if kw in name_lower:
            return LayerGroup.LAYER_NORM

    for kw in _ATTENTION_KEYWORDS:
        if kw in name_lower:
            return LayerGroup.ATTENTION

    for kw in _FFN_KEYWORDS:
        if kw in name_lower:
            return LayerGroup.FFN

    return LayerGroup.OTHER


class TransformerLayerClassifier:
    """Classifies every parameter in a model into a semantic
    :class:`~gradient_pathology.core.LayerGroup`.

    The classifier also resolves the **module-level type name** (e.g.
    ``'Linear'``, ``'LayerNorm'``) by walking ``model.named_modules()``
    and matching on the shared prefix of the parameter name.

    Parameters
    ----------
    model:
        The PyTorch model to classify.

    Example
    -------
    ::

        from transformers import GPT2Model
        from gradient_pathology.pipeline import TransformerLayerClassifier

        model = GPT2Model.from_pretrained("gpt2")
        clf = TransformerLayerClassifier(model)
        meta = clf.build_param_metadata()
        # meta["transformer.h.0.attn.c_attn.weight"]
        # → ("Linear", LayerGroup.ATTENTION)
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        # Pre-build a sorted list of (module_path, module_type) pairs so we
        # can quickly resolve parameter → owning module.
        self._module_types: List[Tuple[str, str]] = [
            (name, type(module).__name__)
            for name, module in model.named_modules()
            if name  # skip the root module (empty string)
        ]
        # Sort longest prefix first so the first match is always the most
        # specific owning module.
        self._module_types.sort(key=lambda t: len(t[0]), reverse=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_param_metadata(
        self,
    ) -> Dict[str, Tuple[str, LayerGroup]]:
        """Build a mapping of parameter names to ``(layer_type, LayerGroup)``.

        Returns
        -------
        dict
            Keys are fully-qualified parameter names (as from
            ``model.named_parameters()``).  Values are ``(layer_type,
            group)`` tuples where *layer_type* is the PyTorch module class
            name and *group* is the inferred :class:`LayerGroup`.
        """
        meta: Dict[str, Tuple[str, LayerGroup]] = {}
        for param_name, _ in self._model.named_parameters():
            layer_type = self._resolve_layer_type(param_name)
            group = _classify_name(param_name)
            meta[param_name] = (layer_type, group)
        return meta

    def classify_param(self, param_name: str) -> LayerGroup:
        """Classify a single parameter name.

        Parameters
        ----------
        param_name:
            Fully-qualified parameter name.

        Returns
        -------
        LayerGroup
        """
        return _classify_name(param_name)

    def group_summary(self) -> Dict[str, List[str]]:
        """Return a dict mapping group names to lists of parameter names.

        Useful for a quick sanity-check of the classification result.

        Returns
        -------
        dict
            e.g. ``{"attention": ["h.0.attn.c_attn.weight", ...], ...}``
        """
        result: Dict[str, List[str]] = {g.value: [] for g in LayerGroup}
        for param_name, _ in self._model.named_parameters():
            group = _classify_name(param_name)
            result[group.value].append(param_name)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_layer_type(self, param_name: str) -> str:
        """Return the module class name that owns *param_name*.

        Walks the pre-sorted ``_module_types`` list and returns the type of
        the first (most specific) module whose path is a prefix of
        *param_name*.
        """
        for module_path, module_type in self._module_types:
            # The param name starts with the module path followed by a dot.
            if param_name.startswith(module_path + ".") or param_name == module_path:
                return module_type
        return "unknown"
