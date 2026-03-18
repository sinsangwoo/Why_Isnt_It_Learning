"""Tests for Phase-1 data pipeline: classifier, snapshot store, enriched stats."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from gradient_pathology.core import LayerGroup, LayerGradientStats, GradientPathology
from gradient_pathology.pipeline.classifier import (
    TransformerLayerClassifier,
    _classify_name,
)
from gradient_pathology.pipeline.snapshot import GradientSnapshotStore
from gradient_pathology.analyzer import GradientAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TinyTransformer(nn.Module):
    """Minimal GPT-like model for testing the classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.wte = nn.Embedding(100, 32)           # EMBEDDING
        self.wpe = nn.Embedding(16, 32)            # EMBEDDING
        self.ln1 = nn.LayerNorm(32)               # LAYER_NORM
        self.attn_q = nn.Linear(32, 32)           # ATTENTION (q)
        self.attn_k = nn.Linear(32, 32)           # ATTENTION (k)
        self.attn_v = nn.Linear(32, 32)           # ATTENTION (v)
        self.attn_out = nn.Linear(32, 32)         # ATTENTION (out_proj)
        self.ln2 = nn.LayerNorm(32)               # LAYER_NORM
        self.mlp_fc = nn.Linear(32, 128)          # FFN (fc)
        self.mlp_proj = nn.Linear(128, 32)        # FFN (mlp)
        self.lm_head = nn.Linear(32, 100)         # HEAD

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Minimal forward — just enough for a loss to be computed.
        tok = self.wte(x)
        pos = self.wpe(torch.arange(x.size(1)))
        h = self.ln1(tok + pos)
        q = self.attn_q(h)
        k = self.attn_k(h)
        v = self.attn_v(h)
        attn = torch.softmax(q @ k.transpose(-2, -1) / 32 ** 0.5, dim=-1) @ v
        h = self.attn_out(attn) + h
        h = self.ln2(h)
        h = self.mlp_proj(nn.functional.gelu(self.mlp_fc(h)))
        return self.lm_head(h)


@pytest.fixture()
def tiny_transformer() -> TinyTransformer:
    return TinyTransformer()


@pytest.fixture()
def simple_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    )


# ---------------------------------------------------------------------------
# 1-C: TransformerLayerClassifier tests
# ---------------------------------------------------------------------------


class TestClassifyName:
    """Unit tests for the _classify_name heuristic."""

    @pytest.mark.parametrize(
        "param_name, expected",
        [
            # Embeddings
            ("wte.weight", LayerGroup.EMBEDDING),
            ("wpe.weight", LayerGroup.EMBEDDING),
            ("embed_tokens.weight", LayerGroup.EMBEDDING),
            # Head
            ("lm_head.weight", LayerGroup.HEAD),
            # Layer norm
            ("transformer.h.0.ln_1.weight", LayerGroup.LAYER_NORM),
            ("model.layers.0.input_layernorm.weight", LayerGroup.LAYER_NORM),
            ("model.norm.weight", LayerGroup.LAYER_NORM),
            # Attention
            ("transformer.h.0.attn.c_attn.weight", LayerGroup.ATTENTION),
            ("model.layers.0.self_attn.q_proj.weight", LayerGroup.ATTENTION),
            ("model.layers.0.self_attn.v_proj.weight", LayerGroup.ATTENTION),
            ("model.layers.0.self_attn.out_proj.weight", LayerGroup.ATTENTION),
            # FFN
            ("transformer.h.0.mlp.c_fc.weight", LayerGroup.FFN),
            ("model.layers.0.mlp.gate_proj.weight", LayerGroup.FFN),
            ("model.layers.0.mlp.up_proj.weight", LayerGroup.FFN),
            ("model.layers.0.mlp.down_proj.weight", LayerGroup.FFN),
            ("encoder.layer.0.intermediate.dense.weight", LayerGroup.FFN),
            # Other
            ("some_custom_layer.weight", LayerGroup.OTHER),
        ],
    )
    def test_classify_name(self, param_name: str, expected: LayerGroup) -> None:
        assert _classify_name(param_name) == expected


class TestTransformerLayerClassifier:
    def test_build_param_metadata_keys_match_named_parameters(
        self, tiny_transformer: TinyTransformer
    ) -> None:
        clf = TransformerLayerClassifier(tiny_transformer)
        meta = clf.build_param_metadata()
        expected_keys = {name for name, _ in tiny_transformer.named_parameters()}
        assert set(meta.keys()) == expected_keys

    def test_attention_params_classified_correctly(
        self, tiny_transformer: TinyTransformer
    ) -> None:
        clf = TransformerLayerClassifier(tiny_transformer)
        meta = clf.build_param_metadata()
        for param_name in ["attn_q.weight", "attn_k.weight", "attn_v.weight"]:
            _, group = meta[param_name]
            assert group == LayerGroup.ATTENTION, (
                f"{param_name} should be ATTENTION, got {group}"
            )

    def test_ffn_params_classified_correctly(
        self, tiny_transformer: TinyTransformer
    ) -> None:
        clf = TransformerLayerClassifier(tiny_transformer)
        meta = clf.build_param_metadata()
        for param_name in ["mlp_fc.weight", "mlp_proj.weight"]:
            _, group = meta[param_name]
            assert group == LayerGroup.FFN, (
                f"{param_name} should be FFN, got {group}"
            )

    def test_layer_norm_params_classified_correctly(
        self, tiny_transformer: TinyTransformer
    ) -> None:
        clf = TransformerLayerClassifier(tiny_transformer)
        meta = clf.build_param_metadata()
        for param_name in ["ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"]:
            _, group = meta[param_name]
            assert group == LayerGroup.LAYER_NORM, (
                f"{param_name} should be LAYER_NORM, got {group}"
            )

    def test_head_classified_correctly(
        self, tiny_transformer: TinyTransformer
    ) -> None:
        clf = TransformerLayerClassifier(tiny_transformer)
        meta = clf.build_param_metadata()
        _, group = meta["lm_head.weight"]
        assert group == LayerGroup.HEAD

    def test_layer_type_resolved(
        self, tiny_transformer: TinyTransformer
    ) -> None:
        clf = TransformerLayerClassifier(tiny_transformer)
        meta = clf.build_param_metadata()
        layer_type, _ = meta["attn_q.weight"]
        assert layer_type == "Linear"

    def test_group_summary_covers_all_groups(
        self, tiny_transformer: TinyTransformer
    ) -> None:
        clf = TransformerLayerClassifier(tiny_transformer)
        summary = clf.group_summary()
        # All LayerGroup values should be keys
        for group in LayerGroup:
            assert group.value in summary

    def test_plain_mlp_falls_back_to_other(
        self, simple_mlp: nn.Module
    ) -> None:
        clf = TransformerLayerClassifier(simple_mlp)
        meta = clf.build_param_metadata()
        for _, (layer_type, group) in meta.items():
            assert group == LayerGroup.OTHER


# ---------------------------------------------------------------------------
# 1-B: GradientSnapshotStore tests
# ---------------------------------------------------------------------------


class TestGradientSnapshotStore:
    def _make_stats(self, n: int = 3) -> list:
        """Create a small list of LayerGradientStats for testing."""
        stats = []
        for i in range(n):
            s = LayerGradientStats(
                layer_name=f"layer_{i}.weight",
                layer_index=i,
                mean=float(np.random.randn()),
                std=float(np.abs(np.random.randn())),
                min=-1.0,
                max=1.0,
                median=0.0,
                num_zeros=0,
                total_params=64,
                layer_type="Linear",
                depth=i,
                group=LayerGroup.FFN,
                grad_norm=float(np.abs(np.random.randn())),
            )
            stats.append(s)
        return stats

    def test_invalid_fmt_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="fmt must be"):
                GradientSnapshotStore(output_dir=tmp, fmt="csv")

    def test_flush_json_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = GradientSnapshotStore(output_dir=tmp, fmt="json", buffer_size=0)
            store.record_from_stats(step=0, layer_stats=self._make_stats())
            out = store.flush()
            assert out is not None
            assert out.exists()
            assert out.suffix == ".json"

    def test_json_content_has_correct_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = GradientSnapshotStore(output_dir=tmp, fmt="json", buffer_size=0)
            stats = self._make_stats(n=2)
            store.record_from_stats(step=5, layer_stats=stats)
            out = store.flush()
            assert out is not None
            with open(out) as fh:
                records = json.load(fh)
            assert len(records) == 2
            for rec in records:
                assert rec["step"] == 5
                assert "layer_name" in rec
                assert "grad_norm" in rec
                assert "group" in rec
                assert "pathology" in rec

    def test_auto_flush_on_buffer_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # buffer_size=3 rows; each record_from_stats call adds n_layers rows
            store = GradientSnapshotStore(output_dir=tmp, fmt="json", buffer_size=3)
            store.record_from_stats(step=0, layer_stats=self._make_stats(n=3))
            # Buffer should have been auto-flushed (3 rows == buffer_size)
            files = list(Path(tmp).glob("*.json"))
            assert len(files) == 1

    def test_flush_returns_none_on_empty_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = GradientSnapshotStore(output_dir=tmp, fmt="json")
            result = store.flush()
            assert result is None

    def test_load_json_raw_returns_all_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = GradientSnapshotStore(output_dir=tmp, fmt="json", buffer_size=0)
            for step in range(3):
                store.record_from_stats(step=step, layer_stats=self._make_stats(n=2))
                store.flush()
            raw = store.load_json_raw()
            assert len(raw) == 6  # 3 steps * 2 layers

    def test_multiple_chunks_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = GradientSnapshotStore(output_dir=tmp, fmt="json", buffer_size=0)
            for step in range(4):
                store.record_from_stats(step=step, layer_stats=self._make_stats(n=1))
                store.flush()
            files = sorted(Path(tmp).glob("*.json"))
            assert len(files) == 4
            # Chunk filenames are zero-padded
            assert files[0].name == "snapshot_00000.json"
            assert files[3].name == "snapshot_00003.json"

    def test_record_from_monitor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = GradientSnapshotStore(output_dir=tmp, fmt="json", buffer_size=0)
            monitor_entry = {
                "fc1.weight": {"mean": 1e-3, "std": 1e-4, "max": 2e-3},
                "fc2.weight": {"mean": 1e-9, "std": 1e-10, "max": 2e-9},
            }
            store.record_from_monitor(step=0, monitor_history_entry=monitor_entry)
            out = store.flush()
            assert out is not None
            with open(out) as fh:
                records = json.load(fh)
            pathologies = {r["layer_name"]: r["pathology"] for r in records}
            assert pathologies["fc1.weight"] == GradientPathology.HEALTHY.value
            assert pathologies["fc2.weight"] == GradientPathology.VANISHING.value

    def test_summary_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = GradientSnapshotStore(output_dir=tmp, fmt="json")
            s = store.summary()
            assert "GradientSnapshotStore" in s
            assert "json" in s


# ---------------------------------------------------------------------------
# 1-A: GradientAnalyzer enriched stats tests
# ---------------------------------------------------------------------------


class TestGradientAnalyzerEnrichedStats:
    def test_layer_stats_have_grad_norm(
        self, tiny_transformer: TinyTransformer
    ) -> None:
        analyzer = GradientAnalyzer(tiny_transformer)
        # Use a simple token-level input
        x = torch.randint(0, 100, (2, 8))
        targets = torch.randint(0, 100, (2, 8))
        loader = [(x, targets)]
        report = analyzer.diagnose(
            dataloader=loader,
            loss_fn=nn.CrossEntropyLoss(),
        )
        for stats in report.layer_stats:
            assert hasattr(stats, "grad_norm")
            assert stats.grad_norm >= 0.0

    def test_layer_stats_have_layer_type(
        self, tiny_transformer: TinyTransformer
    ) -> None:
        analyzer = GradientAnalyzer(tiny_transformer)
        x = torch.randint(0, 100, (2, 8))
        targets = torch.randint(0, 100, (2, 8))
        loader = [(x, targets)]
        report = analyzer.diagnose(
            dataloader=loader,
            loss_fn=nn.CrossEntropyLoss(),
        )
        for stats in report.layer_stats:
            assert stats.layer_type != ""
            assert stats.layer_type is not None

    def test_layer_stats_have_group(
        self, tiny_transformer: TinyTransformer
    ) -> None:
        analyzer = GradientAnalyzer(tiny_transformer)
        x = torch.randint(0, 100, (2, 8))
        targets = torch.randint(0, 100, (2, 8))
        loader = [(x, targets)]
        report = analyzer.diagnose(
            dataloader=loader,
            loss_fn=nn.CrossEntropyLoss(),
        )
        groups_seen = {stats.group for stats in report.layer_stats}
        # A Transformer should have at least Attention and FFN groups
        assert LayerGroup.ATTENTION in groups_seen
        assert LayerGroup.FFN in groups_seen

    def test_synthetic_mode_still_works(
        self, simple_mlp: nn.Module
    ) -> None:
        analyzer = GradientAnalyzer(simple_mlp)
        report = analyzer.diagnose(num_steps=5, input_shape=(16,))
        assert len(report.layer_stats) > 0
        for stats in report.layer_stats:
            assert hasattr(stats, "depth")
            assert hasattr(stats, "group")
