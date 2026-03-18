"""1-B: Step-wise gradient snapshot storage.

This module provides :class:`GradientSnapshotStore`, which records gradient
statistics at every training step and persists them as:

* **JSON** — human-readable, zero extra dependencies, suitable for small runs
  (< 10k steps × 200 layers).
* **Parquet** — columnar binary format via ``pandas`` + ``pyarrow``; compact
  and fast for large LLM runs.  Requires the ``[storage]`` extra.

The stored schema is a flat table where each row is one (step, layer) pair::

    step        int       training step index (0-based)
    layer_name  str       fully-qualified parameter name
    layer_type  str       PyTorch module class, e.g. 'Linear'
    depth       int       0-based layer depth (parameter index)
    group       str       LayerGroup value, e.g. 'attention'
    grad_norm   float     L2 norm of gradient at this step
    mean        float     mean gradient value
    std         float     gradient standard deviation
    min         float     minimum gradient value
    max         float     maximum gradient value
    zero_ratio  float     fraction of zero gradients
    pathology   str       GradientPathology value, e.g. 'healthy'

Typical usage::

    from gradient_pathology.pipeline import GradientSnapshotStore
    from gradient_pathology.callbacks import GradientMonitor

    store = GradientSnapshotStore(output_dir="runs/exp1")
    monitor = GradientMonitor(model)

    for step, batch in enumerate(dataloader):
        loss.backward()
        monitor.record_step()
        store.record_step(step, monitor)

    store.flush()          # write buffered rows to disk
    df = store.load()      # returns a pandas DataFrame
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch.nn as nn

from gradient_pathology.core import GradientPathology, LayerGradientStats


class GradientSnapshotStore:
    """Records and persists step-wise gradient statistics.

    Parameters
    ----------
    output_dir:
        Directory where snapshot files will be written.  Created
        automatically if it does not exist.
    fmt:
        Storage format — ``'json'`` (default) or ``'parquet'``.
        Parquet requires ``pandas`` and ``pyarrow``.
    buffer_size:
        Number of step-rows to accumulate in memory before flushing to
        disk.  Set to ``0`` to flush on every ``record_step`` call.
    """

    _COLUMNS = [
        "step",
        "layer_name",
        "layer_type",
        "depth",
        "group",
        "grad_norm",
        "mean",
        "std",
        "min",
        "max",
        "zero_ratio",
        "pathology",
    ]

    def __init__(
        self,
        output_dir: str = "gradient_snapshots",
        fmt: str = "json",
        buffer_size: int = 500,
    ) -> None:
        if fmt not in ("json", "parquet"):
            raise ValueError(f"fmt must be 'json' or 'parquet', got {fmt!r}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fmt = fmt
        self.buffer_size = buffer_size

        self._buffer: List[Dict[str, Any]] = []
        self._chunk_index: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_from_stats(
        self,
        step: int,
        layer_stats: List[LayerGradientStats],
    ) -> None:
        """Record a snapshot from a list of :class:`LayerGradientStats`.

        This is the primary entry point when integrating with
        :class:`~gradient_pathology.analyzer.GradientAnalyzer`.

        Parameters
        ----------
        step:
            Current training step index (0-based).
        layer_stats:
            Per-layer statistics, typically ``report.layer_stats``.
        """
        for stats in layer_stats:
            row = self._stats_to_row(step, stats)
            self._buffer.append(row)

        if self.buffer_size > 0 and len(self._buffer) >= self.buffer_size:
            self.flush()

    def record_from_monitor(
        self,
        step: int,
        monitor_history_entry: Dict[str, Dict[str, float]],
        param_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a snapshot directly from a
        :class:`~gradient_pathology.callbacks.GradientMonitor` step entry.

        This lightweight path avoids building a full
        :class:`~gradient_pathology.core.GradientReport` and is suited for
        real-time monitoring during training.

        Parameters
        ----------
        step:
            Current training step index.
        monitor_history_entry:
            One entry from ``GradientMonitor.history`` — a dict mapping
            ``layer_name`` to ``{mean, std, max}``.
        param_meta:
            Optional mapping of ``layer_name`` to ``(layer_type, LayerGroup)``
            as returned by
            :meth:`~gradient_pathology.pipeline.classifier.TransformerLayerClassifier.build_param_metadata`.
            When supplied, ``layer_type`` and ``group`` are populated;
            otherwise they default to ``'unknown'`` and ``'other'``.
        """
        for layer_name, grad_stats in monitor_history_entry.items():
            layer_type = "unknown"
            group_value = "other"
            depth = 0

            if param_meta and layer_name in param_meta:
                lt, grp = param_meta[layer_name]
                layer_type = lt
                group_value = grp.value

            mean_val = grad_stats.get("mean", 0.0)
            abs_mean = abs(mean_val)
            if abs_mean < 1e-8:
                pathology = GradientPathology.VANISHING.value
            elif abs_mean > 1e3:
                pathology = GradientPathology.EXPLODING.value
            else:
                pathology = GradientPathology.HEALTHY.value

            row: Dict[str, Any] = {
                "step": step,
                "layer_name": layer_name,
                "layer_type": layer_type,
                "depth": depth,
                "group": group_value,
                "grad_norm": grad_stats.get("max", 0.0),
                "mean": mean_val,
                "std": grad_stats.get("std", 0.0),
                "min": 0.0,   # not tracked in GradientMonitor
                "max": grad_stats.get("max", 0.0),
                "zero_ratio": 0.0,  # not tracked in GradientMonitor
                "pathology": pathology,
            }
            self._buffer.append(row)

        if self.buffer_size > 0 and len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> Optional[Path]:
        """Write buffered rows to disk and clear the buffer.

        Returns
        -------
        Path or None
            Path of the written file, or ``None`` if the buffer was empty.
        """
        if not self._buffer:
            return None

        out_path: Path
        if self.fmt == "json":
            out_path = self._flush_json()
        else:
            out_path = self._flush_parquet()

        self._buffer.clear()
        self._chunk_index += 1
        return out_path

    def load(self, combined: bool = True) -> "Any":
        """Load all persisted snapshots as a ``pandas.DataFrame``.

        Parameters
        ----------
        combined:
            If ``True`` (default), all chunk files are concatenated into a
            single DataFrame.  If ``False``, a list of DataFrames is
            returned (one per chunk file).

        Returns
        -------
        pandas.DataFrame or list[pandas.DataFrame]

        Raises
        ------
        ImportError
            If ``pandas`` is not installed.
        """
        try:
            import pandas as pd  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pandas is required to call GradientSnapshotStore.load(). "
                "Install it with: pip install pandas"
            ) from exc

        files = sorted(self.output_dir.glob(f"*.{self.fmt}"))
        if not files:
            return pd.DataFrame(columns=self._COLUMNS)

        if self.fmt == "json":
            frames = [pd.read_json(f) for f in files]
        else:
            frames = [pd.read_parquet(f) for f in files]

        if combined:
            return pd.concat(frames, ignore_index=True)
        return frames

    def load_json_raw(self) -> List[Dict[str, Any]]:
        """Load all JSON snapshots as a plain list of dicts.

        This is the zero-dependency alternative to :meth:`load` — it does
        not require ``pandas``.

        Returns
        -------
        list[dict]
        """
        records: List[Dict[str, Any]] = []
        for path in sorted(self.output_dir.glob("*.json")):
            with open(path) as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    records.extend(data)
        return records

    def summary(self) -> str:
        """Return a human-readable summary of stored snapshots."""
        files = list(self.output_dir.glob(f"*.{self.fmt}"))
        buffered = len(self._buffer)
        return (
            f"GradientSnapshotStore(dir={self.output_dir}, "
            f"fmt={self.fmt}, chunks={len(files)}, buffered_rows={buffered})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stats_to_row(
        step: int, stats: LayerGradientStats
    ) -> Dict[str, Any]:
        """Convert a :class:`LayerGradientStats` instance to a flat dict row."""
        return {
            "step": step,
            "layer_name": stats.layer_name,
            "layer_type": stats.layer_type,
            "depth": stats.depth,
            "group": stats.group.value,
            "grad_norm": stats.grad_norm,
            "mean": stats.mean,
            "std": stats.std,
            "min": stats.min,
            "max": stats.max,
            "zero_ratio": stats.zero_ratio,
            "pathology": stats.diagnose().value,
        }

    def _flush_json(self) -> Path:
        out_path = self.output_dir / f"snapshot_{self._chunk_index:05d}.json"
        with open(out_path, "w") as fh:
            json.dump(self._buffer, fh, indent=2)
        return out_path

    def _flush_parquet(self) -> Path:
        try:
            import pandas as pd  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pandas + pyarrow are required for Parquet output. "
                "Install with: pip install pandas pyarrow"
            ) from exc

        out_path = self.output_dir / f"snapshot_{self._chunk_index:05d}.parquet"
        df = pd.DataFrame(self._buffer, columns=self._COLUMNS)
        df.to_parquet(out_path, index=False)
        return out_path
