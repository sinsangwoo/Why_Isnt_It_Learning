"""Thread-safe ring-buffer bridge: training loop → Streamlit session_state.

Design goals
------------
* **Zero blocking** — the training loop must never wait for Streamlit.  All
  writes are protected by a :class:`threading.Lock` and complete in O(1).
* **Fixed memory** — a ring-buffer of at most *max_steps* snapshots keeps
  the dashboard responsive regardless of training duration.
* **Streamlit-compatible** — the bridge stores itself in
  ``st.session_state`` under a user-chosen key so all reruns can read the
  latest data without re-creating the object.
* **Standalone** — the bridge has *no* Streamlit import so it can be used
  in pure-Python training scripts and unit-tested without a browser.

Data model
----------
Each :class:`GradientSnapshot` captures one training step::

    step          int       global step index
    loss          float     scalar training loss (or NaN)
    global_mean   float     mean |gradient| across all parameters
    global_std    float     std  |gradient| across all parameters
    layer_norms   dict      {param_name: mean_abs_grad}  (latest window avg)
    alerts        list[str] pathology messages produced this step
    timestamp     float     wall-clock time (time.monotonic)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np


@dataclass
class GradientSnapshot:
    """Immutable record of gradient statistics at one training step."""

    step:        int
    loss:        float
    global_mean: float
    global_std:  float
    layer_norms: Dict[str, float]  = field(default_factory=dict)
    alerts:      List[str]         = field(default_factory=list)
    timestamp:   float             = field(default_factory=time.monotonic)


class LiveGradientBridge:
    """Thread-safe ring-buffer that connects a training loop to Streamlit.

    Parameters
    ----------
    max_steps:
        Maximum number of :class:`GradientSnapshot` objects retained.
        Oldest entries are discarded automatically when the buffer is full.
    alert_threshold:
        Gradient mean below this value triggers a vanishing-gradient alert.
    explode_threshold:
        Gradient mean above this value triggers an exploding-gradient alert.

    Thread safety
    -------------
    * :meth:`push` is called from the **training thread** (or main thread).
    * :meth:`latest_snapshot`, :meth:`all_snapshots`, and :meth:`metrics_series`
      are called from the **Streamlit rerun thread**.
    * All public methods acquire ``_lock`` before reading or writing shared
      state, guaranteeing consistency without busy-waiting.

    Examples
    --------
    ::

        bridge = LiveGradientBridge(max_steps=200)

        # training loop
        for step, (x, y) in enumerate(loader):
            loss = model(x) ; loss.backward()
            bridge.push(step=step, loss=loss.item(), model=model)

        # Streamlit tab
        snap = bridge.latest_snapshot()
        if snap:
            st.metric("Loss", f"{snap.loss:.4f}")
    """

    def __init__(
        self,
        max_steps: int        = 500,
        alert_threshold: float = 1e-7,
        explode_threshold: float = 1e3,
    ) -> None:
        self._max_steps         = max_steps
        self._alert_threshold   = alert_threshold
        self._explode_threshold = explode_threshold
        self._lock: threading.Lock       = threading.Lock()
        self._buffer: Deque[GradientSnapshot] = deque(maxlen=max_steps)
        self._pending_alerts: List[str]  = []
        self._total_pushed: int          = 0

    # ------------------------------------------------------------------
    # Write API (training thread)
    # ------------------------------------------------------------------

    def push(
        self,
        step:  int,
        loss:  float,
        model: Optional[Any] = None,   # nn.Module
        layer_norms: Optional[Dict[str, float]] = None,
        extra_alerts: Optional[List[str]] = None,
    ) -> GradientSnapshot:
        """Collect gradient stats from *model* and append a snapshot.

        Parameters
        ----------
        step:
            Global training step index.
        loss:
            Scalar training loss for this step.
        model:
            ``torch.nn.Module`` whose ``.grad`` tensors will be sampled.
            If ``None``, *layer_norms* must be supplied.
        layer_norms:
            Pre-computed ``{param_name: mean_abs_grad}`` dict.  Used when
            the caller has already computed norms (e.g. from
            :class:`~gradient_pathology.callbacks.GradientMonitor`).
        extra_alerts:
            Additional alert strings to attach to this snapshot.

        Returns
        -------
        GradientSnapshot
            The snapshot that was just pushed.
        """
        norms: Dict[str, float] = {}
        alerts: List[str] = list(extra_alerts or [])

        if model is not None:
            try:
                import torch  # local import — optional outside training
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        g = param.grad.detach().cpu()
                        m = float(g.abs().mean().item())
                        norms[name] = m
                        if m < self._alert_threshold:
                            alerts.append(
                                f"Vanishing gradient: {name} (mean={m:.2e})"
                            )
                        elif m > self._explode_threshold:
                            alerts.append(
                                f"Exploding gradient: {name} (mean={m:.2e})"
                            )
            except Exception:  # pragma: no cover
                pass
        elif layer_norms is not None:
            norms = dict(layer_norms)

        if norms:
            vals        = np.array(list(norms.values()), dtype=float)
            global_mean = float(vals.mean())
            global_std  = float(vals.std())
        else:
            global_mean = float("nan")
            global_std  = float("nan")

        snap = GradientSnapshot(
            step=step,
            loss=float(loss),
            global_mean=global_mean,
            global_std=global_std,
            layer_norms=norms,
            alerts=alerts,
        )

        with self._lock:
            self._buffer.append(snap)
            self._pending_alerts.extend(alerts)
            self._total_pushed += 1

        return snap

    def clear(self) -> None:
        """Reset the ring-buffer (call between training runs)."""
        with self._lock:
            self._buffer.clear()
            self._pending_alerts.clear()
            self._total_pushed = 0

    # ------------------------------------------------------------------
    # Read API (Streamlit rerun thread)
    # ------------------------------------------------------------------

    def latest_snapshot(self) -> Optional[GradientSnapshot]:
        """Return the most recently pushed snapshot, or ``None``."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def all_snapshots(self) -> List[GradientSnapshot]:
        """Return a defensive copy of all retained snapshots."""
        with self._lock:
            return list(self._buffer)

    def metrics_series(
        self,
        field: str = "loss",
    ) -> tuple:
        """Return ``(steps, values)`` arrays for a given snapshot field.

        Parameters
        ----------
        field:
            One of: ``'loss'``, ``'global_mean'``, ``'global_std'``.

        Returns
        -------
        tuple[list[int], list[float]]
        """
        snaps = self.all_snapshots()
        steps  = [s.step         for s in snaps]
        values = [getattr(s, field, float("nan")) for s in snaps]
        return steps, values

    def drain_alerts(self) -> List[str]:
        """Return and clear all pending alert strings."""
        with self._lock:
            alerts = list(self._pending_alerts)
            self._pending_alerts.clear()
        return alerts

    @property
    def total_pushed(self) -> int:
        """Total snapshots ever pushed (not capped by ring-buffer size)."""
        with self._lock:
            return self._total_pushed

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0

    # ------------------------------------------------------------------
    # Streamlit session_state helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_session_state(
        cls,
        key: str = "_gp_bridge",
        **kwargs: Any,
    ) -> "LiveGradientBridge":
        """Return the bridge stored in ``st.session_state[key]``, creating
        one if it doesn’t exist yet.

        This is the recommended way to obtain a bridge inside a Streamlit
        app: call this once at the top of the page and pass the returned
        object to both the training callback and the dashboard tab.

        Parameters
        ----------
        key:
            ``st.session_state`` key to use for persistence.
        **kwargs:
            Passed to the constructor when a new bridge is created.

        Returns
        -------
        LiveGradientBridge
        """
        try:
            import streamlit as st  # type: ignore[import]
            if key not in st.session_state:
                st.session_state[key] = cls(**kwargs)
            return st.session_state[key]  # type: ignore[return-value]
        except ImportError:
            return cls(**kwargs)

    def __repr__(self) -> str:
        with self._lock:
            n = len(self._buffer)
        return (
            f"LiveGradientBridge("
            f"buffered={n}/{self._max_steps}, "
            f"total_pushed={self._total_pushed})"
        )
