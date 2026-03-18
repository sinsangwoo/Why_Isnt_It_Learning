"""LiveGradientBridge — thread-safe in-memory store shared between the
training loop and the Streamlit refresh cycle.

Streamlit re-runs the entire script on each interaction, so state must
live somewhere persistent.  We use two mechanisms in parallel:

1. **Module-level singleton** (`_GLOBAL_BRIDGE`): works when training and
   the dashboard share the *same Python process* (e.g. a notebook or a
   ``multiprocessing.Process`` with shared memory).
2. **``st.session_state`` injection** (optional): the bridge can push
   snapshots directly into Streamlit's session state dict so that the UI
   tab reflects the latest data without a page reload.

Data model
----------
The bridge accumulates three ring buffers:

* ``step_history``    — list of step indices (capped to *max_steps*).
* ``loss_history``    — list of scalar loss values.
* ``grad_snapshots``  — list of dicts ``{layer_name: {mean, std, max}}``
  (one entry per recorded step, same format as
  :meth:`~gradient_pathology.callbacks.GradientMonitor.record_step`).

It also stores:

* ``current_report``  — the most recent
  :class:`~gradient_pathology.core.GradientReport`, rebuilt every
  *report_every_n_steps* steps.
* ``alert_queue``     — list of unacknowledged alert strings.
* ``is_training``     — bool flag, set ``True`` when a callback starts
  and ``False`` when it signals training completion.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional


class LiveGradientBridge:
    """Thread-safe shared memory between a training loop and Streamlit.

    Parameters
    ----------
    max_steps:
        Maximum number of steps kept in the ring buffers.  Older entries
        are dropped automatically to keep memory bounded.
    report_every_n_steps:
        How frequently (in steps) a fresh
        :class:`~gradient_pathology.core.GradientReport` is built from
        the accumulated history.

    Examples
    --------
    ::

        bridge = LiveGradientBridge(max_steps=200)
        # pass *bridge* to StreamlitCallback; Streamlit dashboard reads from it
    """

    def __init__(
        self,
        max_steps: int = 500,
        report_every_n_steps: int = 20,
    ) -> None:
        self.max_steps             = max_steps
        self.report_every_n_steps  = report_every_n_steps

        self._lock: threading.Lock = threading.Lock()

        self.step_history:   Deque[int]              = deque(maxlen=max_steps)
        self.loss_history:   Deque[float]            = deque(maxlen=max_steps)
        self.grad_snapshots: Deque[Dict[str, Any]]   = deque(maxlen=max_steps)

        self.current_report: Optional[Any]           = None   # GradientReport
        self.alert_queue:    List[str]               = []
        self.is_training:    bool                    = False
        self._total_steps:   int                     = 0

    # ------------------------------------------------------------------
    # Write side (called from training thread)
    # ------------------------------------------------------------------

    def push_step(
        self,
        step: int,
        loss: float,
        grad_snapshot: Dict[str, Dict[str, float]],
    ) -> None:
        """Record one training step.

        Parameters
        ----------
        step:
            Global step index.
        loss:
            Scalar loss value for this step.
        grad_snapshot:
            Dict mapping ``layer_name`` to ``{mean, std, max}`` as
            produced by
            :meth:`~gradient_pathology.callbacks.GradientMonitor.record_step`.
        """
        with self._lock:
            self.step_history.append(step)
            self.loss_history.append(float(loss))
            self.grad_snapshots.append(grad_snapshot)
            self._total_steps += 1
            self.is_training = True

    def push_report(self, report: Any) -> None:
        """Store a freshly built :class:`~gradient_pathology.core.GradientReport`."""
        with self._lock:
            self.current_report = report

    def push_alert(self, message: str) -> None:
        """Enqueue an alert string for the UI to consume."""
        with self._lock:
            self.alert_queue.append(message)
            # Cap to 50 unread alerts to avoid unbounded growth.
            if len(self.alert_queue) > 50:
                self.alert_queue.pop(0)

    def signal_done(self) -> None:
        """Signal that training has finished."""
        with self._lock:
            self.is_training = False

    def clear(self) -> None:
        """Reset all buffers (useful for a new training run)."""
        with self._lock:
            self.step_history.clear()
            self.loss_history.clear()
            self.grad_snapshots.clear()
            self.current_report = None
            self.alert_queue.clear()
            self.is_training    = False
            self._total_steps   = 0

    # ------------------------------------------------------------------
    # Read side (called from Streamlit render thread)
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a consistent point-in-time snapshot of all bridge state.

        The returned dict is safe to pass around; it holds copies of the
        ring-buffer lists so subsequent mutations do not affect it.

        Returns
        -------
        dict with keys:
            ``steps``         list[int]
            ``losses``        list[float]
            ``grad_snapshots``list[dict]
            ``current_report`` GradientReport or None
            ``alerts``        list[str]  (copy; alerts are NOT consumed)
            ``is_training``   bool
            ``total_steps``   int
        """
        with self._lock:
            return {
                "steps":          list(self.step_history),
                "losses":         list(self.loss_history),
                "grad_snapshots": list(self.grad_snapshots),
                "current_report": self.current_report,
                "alerts":         list(self.alert_queue),
                "is_training":    self.is_training,
                "total_steps":    self._total_steps,
            }

    def pop_alerts(self) -> List[str]:
        """Return and *clear* the alert queue."""
        with self._lock:
            alerts = list(self.alert_queue)
            self.alert_queue.clear()
            return alerts

    @property
    def total_steps(self) -> int:
        """Total steps recorded since last :meth:`clear`."""
        return self._total_steps

    def inject_session_state(self, st_session: Dict[str, Any]) -> None:
        """Push the current snapshot into a Streamlit ``session_state`` dict.

        Call this inside the training loop after each step if you want the
        Streamlit UI to pick up changes without an explicit page reload.

        Parameters
        ----------
        st_session:
            Typically ``streamlit.session_state``.
        """
        snap = self.snapshot()
        st_session["live_steps"]          = snap["steps"]
        st_session["live_losses"]          = snap["losses"]
        st_session["live_grad_snapshots"]  = snap["grad_snapshots"]
        st_session["live_report"]          = snap["current_report"]
        st_session["live_alerts"]          = snap["alerts"]
        st_session["live_is_training"]     = snap["is_training"]
        st_session["live_total_steps"]     = snap["total_steps"]


# ---------------------------------------------------------------------------
# Module-level singleton (convenience for same-process usage)
# ---------------------------------------------------------------------------

_GLOBAL_BRIDGE: Optional[LiveGradientBridge] = None


def get_global_bridge(
    max_steps: int = 500,
    report_every_n_steps: int = 20,
) -> LiveGradientBridge:
    """Return the process-global :class:`LiveGradientBridge`, creating it if needed."""
    global _GLOBAL_BRIDGE
    if _GLOBAL_BRIDGE is None:
        _GLOBAL_BRIDGE = LiveGradientBridge(
            max_steps=max_steps,
            report_every_n_steps=report_every_n_steps,
        )
    return _GLOBAL_BRIDGE


def reset_global_bridge() -> None:
    """Destroy the global bridge (useful in tests)."""
    global _GLOBAL_BRIDGE
    _GLOBAL_BRIDGE = None
