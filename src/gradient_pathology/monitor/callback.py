"""StreamlitCallback — connects the training loop to :class:`LiveGradientBridge`.

Usage in a plain PyTorch training loop::

    from gradient_pathology.monitor import LiveGradientBridge, StreamlitCallback

    bridge   = LiveGradientBridge(max_steps=300)
    callback = StreamlitCallback(
        model,
        bridge=bridge,
        report_every_n_steps=25,
        alert_threshold=1e-7,
    )

    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        callback.on_batch_end(optimizer, loss=loss.item(), step=step)

    callback.on_train_end()

Usage with HuggingFace Trainer::

    from gradient_pathology.monitor import StreamlitCallback
    trainer = Trainer(
        model=model,
        callbacks=[StreamlitCallback.as_hf_callback(bridge=bridge)],
        ...)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from gradient_pathology.monitor.bridge import LiveGradientBridge, get_global_bridge


class StreamlitCallback:
    """Collects gradients after each backward pass and forwards them to
    :class:`LiveGradientBridge`.

    Parameters
    ----------
    model:
        The PyTorch model being trained.
    bridge:
        :class:`LiveGradientBridge` instance to push data to.  When
        ``None``, the module-level global bridge is used (or created).
    report_every_n_steps:
        How often to build a full :class:`~gradient_pathology.core.GradientReport`
        and push it to the bridge.  Building a full report is more
        expensive than recording a snapshot, so set this to a value
        that balances responsiveness vs. overhead (default: 20).
    alert_threshold:
        Gradient mean below this value triggers a vanishing-gradient
        alert in the bridge's alert queue.
    check_every_n_steps:
        How frequently to check for alerts (default: 1 = every step).
    """

    def __init__(
        self,
        model: nn.Module,
        bridge: Optional[LiveGradientBridge] = None,
        report_every_n_steps: int  = 20,
        alert_threshold: float     = 1e-7,
        check_every_n_steps: int   = 1,
    ) -> None:
        self.model                 = model
        self.bridge                = bridge or get_global_bridge()
        self.report_every_n_steps  = report_every_n_steps
        self.alert_threshold       = alert_threshold
        self.check_every_n_steps   = check_every_n_steps
        self._step_count:  int     = 0

    # ------------------------------------------------------------------
    # Primary hook — call after loss.backward() and optimizer.step()
    # ------------------------------------------------------------------

    def on_batch_end(
        self,
        optimizer: Optional[Any] = None,
        *,
        loss: float = 0.0,
        step: Optional[int] = None,
    ) -> None:
        """Record gradients and push to bridge.

        Parameters
        ----------
        optimizer:
            The optimizer (unused at the moment, kept for API symmetry).
        loss:
            Scalar loss value for logging.
        step:
            Explicit step index.  When ``None``, an internal counter is used.
        """
        if step is None:
            step = self._step_count
        self._step_count += 1

        # Collect per-layer gradient snapshot.
        snapshot = self._collect_snapshot()

        # Push raw snapshot + loss to bridge.
        self.bridge.push_step(step=step, loss=loss, grad_snapshot=snapshot)

        # Periodic alert check.
        if step % self.check_every_n_steps == 0:
            self._check_alerts(snapshot)

        # Periodic full report build.
        if step % self.report_every_n_steps == 0 and step > 0:
            self._rebuild_report()

    def on_train_end(self) -> None:
        """Signal training completion to the bridge."""
        self._rebuild_report()  # final report
        self.bridge.signal_done()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Read current .grad tensors and return a snapshot dict."""
        snap: Dict[str, Dict[str, float]] = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                g = param.grad.detach().cpu().numpy()
                snap[name] = {
                    "mean": float(np.mean(np.abs(g))),
                    "std":  float(np.std(g)),
                    "max":  float(np.max(np.abs(g))),
                }
        return snap

    def _check_alerts(self, snapshot: Dict[str, Dict[str, float]]) -> None:
        """Enqueue alerts for any pathological gradients."""
        for name, stats in snapshot.items():
            mean = stats.get("mean", 0.0)
            if mean < self.alert_threshold:
                self.bridge.push_alert(
                    f"🔴 VANISHING | {name} | mean={mean:.2e}"
                )
            elif mean > 1e3:
                self.bridge.push_alert(
                    f"🟠 EXPLODING | {name} | mean={mean:.2e}"
                )

    def _rebuild_report(self) -> None:
        """Build a GradientReport from recent history and push to bridge."""
        from gradient_pathology.analyzer import GradientAnalyzer  # lazy import
        try:
            analyzer = GradientAnalyzer(self.model)
            report   = analyzer.diagnose(
                num_steps=min(self.report_every_n_steps, 10),
                input_shape=(10,),
            )
            self.bridge.push_report(report)
        except Exception:  # noqa: BLE001
            pass  # never crash the training loop

    # ------------------------------------------------------------------
    # HuggingFace Trainer compatibility shim
    # ------------------------------------------------------------------

    @classmethod
    def as_hf_callback(cls, **kwargs: Any) -> Any:
        """Return a HuggingFace-compatible ``TrainerCallback`` wrapper.

        Requires ``transformers`` to be installed.

        Parameters
        ----------
        **kwargs:
            Passed verbatim to :class:`StreamlitCallback.__init__` *except*
            ``model`` (which is injected at runtime by the Trainer).
        """
        try:
            from transformers import TrainerCallback  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "transformers is required for StreamlitCallback.as_hf_callback(). "
                "Install with: pip install transformers"
            ) from exc

        outer_kwargs = kwargs

        class _HFAdapter(TrainerCallback):
            def __init__(self) -> None:
                self._cb: Optional[StreamlitCallback] = None

            def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None, **kw: Any) -> None:  # type: ignore[override]
                if model is not None:
                    self._cb = StreamlitCallback(model, **outer_kwargs)

            def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:  # type: ignore[override]
                if self._cb is not None:
                    loss = kw.get("logs", {}).get("loss", 0.0)
                    self._cb.on_batch_end(loss=float(loss), step=state.global_step)

            def on_train_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:  # type: ignore[override]
                if self._cb is not None:
                    self._cb.on_train_end()

        return _HFAdapter()
