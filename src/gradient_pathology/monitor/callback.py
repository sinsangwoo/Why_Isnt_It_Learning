"""Phase-4 training callbacks that push to :class:`LiveGradientBridge`.

Two adapters are provided:

:class:`StreamlitCallback`
    Vanilla PyTorch training-loop callback.  Call :meth:`on_batch_end`
    once per gradient step *after* ``loss.backward()``.

:class:`HuggingFaceCallbackAdapter`
    Wraps ``StreamlitCallback`` as a HuggingFace
    ``transformers.TrainerCallback`` so it drops straight into any
    ``Trainer``-based workflow with zero boilerplate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch.nn as nn

from gradient_pathology.monitor.bridge import LiveGradientBridge


class StreamlitCallback:
    """Vanilla PyTorch callback that pushes gradient snapshots to a bridge.

    Parameters
    ----------
    model:
        The model being trained (must have ``.named_parameters()``).
    bridge:
        :class:`LiveGradientBridge` instance shared with the dashboard.
    push_every_n_steps:
        Only push every *n* steps to reduce overhead.  Defaults to 1
        (push on every step).

    Examples
    --------
    ::

        bridge   = LiveGradientBridge(max_steps=300)
        callback = StreamlitCallback(model, bridge, push_every_n_steps=5)

        for step, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            callback.on_batch_end(step=step, loss=loss.item())
    """

    def __init__(
        self,
        model:                nn.Module,
        bridge:               LiveGradientBridge,
        push_every_n_steps:   int = 1,
    ) -> None:
        self.model               = model
        self.bridge              = bridge
        self.push_every_n_steps  = max(1, push_every_n_steps)
        self._step_counter: int  = 0

    def on_batch_end(
        self,
        step: int,
        loss: float,
        extra_alerts: Optional[List[str]] = None,
    ) -> None:
        """Call *after* ``optimizer.step()`` (or at least after ``backward()``).

        Parameters
        ----------
        step:
            Global training step index.
        loss:
            Scalar loss value for this step.
        extra_alerts:
            Optional additional alert strings to attach to this snapshot.
        """
        self._step_counter += 1
        if self._step_counter % self.push_every_n_steps != 0:
            return
        self.bridge.push(
            step=step,
            loss=loss,
            model=self.model,
            extra_alerts=extra_alerts,
        )

    def reset(self) -> None:
        """Reset internal step counter (call between epochs or runs)."""
        self._step_counter = 0


class HuggingFaceCallbackAdapter:
    """Wraps :class:`StreamlitCallback` as a HuggingFace ``TrainerCallback``.

    The class intentionally does **not** import
    ``transformers.TrainerCallback`` at module level so that
    ``gradient-pathology`` can be installed without the HuggingFace
    ecosystem.  The HF base class is resolved lazily at instantiation time.

    Parameters
    ----------
    model:
        The model passed to ``Trainer``.  Can also be ``None`` when used
        with a Trainer that sets ``self.model`` after init; the adapter
        will pick it up from the Trainer state at the first callback call.
    bridge:
        Shared :class:`LiveGradientBridge`.
    push_every_n_steps:
        Push frequency.

    Examples
    --------
    ::

        from transformers import Trainer, TrainingArguments
        from gradient_pathology.monitor import LiveGradientBridge, HuggingFaceCallbackAdapter

        bridge    = LiveGradientBridge()
        hf_cb     = HuggingFaceCallbackAdapter(model=None, bridge=bridge)
        trainer   = Trainer(
            model=model,
            args=TrainingArguments(...),
            callbacks=[hf_cb],
        )
        trainer.train()
    """

    def __init__(
        self,
        model:               Optional[nn.Module],
        bridge:              LiveGradientBridge,
        push_every_n_steps:  int = 1,
    ) -> None:
        # Lazy-resolve the HF base class
        try:
            from transformers import TrainerCallback  # type: ignore[import]
            self.__class__ = type(
                "_HFAdapter",
                (HuggingFaceCallbackAdapter, TrainerCallback),
                {},
            )
        except ImportError:
            pass  # Used in non-HF environments; on_log will never be called

        self._model              = model
        self.bridge              = bridge
        self.push_every_n_steps  = max(1, push_every_n_steps)

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """HuggingFace ``TrainerCallback.on_log`` hook."""
        if state.global_step % self.push_every_n_steps != 0:
            return

        model = self._model or kwargs.get("model")
        loss  = (logs or {}).get("loss", float("nan"))

        self.bridge.push(
            step=state.global_step,
            loss=loss,
            model=model,
        )

    def on_train_begin(self, *args: Any, **kwargs: Any) -> None:
        """Clear the bridge when a new Trainer.train() call starts."""
        self.bridge.clear()
