"""Phase 2 — Non-invasive gradient diagnosis via a Context Manager.

Public API
----------
::

    import gradient_pathology as gp

    # ── Option A: Context Manager (zero code change to train loop) ──
    with gp.watch(model) as watcher:
        train()                        # your existing loop, untouched
    # Report is printed automatically on __exit__.
    report  = watcher.report          # GradientReport
    findings = watcher.findings       # List[ExpertFinding]

    # ── Option B: explicit start/stop (e.g. inside a Trainer) ────────
    watcher = gp.watch(model, auto_print=False)
    watcher.start()
    ...                                # training
    watcher.stop()
    print(watcher.report.summary())

Design notes
------------
* Hook injection uses ``register_full_backward_hook`` (PyTorch ≥ 2.0).
  On older versions we fall back to ``register_backward_hook``.
* Hooks store the mean L2 norm of the incoming gradient tensor per module
  per step, kept in a bounded deque so memory stays constant regardless
  of training length.
* No monkey-patching of ``nn.Module`` is performed; hooks are removed
  cleanly in ``__exit__`` / ``stop()`` even if an exception is raised.
* Thread-safety: the deque append is GIL-protected and therefore safe
  for single-forward-pass-per-step patterns (the common case).
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from gradient_pathology.core import (
    GradientPathology,
    GradientReport,
    LayerGradientStats,
    LayerGroup,
)
from gradient_pathology.expert.engine import ExpertEngine, ExpertFinding
from gradient_pathology.pipeline.classifier import TransformerLayerClassifier


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _register_backward_hook(
    module: nn.Module,
    fn: Any,
) -> Any:
    """Register a full backward hook, falling back gracefully."""
    if hasattr(module, "register_full_backward_hook"):
        return module.register_full_backward_hook(fn)
    return module.register_backward_hook(fn)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# ModelWatcher
# ---------------------------------------------------------------------------

class ModelWatcher:
    """Non-invasive gradient watcher for a PyTorch model.

    Instantiate via :func:`watch` rather than directly.
    """

    # Maximum gradient-norm samples kept per layer (memory guard)
    _MAX_SAMPLES: int = 2_000

    def __init__(
        self,
        model: nn.Module,
        *,
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 1e3,
        auto_print: bool = True,
        report_format: str = "markdown",
        output_dir: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._vanishing_threshold = vanishing_threshold
        self._exploding_threshold = exploding_threshold
        self._auto_print = auto_print
        self._report_format = report_format
        self._output_dir = output_dir
        self._device = device

        # Per-module gradient norm history: module_name -> deque[float]
        self._grad_norms: Dict[str, Deque[float]] = {}
        self._hooks: List[Any] = []
        self._running: bool = False
        self._step: int = 0

        # Classifier for Transformer-aware layer metadata
        self._classifier = TransformerLayerClassifier(model)
        self._param_meta = self._classifier.build_param_metadata()

        # Results (populated after stop())
        self._report: Optional[GradientReport] = None
        self._findings: Optional[List[ExpertFinding]] = None

    # ------------------------------------------------------------------ #
    # Context manager protocol
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "ModelWatcher":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> bool:
        self.stop()
        if exc_type is None and self._auto_print:
            self._print_report()
        return False  # never suppress exceptions

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Inject backward hooks into every module that has parameters."""
        if self._running:
            return
        self._grad_norms.clear()
        self._hooks.clear()
        self._step = 0

        for name, module in self._model.named_modules():
            # Only modules that own at least one parameter are interesting
            if not list(module.parameters(recurse=False)):
                continue
            # Initialise the deque slot
            self._grad_norms[name] = deque(maxlen=self._MAX_SAMPLES)
            # Capture `name` by value in the closure
            hook = self._make_hook(name)
            handle = _register_backward_hook(module, hook)
            self._hooks.append(handle)

        self._running = True

    def stop(self) -> None:
        """Remove all hooks and build the final report."""
        if not self._running:
            return
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._running = False
        self._report, self._findings = self._build_report()

    # ------------------------------------------------------------------ #
    # Hook factory
    # ------------------------------------------------------------------ #

    def _make_hook(self, module_name: str) -> Any:
        """Return a closure that records the mean L2 grad norm for *module_name*."""

        def _hook(
            module: nn.Module,
            grad_input: Tuple[Optional[torch.Tensor], ...],
            grad_output: Tuple[Optional[torch.Tensor], ...],
        ) -> None:
            self._step += 1
            # Use the first non-None element from grad_input as the signal
            for g in (*grad_input, *grad_output):
                if g is not None and g.numel() > 0:
                    norm = float(g.detach().norm().item())
                    self._grad_norms[module_name].append(norm)
                    return

        return _hook

    # ------------------------------------------------------------------ #
    # Report builder
    # ------------------------------------------------------------------ #

    def _build_report(self) -> Tuple[GradientReport, List[ExpertFinding]]:
        """Aggregate per-module norms into a :class:`GradientReport`."""
        import numpy as np

        layer_stats: List[LayerGradientStats] = []
        all_norms: List[float] = []

        # Iterate in the same order as named_parameters so indices are stable
        param_names = [n for n, _ in self._model.named_parameters()]

        # Build a mapping: module_name -> (first_param_name_under_it)
        # We use this to pull layer_type / group from the param_meta dict.
        mod_to_param: Dict[str, str] = {}
        for pname in param_names:
            # The module name is everything up to the last '.' in the param name
            parts = pname.rsplit(".", 1)
            mod = parts[0] if len(parts) == 2 else ""
            if mod not in mod_to_param:
                mod_to_param[mod] = pname

        for idx, (mod_name, norms_deque) in enumerate(self._grad_norms.items()):
            norms = list(norms_deque)
            if not norms:
                continue
            arr = np.array(norms, dtype=float)
            grad_norm = float(np.mean(arr))
            all_norms.extend(norms)

            # Resolve metadata via the param_meta map
            rep_param = mod_to_param.get(mod_name, "")
            layer_type, group = self._param_meta.get(
                rep_param, ("unknown", LayerGroup.OTHER)
            )

            # Synthesise mean / std from the norm distribution
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))

            stats = LayerGradientStats(
                layer_name=mod_name or "<root>",
                layer_index=idx,
                mean=mean_val,
                std=std_val,
                min=float(np.min(arr)),
                max=float(np.max(arr)),
                median=float(np.median(arr)),
                num_zeros=int(np.sum(arr == 0.0)),
                total_params=len(arr),
                layer_type=layer_type,
                depth=idx,
                group=group,
                grad_norm=grad_norm,
            )
            layer_stats.append(stats)

        if all_norms:
            g_mean = float(np.mean(all_norms))
            g_std = float(np.std(all_norms))
        else:
            g_mean = g_std = 0.0

        report = GradientReport(
            layer_stats=layer_stats,
            global_mean=g_mean,
            global_std=g_std,
            num_steps=self._step,
            data_source="watch_hook",
        )

        engine = ExpertEngine(
            vanishing_threshold=self._vanishing_threshold,
            exploding_threshold=self._exploding_threshold,
        )
        findings = engine.analyse(report)
        return report, findings

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    def _print_report(self) -> None:
        """Print the report to stdout (and optionally save artefacts)."""
        if self._report is None:
            return
        findings = self._findings or []

        if self._report_format == "json":
            import json
            from gradient_pathology.cli.report import save_json_report
            from pathlib import Path
            out = Path(self._output_dir or "pathology_output")
            out.mkdir(parents=True, exist_ok=True)
            dest = save_json_report(self._report, findings, out)
            print(dest.read_text())
        else:
            from gradient_pathology.cli.report import render_markdown
            from pathlib import Path
            out = Path(self._output_dir or "pathology_output")
            out.mkdir(parents=True, exist_ok=True)
            print(render_markdown(self._report, findings, out))

        if self._output_dir:
            from gradient_pathology.cli.report import save_json_report, save_parquet_report
            from pathlib import Path
            out = Path(self._output_dir)
            out.mkdir(parents=True, exist_ok=True)
            save_json_report(self._report, findings, out)
            try:
                save_parquet_report(self._report, out)
            except ImportError:
                pass

    # ------------------------------------------------------------------ #
    # Public read-only properties
    # ------------------------------------------------------------------ #

    @property
    def report(self) -> Optional[GradientReport]:
        """The :class:`~gradient_pathology.core.GradientReport` after ``stop()``."""
        return self._report

    @property
    def findings(self) -> Optional[List[ExpertFinding]]:
        """The list of :class:`~gradient_pathology.expert.engine.ExpertFinding` after ``stop()``."""
        return self._findings

    @property
    def is_running(self) -> bool:
        """``True`` while hooks are active."""
        return self._running

    @property
    def step_count(self) -> int:
        """Number of backward-pass hook calls recorded so far."""
        return self._step

    def quick_summary(self) -> str:
        """One-line health string (requires ``stop()`` to have been called)."""
        if self._report is None or self._findings is None:
            return "⏳ Watcher not stopped yet — call .stop() first."
        engine = ExpertEngine(
            vanishing_threshold=self._vanishing_threshold,
            exploding_threshold=self._exploding_threshold,
        )
        return engine.quick_summary(self._report)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def watch(
    model: nn.Module,
    *,
    vanishing_threshold: float = 1e-7,
    exploding_threshold: float = 1e3,
    auto_print: bool = True,
    report_format: str = "markdown",
    output_dir: Optional[str] = None,
    device: str = "cpu",
) -> ModelWatcher:
    """Create a :class:`ModelWatcher` for *model*.

    Parameters
    ----------
    model:
        The ``nn.Module`` to watch.  No modifications are made to the model
        itself; hooks are attached and detached cleanly.
    vanishing_threshold:
        Layers whose mean gradient norm falls below this value are flagged
        as vanishing.  Default: ``1e-7``.
    exploding_threshold:
        Layers whose mean gradient norm exceeds this value are flagged as
        exploding.  Default: ``1e3``.
    auto_print:
        When used as a context manager, print the report on ``__exit__``.
        Default: ``True``.
    report_format:
        ``'markdown'`` (default) or ``'json'``.
    output_dir:
        If set, artefacts (``report.json``, ``layer_stats.parquet``) are
        written here in addition to stdout output.
    device:
        Device hint for future use.  Currently unused (hooks are
        device-agnostic).  Default: ``'cpu'``.

    Returns
    -------
    ModelWatcher
        A watcher instance that can be used as a context manager or
        controlled explicitly via :meth:`ModelWatcher.start` and
        :meth:`ModelWatcher.stop`.

    Examples
    --------
    Context manager::

        with gp.watch(model):
            train()                  # untouched training loop

    Explicit start/stop::

        watcher = gp.watch(model, auto_print=False)
        watcher.start()
        train()
        watcher.stop()
        print(watcher.quick_summary())

    Custom thresholds + save artefacts::

        with gp.watch(
            model,
            vanishing_threshold=1e-6,
            output_dir="./diagnostics",
        ):
            train()
    """
    return ModelWatcher(
        model,
        vanishing_threshold=vanishing_threshold,
        exploding_threshold=exploding_threshold,
        auto_print=auto_print,
        report_format=report_format,
        output_dir=output_dir,
        device=device,
    )
