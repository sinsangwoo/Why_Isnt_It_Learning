"""Phase-4 real-time monitoring bridge.

This package connects a live PyTorch training loop to the Streamlit dashboard
so that gradient statistics, loss curves, and Expert-System alerts update
automatically as training progresses.

Public surface::

    from gradient_pathology.monitor import (
        LiveGradientBridge,
        StreamlitCallback,
        HuggingFaceCallbackAdapter,
    )

    # --- training side ---
    bridge = LiveGradientBridge(max_steps=500)
    callback = StreamlitCallback(model, bridge)

    for step, (x, y) in enumerate(loader):
        loss = train_step(x, y)
        callback.on_batch_end(step=step, loss=loss.item())

    # --- dashboard side (inside render_realtime_tab) ---
    snapshot = bridge.latest_snapshot()   # thread-safe read
"""

from gradient_pathology.monitor.bridge import LiveGradientBridge, GradientSnapshot
from gradient_pathology.monitor.callback import StreamlitCallback, HuggingFaceCallbackAdapter

__all__ = [
    "LiveGradientBridge",
    "GradientSnapshot",
    "StreamlitCallback",
    "HuggingFaceCallbackAdapter",
]
