"""Phase-4 real-time monitoring bridge.

Connects :class:`~gradient_pathology.callbacks.GradientMonitor` to the
Streamlit session state so that the live dashboard tab auto-refreshes
as training progresses.

Public surface::

    from gradient_pathology.monitor import LiveGradientBridge, StreamlitCallback

    bridge   = LiveGradientBridge()
    callback = StreamlitCallback(model, bridge=bridge)

    for step, (x, y) in enumerate(loader):
        loss = criterion(model(x), y)
        loss.backward()
        callback.on_batch_end(optimizer, loss=loss.item(), step=step)
"""

from gradient_pathology.monitor.bridge import LiveGradientBridge
from gradient_pathology.monitor.callback import StreamlitCallback

__all__ = ["LiveGradientBridge", "StreamlitCallback"]
