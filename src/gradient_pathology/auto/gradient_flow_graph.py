"""Gradient flow graph — legacy visualiser, now extended with Heatmap integration.

The original :class:`GradientFlowGraph` API is preserved unchanged.  Two new
methods are added:

* :meth:`GradientFlowGraph.plot_heatmap` — build and return an interactive
  Plotly Heatmap using the Phase-2 :class:`~gradient_pathology.heatmap.GradientHeatmapRenderer`.
* :meth:`GradientFlowGraph.build_report` — convenience wrapper that runs
  :class:`~gradient_pathology.analyzer.GradientAnalyzer` on recorded flow data
  and returns a :class:`~gradient_pathology.core.GradientReport`.

All original behaviour (record_flow, plot_flow_simple, plot_flow_network,
find_bottlenecks) is kept intact so existing user code is not broken.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class GradientFlowGraph:
    """Visualise and analyse gradient flow through a PyTorch model.

    Parameters
    ----------
    model:
        The PyTorch model to inspect.

    Examples
    --------
    Legacy usage (unchanged)::

        gfg = GradientFlowGraph(model)
        gfg.record_flow(dataloader, loss_fn)
        gfg.plot_flow_simple()
        gfg.find_bottlenecks()

    Phase-2 Heatmap::

        gfg = GradientFlowGraph(model)
        gfg.record_flow(dataloader, loss_fn)
        fig = gfg.plot_heatmap()   # interactive Plotly figure
        fig.show()
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.flow_data: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Original API (preserved exactly)
    # ------------------------------------------------------------------

    def record_flow(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        num_samples: int = 5,
    ) -> None:
        """Record gradient flow statistics over *num_samples* batches."""
        gradient_norms: Dict[str, List[float]] = {
            name: [] for name, _ in self.model.named_parameters()
        }

        self.model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= num_samples:
                break

            self.model.zero_grad()
            output = self.model(data)
            loss = loss_fn(output, target)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    norm = param.grad.norm().item()
                    gradient_norms[name].append(norm)

        self.flow_data = {
            name: float(np.mean(norms)) if norms else 0.0
            for name, norms in gradient_norms.items()
        }

    def plot_flow_simple(self, save_path: Optional[str] = None) -> None:
        """Bar chart (log scale) of gradient norms per layer."""
        if not self.flow_data:
            print("먼저 record_flow()를 실행하세요")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        names  = list(self.flow_data.keys())
        values = list(self.flow_data.values())
        log_values = [np.log10(v + 1e-10) for v in values]
        colors = plt.cm.RdYlGn(
            np.array(log_values) / max(log_values) if max(log_values) != 0 else [0.5] * len(log_values)
        )
        ax.barh(range(len(names)), log_values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('그래디언트 크기 (log10)')
        ax.set_title('레이어별 그래디언트 흐름')
        ax.axvline(x=np.log10(1e-7), color='r', linestyle='--', label='매우 작음 (문제 가능)')
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_flow_network(self, save_path: Optional[str] = None) -> None:
        """NetworkX spring-layout network graph of gradient flow."""
        if not NETWORKX_AVAILABLE:
            print("NetworkX가 설치되지 않음. pip install networkx")
            return
        if not self.flow_data:
            print("먼저 record_flow()를 실행하세요")
            return

        G = nx.DiGraph()
        layer_names = list(self.flow_data.keys())
        for idx, name in enumerate(layer_names):
            G.add_node(idx, name=name, gradient=self.flow_data[name])
        for i in range(len(layer_names) - 1):
            G.add_edge(i + 1, i)

        fig, ax = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)
        node_colors = [
            np.log10(self.flow_data[layer_names[node]] + 1e-10)
            for node in G.nodes()
        ]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000,
                               cmap=plt.cm.RdYlGn, vmin=min(node_colors),
                               vmax=max(node_colors), ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, ax=ax)
        labels = {idx: name.split('.')[-1][:10] for idx, name in enumerate(layer_names)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        ax.set_title('그래디언트 흐름 네트워크 (화살표 = 역전파 방향)')
        ax.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def find_bottlenecks(self, threshold: float = 1e-7) -> List[str]:
        """Return list of layer names whose gradient norm is below *threshold*."""
        bottlenecks = [name for name, value in self.flow_data.items() if value < threshold]
        if bottlenecks:
            print("\n⚠️ 병목 지점 발견:")
            for name in bottlenecks:
                print(f"  - {name}: {self.flow_data[name]:.2e}")
            print("\n해결 방법:")
            print("  1. 해당 레이어의 학습률을 높이기")
            print("  2. 초기화 방법 바꾸기 (He, Xavier 등)")
            print("  3. BatchNorm 또는 LayerNorm 추가")
        else:
            print("✅ 병목 없음 - 그래디언트가 잘 흐르고 있음")
        return bottlenecks

    # ------------------------------------------------------------------
    # Phase-2 additions
    # ------------------------------------------------------------------

    def build_report(
        self,
        num_steps: int = 50,
        input_shape: Tuple[int, ...] = (10,),
    ) -> "GradientReport":  # type: ignore[return]
        """Run :class:`~gradient_pathology.analyzer.GradientAnalyzer` on the
        model and return a :class:`~gradient_pathology.core.GradientReport`.

        This is a convenience shortcut so you can go from
        :class:`GradientFlowGraph` directly to a
        :class:`~gradient_pathology.heatmap.GradientHeatmapRenderer` without
        importing the analyser separately.

        Parameters
        ----------
        num_steps:
            Number of synthetic forward/backward passes.
        input_shape:
            Input tensor shape (excluding batch dimension).

        Returns
        -------
        GradientReport
        """
        # Lazy import to avoid circular dependency.
        from gradient_pathology.analyzer import GradientAnalyzer  # noqa: PLC0415
        analyzer = GradientAnalyzer(self.model)
        return analyzer.diagnose(num_steps=num_steps, input_shape=input_shape)

    def plot_heatmap(
        self,
        report: Optional["GradientReport"] = None,  # type: ignore[name-defined]
        scheme: str = "viridis",
        layout: str = "sequential",
        vanishing_threshold: float = 1e-7,
        show_edges: bool = True,
        num_steps: int = 50,
        input_shape: Tuple[int, ...] = (10,),
    ) -> "go.Figure":  # type: ignore[return]
        """Build and return a Plotly Heatmap figure.

        If *report* is ``None`` the method calls :meth:`build_report` first
        using *num_steps* and *input_shape*.

        Parameters
        ----------
        report:
            Pre-computed :class:`~gradient_pathology.core.GradientReport`.
            Pass ``None`` to auto-compute.
        scheme:
            Colormap name — ``'viridis'`` or ``'rdylgn'``.
        layout:
            Layout strategy — ``'sequential'``, ``'grouped'``, or ``'spring'``.
        vanishing_threshold:
            Threshold below which layers are flagged as vanishing.
        show_edges:
            Whether to draw gradient-flow arrows.
        num_steps:
            Passed to :meth:`build_report` when *report* is ``None``.
        input_shape:
            Passed to :meth:`build_report` when *report* is ``None``.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from gradient_pathology.heatmap import GradientHeatmapRenderer  # noqa: PLC0415
        from gradient_pathology.heatmap.colormap import ColorScheme     # noqa: PLC0415
        from gradient_pathology.heatmap.layout import LayoutStrategy    # noqa: PLC0415

        if report is None:
            report = self.build_report(num_steps=num_steps, input_shape=input_shape)

        scheme_map = {"viridis": ColorScheme.VIRIDIS, "rdylgn": ColorScheme.RDYLGN}
        layout_map = {
            "sequential": LayoutStrategy.SEQUENTIAL,
            "grouped":    LayoutStrategy.GROUPED,
            "spring":     LayoutStrategy.SPRING,
        }

        renderer = GradientHeatmapRenderer(
            report,
            scheme=scheme_map.get(scheme.lower(), ColorScheme.VIRIDIS),
            layout_strategy=layout_map.get(layout.lower(), LayoutStrategy.SEQUENTIAL),
            vanishing_threshold=vanishing_threshold,
            show_edges=show_edges,
        )
        return renderer.build()
