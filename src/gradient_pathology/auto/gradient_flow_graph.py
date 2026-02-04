"""그래디언트가 어떻게 흘러가는지 그래프로 시각화.

쉬운 설명:
- 신경망은 레이어들이 연결된 그래프
- 그래디언트는 뒤에서 앞으로 흐름 (역전파)
- 어디서 막히는지, 어디가 중요한지 그림으로 보여줌
- 병목 지점을 찾아서 고칠 수 있음
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class GradientFlowGraph:
    """그래디언트 흐름을 그래프로 시각화."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.flow_data: Dict[str, float] = {}

    def record_flow(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        num_samples: int = 5,
    ) -> None:
        """그래디언트 흐름 기록."""
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
            
            # 각 레이어의 그래디언트 크기 기록
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    norm = param.grad.norm().item()
                    gradient_norms[name].append(norm)
        
        # 평균 계산
        self.flow_data = {
            name: np.mean(norms) 
            for name, norms in gradient_norms.items()
        }

    def plot_flow_simple(self, save_path: str = None) -> None:
        """간단한 막대 그래프로 시각화."""
        if not self.flow_data:
            print("먼저 record_flow()를 실행하세요")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        names = list(self.flow_data.keys())
        values = list(self.flow_data.values())
        
        # 로그 스케일로 변환
        log_values = [np.log10(v + 1e-10) for v in values]
        
        # 색상: 작으면 빨강, 크면 초록
        colors = plt.cm.RdYlGn(np.array(log_values) / max(log_values))
        
        ax.barh(range(len(names)), log_values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('그래디언트 크기 (log10)')
        ax.set_title('레이어별 그래디언트 흐름')
        ax.axvline(x=np.log10(1e-7), color='r', linestyle='--', 
                   label='매우 작음 (문제 가능)')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()

    def plot_flow_network(self, save_path: str = None) -> None:
        """NetworkX로 네트워크 그래프 시각화."""
        if not NETWORKX_AVAILABLE:
            print("NetworkX가 설치되지 않음. pip install networkx")
            return
        
        if not self.flow_data:
            print("먼저 record_flow()를 실행하세요")
            return
        
        G = nx.DiGraph()
        
        # 노드 추가 (레이어)
        layer_names = list(self.flow_data.keys())
        for idx, name in enumerate(layer_names):
            G.add_node(idx, name=name, gradient=self.flow_data[name])
        
        # 엣지 추가 (레이어 간 연결)
        for i in range(len(layer_names) - 1):
            G.add_edge(i + 1, i)  # 역전파 방향
        
        # 시각화
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 레이아웃
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 노드 색상 (그래디언트 크기)
        node_colors = [
            np.log10(self.flow_data[layer_names[node]] + 1e-10)
            for node in G.nodes()
        ]
        
        nx.draw_networkx_nodes(
            G, pos, 
            node_color=node_colors,
            node_size=1000,
            cmap=plt.cm.RdYlGn,
            vmin=min(node_colors),
            vmax=max(node_colors),
            ax=ax,
        )
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            ax=ax,
        )
        
        # 레이블 (짧게)
        labels = {
            idx: name.split('.')[-1][:10]
            for idx, name in enumerate(layer_names)
        }
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title('그래디언트 흐름 네트워크 (화살표 = 역전파 방향)')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()

    def find_bottlenecks(self, threshold: float = 1e-7) -> List[str]:
        """병목 지점 찾기."""
        bottlenecks = [
            name 
            for name, value in self.flow_data.items()
            if value < threshold
        ]
        
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
