"""그래디언트가 실제로 몇 개 방향으로 업데이트되는지 측정.

쉬운 설명:
- 모델 파라미터가 100만개 있어도, 실제로는 10개 방향으로만 움직일 수 있음
- 이걸 '유효 차원(Effective Rank)'이라고 부름
- 낮으면 = 학습이 비효율적 (파라미터 낭비)
- 높으면 = 학습이 효율적 (파라미터 잘 활용)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List


class EffectiveRankAnalyzer:
    """그래디언트의 유효 차원 측정."""

    def __init__(self, model: nn.Module):
        self.model = model

    def compute_effective_rank(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        num_samples: int = 10,
    ) -> dict:
        """유효 차원 계산.
        
        Args:
            dataloader: 학습 데이터
            loss_fn: 손실 함수
            num_samples: 계산할 배치 개수
            
        Returns:
            결과 딕셔너리
        """
        # 여러 배치에서 그래디언트 수집
        gradients: List[torch.Tensor] = []
        
        self.model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
            
            self.model.zero_grad()
            output = self.model(data)
            loss = loss_fn(output, target)
            loss.backward()
            
            # 모든 레이어 그래디언트를 하나로 합침
            grad_vector = torch.cat([
                p.grad.flatten() 
                for p in self.model.parameters() 
                if p.grad is not None
            ])
            
            gradients.append(grad_vector.detach().cpu())
        
        # 그래디언트들을 행렬로 만듦 [num_samples, num_params]
        grad_matrix = torch.stack(gradients).numpy()
        
        # 특이값 분해 (SVD) - 수학적으로 "주요 방향" 찾기
        # 여기서 singular values = 각 방향의 중요도
        U, singular_values, Vt = np.linalg.svd(grad_matrix, full_matrices=False)
        
        # 유효 차원 계산 (Shannon entropy 방식)
        # 수식: exp(-sum(p * log(p))) where p = normalized singular values
        sv_normalized = singular_values / singular_values.sum()
        sv_normalized = sv_normalized[sv_normalized > 0]  # log(0) 방지
        
        entropy = -np.sum(sv_normalized * np.log(sv_normalized))
        effective_rank = int(np.exp(entropy))
        
        # 상위 90% 에너지를 차지하는 차원 수
        cumsum = np.cumsum(sv_normalized)
        rank_90 = int(np.searchsorted(cumsum, 0.9) + 1)
        
        return {
            "effective_rank": effective_rank,
            "rank_90_percent": rank_90,
            "total_params": len(grad_vector),
            "singular_values": singular_values[:10].tolist(),  # 상위 10개만
            "efficiency": effective_rank / len(grad_vector),  # 0~1 사이
        }

    def diagnose(self, results: dict) -> str:
        """결과 해석."""
        total = results["total_params"]
        effective = results["effective_rank"]
        efficiency = results["efficiency"]
        
        lines = []
        lines.append(f"전체 파라미터: {total:,}개")
        lines.append(f"유효 차원: {effective}개")
        lines.append(f"효율성: {efficiency:.1%}")
        lines.append("")
        
        if efficiency < 0.01:
            lines.append("⚠️ 경고: 매우 비효율적!")
            lines.append("  → 100만개 파라미터 중 1%도 안 쓰는 중")
            lines.append("  → 해결: 모델을 작게 만들거나 정규화 줄이기")
        elif efficiency < 0.1:
            lines.append("⚠️ 주의: 비효율적")
            lines.append("  → 파라미터의 10%만 활용 중")
            lines.append("  → 해결: 학습률 올리기 또는 배치 크기 조정")
        else:
            lines.append("✅ 정상: 파라미터를 잘 활용 중")
        
        return "\n".join(lines)
