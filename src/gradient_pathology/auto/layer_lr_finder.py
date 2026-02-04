"""레이어마다 최적 학습률이 다름 → 자동으로 찾아주기.

쉬운 설명:
- 첫 번째 레이어는 LR=0.001이 좋고
- 마지막 레이어는 LR=0.1이 좋을 수 있음
- 보통은 전체 모델에 하나의 LR만 쓰는데, 이건 비효율적
- 이 도구는 각 레이어마다 최적 LR을 자동으로 찾아줌
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
from copy import deepcopy


class LayerLRFinder:
    """레이어별 최적 학습률 찾기."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_names = [name for name, _ in model.named_parameters()]

    def find_layer_lrs(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        start_lr: float = 1e-6,
        end_lr: float = 1.0,
        num_steps: int = 50,
    ) -> Dict[str, float]:
        """각 레이어의 최적 LR 찾기.
        
        Args:
            dataloader: 학습 데이터
            loss_fn: 손실 함수
            start_lr: 시작 LR
            end_lr: 끝 LR
            num_steps: 테스트 횟수
            
        Returns:
            레이어별 추천 LR 딕셔너리
        """
        # 원본 상태 저장
        original_state = deepcopy(self.model.state_dict())
        
        layer_gradients: Dict[str, List[float]] = {
            name: [] for name in self.layer_names
        }
        
        lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_steps)
        
        data_iter = iter(dataloader)
        
        for lr in lrs:
            try:
                data, target = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                data, target = next(data_iter)
            
            self.model.zero_grad()
            output = self.model(data)
            loss = loss_fn(output, target)
            loss.backward()
            
            # 각 레이어의 그래디언트 크기 기록
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    layer_gradients[name].append(grad_norm)
            
            # 실제로 파라미터 업데이트 (LR 테스트)
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param -= lr * param.grad
        
        # 원본 상태로 복원
        self.model.load_state_dict(original_state)
        
        # 각 레이어마다 최적 LR 결정
        optimal_lrs = {}
        
        for layer_name, grads in layer_gradients.items():
            if not grads:
                continue
            
            # 그래디언트가 가장 안정적인 구간의 LR 찾기
            # 방법: 그래디언트 크기의 변화율이 적은 구간
            grad_changes = np.abs(np.diff(grads))
            
            # 너무 작지도, 너무 크지도 않은 구간
            valid_indices = np.where(
                (np.array(grads[:-1]) > 1e-8) & 
                (np.array(grads[:-1]) < 1e2)
            )[0]
            
            if len(valid_indices) > 0:
                # 변화율이 가장 작은 지점
                stable_idx = valid_indices[np.argmin(grad_changes[valid_indices])]
                optimal_lrs[layer_name] = float(lrs[stable_idx])
            else:
                optimal_lrs[layer_name] = start_lr
        
        return optimal_lrs

    def suggest_optimizer_groups(self, optimal_lrs: Dict[str, float]) -> List[dict]:
        """PyTorch optimizer에 바로 쓸 수 있는 형태로 변환.
        
        Returns:
            optimizer = Adam(param_groups) 형태로 쓸 수 있음
        """
        param_groups = []
        
        for name, param in self.model.named_parameters():
            if name in optimal_lrs:
                param_groups.append({
                    "params": [param],
                    "lr": optimal_lrs[name],
                })
        
        return param_groups

    def print_summary(self, optimal_lrs: Dict[str, float]) -> None:
        """결과 요약 출력."""
        print("\n" + "="*60)
        print("레이어별 추천 학습률")
        print("="*60)
        
        # LR 값으로 그룹화
        lr_groups: Dict[float, List[str]] = {}
        for name, lr in optimal_lrs.items():
            if lr not in lr_groups:
                lr_groups[lr] = []
            lr_groups[lr].append(name)
        
        # 큰 LR부터 출력
        for lr in sorted(lr_groups.keys(), reverse=True):
            layers = lr_groups[lr]
            print(f"\nLR = {lr:.2e} ({len(layers)}개 레이어)")
            for layer in layers[:3]:  # 처음 3개만
                print(f"  - {layer}")
            if len(layers) > 3:
                print(f"  ... 외 {len(layers)-3}개")
