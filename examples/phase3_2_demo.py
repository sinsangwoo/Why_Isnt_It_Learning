#!/usr/bin/env python3
"""Phase 3.2 자동 진단 도구 데모.

이 스크립트는:
1. 유효 차원 (Effective Rank) - 파라미터 활용도 측정
2. 레이어별 LR Finder - 각 레이어마다 최적 학습률 찾기
3. 그래디언트 흐름 그래프 - 어디서 막히는지 시각화

를 보여줍니다.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gradient_pathology.auto.effective_rank import EffectiveRankAnalyzer
from gradient_pathology.auto.layer_lr_finder import LayerLRFinder
from gradient_pathology.auto.gradient_flow_graph import GradientFlowGraph


def create_test_model():
    """테스트용 모델 생성."""
    return nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def create_test_data():
    """테스트용 데이터 생성."""
    X = torch.randn(200, 20)
    y = torch.randn(200, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16, shuffle=True)


def demo_effective_rank():
    """유효 차원 분석 데모."""
    print("\n" + "="*60)
    print("1. 유효 차원 분석")
    print("="*60)
    print("\n설명: 모델이 파라미터를 얼마나 효율적으로 쓰는지 측정")
    
    model = create_test_model()
    dataloader = create_test_data()
    
    analyzer = EffectiveRankAnalyzer(model)
    results = analyzer.compute_effective_rank(
        dataloader,
        nn.MSELoss(),
        num_samples=10
    )
    
    print("\n" + analyzer.diagnose(results))


def demo_layer_lr_finder():
    """레이어별 LR 찾기 데모."""
    print("\n" + "="*60)
    print("2. 레이어별 학습률 찾기")
    print("="*60)
    print("\n설명: 각 레이어마다 다른 학습률을 쓰면 더 빨리 학습")
    
    model = create_test_model()
    dataloader = create_test_data()
    
    finder = LayerLRFinder(model)
    
    print("\n학습률 테스트 중... (30초 정도 걸림)")
    optimal_lrs = finder.find_layer_lrs(
        dataloader,
        nn.MSELoss(),
        num_steps=30
    )
    
    finder.print_summary(optimal_lrs)
    
    print("\n사용 예시:")
    print("```python")
    print("param_groups = finder.suggest_optimizer_groups(optimal_lrs)")
    print("optimizer = torch.optim.Adam(param_groups)")
    print("```")


def demo_gradient_flow():
    """그래디언트 흐름 시각화 데모."""
    print("\n" + "="*60)
    print("3. 그래디언트 흐름 분석")
    print("="*60)
    print("\n설명: 그래디언트가 어디서 막히는지 그림으로 확인")
    
    model = create_test_model()
    dataloader = create_test_data()
    
    flow = GradientFlowGraph(model)
    
    print("\n그래디언트 기록 중...")
    flow.record_flow(dataloader, nn.MSELoss(), num_samples=5)
    
    print("\n병목 지점 찾기:")
    bottlenecks = flow.find_bottlenecks()
    
    print("\n막대 그래프 생성...")
    flow.plot_flow_simple()
    
    print("\n네트워크 그래프 생성...")
    try:
        flow.plot_flow_network()
    except Exception as e:
        print(f"NetworkX 그래프 생성 실패: {e}")
        print("pip install networkx 실행하면 볼 수 있습니다")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# Phase 3.2: 자동 진단 도구 데모")
    print("#"*60)
    
    demo_effective_rank()
    demo_layer_lr_finder()
    demo_gradient_flow()
    
    print("\n" + "#"*60)
    print("# 데모 완료!")
    print("#"*60)
    print("\n이 도구들을 쓰면:")
    print("  - 수동으로 학습률 튜닝할 필요 없음")
    print("  - 어느 레이어가 문제인지 바로 알 수 있음")
    print("  - 모델 효율성을 숫자로 확인 가능")
