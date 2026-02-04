#!/usr/bin/env python3
"""Expert system demonstration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch.nn as nn

from gradient_pathology.expert import ExpertSystem


def demo_problematic_architecture():
    """Demonstrate diagnosis of problematic architecture."""
    print("\n" + "=" * 70)
    print("DEMO 1: Problematic Deep Sigmoid Network")
    print("=" * 70)
    
    # Create problematic model
    model = nn.Sequential(
        *[nn.Linear(64, 64), nn.Sigmoid()] * 30,  # 30 sigmoid layers!
        nn.Linear(64, 1),
    )
    
    expert = ExpertSystem()
    diagnoses = expert.diagnose_architecture(model)
    
    print(expert.generate_report())
    print("\n💡 Quick Fix:", expert.get_quick_fix())


def demo_healthy_architecture():
    """Demonstrate diagnosis of healthy architecture."""
    print("\n" + "=" * 70)
    print("DEMO 2: Healthy Modern Architecture")
    print("=" * 70)
    
    # Create healthy model
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Linear(128, 128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Linear(128, 64),
        nn.LayerNorm(64),
        nn.GELU(),
        nn.Linear(64, 1),
    )
    
    expert = ExpertSystem()
    diagnoses = expert.diagnose_architecture(model)
    
    print(expert.generate_report())


def demo_gradient_based_diagnosis():
    """Demonstrate gradient-based diagnosis."""
    print("\n" + "=" * 70)
    print("DEMO 3: Gradient-Based Diagnosis")
    print("=" * 70)
    
    model = nn.Sequential(
        *[nn.Linear(64, 64), nn.ReLU()] * 10,
        nn.Linear(64, 1),
    )
    
    # Simulate gradient statistics
    gradient_stats = {
        "global_mean": 2.3e-8,  # Very small - vanishing!
        "global_std": 1.2e-7,
    }
    
    expert = ExpertSystem()
    diagnoses = expert.diagnose_architecture(model, gradient_stats)
    
    print(expert.generate_report())
    print("\n💡 Quick Fix:", expert.get_quick_fix())


def demo_lr_problem():
    """Demonstrate learning rate problem diagnosis."""
    print("\n" + "=" * 70)
    print("DEMO 4: Learning Rate Too High")
    print("=" * 70)
    
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    
    # Simulate exploding gradients from high LR
    gradient_stats = {
        "global_mean": 150.0,  # Very large!
        "global_std": 200.0,
    }
    
    expert = ExpertSystem()
    diagnoses = expert.diagnose_architecture(model, gradient_stats)
    
    print(expert.generate_report())
    print("\n💡 Quick Fix:", expert.get_quick_fix())


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# EXPERT SYSTEM DEMONSTRATIONS")
    print("#" * 70)
    
    demo_problematic_architecture()
    demo_healthy_architecture()
    demo_gradient_based_diagnosis()
    demo_lr_problem()
    
    print("\n" + "#" * 70)
    print("# All demonstrations completed")
    print("#" * 70)
