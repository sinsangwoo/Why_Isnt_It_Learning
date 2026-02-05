"""Rule-based expert system for gradient pathology diagnosis."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch.nn as nn


@dataclass
class Diagnosis:
    """Diagnosis result with recommendations."""

    problem: str
    severity: str  # "critical", "warning", "info"
    recommendations: List[str]
    confidence: float  # 0-1


class ExpertSystem:
    """Automated expert system for training problem diagnosis.
    
    Analyzes model architecture and gradient statistics to provide
    actionable recommendations based on established best practices.
    """

    def __init__(self):
        self.diagnoses: List[Diagnosis] = []

    def diagnose_architecture(
        self,
        model: nn.Module,
        gradient_stats: Optional[Dict[str, float]] = None,
    ) -> List[Diagnosis]:
        """Analyze model architecture and provide recommendations.
        
        Args:
            model: PyTorch model to analyze
            gradient_stats: Optional gradient statistics from training
            
        Returns:
            List of diagnosis results
        """
        self.diagnoses = []
        
        # Architecture analysis
        self._check_depth(model)
        self._check_activation_functions(model)
        self._check_normalization(model)
        
        # Gradient-based analysis
        if gradient_stats:
            self._check_gradient_flow(gradient_stats)
            self._check_learning_rate(gradient_stats)
        
        return self.diagnoses

    def _check_depth(self, model: nn.Module) -> None:
        """Check if model depth is appropriate for architecture."""
        # Count layers
        num_layers = sum(1 for _ in model.modules() if isinstance(_, nn.Linear))
        
        # Check for normalization
        has_norm = any(
            isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d))
            for m in model.modules()
        )
        
        if num_layers > 20 and not has_norm:
            self.diagnoses.append(
                Diagnosis(
                    problem="Deep network without normalization",
                    severity="critical",
                    recommendations=[
                        "Add LayerNorm after each layer",
                        "Or use BatchNorm for CNN architectures",
                        "Consider PreLN Transformer architecture for very deep models",
                    ],
                    confidence=0.95,
                )
            )
        elif num_layers > 50:
            self.diagnoses.append(
                Diagnosis(
                    problem="Extremely deep network detected",
                    severity="warning",
                    recommendations=[
                        "Consider ResNet-style skip connections",
                        "Use PreLN (Pre-Layer Normalization) instead of PostLN",
                        "Implement gradient checkpointing to save memory",
                    ],
                    confidence=0.85,
                )
            )

    def _check_activation_functions(self, model: nn.Module) -> None:
        """Check activation function choices."""
        activations = [m for m in model.modules() if isinstance(m, nn.Module)]
        
        sigmoid_count = sum(1 for m in activations if isinstance(m, nn.Sigmoid))
        tanh_count = sum(1 for m in activations if isinstance(m, nn.Tanh))
        relu_count = sum(1 for m in activations if isinstance(m, nn.ReLU))
        
        total_activations = sigmoid_count + tanh_count + relu_count
        
        if sigmoid_count > 5:
            self.diagnoses.append(
                Diagnosis(
                    problem=f"Heavy use of Sigmoid activations ({sigmoid_count} layers)",
                    severity="critical",
                    recommendations=[
                        "Replace with ReLU or GELU for hidden layers",
                        "Keep Sigmoid only for final output layer (binary classification)",
                        "Use He initialization with ReLU",
                    ],
                    confidence=0.9,
                )
            )
        
        if tanh_count > 5:
            self.diagnoses.append(
                Diagnosis(
                    problem=f"Heavy use of Tanh activations ({tanh_count} layers)",
                    severity="warning",
                    recommendations=[
                        "Consider GELU or SiLU for better gradient flow",
                        "Or use ReLU with proper initialization",
                    ],
                    confidence=0.8,
                )
            )

    def _check_normalization(self, model: nn.Module) -> None:
        """Check normalization layer usage."""
        linear_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        norm_layers = sum(
            1
            for m in model.modules()
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d))
        )
        
        if linear_layers > 10 and norm_layers == 0:
            self.diagnoses.append(
                Diagnosis(
                    problem="No normalization in deep network",
                    severity="critical",
                    recommendations=[
                        "Add LayerNorm after each Linear layer",
                        "Pattern: Linear → LayerNorm → Activation",
                        "Or use BatchNorm for mini-batch training",
                    ],
                    confidence=0.95,
                )
            )

    def _check_gradient_flow(self, stats: Dict[str, float]) -> None:
        """Analyze gradient flow patterns."""
        mean_grad = stats.get("global_mean", 0)
        std_grad = stats.get("global_std", 0)
        
        # Vanishing gradients
        if abs(mean_grad) < 1e-7:
            self.diagnoses.append(
                Diagnosis(
                    problem="Vanishing gradients detected",
                    severity="critical",
                    recommendations=[
                        "Use ReLU or GELU activation instead of Sigmoid/Tanh",
                        "Implement residual connections (skip connections)",
                        "Try Xavier/He initialization",
                        "Add LayerNorm or BatchNorm",
                    ],
                    confidence=0.95,
                )
            )
        
        # Exploding gradients
        if abs(mean_grad) > 1e2:
            self.diagnoses.append(
                Diagnosis(
                    problem="Exploding gradients detected",
                    severity="critical",
                    recommendations=[
                        "Implement gradient clipping (clip_grad_norm)",
                        "Reduce learning rate by 5-10x",
                        "Check weight initialization scale",
                        "Add normalization layers",
                    ],
                    confidence=0.9,
                )
            )

    def _check_learning_rate(self, stats: Dict[str, float]) -> None:
        """Analyze learning rate appropriateness."""
        mean_grad = stats.get("global_mean", 0)
        
        # Very small gradients might indicate LR too low
        if 1e-8 < abs(mean_grad) < 1e-6:
            self.diagnoses.append(
                Diagnosis(
                    problem="Very small gradients - LR might be too low",
                    severity="warning",
                    recommendations=[
                        "Try increasing learning rate by 2-5x",
                        "Use learning rate warmup for first few epochs",
                        "Consider cosine annealing schedule",
                    ],
                    confidence=0.7,
                )
            )
        
        # Large gradients might indicate LR too high
        if abs(mean_grad) > 10:
            self.diagnoses.append(
                Diagnosis(
                    problem="Large gradients - LR might be too high",
                    severity="critical",
                    recommendations=[
                        "Reduce learning rate by 5-10x",
                        "Implement warmup: start with LR/100 for first 1000 steps",
                        "Use gradient clipping as safety measure",
                    ],
                    confidence=0.85,
                )
            )

    def generate_report(self) -> str:
        """Generate human-readable diagnostic report."""
        if not self.diagnoses:
            return "✅ No major issues detected. Model architecture looks healthy."
        
        lines = ["=" * 70]
        lines.append("EXPERT SYSTEM DIAGNOSTIC REPORT")
        lines.append("=" * 70)
        
        # Group by severity
        critical = [d for d in self.diagnoses if d.severity == "critical"]
        warnings = [d for d in self.diagnoses if d.severity == "warning"]
        info = [d for d in self.diagnoses if d.severity == "info"]
        
        if critical:
            lines.append("\n🚨 CRITICAL ISSUES:")
            lines.append("-" * 70)
            for diag in critical:
                lines.append(f"\n{diag.problem} (confidence: {diag.confidence:.0%})")
                lines.append("Recommendations:")
                for rec in diag.recommendations:
                    lines.append(f"  • {rec}")
        
        if warnings:
            lines.append("\n⚠️  WARNINGS:")
            lines.append("-" * 70)
            for diag in warnings:
                lines.append(f"\n{diag.problem} (confidence: {diag.confidence:.0%})")
                lines.append("Recommendations:")
                for rec in diag.recommendations:
                    lines.append(f"  • {rec}")
        
        if info:
            lines.append("\nℹ️  SUGGESTIONS:")
            lines.append("-" * 70)
            for diag in info:
                lines.append(f"\n{diag.problem}")
                for rec in diag.recommendations:
                    lines.append(f"  • {rec}")
        
        return "\n".join(lines)

    def get_quick_fix(self) -> Optional[str]:
        """Get single most important recommendation."""
        if not self.diagnoses:
            return None
        
        # Sort by severity and confidence
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        sorted_diags = sorted(
            self.diagnoses,
            key=lambda d: (severity_order[d.severity], -d.confidence),
        )
        
        top_diag = sorted_diags[0]
        return f"{top_diag.problem} → {top_diag.recommendations[0]}"
