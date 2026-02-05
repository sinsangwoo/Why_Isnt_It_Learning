"""Benchmark runner for standardized experiments."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from gradient_pathology.analyzer import GradientAnalyzer
from gradient_pathology.core import GradientReport


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""

    model_name: str
    num_layers: int
    hidden_dim: int
    activation: str
    use_normalization: bool
    num_diagnostic_steps: int = 100
    seed: int = 42


class BenchmarkRunner:
    """Run standardized benchmark experiments."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results: List[Dict] = []

    def run_benchmark(
        self,
        config: BenchmarkConfig,
        save_report: bool = True,
    ) -> GradientReport:
        """Run single benchmark experiment.
        
        Args:
            config: Benchmark configuration
            save_report: Whether to save detailed report
            
        Returns:
            Gradient analysis report
        """
        # Set seed for reproducibility
        torch.manual_seed(config.seed)
        
        # Build model
        model = self._build_model(config)
        
        # Run analysis
        analyzer = GradientAnalyzer(model, device=self.device)
        
        start_time = time.time()
        report = analyzer.diagnose(
            num_steps=config.num_diagnostic_steps,
            input_shape=(config.hidden_dim,),
        )
        elapsed_time = time.time() - start_time
        
        # Store results
        result = {
            "config": config,
            "report": report,
            "elapsed_time": elapsed_time,
            "timestamp": time.time(),
        }
        self.results.append(result)
        
        return report

    def _build_model(self, config: BenchmarkConfig) -> nn.Module:
        """Build model from configuration."""
        from gradient_pathology.experiments import create_deep_network
        
        return create_deep_network(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            activation=config.activation,
            use_normalization=config.use_normalization,
        )

    def run_standard_suite(self) -> Dict[str, GradientReport]:
        """Run standard benchmark suite.
        
        Returns:
            Dictionary of benchmark results
        """
        configs = [
            BenchmarkConfig(
                model_name="shallow_relu",
                num_layers=3,
                hidden_dim=64,
                activation="relu",
                use_normalization=False,
            ),
            BenchmarkConfig(
                model_name="deep_relu_no_norm",
                num_layers=20,
                hidden_dim=64,
                activation="relu",
                use_normalization=False,
            ),
            BenchmarkConfig(
                model_name="deep_relu_with_norm",
                num_layers=20,
                hidden_dim=64,
                activation="relu",
                use_normalization=True,
            ),
            BenchmarkConfig(
                model_name="deep_sigmoid",
                num_layers=30,
                hidden_dim=64,
                activation="sigmoid",
                use_normalization=False,
            ),
            BenchmarkConfig(
                model_name="deep_gelu_with_norm",
                num_layers=20,
                hidden_dim=64,
                activation="gelu",
                use_normalization=True,
            ),
        ]
        
        results = {}
        for config in configs:
            print(f"Running benchmark: {config.model_name}...")
            report = self.run_benchmark(config)
            results[config.model_name] = report
        
        return results

    def generate_summary(self) -> str:
        """Generate summary of all benchmarks."""
        if not self.results:
            return "No benchmarks run yet."
        
        lines = ["=" * 70]
        lines.append("BENCHMARK SUMMARY")
        lines.append("=" * 70)
        
        for result in self.results:
            config = result["config"]
            report = result["report"]
            elapsed = result["elapsed_time"]
            
            lines.append(f"\n{config.model_name}:")
            lines.append(f"  Configuration: {config.num_layers} layers, "
                        f"{config.activation}, "
                        f"norm={config.use_normalization}")
            lines.append(f"  Analysis time: {elapsed:.2f}s")
            lines.append(f"  Global gradient: mean={report.global_mean:.2e}, "
                        f"std={report.global_std:.2e}")
        
        return "\n".join(lines)
