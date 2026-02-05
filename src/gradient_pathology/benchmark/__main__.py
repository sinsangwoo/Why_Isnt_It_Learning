"""CLI entry point for benchmarks."""

import argparse

from gradient_pathology.benchmark.runner import BenchmarkRunner


def main() -> None:
    """Run benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Run gradient pathology benchmarks"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (cpu/cuda)",
    )
    parser.add_argument(
        "--suite",
        default="standard",
        help="Benchmark suite to run",
    )
    
    args = parser.parse_args()
    
    print("Starting Gradient Pathology Benchmark Suite")
    print("=" * 70)
    
    runner = BenchmarkRunner(device=args.device)
    
    if args.suite == "standard":
        results = runner.run_standard_suite()
    
    print("\n" + runner.generate_summary())
    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
