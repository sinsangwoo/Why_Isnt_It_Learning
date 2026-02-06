# Reproducibility Checklist

This document ensures experiments using Gradient Pathology are fully reproducible.

## Environment Setup

### Docker (Recommended)

```bash
# Build image
docker build -t gradient-pathology .

# Run standard benchmark
docker run gradient-pathology

# Run custom benchmark
docker run -v $(pwd)/experiments:/app/experiments gradient-pathology \
    python -m gradient_pathology.benchmark --device cpu
```

### Local Installation

```bash
# Install exact versions
pip install gradient-pathology==0.3.0

# Verify installation
python -c "import gradient_pathology; print(gradient_pathology.__version__)"
```

## Experiment Configuration

### Required Information

Every experiment should specify:

1. **Package Version**: `gradient-pathology==0.3.0`
2. **PyTorch Version**: `torch==2.0.0` or later
3. **Python Version**: `3.9+`
4. **Random Seed**: Always set `torch.manual_seed(seed)`
5. **Device**: CPU vs GPU can produce slightly different results

### Example Configuration

```python
import torch
from gradient_pathology.benchmark import BenchmarkRunner, BenchmarkConfig

# Set seed for reproducibility
torch.manual_seed(42)

# Define configuration
config = BenchmarkConfig(
    model_name="my_experiment",
    num_layers=20,
    hidden_dim=64,
    activation="relu",
    use_normalization=True,
    num_diagnostic_steps=100,
    seed=42,
)

# Run benchmark
runner = BenchmarkRunner(device="cpu")
report = runner.run_benchmark(config)
```

## Experiment Tracking

### With MLflow (Recommended)

```python
from gradient_pathology.benchmark import ExperimentTracker

with ExperimentTracker(experiment_name="my_experiments") as tracker:
    tracker.log_params({
        "num_layers": 20,
        "hidden_dim": 64,
        "activation": "relu",
    })
    
    # Run experiment
    report = runner.run_benchmark(config)
    
    tracker.log_metrics({
        "global_mean": report.global_mean,
        "global_std": report.global_std,
    })
```

### Without MLflow (JSON Fallback)

```python
# Automatically saves to experiments/my_experiments/run_*.json
with ExperimentTracker(experiment_name="my_experiments", use_mlflow=False) as tracker:
    # Same API as above
    pass
```

## Standard Benchmark Suite

### Running the Suite

```bash
# Command line
python -m gradient_pathology.benchmark --device cpu --suite standard

# Or in Python
from gradient_pathology.benchmark import BenchmarkRunner

runner = BenchmarkRunner()
results = runner.run_standard_suite()
```

### Expected Results

The standard suite includes:

1. **shallow_relu**: 3 layers, ReLU, no norm
   - Expected: Healthy gradients
   
2. **deep_relu_no_norm**: 20 layers, ReLU, no norm
   - Expected: Some instability
   
3. **deep_relu_with_norm**: 20 layers, ReLU, with LayerNorm
   - Expected: Healthy gradients
   
4. **deep_sigmoid**: 30 layers, Sigmoid, no norm
   - Expected: Vanishing or unstable gradients
   
5. **deep_gelu_with_norm**: 20 layers, GELU, with LayerNorm
   - Expected: Healthy gradients

## Publication Checklist

Before publishing results:

- [ ] Package version specified
- [ ] Random seed documented
- [ ] Device (CPU/GPU) specified
- [ ] Complete configuration saved
- [ ] Results tracked (MLflow or JSON)
- [ ] Code available (GitHub repo)
- [ ] Docker image available (optional but recommended)
- [ ] Benchmark results match expected values

## Contact

For reproducibility issues, please open an issue on GitHub:
https://github.com/sinsangwoo/Why_Isnt_It_Learning/issues
