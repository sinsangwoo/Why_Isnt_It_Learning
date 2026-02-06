"""Experiment tracking integration."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """Track experiments with MLflow or local JSON."""

    def __init__(
        self,
        experiment_name: str = "gradient-pathology",
        tracking_uri: Optional[str] = None,
        use_mlflow: bool = True,
    ) -> None:
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        
        if self.use_mlflow:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
        else:
            # Fallback to local JSON tracking
            self.log_dir = Path("experiments") / experiment_name
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.current_run_data: Dict[str, Any] = {}

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start new experiment run."""
        if self.use_mlflow:
            mlflow.start_run(run_name=run_name)
        else:
            self.current_run_data = {}

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log experiment parameters."""
        if self.use_mlflow:
            mlflow.log_params(params)
        else:
            self.current_run_data["params"] = params

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log experiment metrics."""
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        else:
            if "metrics" not in self.current_run_data:
                self.current_run_data["metrics"] = []
            self.current_run_data["metrics"].append({"step": step, "values": metrics})

    def log_artifact(self, artifact_path: str) -> None:
        """Log artifact file."""
        if self.use_mlflow:
            mlflow.log_artifact(artifact_path)
        else:
            # Copy to log directory
            import shutil

            dest = self.log_dir / Path(artifact_path).name
            shutil.copy(artifact_path, dest)

    def end_run(self) -> None:
        """End current experiment run."""
        if self.use_mlflow:
            mlflow.end_run()
        else:
            # Save to JSON
            import time

            filename = f"run_{int(time.time())}.json"
            with open(self.log_dir / filename, "w") as f:
                json.dump(self.current_run_data, f, indent=2)

    def __enter__(self) -> "ExperimentTracker":
        self.start_run()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_run()
