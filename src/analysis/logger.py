# Logging utilities

import logging
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("f1_digital_twin")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """Structured metrics logging for analysis."""
    
    def __init__(self, log_dir: Path):
        """Initialize metrics logger.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.log_dir / "metrics.csv"
        self.json_path = self.log_dir / "metrics.json"
        
        self._metrics_history: List[Dict[str, Any]] = []
        self._csv_initialized = False
        self._fieldnames: List[str] = []
    
    def log(self, step: int, metrics: Dict[str, float]) -> None:
        """Log metrics for a training step.
        
        Args:
            step: Training step
            metrics: Dict of metric values
        """
        record = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        self._metrics_history.append(record)
        
        # Initialize CSV with fieldnames from first record
        if not self._csv_initialized:
            self._fieldnames = list(record.keys())
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
            self._csv_initialized = True
        
        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            writer.writerow(record)
    
    def save_summary(self) -> None:
        """Save complete metrics history as JSON."""
        with open(self.json_path, "w") as f:
            json.dump(self._metrics_history, f, indent=2)
    
    def get_metric_series(self, metric_name: str) -> List[float]:
        """Get time series of a specific metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            List of metric values
        """
        return [
            m.get(metric_name)
            for m in self._metrics_history
            if metric_name in m
        ]
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value of a metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Latest value or None
        """
        for m in reversed(self._metrics_history):
            if metric_name in m:
                return m[metric_name]
        return None
    
    def get_summary_stats(self, metric_name: str, window: int = 100) -> Dict[str, float]:
        """Get summary statistics for a metric.
        
        Args:
            metric_name: Name of metric
            window: Number of recent values to consider
            
        Returns:
            Dict with mean, std, min, max
        """
        import numpy as np
        
        values = self.get_metric_series(metric_name)[-window:]
        if not values:
            return {}
        
        values = [v for v in values if v is not None]
        if not values:
            return {}
        
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }


class ExperimentLogger:
    """Logger for complete experiment tracking."""
    
    def __init__(
        self,
        experiment_name: str,
        base_dir: Path = Path("experiments"),
    ):
        """Initialize experiment logger.
        
        Args:
            experiment_name: Name of experiment
            base_dir: Base directory for experiments
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = base_dir / f"{timestamp}_{experiment_name}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs_dir = self.experiment_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.analysis_dir = self.experiment_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(
            level="INFO",
            log_file=self.logs_dir / "train.log",
        )
        
        # Setup metrics
        self.metrics = MetricsLogger(self.logs_dir)
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save experiment configuration.
        
        Args:
            config: Configuration dict
        """
        import yaml
        
        config_path = self.experiment_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def save_git_info(self) -> None:
        """Save git commit information."""
        import subprocess
        
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            
            dirty = subprocess.call(
                ["git", "diff", "--quiet"],
                stderr=subprocess.DEVNULL,
            ) != 0
            
            git_info = {
                "commit": commit,
                "branch": branch,
                "dirty": dirty,
            }
            
            with open(self.experiment_dir / "git_info.json", "w") as f:
                json.dump(git_info, f, indent=2)
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Git not available
            pass
    
    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Log training metrics.
        
        Args:
            step: Training step
            metrics: Metrics dict
        """
        self.metrics.log(step, metrics)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
