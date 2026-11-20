"""Compare machine-learning and deep-learning forecasts for one input context."""

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "utils"))

from src.model_inference import ModelInference
from utils.config_utils import get_preprocessing


class InferencePipeline:
    """Load trained models and compare predictions for the same chronological input."""

    def __init__(self) -> None:
        scaler_dir = ROOT_DIR / "artifacts/scalers"
        self.ml_model_path = ROOT_DIR / "artifacts/models/best_ml_model.pkl"
        self.scaler_x_path = scaler_dir / "scaler_X.pkl"
        self.scaler_y_path = scaler_dir / "scaler_y.pkl"
        self.selected_features_path = scaler_dir / "selected_features.pkl"
        self.time_steps = int(get_preprocessing()["sequence"]["time_steps"])

    def predict_for_context(self, history: pd.DataFrame) -> dict[str, Any]:
        """
        Generate ML and available DL predictions for the latest input row.

        Args:
            history: Chronological raw rows ending with the row to forecast.

        Returns:
            Comparison dictionary containing model predictions and differences.
        """
        results: dict[str, Any] = {}
        architectures = ("lstm", "gru")

        for architecture in architectures:
            deep_model_path = ROOT_DIR / f"artifacts/models/{architecture}_model.pth"
            if not deep_model_path.exists():
                continue
            inference = ModelInference(
                ml_model_path=self.ml_model_path,
                deep_model_path=deep_model_path,
                scaler_x_path=self.scaler_x_path,
                scaler_y_path=self.scaler_y_path,
                selected_features_path=self.selected_features_path,
                time_steps=self.time_steps,
            )
            results[architecture.upper()] = inference.predict(history)

        if not results:
            raise FileNotFoundError("No deep-learning model was found in artifacts/models.")
        return results


def main() -> dict[str, Any]:
    """CLI entry point using the most recent test-period context."""
    raw_path = ROOT_DIR / "data/raw/energy_data_set.csv"
    frame = pd.read_csv(raw_path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    required_rows = 36 + int(get_preprocessing()["sequence"]["time_steps"])
    history = frame.tail(required_rows).copy()
    return InferencePipeline().predict_for_context(history)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))