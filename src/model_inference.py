from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from src.feature_engineering import FeatureEngineer


class ModelInference:
    """Generate ML and deep-learning predictions from the same input."""

    def __init__(
        self,
        ml_model_path: str | Path,
        deep_model_path: str | Path,
        scaler_x_path: str | Path,
        scaler_y_path: str | Path,
        selected_features_path: str | Path,
        time_steps: int = 10,
        device: torch.device | None = None,
    ) -> None:
        self.ml_model = joblib.load(ml_model_path)
        self.scaler_x_path = Path(scaler_x_path)
        self.scaler_x = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        self.selected_features = joblib.load(selected_features_path)
        self.time_steps = time_steps
        self.feature_engineer = FeatureEngineer()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from src.deep_learning_model import RecurrentForecaster

        self.deep_model = RecurrentForecaster.load(deep_model_path, self.device)

    def preprocess(self, history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Engineer and scale a historical context containing the input row.

        Args:
            history: Chronological raw rows ending with the target input row.

        Returns:
            Tuple containing the ML feature row and the deep-learning sequence.
        """
        required_history = 36 + self.time_steps
        if len(history) < required_history:
            raise ValueError(
                f"Inference requires at least {required_history} chronological rows; "
                f"received {len(history)}."
            )

        frame = history.copy()
        clipper_path = self.scaler_x_path.parent / "outlier_clipper.pkl"
        if clipper_path.exists():
            frame = joblib.load(clipper_path).transform(frame)
        frame["Appliances"] = frame["Appliances"].astype(float)
        frame.iloc[-1, frame.columns.get_loc("Appliances")] = 0.0
        target_index = frame.index[-1]
        frame.loc[target_index, "Appliances"] = 0.0
        engineered = self.feature_engineer.transform(frame).dropna()

        if target_index not in engineered.index:
            raise ValueError("The target input row does not contain enough historical context.")

        features = engineered[self.selected_features].tail(self.time_steps + 1)
        ml_features = features.iloc[-1:]
        ml_features_scaled = self.scaler_x.transform(ml_features)

        scaled_sequence = self.scaler_x.transform(features.tail(self.time_steps))
        deep_sequence = scaled_sequence[np.newaxis, :, :]
        return ml_features_scaled, deep_sequence.astype(np.float32)

    def predict(self, history: pd.DataFrame) -> dict[str, Any]:
        """
        Predict with the ML and deep-learning models for the same input.

        Args:
            history: Chronological raw context ending with the row to predict.

        Returns:
            Dictionary with ML prediction, DL prediction, and absolute difference.
        """
        ml_input, deep_input = self.preprocess(history)
        ml_prediction_scaled = float(np.asarray(self.ml_model.predict(ml_input)).reshape(-1)[0])
        ml_prediction = ml_prediction_scaled

        with torch.no_grad():
            tensor = torch.tensor(deep_input, dtype=torch.float32, device=self.device)
            dl_scaled = float(self.deep_model(tensor).cpu().numpy().reshape(-1)[0])
        dl_prediction = float(
            self.scaler_y.inverse_transform([[dl_scaled]])[0, 0]
        )

        if hasattr(self.ml_model, "feature_names_in_"):
            ml_prediction = ml_prediction

        return {
            "ML prediction": round(ml_prediction, 4),
            "DL prediction": round(dl_prediction, 4),
            "Absolute difference": round(abs(ml_prediction - dl_prediction), 4),
        }
