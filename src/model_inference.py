import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from src.feature_engineering import FeatureEngineer

logger = logging.getLogger("inference")


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
        
        logger.info("Loading ML model from %s", ml_model_path)
        self.ml_model = joblib.load(ml_model_path)
        self.scaler_x_path = Path(scaler_x_path)
        logger.info("Loading feature scaler and selected feature list")
        self.scaler_x = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        self.selected_features = joblib.load(selected_features_path)
        self.time_steps = time_steps
        self.feature_engineer = FeatureEngineer()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from src.deep_learning_model import RecurrentForecaster

        logger.info("Loading DL checkpoint from %s on %s", deep_model_path, self.device)
        self.deep_model = RecurrentForecaster.load(deep_model_path, self.device)
        logger.info(
            "Models ready: ML=%s, DL architecture=%s, selected features=%d, device=%s",
            type(self.ml_model).__name__,
            self.deep_model.architecture.upper(),
            len(self.selected_features),
            self.device,
        )

    def preprocess(self, history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Engineer and scale a historical context containing the input row.

        Args:
            history: Chronological raw rows ending with the target input row.

        Returns:
            Tuple containing the ML feature row and the deep-learning sequence.
        """
        required_history = 36 + self.time_steps
        logger.info(
            "Preprocessing input: rows=%d, columns=%d, missing cells=%d, target time=%s",
            len(history),
            len(history.columns),
            int(history.isna().sum().sum()),
            history.index[-1] if not history.empty else "unavailable",
        )
        if len(history) < required_history:
            logger.error(
                "Insufficient inference history: required=%d, received=%d",
                required_history,
                len(history),
            )
            raise ValueError(
                f"Inference requires at least {required_history} chronological rows; "
                f"received {len(history)}."
            )

        frame = history.copy()
        clipper_path = self.scaler_x_path.parent / "outlier_clipper.pkl"
        if clipper_path.exists():
            logger.info("Applying fitted outlier clipping from %s", clipper_path)
            frame = joblib.load(clipper_path).transform(frame)
        else:
            logger.warning("Outlier clipping skipped; artifact not found: %s", clipper_path)

        frame["Appliances"] = frame["Appliances"].astype(float)
        frame.iloc[-1, frame.columns.get_loc("Appliances")] = 0.0
        target_index = frame.index[-1]
        frame.loc[target_index, "Appliances"] = 0.0
        logger.info("Masked target value at %s to prevent target leakage", target_index)

        logger.info("Engineering time, lag, rolling, and domain features")
        engineered_all = self.feature_engineer.transform(frame)
        engineered = engineered_all.dropna()
        logger.info(
            "Feature engineering complete: rows before dropna=%d, rows after=%d, "
            "rows removed=%d, feature columns=%d",
            len(engineered_all),
            len(engineered),
            len(engineered_all) - len(engineered),
            len(engineered.columns),
        )

        if target_index not in engineered.index:
            logger.error("Target row %s was removed during feature engineering", target_index)
            raise ValueError("The target input row does not contain enough historical context.")

        features = engineered[self.selected_features].tail(self.time_steps + 1)
        ml_features = features.iloc[-1:]
        logger.info(
            "Selected model features: count=%d, ML row shape=%s, DL lookback rows=%d",
            len(self.selected_features),
            ml_features.shape,
            self.time_steps,
        )
        logger.info("Scaling ML row and DL sequence with the fitted feature scaler")
        ml_features_scaled = self.scaler_x.transform(ml_features)

        scaled_sequence = self.scaler_x.transform(features.tail(self.time_steps))
        deep_sequence = scaled_sequence[np.newaxis, :, :]
        logger.info(
            "Preprocessing complete: scaled ML input shape=%s, scaled DL input shape=%s",
            ml_features_scaled.shape,
            deep_sequence.shape,
        )
        return ml_features_scaled, deep_sequence.astype(np.float32)

    def predict(self, history: pd.DataFrame) -> dict[str, Any]:
        """
        Predict with the ML and deep-learning models for the same input.

        Args:
            history: Chronological raw context ending with the row to predict.

        Returns:
            Dictionary with ML prediction, DL prediction, and absolute difference.
        """
        logger.info("Running ML and %s prediction", self.deep_model.architecture.upper())
        ml_input, deep_input = self.preprocess(history)
        ml_prediction_scaled = float(np.asarray(self.ml_model.predict(ml_input)).reshape(-1)[0])
        ml_prediction = ml_prediction_scaled

        with torch.no_grad():
            tensor = torch.tensor(deep_input, dtype=torch.float32, device=self.device)
            dl_scaled = float(self.deep_model(tensor).cpu().numpy().reshape(-1)[0])
        dl_prediction = float(
            self.scaler_y.inverse_transform([[dl_scaled]])[0, 0]
        )

        difference = abs(ml_prediction - dl_prediction)
        logger.info(
            "Prediction complete (%s): ML=%.4f, DL=%.4f, absolute difference=%.4f",
            self.deep_model.architecture.upper(),
            ml_prediction,
            dl_prediction,
            difference,
        )

        return {
            "ML prediction": round(ml_prediction, 4),
            "DL prediction": round(dl_prediction, 4),
            "Absolute difference": round(difference, 4),
        }
