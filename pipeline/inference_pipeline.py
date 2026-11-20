"""Predict appliance energy from a user-provided sensor reading."""

import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "utils"))

from src.deep_learning_model import RecurrentForecaster
from src.model_inference import ModelInference
from utils.config_utils import get_columns, get_logging, get_preprocessing


logger = logging.getLogger("inference")


class InferencePipeline:
    """Load trained models and compare predictions for the same chronological input."""

    def __init__(self) -> None:
        logging_config = get_logging()
        level_name = str(logging_config.get("level", "INFO")).upper()
        try:
            logger.setLevel(level_name)
        except ValueError as exc:
            raise ValueError(
                f"Invalid logging level configured for inference: {level_name}"
            ) from exc

        log_path = ROOT_DIR / logging_config.get(
            "inference_file", "artifacts/logs/inference.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.propagate = False
        log_format = logging_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        if not any(
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
            for handler in logger.handlers
        ):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logger.level)
            console_handler.setFormatter(logging.Formatter(f"{log_format}\n"))
            logger.addHandler(console_handler)

        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_path.resolve()
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(logger.level)
            file_handler.setFormatter(logging.Formatter(f"{log_format}\n"))
            logger.addHandler(file_handler)

        scaler_dir = ROOT_DIR / "artifacts/scalers"
        self.ml_model_path = ROOT_DIR / "artifacts/models/best_ml_model.pkl"
        self.scaler_x_path = scaler_dir / "scaler_X.pkl"
        self.scaler_y_path = scaler_dir / "scaler_y.pkl"
        self.selected_features_path = scaler_dir / "selected_features.pkl"
        self.time_steps = int(get_preprocessing()["sequence"]["time_steps"])

        logger.info(
            "Inference initialized: ML model=%s, feature scaler=%s, target scaler=%s, "
            "sequence length=%d, log file=%s",
            self.ml_model_path,
            self.scaler_x_path,
            self.scaler_y_path,
            self.time_steps,
            log_path,
        )

    def predict_for_context(self, history: pd.DataFrame) -> dict[str, float]:
        """Return predictions from the best ML, Random Forest, XGBoost, LSTM, and GRU."""
        model_dir = ROOT_DIR / "artifacts/models"
        deep_models = {
            name: model_dir / f"{name.lower()}_model.pth" for name in ("LSTM", "GRU")
        }
        available_deep_models = {
            name: path for name, path in deep_models.items() if path.exists()
        }
        if not available_deep_models:
            raise FileNotFoundError("No LSTM or GRU model was found in artifacts/models.")

        first_deep_path = next(iter(available_deep_models.values()))
        inference = ModelInference(
            self.ml_model_path,
            first_deep_path,
            self.scaler_x_path,
            self.scaler_y_path,
            self.selected_features_path,
            self.time_steps,
        )
        ml_input, deep_input = inference.preprocess(history)
        results: dict[str, float] = {}

        ml_models = {
            f"{type(inference.ml_model).__name__} (Best ML)": self.ml_model_path,
            "Random Forest (ML)": model_dir / "random_forest_model.pkl",
            "XGBoost (ML)": model_dir / "xgboost_model.pkl",
        }
        for name, path in ml_models.items():
            if not path.exists():
                logger.warning("Skipping missing ML model: %s", path)
                continue
            model = inference.ml_model if path == self.ml_model_path else joblib.load(path)
            results[name] = round(float(np.asarray(model.predict(ml_input)).reshape(-1)[0]), 4)
            logger.info("%s prediction: %.4f", name, results[name])

        for name, path in available_deep_models.items():
            model = (
                inference.deep_model
                if path == first_deep_path
                else RecurrentForecaster.load(path, inference.device)
            )
            with torch.no_grad():
                tensor = torch.tensor(deep_input, dtype=torch.float32, device=inference.device)
                scaled_prediction = float(model(tensor).cpu().numpy().reshape(-1)[0])
            prediction = float(inference.scaler_y.inverse_transform([[scaled_prediction]])[0, 0])
            results[f"{name} (DL)"] = round(prediction, 4)
            logger.info("%s prediction: %.4f", name, results[f"{name} (DL)"])

        logger.info("All available predictions: %s", results)
        return results


def build_input_context(
    data: Mapping[str, Any],
    dataset: pd.DataFrame,
    required_rows: int,
    drop_columns: list[str],
) -> pd.DataFrame:
    """Combine the user-provided target row with preceding dataset history."""
    if "date" not in data:
        raise ValueError("Inference input must include a 'date' field.")

    try:
        target_time = pd.Timestamp(data["date"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid inference date: {data['date']!r}") from exc
    if pd.isna(target_time):
        raise ValueError("Inference date must not be empty.")
    if target_time.tz is not None:
        raise ValueError("Timezone-aware inference dates are not supported.")

    available_columns = set(dataset.columns)
    allowed_input_columns = available_columns | set(drop_columns) | {"date"}
    unknown_columns = set(data).difference(allowed_input_columns)
    if unknown_columns:
        raise ValueError(f"Unexpected inference fields: {sorted(unknown_columns)}")

    required_input_columns = available_columns.difference({"Appliances"})
    missing_columns = required_input_columns.difference(data)
    if missing_columns:
        raise ValueError(f"Missing required inference fields: {sorted(missing_columns)}")

    input_row: dict[str, float] = {}
    for column in dataset.columns:
        raw_value = data.get(column, 0.0)
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Inference field '{column}' must be numeric.") from exc
        if not np.isfinite(numeric_value):
            raise ValueError(f"Inference field '{column}' must be finite.")
        input_row[column] = numeric_value

    prior_history = dataset.loc[dataset.index < target_time].tail(required_rows - 1)
    if len(prior_history) < required_rows - 1:
        raise ValueError(
            f"Inference requires {required_rows - 1} historical rows before {target_time}; "
            f"found {len(prior_history)}."
        )

    target_row = pd.DataFrame(
        [input_row],
        index=pd.DatetimeIndex([target_time], name=dataset.index.name),
    )
    history = pd.concat([prior_history, target_row]).sort_index()
    logger.info(
        "Built input context: user row=%s, prior history rows=%d, total rows=%d",
        target_time,
        len(prior_history),
        len(history),
    )
    return history


def main(data: Mapping[str, Any]) -> dict[str, float]:
    """Run inference for a user-provided row using prior CSV records as context."""

    pipeline = InferencePipeline()
    raw_path = ROOT_DIR / "data/raw/energy_data_set.csv"

    logger.info("Loading inference dataset from %s", raw_path)
    frame = pd.read_csv(raw_path)

    logger.info(
        "Dataset loaded: rows=%d, columns=%d, missing cells=%d",
        len(frame),
        len(frame.columns),
        int(frame.isna().sum().sum()),
    )

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.set_index("date").sort_index()
    drop_columns = get_columns().get("drop_columns", [])
    frame = frame.drop(
        columns=[column for column in drop_columns if column in frame.columns]
    )
    required_rows = 36 + int(get_preprocessing()["sequence"]["time_steps"])
    history = build_input_context(data, frame, required_rows, drop_columns)
    logger.info(
        "Inference context ready: rows=%d, time range=%s to %s",
        len(history),
        history.index.min(),
        history.index.max(),
    )

    return pipeline.predict_for_context(history)


if __name__ == "__main__":
    data = {
    "date": "2016-03-18 20:30:00",

    # Energy related
    "lights": 30,

    # Indoor temperatures (°C)
    "T1": 21.50,   # Kitchen
    "T2": 21.20,   # Living room
    "T3": 20.80,   # Laundry
    "T4": 20.40,   # Office
    "T5": 19.90,   # Bathroom
    "T6": 8.70,    # Outside north side
    "T7": 20.10,   # Ironing room
    "T8": 20.60,   # Teenager room
    "T9": 19.80,   # Parents room

    # Indoor humidity (%)
    "RH_1": 44.20,
    "RH_2": 43.50,
    "RH_3": 42.80,
    "RH_4": 41.90,
    "RH_5": 50.40,
    "RH_6": 78.10,
    "RH_7": 40.60,
    "RH_8": 45.20,
    "RH_9": 43.70,

    # Outdoor / weather
    "T_out": 7.80,
    "Press_mm_hg": 756.20,
    "RH_out": 85.00,
    "Windspeed": 4.50,
    "Visibility": 40.00,
    "Tdewpoint": 5.10
}

    predictions = main(data)
    for model_name, prediction in predictions.items():
        print(f"{model_name} : {prediction:.4f} Wh")