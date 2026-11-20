"""Configuration-driven data preprocessing pipeline."""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "utils"))

from src.data_ingestion import DataIngestorCSV
from src.feature_engineering import FeatureEngineer
from src.outlier_handler import OutlierClipper
from utils.config_utils import (
    get_columns,
    get_logging,
    get_preprocessing,
    get_paths,
    get_reproducibility,
)


logger = logging.getLogger("data_pipeline")


def configure_logging() -> None:
    """Write spaced data-pipeline logs to the console and configured log file."""
    config = get_logging()
    log_path = ROOT_DIR / config.get(
        "data_pipeline_file", "artifacts/logs/data_pipeline.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(config.get("level", "INFO"))
    logger.propagate = False

    log_format = f"{config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')}\n"
    if not any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    ):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console)
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == log_path.resolve()
        for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)


class TemporalDataPipeline:
    """Prepare energy data for tabular and sequence models."""

    def __init__(self) -> None:
        configure_logging()
        self.paths = get_paths()
        self.columns = get_columns()
        self.preprocessing = get_preprocessing()
        self.reproducibility = get_reproducibility()
        self.feature_engineer = FeatureEngineer()
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.selected_features: list[str] = []

    def build(self, force_rebuild: bool = False) -> dict[str, object]:
        """
        Execute the complete chronological data pipeline.

        Args:
            force_rebuild: Retained for API compatibility; pipeline is deterministic.

        Returns:
            Paths and arrays describing the prepared data.
        """
        raw_path = ROOT_DIR / self.paths["data"]["raw"]
        logger.info("Starting data pipeline")
        logger.info("Loading raw data from %s", raw_path)
        frame = DataIngestorCSV().ingest(raw_path)
        logger.info(
            "Data loaded: rows=%d, columns=%d, missing cells=%d",
            len(frame),
            len(frame.columns),
            int(frame.isna().sum().sum()),
        )

        drop_columns = self.columns.get("drop_columns", [])
        frame = frame.drop(columns=[column for column in drop_columns if column in frame.columns])
        logger.info("Removed configured columns: %s", drop_columns)
        frame = frame.interpolate(method="time").ffill().bfill()
        logger.info("Missing-value processing complete: remaining=%d", int(frame.isna().sum().sum()))

        split_ratio = 1.0 - float(self.preprocessing["split"]["test_size"])
        split_index = int(len(frame) * split_ratio)
        train_raw = frame.iloc[:split_index].copy()
        test_raw = frame.iloc[split_index:].copy()
        logger.info(
            "Chronological split complete: train_rows=%d, test_rows=%d",
            len(train_raw),
            len(test_raw),
        )

        clip_config = self.preprocessing["outlier_clipping"]
        logger.info("Fitting outlier clipper on training data")
        clipper = OutlierClipper(
            columns=clip_config["columns"],
            lower_quantile=clip_config["lower_quantile"],
            upper_quantile=clip_config["upper_quantile"],
        )
        train_raw = clipper.fit_transform(train_raw)
        test_raw = clipper.transform(test_raw)
        logger.info("Outlier clipping applied to train and test data")

        combined = pd.concat([train_raw, test_raw], axis=0).sort_index()
        logger.info("Engineering time, lag, rolling, and domain features")
        featured = self.feature_engineer.transform(combined).dropna()
        logger.info("Feature engineering complete: rows=%d, columns=%d", *featured.shape)

        featured_split = int(len(featured) * split_ratio)
        train = featured.iloc[:featured_split]
        test = featured.iloc[featured_split:]
        target = self.columns["target"]

        drop_features = [target, "hour", "day_of_week", "month"]
        available_features = [column for column in featured.columns if column not in drop_features]
        X_train = train[available_features]
        y_train = train[target]
        X_test = test[available_features]
        y_test = test[target]
        logger.info(
            "Prepared model splits: train=%s, test=%s, candidate_features=%d",
            X_train.shape,
            X_test.shape,
            len(available_features),
        )

        selector_config = self.preprocessing["feature_selection"]
        logger.info("Selecting features using Random Forest importance and correlation")
        forest = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=self.reproducibility["random_state"],
            n_jobs=-1,
        )
        forest.fit(X_train, y_train)
        importance = pd.Series(forest.feature_importances_, index=available_features).sort_values(
            ascending=False
        )

        correlation = X_train.corr().abs()
        upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        threshold = float(selector_config["redundant_correlation_threshold"])
        redundant = [
            column for column in upper.columns if any(upper[column] > threshold)
        ]
        top_count = int(selector_config["top_features"])
        self.selected_features = [
            feature
            for feature in importance.head(top_count).index
            if feature not in redundant
        ]
        logger.info(
            "Feature selection complete: selected=%d, removed_correlated=%d",
            len(self.selected_features),
            len(redundant),
        )

        X_train_selected = X_train[self.selected_features]
        X_test_selected = X_test[self.selected_features]

        X_train_scaled = self.scaler_x.fit_transform(X_train_selected)
        X_test_scaled = self.scaler_x.transform(X_test_selected)
        y_train_scaled = self.scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1))
        y_test_scaled = self.scaler_y.transform(y_test.to_numpy().reshape(-1, 1))
        logger.info(
            "Scaling complete: X_train=%s, X_test=%s",
            X_train_scaled.shape,
            X_test_scaled.shape,
        )

        processed_dir = ROOT_DIR / self.paths["data"]["processed_dir"]
        artifact_scalers = ROOT_DIR / self.paths["artifacts"]["scalers"]
        processed_dir.mkdir(parents=True, exist_ok=True)
        artifact_scalers.mkdir(parents=True, exist_ok=True)

        np.save(processed_dir / "X_train_scaled.npy", X_train_scaled)
        np.save(processed_dir / "X_test_scaled.npy", X_test_scaled)
        np.save(processed_dir / "y_train_scaled.npy", y_train_scaled)
        np.save(processed_dir / "y_test_scaled.npy", y_test_scaled)
        np.save(processed_dir / "y_train.npy", y_train.to_numpy())
        np.save(processed_dir / "y_test.npy", y_test.to_numpy())

        featured.to_csv(ROOT_DIR / self.paths["data"]["featured"], index=True)
        train.to_csv(ROOT_DIR / self.paths["data"]["splits"]["train"], index=True)
        test.to_csv(ROOT_DIR / self.paths["data"]["splits"]["test"], index=True)

        joblib.dump(self.scaler_x, artifact_scalers / "scaler_X.pkl")
        joblib.dump(self.scaler_y, artifact_scalers / "scaler_y.pkl")
        joblib.dump(self.selected_features, artifact_scalers / "selected_features.pkl")
        joblib.dump(clipper, artifact_scalers / "outlier_clipper.pkl")
        logger.info("Saved processed datasets and preprocessing artifacts")

        result = {
            "train_shape": X_train_scaled.shape,
            "test_shape": X_test_scaled.shape,
            "selected_features": self.selected_features,
        }
        logger.info(
            "Data pipeline complete: train_shape=%s, test_shape=%s, selected_features=%d",
            result["train_shape"],
            result["test_shape"],
            len(self.selected_features),
        )
        return result


def build_data_pipeline(force_rebuild: bool = False) -> dict[str, object]:
    """Run the temporal data pipeline."""
    return TemporalDataPipeline().build(force_rebuild=force_rebuild)


def main() -> None:
    result = build_data_pipeline(force_rebuild=True)
    print(result)


if __name__ == "__main__":
    main()