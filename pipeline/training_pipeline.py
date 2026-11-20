"""Train tabular and recurrent models without experiment-tracking dependencies."""

import json
import sys
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "utils"))

from src.deep_learning_model import RecurrentForecaster
from src.model_evaluation import ModelEvaluator
from src.model_training import ModelTrainer
from utils.config_utils import get_modeling, get_paths, get_preprocessing, get_reproducibility


class DeepLearningTrainer:
    """Train and select LSTM or GRU recurrent forecasters."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def create_sequences(
        features: np.ndarray,
        targets: np.ndarray,
        time_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert tabular arrays into fixed-length temporal sequences.

        Args:
            features: Two-dimensional feature array.
            targets: One-dimensional or one-column target array.
            time_steps: Number of preceding observations in each sequence.

        Returns:
            Sequence features and corresponding targets.
        """
        X_sequences = []
        y_sequences = []
        targets = targets.reshape(-1)
        for index in range(time_steps, len(features)):
            X_sequences.append(features[index - time_steps:index])
            y_sequences.append(targets[index])
        return np.asarray(X_sequences, dtype=np.float32), np.asarray(y_sequences, dtype=np.float32).reshape(-1, 1)

    def train(
        self,
        architecture: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test_scaled: np.ndarray,
        y_test_original: np.ndarray,
        scaler_y,
        time_steps: int,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        patience: int = 10,
    ) -> dict[str, object]:
        """
        Train a recurrent model and evaluate it on the held-out test period.

        Args:
            architecture: Either ``lstm`` or ``gru``.
            X_train: Scaled training features.
            y_train: Scaled training target.
            X_test: Scaled test features.
            y_test_scaled: Scaled test target.
            y_test_original: Test target in original units.
            scaler_y: Fitted target scaler.
            time_steps: Number of observations per sequence.
            epochs: Maximum training epochs.
            batch_size: Training batch size.
            learning_rate: Adam learning rate.
            patience: Early-stopping patience.

        Returns:
            Metrics, predictions, loss history, and model save path.
        """
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, time_steps)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test_scaled, time_steps)

        validation_count = max(1, int(len(X_train_seq) * 0.15))
        train_count = len(X_train_seq) - validation_count
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_seq[:train_count]),
            torch.from_numpy(y_train_seq[:train_count]),
        )
        validation_dataset = TensorDataset(
            torch.from_numpy(X_train_seq[train_count:]),
            torch.from_numpy(y_train_seq[train_count:]),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        model = RecurrentForecaster(
            input_size=X_train_seq.shape[2],
            architecture=architecture,
            hidden_size1=64,
            hidden_size2=32,
            dense_units=16,
            dropout=0.2,
        ).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_validation_loss = float("inf")
        patience_counter = 0
        best_state = None
        train_losses: list[float] = []
        validation_losses: list[float] = []

        for _ in range(epochs):
            model.train()
            total_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                prediction = model(X_batch)
                loss = criterion(prediction, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / max(1, len(train_loader))

            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in validation_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    validation_loss += criterion(model(X_batch), y_batch).item()
            validation_loss /= max(1, len(validation_loader))

            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            scheduler.step(validation_loss)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                patience_counter = 0
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        test_dataset = TensorDataset(torch.from_numpy(X_test_seq), torch.from_numpy(y_test_seq))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        scaled_predictions = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                predictions = model(X_batch.to(self.device)).cpu().numpy()
                scaled_predictions.extend(predictions.reshape(-1))

        predictions_original = scaler_y.inverse_transform(
            np.asarray(scaled_predictions).reshape(-1, 1)
        ).reshape(-1)
        aligned_original = y_test_original[time_steps:]
        metrics = ModelEvaluator.evaluate(aligned_original, predictions_original)

        model_path = ROOT_DIR / get_paths()["model"][f"{architecture}_model"]
        model.save(model_path, X_train_seq.shape[2])
        return {
            "architecture": architecture,
            "metrics": metrics,
            "predictions": predictions_original,
            "train_losses": train_losses,
            "validation_losses": validation_losses,
            "model_path": str(model_path),
        }


class TrainingPipeline:
    """Train, evaluate, compare, and save ML and deep-learning models."""

    def run(self) -> dict[str, object]:
        """Execute training for all configured model families."""
        processed = ROOT_DIR / "data" / "processed"
        X_train = np.load(processed / "X_train_scaled.npy")
        X_test = np.load(processed / "X_test_scaled.npy")
        y_train_scaled = np.load(processed / "y_train_scaled.npy")
        y_test_scaled = np.load(processed / "y_test_scaled.npy")
        y_train = np.load(processed / "y_train.npy")
        y_test = np.load(processed / "y_test.npy")

        random_state = int(get_reproducibility()["random_state"])
        trainer = ModelTrainer()
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
            "XGBoost": XGBRegressor(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
        }

        ml_results: list[dict[str, object]] = []
        for name, model in models.items():
            fitted = trainer.train(model, X_train, y_train)
            predictions = fitted.predict(X_test)
            metrics = ModelEvaluator.evaluate(y_test, predictions)
            ml_results.append({"Model": name, **metrics, "predictions": predictions})

            safe_name = name.lower().replace(" ", "_")
            model_path = ROOT_DIR / f"artifacts/models/{safe_name}_model.pkl"
            trainer.save(fitted, model_path)

        # Tune Random Forest and XGBoost using chronological cross-validation.
        training_config = get_modeling()
        tuning_enabled = bool(training_config.get("baseline_models"))
        if tuning_enabled:
            cv = TimeSeriesSplit(n_splits=3)
            rf_search = RandomizedSearchCV(
                RandomForestRegressor(random_state=random_state, n_jobs=-1),
                {
                    "n_estimators": [50, 100, 150, 200],
                    "max_depth": [6, 8, 10, 12, 15],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                },
                n_iter=10,
                scoring="neg_mean_absolute_error",
                cv=cv,
                random_state=random_state,
                n_jobs=-1,
            )
            rf_search.fit(X_train, y_train)
            tuned_rf = rf_search.best_estimator_
            rf_predictions = tuned_rf.predict(X_test)
            ml_results.append(
                {"Model": "Tuned Random Forest", **ModelEvaluator.evaluate(y_test, rf_predictions),
                 "predictions": rf_predictions}
            )
            trainer.save(tuned_rf, ROOT_DIR / "artifacts/models/tuned_random_forest_model.pkl")

            xgb_search = RandomizedSearchCV(
                XGBRegressor(random_state=random_state, n_jobs=-1),
                {
                    "n_estimators": [50, 100, 150, 200],
                    "max_depth": [3, 4, 5, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                },
                n_iter=10,
                scoring="neg_mean_absolute_error",
                cv=cv,
                random_state=random_state,
                n_jobs=-1,
            )
            xgb_search.fit(X_train, y_train)
            tuned_xgb = xgb_search.best_estimator_
            xgb_predictions = tuned_xgb.predict(X_test)
            ml_results.append(
                {"Model": "Tuned XGBoost", **ModelEvaluator.evaluate(y_test, xgb_predictions),
                 "predictions": xgb_predictions}
            )
            trainer.save(tuned_xgb, ROOT_DIR / "artifacts/models/tuned_xgboost_model.pkl")

        ml_results.sort(key=lambda result: float(result["MAE"]))
        best_ml = ml_results[0]
        best_model_name = str(best_ml["Model"])
        if "Random Forest" in best_model_name:
            best_model_path = ROOT_DIR / (
                "artifacts/models/tuned_random_forest_model.pkl"
                if "Tuned" in best_model_name
                else "artifacts/models/random_forest_model.pkl"
            )
        elif "XGBoost" in best_model_name:
            best_model_path = ROOT_DIR / (
                "artifacts/models/tuned_xgboost_model.pkl"
                if "Tuned" in best_model_name
                else "artifacts/models/xgboost_model.pkl"
            )
        else:
            best_model_path = ROOT_DIR / "artifacts/models/linear_regression_model.pkl"
        trainer.save(trainer.load(best_model_path), ROOT_DIR / "artifacts/models/best_ml_model.pkl")

        deep_config = get_modeling()["deep_learning"]
        time_steps = int(get_preprocessing()["sequence"]["time_steps"])
        scaler_y = joblib.load(ROOT_DIR / "artifacts/scalers/scaler_y.pkl")
        deep_trainer = DeepLearningTrainer(random_state=random_state)
        deep_results = []
        for architecture in deep_config["architectures"]:
            if architecture not in ("lstm", "gru"):
                continue
            result = deep_trainer.train(
                architecture=architecture,
                X_train=X_train,
                y_train=y_train_scaled,
                X_test=X_test,
                y_test_scaled=y_test_scaled,
                y_test_original=y_test,
                scaler_y=scaler_y,
                time_steps=time_steps,
                epochs=int(deep_config["epochs"]),
                batch_size=int(deep_config["batch_size"]),
                learning_rate=float(deep_config["learning_rate"]),
                patience=int(deep_config["early_stopping"]["patience"]),
            )
            deep_results.append(result)

        model_metrics = [
            {
                "Model": result["Model"],
                "Family": "ML",
                **{key: value for key, value in result.items() if key in ("MAE", "RMSE", "MAPE", "R2")},
            }
            for result in ml_results
        ]
        model_metrics.extend(
            {
                "Model": result["architecture"].upper(),
                "Family": "DL",
                **result["metrics"],
            }
            for result in deep_results
        )

        metrics_dir = ROOT_DIR / "artifacts/metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(model_metrics).to_csv(metrics_dir / "model_comparison.csv", index=False)
        loss_dir = metrics_dir
        for result in deep_results:
            with open(loss_dir / f"{result['architecture']}_loss_history.json", "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "train_losses": result["train_losses"],
                        "validation_losses": result["validation_losses"],
                    },
                    file,
                    indent=2,
                )

        return {
            "best_ml_model": best_model_name,
            "ml_results": model_metrics[:len(ml_results)],
            "deep_results": [{key: value for key, value in result.items() if key != "predictions"} for result in deep_results],
        }


def main() -> dict[str, object]:
    """CLI entry point for training."""
    return TrainingPipeline().run()


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))