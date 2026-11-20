from pathlib import Path
from typing import Any

import joblib


class ModelTrainer:
    """Train and persist scikit-learn-compatible regression models."""

    def train(self, model: Any, X_train, y_train) -> Any:
        """
        Fit a model.

        Args:
            model: Regressor implementing fit.
            X_train: Training features.
            y_train: Training targets.

        Returns:
            The fitted model.
        """
        return model.fit(X_train, y_train)

    @staticmethod
    def save(model: Any, file_path: str | Path) -> None:
        """Serialize a fitted model with joblib."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)

    @staticmethod
    def load(file_path: str | Path) -> Any:
        """Load a fitted joblib model."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model does not exist: {path}")
        return joblib.load(path)
