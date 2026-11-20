import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    """Calculate regression evaluation metrics."""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """
        Calculate MAE, RMSE, MAPE, and R².

        Args:
            y_true: Actual target values.
            y_pred: Predicted target values.

        Returns:
            Dictionary containing named metric values.
        """
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = r2_score(y_true, y_pred)
        epsilon = np.finfo(np.float64).eps
        mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100)
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
