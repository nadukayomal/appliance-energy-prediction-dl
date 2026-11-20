import numpy as np
import pandas as pd


class FeatureEngineer:
    """Create time, lag, rolling, and domain features."""

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a datetime-indexed frame into model features.

        Args:
            frame: Data ordered by a DatetimeIndex.

        Returns:
            Feature frame with lag and rolling values. Initial NaNs are retained.
        """
        result = frame.copy()

        result["hour"] = result.index.hour
        result["day_of_week"] = result.index.dayofweek
        result["month"] = result.index.month
        result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
        result["NSM"] = result["hour"] * 3600
        result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
        result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
        result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
        result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)

        for lag in (1, 2, 3, 6, 12, 18, 36):
            result[f"lag_{lag}"] = result["Appliances"].shift(lag)

        for window, name in ((6, "1h"), (12, "2h"), (18, "3h")):
            rolling = result["Appliances"].shift(1).rolling(window=window)
            result[f"roll_mean_{name}"] = rolling.mean()
            result[f"roll_std_{name}"] = rolling.std()

        temp_columns = ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"]
        humidity_columns = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"]
        result["temp_in_mean"] = result[temp_columns].mean(axis=1)
        result["rh_in_mean"] = result[humidity_columns].mean(axis=1)
        result["temp_x_humidity"] = result["temp_in_mean"] * result["rh_in_mean"]
        result["in_out_temp_diff"] = result["temp_in_mean"] - result["T_out"]
        result["lights_on"] = (result["lights"] > 0).astype(int)
        
        return result
