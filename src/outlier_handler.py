from dataclasses import dataclass

import pandas as pd


@dataclass
class ClipBounds:
    """Lower and upper clipping bounds for one feature."""

    lower: float
    upper: float


class OutlierClipper:
    """Clip selected features using bounds fitted only on training data."""

    def __init__(
        self,
        columns: list[str],
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> None:
        self.columns = columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.bounds: dict[str, ClipBounds] = {}

    def fit(self, training_frame: pd.DataFrame) -> "OutlierClipper":
        """
        Fit clipping bounds from training data.

        Args:
            training_frame: Chronologically earlier training rows.

        Returns:
            This fitted clipper.
        """
        for column in self.columns:
            if column not in training_frame.columns:
                continue
            lower = float(training_frame[column].quantile(self.lower_quantile))
            upper = float(training_frame[column].quantile(self.upper_quantile))
            self.bounds[column] = ClipBounds(lower=lower, upper=upper)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted clipping bounds to a frame.

        Args:
            frame: DataFrame to transform.

        Returns:
            A clipped copy of the frame.
        """
        result = frame.copy()
        for column, bounds in self.bounds.items():
            if column in result.columns:
                result[column] = result[column].clip(bounds.lower, bounds.upper)
        return result

    def fit_transform(self, training_frame: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and immediately transform it."""
        return self.fit(training_frame).transform(training_frame)
