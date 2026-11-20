from pathlib import Path

import pandas as pd


class DataIngestorCSV:
    """Load and validate a CSV data source."""

    def __init__(self, required_columns: tuple[str, ...] = ("date", "Appliances")) -> None:
        self.required_columns = required_columns

    def ingest(self, file_path: str | Path) -> pd.DataFrame:
        """
        Load a CSV and validate the required columns.

        Args:
            file_path: Path to the CSV file.

        Returns:
            DataFrame sorted by the datetime index.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file does not exist: {path}")

        frame = pd.read_csv(path)
        missing = set(self.required_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame = frame.set_index("date").sort_index()
        return frame
