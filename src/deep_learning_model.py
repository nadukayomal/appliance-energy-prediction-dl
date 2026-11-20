from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn


class RecurrentForecaster(nn.Module):
    """Configurable LSTM or GRU regressor."""

    def __init__(
        self,
        input_size: int,
        architecture: Literal["lstm", "gru"] = "lstm",
        hidden_size1: int = 64,
        hidden_size2: int = 32,
        dense_units: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        recurrent = nn.LSTM if architecture == "lstm" else nn.GRU
        self.recurrent1 = recurrent(input_size, hidden_size1, batch_first=True)
        self.recurrent2 = recurrent(hidden_size1, hidden_size2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size2, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass and predict one target value per sequence."""
        output, _ = self.recurrent1(features)
        output = self.dropout(output)
        output, _ = self.recurrent2(output)
        output = self.dropout(output[:, -1, :])
        output = torch.relu(self.fc1(output))
        return self.fc2(output)

    def save(self, file_path: str | Path, input_size: int) -> None:
        """Save model weights and the architecture metadata."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "architecture": self.architecture,
                "input_size": input_size,
                "hidden_size1": self.hidden_size1,
                "hidden_size2": self.hidden_size2,
            },
            path,
        )

    @classmethod
    def load(cls, file_path: str | Path, device: torch.device) -> "RecurrentForecaster":
        """Load a recurrent forecaster from a saved checkpoint."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Deep learning model does not exist: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            input_size=checkpoint["input_size"],
            architecture=checkpoint["architecture"],
            hidden_size1=checkpoint["hidden_size1"],
            hidden_size2=checkpoint["hidden_size2"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device).eval()
