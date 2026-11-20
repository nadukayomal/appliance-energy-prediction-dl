import numpy as np
import torch

from src.deep_learning_model import RecurrentForecaster
from src.model_evaluation import ModelEvaluator
from pipeline.training_pipeline import DeepLearningTrainer


def test_create_sequences_shapes() -> None:
    features = np.arange(50).reshape(10, 5)
    targets = np.arange(10)
    X_sequences, y_sequences = DeepLearningTrainer.create_sequences(features, targets, time_steps=3)

    assert X_sequences.shape == (7, 3, 5)
    assert y_sequences.shape == (7, 1)


def test_model_evaluator_returns_expected_metrics() -> None:
    metrics = ModelEvaluator.evaluate(
        np.array([100.0, 200.0, 300.0]),
        np.array([110.0, 190.0, 290.0]),
    )

    assert metrics["MAE"] == 10.0
    assert metrics["RMSE"] == 10.0
    assert metrics["R2"] > 0.9


def test_recurrent_forecaster_forward_shape() -> None:
    model = RecurrentForecaster(input_size=5, architecture="lstm")
    sequence = torch.zeros((4, 10, 5), dtype=torch.float32)

    prediction = model(sequence)

    assert prediction.shape == (4, 1)
