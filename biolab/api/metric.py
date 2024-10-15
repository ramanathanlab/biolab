from __future__ import annotations  # noqa: D100

from typing import Any
from typing import Protocol

import numpy as np


# TODO: might need to add a report method?
# TODO: add method to save metrics (or a chain of metrics?)
class Metric(Protocol):
    """Interface for a metric."""

    _train_acc: list[float]
    _test_acc: list[float]

    def __init__(self, *args, **kwargs):
        """Initialize the metric."""
        self._train_acc = []
        self._test_acc = []

    def evaluate(self, predicted: Any, *args, **kwargs) -> None:
        """Evaluate the model and store results in the metric object."""
        ...

    @property
    def result(self) -> float | None:
        """Return the result of the metric."""
        return self.test_acc

    @property
    def train_acc(self) -> float | None:
        """Return the training accuracy."""
        return np.mean(self._train_acc) if len(self._train_acc) > 0 else None

    @property
    def test_acc(self) -> float | None:
        """Return the testing accuracy."""
        return np.mean(self._test_acc) if len(self._test_acc) > 0 else None
