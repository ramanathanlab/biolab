"""API for metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Protocol

import numpy as np


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

    def save(self, path: Path):
        """Save the metric to a json file."""
        output_data = {
            '_train_acc': self._train_acc,
            '_test_acc': self._test_acc,
        }
        with open(path, 'w') as f:
            json.dump(output_data, f)

    def load(self, path: Path):
        """Load the metric from a json report file."""
        with open(path) as f:
            input_data = json.load(f)
        self._train_acc = input_data['_train_acc']
        self._test_acc = input_data['_test_acc']

    # TODO: this needs a bit of work to interact with reporting script
    def report(self, format: str | None = None) -> str:
        """Return a formatted report of the metric."""
        if format is None:
            return (
                f'Metric: {self.__class__.__name__}\t'
                f'Train: {self.train_acc:0.3f}\tTest: {self.test_acc:0.3f}'
            )
        else:
            raise NotImplementedError(f'Format {format} is not supported')


# TODO add output formatters for metrics? This might live in reporting module
