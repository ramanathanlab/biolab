from __future__ import annotations  # noqa: D100

from typing import Any
from typing import Protocol


# TODO: might need to add a report method?
# TODO: add method to save metrics (or a chain of metrics?)
class Metric(Protocol):
    """Interface for a metric."""

    result: float | None
    train_acc: float | None
    test_acc: float | None

    def __init__(self, *args, **kwargs):
        """Initialize the metric."""
        ...

    def evaluate(self, input: Any, *args, **kwargs) -> None:
        """Evaluate the model and store results in the metric object."""
        ...
