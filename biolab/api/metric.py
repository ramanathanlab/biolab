from typing import Protocol, Any


# TODO: might need to add a report method?
# TODO: add method to save metrics (or a chain of metrics?)
class Metric(Protocol):
    def __init__(self, *args, **kwargs):
        """Initialize the metric"""
        ...

    def evaluate(self, input: Any, *args, **kwargs) -> None:
        """Evaluate the model and store results in the metric object."""
        ...
