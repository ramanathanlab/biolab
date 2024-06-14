from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    accuracy_score,
    r2_score,
    f1_score,
)
import torch

from biolab.api.metric import Metric
from biolab import metric_registry

# TODO: Limit certain metrics to classification/regression/(other?) tasks
# TODO: generalize metrics + make container for storing groups of them
# or initializing groups of them
# TODO: add train/test fields to metrics


@metric_registry.register("mse")
class MSE(Metric):

    result: float | None

    def __init__(self):
        """Initialize the Mean Squared Error metric."""
        self.result = None

    def evaluate(self, input: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        """Evaluate the model and store results in the metric object."""
        self.result = mean_squared_error(labels, input)


@metric_registry.register("rmse")
class RMSE(Metric):

    result: float | None

    def __init__(self):
        """Initialize the Root Mean Squared Error metric."""
        self.result = None

    def evaluate(self, input: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        """Evaluate the model and store results in the metric object."""
        self.result = root_mean_squared_error(labels, input)


@metric_registry.register("r2")
class R2(Metric):

    result: float | None

    def __init__(self):
        """Initialize the R2 metric."""
        self.result = None

    def evaluate(self, input: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        """Evaluate the model and store results in the metric object."""
        self.result = r2_score(labels, input)


@metric_registry.register("accuracy")
class Accuracy(Metric):

    result: float | None

    def __init__(self):
        """Initialize the Accuracy metric."""
        self.result = None

    def evaluate(self, input: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        """Evaluate the model and store results in the metric object."""
        self.result = accuracy_score(labels, input, normalize=True)


# TODO: figure out if micro will return the same for binary as 'binary'
@metric_registry.register("f1")
class F1(Metric):

    result: float | None

    def __init__(self):
        """Initialize the F1 metric."""
        self.result = None

    def evaluate(self, input: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        """Evaluate the model and store results in the metric object."""
        self.result = f1_score(labels, input, average="micro")
