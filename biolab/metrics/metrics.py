from __future__ import annotations  # noqa: D100

import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr

from biolab import metric_registry
from biolab.api.metric import Metric

# TODO: Limit certain metrics to classification/regression/(other?) tasks
# TODO: generalize metrics + make container for storing groups of them
# or initializing groups of them
# TODO: add train/test fields to metrics
# TODO: rename labels and input to be pred, and target


@metric_registry.register('mse')
class MSE(Metric):
    """Regression mean squared error."""

    result: float | None
    train_acc: float | None
    test_acc: float | None

    def __init__(self):
        """Initialize the Mean Squared Error metric."""
        self.result = None
        self.train_acc = None
        self.test_acc = None

    def evaluate(self, input: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        """Evaluate the model and store results in the metric object."""
        self.result = mean_squared_error(labels, input)


@metric_registry.register('rmse')
class RMSE(Metric):
    """Regression RMSE metric."""

    result: float | None

    def __init__(self):
        """Initialize the Root Mean Squared Error metric."""
        self.result = None

    def evaluate(self, input: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        """Evaluate the model and store results in the metric object."""
        self.result = root_mean_squared_error(labels, input)


@metric_registry.register('r2')
class R2(Metric):
    """Regression R2 metric."""

    result: float | None

    def __init__(self):
        """Initialize the R2 metric."""
        self.result = None

    def evaluate(self, input: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        """Evaluate the model and store results in the metric object."""
        self.result = r2_score(labels, input)


@metric_registry.register('accuracy')
class Accuracy(Metric):
    """Classification accuracy."""

    result: float | None

    def __init__(self):
        """Initialize the Accuracy metric."""
        self.result = None

    def evaluate(self, input: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        """Evaluate the model and store results in the metric object."""
        self.result = accuracy_score(labels, input, normalize=True)


# TODO: figure out if average=micro will return the same for binary as 'binary'
@metric_registry.register('f1')
class F1(Metric):
    """F1 accuracy metric."""

    result: float | None

    def __init__(self):
        """Initialize the F1 metric."""
        self.result = None

    def evaluate(self, input: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        """Evaluate the model and store results in the metric object."""
        self.result = f1_score(labels, input, average='micro')


@metric_registry.register('pearson')
class PearsonCorrelation(Metric):
    """Pearson correlation coefficient."""

    result: float | None

    def __init__(self):
        """Initialize the Pearson correlation coefficient metric."""
        self.result = None

    def evaluate(self, input: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        """Evaluate the model and store results in the metric object."""
        pearson_r, _ = pearsonr(input, labels)
        self.result = pearson_r
