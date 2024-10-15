from __future__ import annotations  # noqa: D100

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

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

    def __init__(self):
        """Initialize the Mean Squared Error metric."""
        super().__init__()

    def evaluate(self, predicted: np.ndarray, labels: np.ndarray, train: bool = False):
        """Evaluate the model and store results in the metric object."""
        result = mean_squared_error(labels, predicted)
        if train:
            self._train_acc.append(result)
        else:
            self._test_acc.append(result)


@metric_registry.register('rmse')
class RMSE(Metric):
    """Regression RMSE metric."""

    result: float | None

    def __init__(self):
        """Initialize the Root Mean Squared Error metric."""
        super().__init__()

    def evaluate(self, predicted: np.ndarray, labels: np.ndarray, train: bool = False):
        """Evaluate the model and store results in the metric object."""
        result = root_mean_squared_error(labels, predicted)
        if train:
            self._train_acc.append(result)
        else:
            self._test_acc.append(result)


@metric_registry.register('r2')
class R2(Metric):
    """Regression R2 metric."""

    def __init__(self):
        """Initialize the R2 metric."""
        super().__init__()

    def evaluate(self, predicted: np.ndarray, labels: np.ndarray, train: bool = False):
        """Evaluate the model and store results in the metric object."""
        result = r2_score(labels, predicted)
        if train:
            self._train_acc.append(result)
        else:
            self._test_acc.append(result)


@metric_registry.register('accuracy')
class Accuracy(Metric):
    """Classification accuracy."""

    def __init__(self):
        """Initialize the Accuracy metric."""
        super().__init__()

    def evaluate(self, predicted: np.ndarray, labels: np.ndarray, train: bool = False):
        """Evaluate the model and store results in the metric object."""
        result = accuracy_score(labels, predicted, normalize=True)
        if train:
            self._train_acc.append(result)
        else:
            self._test_acc.append(result)


# TODO: figure out if average=micro will return the same for binary as 'binary'
@metric_registry.register('f1')
class F1(Metric):
    """F1 accuracy metric."""

    def __init__(self):
        """Initialize the F1 metric."""
        super().__init__()

    def evaluate(self, predicted: np.ndarray, labels: np.ndarray, train: bool = False):
        """Evaluate the model and store results in the metric object."""
        result = f1_score(labels, predicted, average='micro')
        if train:
            self._train_acc.append(result)
        else:
            self._test_acc.append(result)


@metric_registry.register('pearson')
class PearsonCorrelation(Metric):
    """Pearson correlation coefficient."""

    def __init__(self):
        """Initialize the Pearson correlation coefficient metric."""
        super().__init__()

    def evaluate(self, predicted: np.ndarray, labels: np.ndarray, train: bool = False):
        """Evaluate the model and store results in the metric object."""
        pearson_r, _ = pearsonr(predicted, labels)
        if train:
            self._train_acc.append(pearson_r)
        else:
            self._test_acc.append(pearson_r)
