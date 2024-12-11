"""Implementation of metrics for downstream model performance."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

from biolab.api.metric import Metric

# TODO: Limit certain metrics to classification/regression/(other?) tasks
# TODO: Store higher performing direction as a field (is_higher_better?)
# TODO: Make container for storing groups of them or initializing groups of them


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


class SpearmanCorrelation(Metric):
    """Spearman correlation.

    Spearman correlation is a non-parametric measure of rank correlation.
    """

    def __init__(self):
        """Initialize the Spearman correlation metric."""
        super().__init__()

    def evaluate(self, predicted: np.ndarray, labels: np.ndarray, train: bool = False):
        """Evaluate the model and store results in the metric object."""
        spearman_r, _ = spearmanr(predicted, labels)
        if train:
            self._train_acc.append(spearman_r)
        else:
            self._test_acc.append(spearman_r)


metric_registry = {
    'mse': MSE,
    'rmse': RMSE,
    'r2': R2,
    'accuracy': Accuracy,
    'f1': F1,
    'pearson': PearsonCorrelation,
    'spearman': SpearmanCorrelation,
}
