from __future__ import annotations  # noqa: D100

import numpy as np
from datasets import Dataset
from sklearn.svm import SVR

from biolab.api.logging import logger
from biolab.api.metric import Metric


def sklearn_svr(
    task_dset: Dataset, input_col: str, target_col: str, metrics: list[Metric]
):
    """Train a Support Vector Regressor (SVR) using the embeddings and target values.

    Parameters
    ----------
    task_dset : Dataset
        The dataset containing the input features and target labels
    input_col : str
        The name of the column containing the input features
    target_col : str
        The name of the column containing the target labels

    Returns
    -------
    Tuple
        The trained SVR regressor, train mse, and test mse
    """
    # set dset to numpy for this function, we can return it to original later
    dset_format = task_dset.format
    task_dset.set_format('numpy')

    # Split the data into train and test sets
    # TODO: take the seed to somewhere more central this is in the weeds
    svr_dset = task_dset.train_test_split(test_size=0.2, seed=42)

    X_train = svr_dset['train'][input_col]  # noqa: N806
    y_train = svr_dset['train'][target_col]
    X_test = svr_dset['test'][input_col]  # noqa: N806
    y_test = svr_dset['test'][target_col]

    # Remove NaN values and issue warning
    # TODO: this is a common pattern, should be a utility function
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        logger.warning('NaN values present in the input features')
        train_mask = ~np.isnan(X_train).any(axis=1)
        test_mask = ~np.isnan(X_test).any(axis=1)

        X_train = X_train[train_mask]  # noqa N806
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]  # noqa N806
        y_test = y_test[test_mask]

    # Train the SVR regressor
    regressor = SVR()
    regressor.fit(X_train, y_train)

    # Calculate metrics
    y_train_pred = regressor.predict(X_train)  # noqa: F841
    y_test_pred = regressor.predict(X_test)

    for metric in metrics:
        metric.evaluate(input=y_test_pred, labels=y_test)

    # return dset to original format
    task_dset.set_format(**dset_format)

    return metrics
