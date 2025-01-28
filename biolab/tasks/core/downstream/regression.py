"""General regression models and utilities for downstream tasks."""

from __future__ import annotations

import numpy as np
from datasets import Dataset
from datasets import DatasetDict
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from biolab import SEED
from biolab import SKLEARN_RANDOM_STATE
from biolab.api.logging import logger
from biolab.api.metric import Metric
from biolab.api.task import DownstreamModel
from biolab.tasks.core.utils import mask_nan


def _run_and_evaluate_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics: list[Metric],
) -> tuple[DownstreamModel, list[Metric]]:
    """Train an SVR regressor and evaluate it using the given metrics.

    Parameters
    ----------
    X_train : np.ndarray
        The input features for the training set
    y_train : np.ndarray
        The target labels for the training set
    X_test : np.ndarray
        The input features for the test set
    y_test : np.ndarray
        The target labels for the test set
    metrics : list[Metric]
        The metrics to evaluate the model

    Returns
    -------
    tuple[DownstreamModel, list[Metric]]
        The trained SVR regressor and metrics evaluated on the training and test sets
    """
    # Remove NaN values and issue warning
    train_mask = mask_nan(X_train)
    test_mask = mask_nan(X_test)
    if ~train_mask.any() or ~test_mask.any():
        logger.warning('NaN values present in the input features')

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    # Train the SVR regressor
    regressor = SVR()
    regressor.fit(X_train, y_train)

    # Calculate metrics
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)

    for metric in metrics:
        metric.evaluate(predicted=y_train_pred, labels=y_train, train=True)
        metric.evaluate(predicted=y_test_pred, labels=y_test, train=False)

    return regressor, metrics


def sklearn_svr(
    task_dataset: Dataset | DatasetDict,
    input_col: str,
    target_col: str,
    metrics: list[Metric],
    k_fold: int = 0,
) -> tuple[dict[str, DownstreamModel | None], list[Metric]]:
    """Train a Support Vector Regressor (SVR) using the embeddings and target values.

    NOTE: If a dataset is passed that already has a train test split AND k_fold is 0,
    the train and test split will be used. Currently will fail if dataset is already
    split and k_fold is greater than 0. (TODO)

    Parameters
    ----------
    task_dataset : Dataset
        The dataset containing the input features and target labels
    input_col : str
        The name of the column containing the input features
    target_col : str
        The name of the column containing the target labels

    Returns
    -------
    tuple[dict[str, DownstreamModel | None], list[Metric]]
        The trained SVR regressor(s) and the evaluated metrics
    """
    logger.info('Evaluating with Support Vector Regressor')
    # Set dset to numpy for this function, we can return it to original later
    if isinstance(task_dataset, Dataset):
        dset_format = task_dataset.format
        task_dataset.set_format('numpy')
    elif isinstance(task_dataset, DatasetDict):
        formats = {}
        for key in task_dataset:
            formats[key] = task_dataset[key].format
            task_dataset[key].set_format('numpy')

    downstream_models = {}
    if k_fold > 0:
        # TODO: Potential bug if dataset is already split when passed into this function
        # and k_fold is greater than 0. Either check in this if statement or manually
        # force this above (assume that if train test split is present it's intended)
        X = task_dataset[input_col]
        y = task_dataset[target_col]

        kf = KFold(n_splits=k_fold, shuffle=True, random_state=SKLEARN_RANDOM_STATE)

        logger.info(f'K-Fold CV with {k_fold} folds')
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model, metrics = _run_and_evaluate_svr(
                X_train, y_train, X_test, y_test, metrics
            )
            downstream_models[f'fold_{fold_idx}'] = model
            logger.info(f'\tFold {fold_idx} completed')

    elif k_fold == 0:
        # If we are able to, split the data into train and test sets
        # If there is no `train_test_split` method, we will assume the dataset
        # is already split into train and test sets
        # TODO: this method for injecting manual splits should be more explicitly defined

        if hasattr(task_dataset, 'train_test_split') and callable(
            task_dataset.train_test_split
        ):
            svr_dset = task_dataset.train_test_split(test_size=0.2, seed=SEED)
        else:
            svr_dset = task_dataset
            assert 'train' in svr_dset and 'test' in svr_dset, (  # noqa PT018
                'The dataset does not have a train_test_split method and '
                'does not contain a train and test split'
            )
            assert len(svr_dset['train']) != 0, 'Downstream model train set is empty'
            assert len(svr_dset['test']) != 0, 'Downstream model test set is empty'

        X_train = svr_dset['train'][input_col]
        y_train = svr_dset['train'][target_col]
        X_test = svr_dset['test'][input_col]
        y_test = svr_dset['test'][target_col]

        model, metrics = _run_and_evaluate_svr(
            X_train, y_train, X_test, y_test, metrics
        )
        downstream_models['default'] = model

    # return dset to original format
    if isinstance(task_dataset, Dataset):
        task_dataset.set_format(**dset_format)
    elif isinstance(task_dataset, DatasetDict):
        for key in task_dataset:
            task_dataset[key].set_format(**formats[key])

    return downstream_models, metrics


def _run_and_evaluate_mlp_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics: list[Metric],
) -> tuple[DownstreamModel, list[Metric]]:
    """Train an MLP regressor and evaluate it using the given metrics.

    Parameters
    ----------
    X_train : np.ndarray
        The input features for the training set
    y_train : np.ndarray
        The target labels for the training set
    X_test : np.ndarray
        The input features for the test set
    y_test : np.ndarray
        The target labels for the test set
    metrics : list[Metric]
        The metrics to evaluate the model

    Returns
    -------
    Tuple[DownstreamModel, list[Metric]]
        The trained SVR regressor(s) and the evaluated metrics
    """
    # Remove NaN values and issue warning
    train_mask = mask_nan(X_train)
    test_mask = mask_nan(X_test)
    if ~train_mask.any() or ~test_mask.any():
        logger.warning('NaN values present in the input features')

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    # Train the SVR regressor
    hidden_size = X_train.shape[1]
    regressor = MLPRegressor(
        hidden_layer_sizes=(hidden_size // 2, hidden_size // 4),
        activation='relu',
        solver='adam',
        early_stopping=True,
        n_iter_no_change=5,
        random_state=SKLEARN_RANDOM_STATE,
    )
    regressor.fit(X_train, y_train)

    # Calculate metrics
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)

    for metric in metrics:
        metric.evaluate(predicted=y_train_pred, labels=y_train, train=True)
        metric.evaluate(predicted=y_test_pred, labels=y_test, train=False)

    return regressor, metrics


def sklearn_mlp_regressor(
    task_dataset: Dataset | DatasetDict,
    input_col: str,
    target_col: str,
    metrics: list[Metric],
    k_fold: int = 0,
) -> tuple[dict[str, DownstreamModel | None], list[Metric]]:
    """Train a Multi Layer Perceptron using the embeddings and target values.

    NOTE: If a dataset is passed that already has a train test split AND k_fold is 0,
    the train and test split will be used. Currently will fail if dataset is already
    split and k_fold is greater than 0. (TODO)

    Parameters
    ----------
    task_dataset : Dataset
        The dataset containing the input features and target labels
    input_col : str
        The name of the column containing the input features
    target_col : str
        The name of the column containing the target labels

    Returns
    -------
    tuple[dict[str, DownstreamModel | None], list[Metric]]
        The trained MLP regressor(s) and the evaluated metrics
    """
    logger.info('Evaluating with Multi Layer Perceptron Regressor')
    # Set dset to numpy for this function, we can return it to original later
    if isinstance(task_dataset, Dataset):
        dset_format = task_dataset.format
        task_dataset.set_format('numpy')
    elif isinstance(task_dataset, DatasetDict):
        formats = {}
        for key in task_dataset:
            formats[key] = task_dataset[key].format
            task_dataset[key].set_format('numpy')

    downstream_models = {}
    if k_fold > 0:
        # TODO: Potential bug if dataset is already split when passed into this function
        # and k_fold is greater than 0. Either check in this if statement or manually
        # force this above (assume that if train test split is present it's intended)
        X = task_dataset[input_col]
        y = task_dataset[target_col]

        kf = KFold(n_splits=k_fold, shuffle=True, random_state=SKLEARN_RANDOM_STATE)

        logger.info(f'K-Fold CV with {k_fold} folds')
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model, metrics = _run_and_evaluate_mlp_regressor(
                X_train, y_train, X_test, y_test, metrics
            )
            downstream_models[f'fold_{fold_idx}'] = model
            logger.info(f'\tFold {fold_idx} completed')

    elif k_fold == 0:
        # If we are able to, split the data into train and test sets
        # If there is no `train_test_split` method, we will assume the dataset
        # is already split into train and test sets
        # TODO: this method for injecting manual splits should be more explicitly defined

        if hasattr(task_dataset, 'train_test_split') and callable(
            task_dataset.train_test_split
        ):
            svr_dset = task_dataset.train_test_split(test_size=0.2, seed=SEED)
        else:
            svr_dset = task_dataset
            assert 'train' in svr_dset and 'test' in svr_dset, (  # noqa PT018
                'The dataset does not have a train_test_split method and '
                'does not contain a train and test split'
            )
            assert len(svr_dset['train']) != 0, 'Downstream model train set is empty'
            assert len(svr_dset['test']) != 0, 'Downstream model test set is empty'

        X_train = svr_dset['train'][input_col]
        y_train = svr_dset['train'][target_col]
        X_test = svr_dset['test'][input_col]
        y_test = svr_dset['test'][target_col]

        model, metrics = _run_and_evaluate_mlp_regressor(
            X_train, y_train, X_test, y_test, metrics
        )
        downstream_models['default'] = model

    # return dset to original format
    if isinstance(task_dataset, Dataset):
        task_dataset.set_format(**dset_format)
    elif isinstance(task_dataset, DatasetDict):
        for key in task_dataset:
            task_dataset[key].set_format(**formats[key])

    return downstream_models, metrics
