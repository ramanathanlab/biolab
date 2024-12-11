"""General classification utilities."""

from __future__ import annotations

import numpy as np
import torch
from datasets import Dataset
from datasets import DatasetDict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils import resample

from biolab.api.logging import logger
from biolab.api.metric import Metric
from biolab.tasks.core.utils import mask_nan


def balance_classes(task_dset: Dataset, input_col: str, target_col: str) -> Dataset:
    """Balance classes by undersampling the majority classes.

    Parameters
    ----------
    task_dset : datasets.Dataset
        The dataset containing the input features and target labels
    input_col : str
        The name of the column containing the input features
    target_col : str
        The name of the column containing the target labels

    Returns
    -------
    datasets.Dataset
        The balanced dataset
    """
    # Extract the input features and target labels
    X = task_dset[input_col]
    y = task_dset[target_col]
    # TODO: this feels a bit if else-y can we generalize or enforce formats earlier?
    # This cast is because if labels are already numeric it will fail, should be list?
    if isinstance(y, torch.Tensor):
        y = y.tolist()

    # Identify unique classes and their counts
    unique_classes, counts = np.unique(y, return_counts=True)
    # class_counts = dict(zip(unique_classes, counts))
    min_class_size = counts.min()

    balanced_X = []
    balanced_y = []

    # Undersample each class to the size of the smallest class
    for class_value in unique_classes:
        class_indices = [i for i, label in enumerate(y) if label == class_value]
        class_sampled_indices = resample(
            class_indices, replace=False, n_samples=min_class_size, random_state=42
        )
        balanced_X.extend([X[i] for i in class_sampled_indices])
        balanced_y.extend([y[i] for i in class_sampled_indices])

    # Create a new balanced dataset
    balanced_dataset = Dataset.from_dict(
        {input_col: balanced_X, target_col: balanced_y}
    )

    return balanced_dataset


def object_to_label(semantic_labels: list[object]) -> np.ndarray:
    """Convert a list of objects into class labels appropriate for SVC.

    Parameters
    ----------
    objects_list : list
        List of objects to be converted into class labels

    Returns
    -------
    list
        List of class labels
    """
    le = LabelEncoder()
    return le.fit_transform(semantic_labels)


def _run_and_evaluate_svc(X_train, y_train, X_test, y_test, metrics):
    """Train a Support Vector Classifier (SVC) using the embeddings and target values.

    Parameters
    ----------
    X_train : np.ndarray
        The input features of the training set
    y_train : np.ndarray
        The target labels of the training set
    X_test : np.ndarray
        The input features of the test set
    y_test : np.ndarray
        The target labels of the test set
    metrics : list
        List of metrics to evaluate the classifier

    Returns
    -------
    list[Metric]
        List of metrics
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

    # Train the SVC classifier
    classifier = SVC()
    classifier.fit(X_train, y_train)

    # Get dataset predictions
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    # Evaluate the classifier
    for metric in metrics:
        metric.evaluate(predicted=y_train_pred, labels=y_train, train=True)
        metric.evaluate(predicted=y_test_pred, labels=y_test, train=False)

    return metrics


def sklearn_svc(
    task_dset: Dataset,
    input_col: str,
    target_col: str,
    metrics: list[Metric],
    k_fold: int = 0,
):
    """Train a Support Vector Classifier (SVC) using the embeddings and target values.

    NOTE: If a dataset is passed that already has a train test split AND k_fold is 0,
    the train and test split will be used. Currently will fail if dataset is already
    split and k_fold is greater than 0. (TODO)

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
        The trained SVC classifier, train accuracy, and test accuracy
    """
    # Set dset to numpy for this function, we can return it to original later
    if isinstance(task_dset, Dataset):
        dset_format = task_dset.format
        task_dset.set_format('numpy')
    elif isinstance(task_dset, DatasetDict):
        formats = {}
        for key in task_dset:
            formats[key] = task_dset[key].format
            task_dset[key].set_format('numpy')

    if k_fold > 0:
        # TODO: Potential bug if dataset is already split when passed into this function
        # and k_fold is greater than 0. Either check in this if statement or manually
        # force this above (assume that if train test split is present it's intended)
        X = task_dset[input_col]
        y = object_to_label(task_dset[target_col])

        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)

        logger.info(f'K-Fold CV with {k_fold} folds')
        for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            logger.info(f'\tFold {fold_idx}')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            metrics = _run_and_evaluate_svc(X_train, y_train, X_test, y_test, metrics)

    elif k_fold == 0:
        # If we are able to, split the data into train and test sets
        # If there is no `train_test_split` method, we will assume the dataset
        # is already split into train and test sets
        # TODO: take the seed to somewhere more central this is in the weeds
        # TODO: this method for injecting manual splits should be more explicitly defined

        if hasattr(task_dset, 'train_test_split') and callable(
            task_dset.train_test_split
        ):
            svc_dset = task_dset.train_test_split(test_size=0.2, seed=42)
        else:
            svc_dset = task_dset
            assert 'train' in svc_dset and 'test' in svc_dset, (  # noqa PT018
                'The dataset does not have a train_test_split method and '
                'does not contain a train and test split'
            )

        X_train = svc_dset['train'][input_col]
        X_test = svc_dset['test'][input_col]
        # This is a null transform if already in labels
        y_test = object_to_label(svc_dset['test'][target_col])
        y_train = object_to_label(svc_dset['train'][target_col])

        metrics = _run_and_evaluate_svc(X_train, y_train, X_test, y_test, metrics)

    # Return dset to original format
    if isinstance(task_dset, Dataset):
        task_dset.set_format(**dset_format)
    elif isinstance(task_dset, DatasetDict):
        for key in task_dset:
            task_dset[key].set_format(**formats[key])

    return metrics
