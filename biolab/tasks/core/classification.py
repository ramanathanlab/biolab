from __future__ import annotations  # noqa: D100

import numpy as np
import torch
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils import resample

from biolab.api.logging import logger
from biolab.api.metric import Metric


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
    X = task_dset[input_col]  # noqa: N806
    y = task_dset[target_col]
    # TODO: this feels a bit if else-y can we generalize or enforce formats earlier?
    # This cast is because if i already have numeric labels it will fail, it should be a list?
    if isinstance(y, torch.Tensor):
        y = y.tolist()

    # Identify unique classes and their counts
    unique_classes = list(set(y))
    class_counts = {cls: y.count(cls) for cls in unique_classes}
    min_class_size = min(class_counts.values())

    balanced_X = []  # noqa: N806
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


def limit_training_samples(
    task_dset: Dataset, max_samples: int, input_col: str, target_col: str
) -> Dataset:
    """Limit the total number of training examples respecting the class balance.

    Parameters
    ----------
    task_dset : datasets.Dataset
        The dataset containing the input features and target labels
    max_samples : int
        Maximum number of samples to use for training
    input_col : str
        The name of the column containing the input features
    target_col : str
        The name of the column containing the target labels


    Returns
    -------
    datasets.Dataset
        The dataset with a limited number of training examples
    """
    # Short circuit if the dataset is already smaller than the maximum number of samples
    if max_samples >= len(task_dset):
        return task_dset

    # Extract the input features and target labels
    X = task_dset[input_col]  # noqa: N806
    y = task_dset[target_col]

    # Calculate the proportion of each class
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total_samples = sum(class_counts)
    class_proportions = {
        cls: count / total_samples
        for cls, count in zip(unique_classes, class_counts, strict=False)
    }

    # Determine the number of samples for each class
    class_sample_counts = {
        cls: int(round(proportion * max_samples))
        for cls, proportion in class_proportions.items()
    }

    # Ensure the total number of samples is exactly max_samples
    while sum(class_sample_counts.values()) != max_samples:
        diff = max_samples - sum(class_sample_counts.values())
        for cls in unique_classes:
            if diff == 0:
                break
            if diff > 0:
                class_sample_counts[cls] += 1
                diff -= 1
            elif class_sample_counts[cls] > 0:
                class_sample_counts[cls] -= 1
                diff += 1

    limited_X = []  # noqa: N806
    limited_y = []

    # Sample the dataset to limit the total number of training examples
    # and respecting class balance
    for class_value in unique_classes:
        class_indices = [i for i, label in enumerate(y) if label == class_value]
        class_sampled_indices = resample(
            class_indices,
            replace=False,
            n_samples=class_sample_counts[class_value],
            random_state=42,
        )
        limited_X.extend([X[i] for i in class_sampled_indices])
        limited_y.extend([y[i] for i in class_sampled_indices])

    # Create a new dataset with limited samples
    limited_dataset = Dataset.from_dict({input_col: limited_X, target_col: limited_y})

    return limited_dataset


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


def sklearn_svc(
    task_dset: Dataset, input_col: str, target_col: str, metrics: list[Metric]
):
    """Train a Support Vector Classifier (SVC) using the embeddings and target values.

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
    dset_format = task_dset.format
    task_dset.set_format('numpy')

    # Split the data into train and test sets
    svc_dset = task_dset.train_test_split(test_size=0.2, seed=42)

    X_train = svc_dset['train'][input_col]  # noqa: N806
    X_test = svc_dset['test'][input_col]  # noqa: N806
    # This is a null transform if already in labels
    y_test = object_to_label(svc_dset['test'][target_col])
    y_train = object_to_label(svc_dset['train'][target_col])

    # Remove NaN values and issue warning
    # TODO: this is a common pattern, should be a utility function
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        logger.warning('NaN values present in the input features')
        train_mask = ~np.isnan(X_train).any(axis=1)
        test_mask = ~np.isnan(X_test).any(axis=1)

        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

    # Train the SVC classifier
    classifier = SVC()
    classifier.fit(X_train, y_train)

    # Calculate metrics
    y_train_pred = classifier.predict(X_train)  # noqa: F841
    y_test_pred = classifier.predict(X_test)

    for metric in metrics:
        metric.evaluate(input=y_test_pred, labels=y_test)

    # Return dset to original format
    task_dset.set_format(**dset_format)

    return metrics
