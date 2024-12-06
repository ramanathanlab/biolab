"""Downstream tasks utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from datasets import Dataset
from sklearn.utils import resample

from biolab.api.modeling import Transform
from biolab.modeling.transforms import transform_registry


# TODO: use enums for task inputs, and resolutions to make this mapping
# smoother. This will make the repeitition less error prone.
def find_transformation(
    model_input: str, model_resolution: str, task_resolution: str
) -> Sequence[Transform]:
    """Map task input, model resolution, and task resolution to a transformation.

    Parameters
    ----------
    task_input : str
        task input type, must be 'dna', or 'aminoacid'
    model_resolution : str
        resolution of the tokens level representations from the model
    task_resolution : str
        required hidden representation granularity

    Returns
    -------
    Sequence[Transform]
        Sequence of transformations to be applied iteratively

    Raises
    ------
    ValueError
        If the resolution mapping is not found, or transform is not found in registry
    """
    # Map model resolution to task resolution through a transform
    task_transform_mapping = {
        # For each task resolution, map model encoding to transform name
        'dna': {
            'sequence': {
                'bpe': ('average_pool',),
                'char': ('average_pool',),
                '3mer': ('average_pool',),
                '6mer': ('average_pool',),
            },
            'nucleotide': {
                'bpe': ('super_resolution',),
                'char': ('full_sequence',),
                '3mer': ('super_resolution',),
                '6mer': ('super_resolution',),
            },
            'aminoacid': {
                'bpe': ('super_resolution', '3_window'),
                'char': ('3_window',),
                '3mer': ('full_sequence',),
                '6mer': ('super_resolution', '3_window'),
            },
        },
        'aminoacid': {
            'sequence': {
                'bpe': ('average_pool',),
                'char': ('average_pool',),
                '3mer': ('average_pool',),
                '6mer': ('average_pool',),
            },
            'nucleotide': {
                # TODO: this isn't possible, should raise exception elsewhere
            },
            'aminoacid': {
                'bpe': ('super_resolution',),
                'char': ('full_sequence',),
                '3mer': ('super_resolution',),
                '6mer': ('super_resolution',),
            },
        },
    }

    # Retrieve list of transforms from the registry
    transform_names = (
        task_transform_mapping.get(model_input, {})
        .get(task_resolution, {})
        .get(model_resolution, None)
    )

    # Check that we haven't missed a mapping
    if transform_names is None:
        raise ValueError(
            f'Resolution mapping not found for {model_input=}, {task_resolution=},'
            f' and {model_resolution=}'
        )

    # Assert that we have all the transforms registered (TODO: goes away with enums)
    for t_name in transform_names:
        if t_name not in transform_registry:
            raise ValueError(f'Transform {t_name} not found in registry')

    return [transform_registry.get(name) for name in transform_names]


def limit_training_samples(
    task_dset: Dataset,
    max_samples: int,
    input_col: str,
    target_col: str,
    continuous=False,
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
    continuous : bool
        Whether the target labels are continuous, if so will bin and balance


    Returns
    -------
    datasets.Dataset
        The dataset with a limited number of training examples
    """
    # Short circuit if the dataset is already smaller than the maximum number of samples
    if max_samples >= len(task_dset):
        return task_dset

    # Extract the input features and target labels
    y = task_dset[target_col]

    # If there are continuoys labels, bin them to balance with classes
    if continuous:
        y_bins = np.digitize(y, np.histogram_bin_edges(y, bins='auto'))
    else:
        y_bins = y

    # Calculate the proportion of each class
    unique_classes, class_counts = np.unique(y_bins, return_counts=True)
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
        for label_class in unique_classes:
            if diff == 0:
                break
            if diff > 0:
                class_sample_counts[label_class] += 1
                diff -= 1
            elif class_sample_counts[label_class] > 0:
                class_sample_counts[label_class] -= 1
                diff += 1

    # Sample the dataset to limit the total number of training examples
    # and respecting class balance
    sampled_indices = []
    for class_value in unique_classes:
        class_indices = [i for i, label in enumerate(y_bins) if label == class_value]

        # TODO: in the continuous setting sometimes the bins are too
        # small to sample from, right now we skip but maybe revisit bin
        # size calculation.
        if class_sample_counts[class_value] == 0:
            continue

        class_sampled_indices = resample(
            class_indices,
            replace=False,
            n_samples=class_sample_counts[class_value],
            random_state=42,
        )
        sampled_indices.extend(class_sampled_indices)

    return task_dset.select(sampled_indices)


def mask_nan(data: np.ndarray) -> np.ndarray:
    """Return mask of same shape as input array, with True where NaN values are present.

    Parameters
    ----------
    mat : np.ndarray
        The input numpy array

    Returns
    -------
    np.ndarray
        The numpy array with NaN values masked
    """
    return ~np.isnan(data).any(axis=1)
