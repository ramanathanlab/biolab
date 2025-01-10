"""Boilerplate for nucleotide/amino acid level embedding tasks."""

from __future__ import annotations

from typing import Literal

import datasets
import numpy as np

from biolab.api.logging import logger
from biolab.api.metric import Metric
from biolab.api.metric import MetricCollection
from biolab.api.modeling import HDF5CachedList
from biolab.api.modeling import LM
from biolab.api.task import Task
from biolab.api.task import TaskConfig
from biolab.metrics import get_and_instantiate_metric
from biolab.tasks.core.downstream import task_map
from biolab.tasks.core.downstream.classification import balance_classes
from biolab.tasks.core.utils import find_transformation
from biolab.tasks.core.utils import limit_training_samples


class CharTaskConfig(TaskConfig):
    """Configuration for a general nucleotide/amino acid level prediction task."""

    # The implementation of the task should include the name ,
    # task prediction type, and metrics
    name: Literal[''] = ''
    metrics: list[str]
    task_type: Literal['classification', 'regression', 'multi-label-classification']

    # Task specific information just need the label column for now
    target_col: str = 'label'
    # Whether to balance classes
    balance_classes: bool = False
    # Limit to number of training samples
    max_samples: int | None = None
    # K-fold cross validation
    k_folds: int = 5
    # Truncate ends of sequences (common for amino acid resolution tasks)
    truncate_end: bool = False


class CharTask(Task):
    """Boilerplate for nucleotide/amino acid level embedding tasks."""

    # Task needs to define resolution as either 'nucleotide' or 'aminoacid'
    resolution: str

    def __init__(self, config: CharTaskConfig):
        super().__init__(config)

    def evaluate(self, model: LM) -> list[Metric]:
        """Run evaluation loop given a model."""
        # Load the dataset
        task_dataset = datasets.load_from_disk(self.config.dataset_name_or_path)
        task_dataset.set_format('torch')

        # Generate embeddings
        logger.info(f'Generating {model.model_input} embeddings')
        input_sequences = task_dataset[model.model_input]
        with HDF5CachedList(
            self.config.cache_dir / f'{model.config.name}_{self.config.name}.h5'
        ) as model_outputs:
            model_outputs = model.generate_embeddings(input_sequences, model_outputs)

            # find correct transformations for embeddings and apply them
            transforms = find_transformation(
                model.model_input, model.model_encoding, self.resolution
            )
            logger.info(
                f'Found transformation {[transform.name for transform in transforms]}'
            )
            # Apply the transformations
            for transform in transforms:
                logger.info(f'Applying {transform.name} transformation')
                model_outputs.map(
                    transform.apply_h5,
                    sequences=input_sequences,
                    tokenizer=model.tokenizer,
                )
            # Flatten embeddings for residue level, flatten labels to match shape
            # truncate end of sequences if applicable
            end_pos = -1 if self.config.truncate_end else None
            token_embs = np.concatenate(
                [output.embedding[:end_pos] for output in model_outputs], axis=0
            )
            labels = [
                res_label
                for seq_label in task_dataset[self.config.target_col]
                for res_label in seq_label
            ]
            # TODO: think about caching and how to link this with the original dataset
            # ( right now it doesn't have the same length as the original dataset
            # because its flattened )
            task_dict = {'transformed': token_embs, 'flat_labels': labels}
            modeling_dataset = datasets.Dataset.from_dict(task_dict)

            # Balance the classes and limit the number of training samples if applicable
            # TODO: revisit now that we can balance real valued labels
            if (
                self.config.task_type == 'classification'
                and self.config.balance_classes
            ):
                modeling_dataset = balance_classes(
                    modeling_dataset, 'transformed', 'flat_labels'
                )

            if self.config.max_samples:
                modeling_dataset = limit_training_samples(
                    modeling_dataset,
                    self.config.max_samples,
                    'transformed',
                    'flat_labels',
                    continuous=self.config.task_type == 'regression',
                )
            # Setup metrics to pass to downstream prediction model and run modeling
            metrics = MetricCollection(
                [get_and_instantiate_metric(metric) for metric in self.config.metrics]
            )
            # Evaluate with appropriate model
            downstream_modeling = task_map[self.config.task_type]
            metrics = downstream_modeling(
                task_dset=modeling_dataset,
                input_col='transformed',
                target_col='flat_labels',
                metrics=metrics,
                k_fold=self.config.k_folds,
            )

        return metrics
