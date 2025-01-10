"""Boilerplate for sequence level embedding tasks."""

from __future__ import annotations

from typing import Literal

import datasets

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


class SequenceTaskConfig(TaskConfig):
    """Configuration for a general sequence level prediction task."""

    # The implementation of the task should include the name,
    # task prediction type, and metrics
    name: Literal[''] = ''
    metrics: list[str]
    task_type: Literal['classification', 'regression', 'multi-label-classification']

    # Whether to balance the classes
    balance_classes: bool = False
    # Whether to limit the number of training samples
    max_samples: int | None = None

    # Task specific information just need the label col for now
    target_col: str = 'label'
    # K-fold CV
    k_folds: int = 5


class SequenceTask(Task):
    """Boilerplate for sequence level embedding tasks."""

    resolution: str = 'sequence'

    def __init__(self, config: SequenceTaskConfig):
        super().__init__(config)

    def evaluate(self, model: LM) -> list[Metric]:
        """Evaluate task for a given model."""
        # Load the dataset
        task_dataset = datasets.load_from_disk(self.config.dataset_name_or_path)
        task_dataset.set_format('numpy')

        # Preemptively balance the classes and
        # limit the number of training samples if applicable
        if self.config.task_type == 'classification' and self.config.balance_classes:
            task_dataset = balance_classes(
                task_dataset, model.model_input, self.config.target_col
            )

        if self.config.max_samples:
            task_dataset = limit_training_samples(
                task_dataset,
                self.config.max_samples,
                model.model_input,
                self.config.target_col,
                continuous=self.config.task_type == 'regression',
            )
        # Generate embeddings
        logger.info(f'Generating {model.model_input} embeddings')
        input_sequences = task_dataset[model.model_input]

        with HDF5CachedList(
            self.config.cache_dir / f'{model.config.name}_{self.config.name}.h5'
        ) as model_outputs:
            model_outputs = model.generate_embeddings(input_sequences, model_outputs)

            # find and instantiate an output transform object
            transforms = find_transformation(
                model.model_input, model.model_encoding, self.resolution
            )
            logger.info(
                f'Found transformation {[transform.name for transform in transforms]}'
            )
            # Apply the transformations
            for transform in transforms:
                logger.info(f'Applying {transform.name} transformation')
                model_outputs.map(transform.apply_h5)

            # Create the downstream modeling dataset
            embed_dict = {
                'transformed': [output.embedding for output in model_outputs],
            }
            modeling_dataset = datasets.Dataset.from_dict(embed_dict)
            modeling_dataset = datasets.concatenate_datasets(
                [
                    task_dataset,
                    modeling_dataset,
                ],
                axis=1,
            )
            # Setup metrics to pass to downstream prediction model and run modeling
            metrics = MetricCollection(
                [get_and_instantiate_metric(metric) for metric in self.config.metrics]
            )

            # Run the downstream modeling with appropriate model
            downstream_modeling = task_map[self.config.task_type]
            metrics = downstream_modeling(
                task_dset=modeling_dataset,
                input_col='transformed',
                target_col=self.config.target_col,
                metrics=metrics,
                k_fold=self.config.k_folds,
            )

        return metrics
