from __future__ import annotations  # noqa: D100

from typing import Literal

import datasets

from biolab import metric_registry
from biolab import task_registry
from biolab.api.logging import logger
from biolab.api.modeling import HDF5CachedList
from biolab.api.modeling import LM
from biolab.api.task import Task
from biolab.api.task import TaskConfig
from biolab.tasks.core.classification import balance_classes
from biolab.tasks.core.classification import sklearn_svc
from biolab.tasks.core.utils import find_transformation
from biolab.tasks.core.utils import limit_training_samples


class DNAClassificationConfig(TaskConfig):
    """Configuration for the DNA classification task."""

    # Name of the task
    name: Literal['DNAClassification'] = 'DNAClassification'
    # Metrics to measure TODO: should be choice of literals
    metrics: list[str] = ['accuracy', 'f1']

    # Whether to balance the classes
    balance_classes: bool = False
    # Whether to limit the number of training samples
    max_samples: int | None = None

    # Task specific information just need the label spec for now
    target_col: str = 'label'


@task_registry.register(config=DNAClassificationConfig)
class DNAClassification(Task):
    """DNA classification."""

    resolution: str = 'sequence'

    def __init__(self, config: DNAClassificationConfig):
        super().__init__(config)

    def evaluate(self, model: LM):
        """Evaluate task for a given model."""
        # Load the dataset
        task_dataset = datasets.load_from_disk(self.config.dataset_name_or_path)
        task_dataset.set_format('torch')

        # Preemptively balance the classes and
        # limit the number of training samples if applicable
        if self.config.balance_classes:
            task_dataset = balance_classes(
                task_dataset, model.model_input, self.config.target_col
            )

        if self.config.max_samples:
            task_dataset = limit_training_samples(
                task_dataset,
                self.config.max_samples,
                model.model_input,
                self.config.target_col,
            )
        # Generate embeddings
        logger.info(f'Generating {model.model_input} embeddings')
        input_sequences = task_dataset[model.model_input]

        with HDF5CachedList(
            self.config.cache_dir / 'dna_classification_outputs.hdf5'
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

            embed_dict = {
                'transformed': [output.embedding for output in model_outputs],
            }
            task_dataset = datasets.concatenate_datasets(
                [task_dataset, datasets.Dataset.from_dict(embed_dict)], axis=1
            )

            # Setup metrics to pass to classifier
            # TODO: this way of setting up metrics is a bit clunky
            metrics = [metric_registry.get(metric)() for metric in self.config.metrics]
            metrics = sklearn_svc(
                task_dataset, 'transformed', self.config.target_col, metrics
            )

        for metric in metrics:
            logger.info(f'Metric: {metric.__class__.__name__}\tValue: {metric.result}')
