from __future__ import annotations  # noqa: D100

from typing import Literal

import datasets

from biolab import metric_registry
from biolab import task_registry
from biolab.api.logging import logger
from biolab.api.modeling import LM
from biolab.api.task import Task
from biolab.api.task import TaskConfig
from biolab.tasks.core.regression import sklearn_svr
from biolab.tasks.core.utils import find_transformation


class GCContentConfig(TaskConfig):
    """Configuration for MDH GC content. (Debug task)."""

    # Name of the task
    name: Literal['GCContent'] = 'GCContent'
    # embedding transformation
    output_transform: str = 'average_pool'
    # Metrics to measure TODO: should be choice of literals
    metrics: list[str] = ['mse', 'r2']

    # Task specific information:
    target_col: str = 'label'


# TODO: add caching to task (way to store some results/models/intermediates)
@task_registry.register(config=GCContentConfig)
class GCContent(Task):
    """GC content from MDH."""

    resolution: str = 'sequence'

    def __init__(self, config: GCContentConfig):
        self.config = config

    def evaluate(self, model: LM):
        """Evaluate a given model on GC content task."""
        # Load the dataset
        task_dataset = datasets.load_from_disk(self.config.dataset_name_or_path)
        task_dataset.set_format('torch')

        # Generate embeddings
        logger.info(f'Generating {model.model_input} embeddings')
        input_sequences = task_dataset[model.model_input]
        model_outputs = model.generate_embeddings(input_sequences)
        model_outputs.set_format('torch')

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
            model_outputs = model_outputs.map(
                transform.apply_hf,
                batched=True,
                fn_kwargs={
                    'sequences': input_sequences,
                    'tokenizer': model.tokenizer,
                },
            )

        # Concatenate the embeddings with the labels for classification
        task_dataset = datasets.concatenate_datasets(
            [task_dataset, model_outputs], axis=1
        )
        task_dataset.set_format('numpy')

        # Setup metrics to pass to regressor
        metrics = [metric_registry.get(metric)() for metric in self.config.metrics]
        metrics = sklearn_svr(
            task_dataset, 'embedding', self.config.target_col, metrics
        )

        for metric in metrics:
            logger.info(f'Metric: {metric.__class__.__name__}\tValue: {metric.result}')
