from typing import Literal

import datasets

from biolab import task_registry, metric_registry
from biolab.tasks.core.utils import find_transformation
from biolab.api.logging import logger
from biolab.api.task import Task, TaskConfig
from biolab.api.modeling import LM
from biolab.tasks.core.regression import sklearn_svr


class GCContentConfig(TaskConfig):

    # Name of the task
    name: Literal["GCContent"] = "GCContent"
    # embedding transformation
    output_transform: str = "average_pool"
    # Metrics to measure TODO: should be choice of literals
    metrics: list[str] = ["mse", "r2"]

    # Task specific information:
    target_col: str = "label"


# TODO: add caching to task (way to store some results/models/intermediates)
@task_registry.register(config=GCContentConfig)
class GCContent(Task):

    resolution: str = "sequence"

    def __init__(self, config: GCContentConfig):
        self.config = config

    def evaluate(self, model: LM):

        # Load the dataset
        task_dataset = datasets.load_from_disk(self.config.dataset_name_or_path)
        task_dataset.set_format("torch")

        # Generate embeddings
        logger.info(f"Generating {model.model_input} embeddings")
        input_sequences = task_dataset[model.model_input]
        model_outputs = model.generate_embeddings(input_sequences)

        # find and instantiate an output transform object
        transforms = find_transformation(model.model_input, model.model_encoding, self.resolution)
        logger.info(f"Found transformation {[transform.name for transform in transforms]}")
        # Apply the transformations
        for transform in transforms:
            logger.info(f"Applying {transform.name} transformation")
            model_outputs = transform.apply(
                model_outputs, sequences=input_sequences, tokenizer=model.tokenizer
            )

        embed_dict = {
            'transformed': [output.embedding for output in model_outputs],
        }
        task_dataset = datasets.concatenate_datasets(
            [task_dataset, datasets.Dataset.from_dict(embed_dict)], axis=1
        )

        # Setup metrics to pass to regressor
        metrics = [metric_registry.get(metric)() for metric in self.config.metrics]
        metrics = sklearn_svr(
            task_dataset, 'transformed', self.config.target_col, metrics
        )

        for metric in metrics:
            logger.info(f"Metric: {metric.__class__.__name__}\tValue: {metric.result}")
