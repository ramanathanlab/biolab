from typing import Literal

import datasets

from biolab import task_registry, transform_registry, metric_registry
from biolab.api.logging import logger
from biolab.api.task import Task, TaskConfig
from biolab.api.lm import LM
from biolab.modeling.embed import generate_embeddings
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


# TODO: generalize metrics + make container for storing them
# TODO: add caching to task (way to store some results/models/intermediates)
@task_registry.register(config=GCContentConfig)
class GCContent(Task):
    def __init__(self, config: GCContentConfig):
        self.config = config

    def evaluate(self, model: LM):

        # Load the dataset
        task_dataset = datasets.load_from_disk(self.config.dataset_name_or_path)
        task_dataset.set_format("torch")

        # find and instantiate an output transform object
        transform = transform_registry.get(self.config.output_transform)
        assert transform, f"Transform {self.config.output_transform} not found."

        # Generate embeddings
        logger.info("Generating embeddings")
        # TODO: there is some coupling between the task and the generate embeddings
        # in that the tokenizer is hard coded to look for `input_dna`
        embed_dataset = generate_embeddings(model, task_dataset, transform)

        # Setup metrics to pass to regressor
        metrics = [metric_registry.get(metric)() for metric in self.config.metrics]
        metrics = sklearn_svr(
            embed_dataset, transform.name, self.config.target_col, metrics
        )

        for metric in metrics:
            logger.info(f"Metric: {metric.__class__.__name__}\tValue: {metric.result}")
