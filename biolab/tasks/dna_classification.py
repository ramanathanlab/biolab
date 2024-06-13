from typing import Literal, Optional

import datasets
from biolab import task_registry, transform_registry, metric_registry
from biolab.api.logging import logger
from biolab.api.task import Task, TaskConfig
from biolab.api.lm import LM
from biolab.modeling.embed import generate_embeddings
from biolab.tasks.core.classification import (
    sklearn_svc,
    limit_training_samples,
    balance_classes,
)


class DNAClassificationConfig(TaskConfig):
    """Configuration for the DNA classification task."""

    # Name of the task
    name: Literal["DNAClassification"] = "DNAClassification"
    # Embedding transformation
    output_transform: str = "average_pool"
    # Metrics to measure TODO: should be choice of literals
    metrics: list[str] = ["accuracy", "f1"]

    # Wether to balance the classes
    balance_classes: bool = False
    # Whether to limit the number of training samples
    max_samples: Optional[int] = None

    # Task specific information just need the label spec for now
    target_col: str = "label"


@task_registry.register(config=DNAClassificationConfig)
class DNAClassification(Task):
    def __init__(self, config: DNAClassificationConfig):
        self.config = config

    def evaluate(self, model: LM):
        # Load the dataset
        task_dataset = datasets.load_from_disk(self.config.dataset_name_or_path)
        task_dataset.set_format("torch")

        # Preemptively balance the classes and
        # limit the number of training samples if applicable
        if self.config.balance_classes:
            task_dataset = balance_classes(
                task_dataset, "input_dna", self.config.target_col
            )

        if self.config.max_samples:
            task_dataset = limit_training_samples(
                task_dataset,
                self.config.max_samples,
                "input_dna",  # TODO: this should be the LM input spec
                self.config.target_col,
            )

        # find and instantiate an output transform object
        transform = transform_registry.get(self.config.output_transform)
        assert transform, f"Transform {self.config.output_transform} not found"

        # Generate embeddings
        logger.info("Generating embeddings")
        # TODO: there is some coupling between the task and the generate embeddings
        # in that the tokenizer is hard coded to look for `input_dna`
        embed_dataset = generate_embeddings(model, task_dataset, transform)

        # Setup metrics to pass to classifier
        # TODO: this way of setting up metrics is a bit clunky
        metrics = [metric_registry.get(metric)() for metric in self.config.metrics]
        metrics = sklearn_svc(
            embed_dataset, transform.name, self.config.target_col, metrics
        )

        for metric in metrics:
            logger.info(f"Metric: {metric.__class__.__name__}\tValue: {metric.result}")
