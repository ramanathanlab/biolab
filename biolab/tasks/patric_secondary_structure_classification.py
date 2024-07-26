from typing import Literal, Optional

import datasets
from biolab import task_registry, metric_registry
from biolab.tasks.core.utils import find_transformation
from biolab.api.logging import logger
from biolab.api.task import Task, TaskConfig
from biolab.api.modeling import LM
from biolab.tasks.core.classification import (
    sklearn_svc,
    limit_training_samples,
    balance_classes,
)
import numpy as np


class PatricSecondaryStructureClassificationConfig(TaskConfig):
    """Configuration for the PATRIC secondary structure classification task."""

    # Name of the task
    name: Literal["PatricSecondaryStructureClassification"] = (
        "PatricSecondaryStructureClassification"
    )
    # Dataset source
    dataset_name_or_path: str
    # Metrics to measure TODO: should be choice of literals
    metrics: list[str] = ["accuracy", "f1"]

    # Task specific information just need the label spec for now
    target_col: str = "label"
    # Whether to balance classes
    balance_classes: bool = False
    # Limit to number of training samples
    max_samples: Optional[int] = None


@task_registry.register(config=PatricSecondaryStructureClassificationConfig)
class PatricSecondaryStructureClassification(Task):

    resolution: str = "aminoacid"

    def __init__(self, config: PatricSecondaryStructureClassificationConfig):
        self.config = config

    def evaluate(self, model: LM):
        # Load the dataset
        task_dataset = datasets.load_from_disk(self.config.dataset_name_or_path)
        task_dataset.set_format("torch")

        # Generate embeddings
        logger.info(f"Generating {model.model_input} embeddings")
        input_sequences = task_dataset[model.model_input]
        model_outputs = model.generate_embeddings(input_sequences)

        # find correct transformations for embeddings and apply them
        transforms = find_transformation(
            model.model_input, model.model_encoding, self.resolution
        )
        logger.info(
            f"Found transformation {[transform.name for transform in transforms]}"
        )
        # Apply the transformations
        for transform in transforms:
            logger.info(f"Applying {transform.name} transformation")
            model_outputs = transform.apply(
                model_outputs, sequences=input_sequences, tokenizer=model.tokenizer
            )
        # Flatten embeddings for residue level embeddings, flatten labels to match shape
        # Need to take off the end token of each sequence as there is no DSSP output for these positions
        token_embs = np.concatenate(
            [output.embedding[:-1,] for output in model_outputs], axis=0
        )
        labels = [
            res_label
            for seq_label in task_dataset[self.config.target_col]
            for res_label in seq_label
        ]

        # TODO: think about caching and how to link this with the original dataset
        # (right now it doesn't have the same length as the original dataset because its flattened)
        task_dict = {"transformed": token_embs, "flat_labels": labels}
        modeling_dataset = datasets.Dataset.from_dict(task_dict)

        # Balance the classes and limit the number of training samples if applicable
        if self.config.balance_classes:
            modeling_dataset = balance_classes(
                modeling_dataset, "transformed", "flat_labels"
            )

        if self.config.max_samples:
            modeling_dataset = limit_training_samples(
                modeling_dataset,
                self.config.max_samples,
                "transformed",
                "flat_labels",
            )
        logger.info(modeling_dataset)

        # Setup metrics to pass to classifier and evaluate with SVC
        metrics = [metric_registry.get(metric)() for metric in self.config.metrics]
        logger.info("Evaluating with SVC")
        metrics = sklearn_svc(modeling_dataset, "transformed", "flat_labels", metrics)

        for metric in metrics:
            logger.info(f"{metric.__class__.__name__}: {metric.result}")
