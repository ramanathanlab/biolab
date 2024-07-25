from typing import Literal, Optional

import numpy as np
import datasets
from biolab import task_registry, metric_registry
from biolab.tasks.core.utils import find_transformation
from biolab.api.logging import logger
from biolab.api.task import Task, TaskConfig
from biolab.api.modeling import LM


class CharLevelTestConfig(TaskConfig):

    # Name of the task
    name: Literal["CharLevelTest"] = "CharLevelTest"
    # Embedding transformation
    output_transform: str = "char_level"
    # Metrics to measure TODO: should be choice of literals
    metrics: list[str] = ["accuracy", "f1"]

    # Wether to balance the classes
    balance_classes: bool = False
    # Whether to limit the number of training samples
    max_samples: Optional[int] = None

    # Task specific information just need the label spec for now
    target_col: str = "label"


@task_registry.register(config=CharLevelTestConfig)
class CharLevelTest(Task):

    resolution: str = "aminoacid"

    def __init__(self, config: CharLevelTestConfig):
        self.config = config

    def evaluate(self, model: LM):
        # Load the dataset
        task_dataset = datasets.load_from_disk(self.config.dataset_name_or_path)
        task_dataset.set_format("torch")

        task_dataset = task_dataset.select(range(10))
        logger.info(task_dataset)

        logger.info(f"Generating {model.model_input} embeddings")
        input_sequences = task_dataset[model.model_input]
        model_outputs = model.generate_embeddings(input_sequences)

        # find and instantiate an output transform object
        transforms = find_transformation(
            model.model_input, model.model_encoding, self.resolution
        )
        logger.info(
            f"Found transformation {[transform.name for transform in transforms]}"
        )

        for transform in transforms:
            logger.info(f"Applying {transform.name} transformation")
            model_outputs = transform.apply(
                model_outputs, sequences=input_sequences, tokenizer=model.tokenizer
            )

        embed_dict = {
            "transformed": [output.embedding for output in model_outputs],
        }
        breakpoint()
