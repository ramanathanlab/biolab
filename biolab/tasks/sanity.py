"""Task to check all the functions of this model give model-able outputs."""

from __future__ import annotations

import sys
from typing import Literal

import datasets

# from biolab import task_registry
from biolab.api.logging import logger
from biolab.api.metric import Metric
from biolab.api.modeling import HDF5CachedList
from biolab.api.modeling import LM
from biolab.api.task import Task
from biolab.api.task import TaskConfig
from biolab.tasks.core.utils import find_transformation

# TODO: Actually make the dataset instead of harvesting a different dataset


class Feasible(Metric):
    """Dummy metric to report if the tasks are feasible for this model."""

    def __init__(self):
        """Initialize the Feasible metric."""
        super().__init__()

        self.tests = []

    def evaluate(self, passed: bool):
        """Evaluate the model and store results in the metric object."""
        self.tests.append(passed)

    def report(self, format: str | None = None) -> str:
        """Return a formatted report of the metric."""
        is_feasible = 'are' if all(self.tests) else 'are not'
        return (
            f'Metric: {self.__class__.__name__}:\t'
            f'Tasks {is_feasible} feasible for this model.'
        )

    def save(self, path):
        """Do NOT save the metric to a json file, it will be picked up by reporting."""
        pass


class SanityConfig(TaskConfig):
    """Configuration for a general nucleotide/amino acid level prediction task."""

    name: Literal['Sanity'] = 'Sanity'


class Sanity(Task):
    """Task to check all the functions of this model give model-able outputs."""

    def __init__(self, config: SanityConfig):
        super().__init__(config)

    def evaluate(self, model: LM) -> None:
        """Evaluate the task and return to make sure it runs appropriately."""
        logger.info('Sanity check: running all the functions of the model.')
        sanity_metric = Feasible()

        task_dataset = datasets.load_from_disk(self.config.dataset_name_or_path)
        task_dataset.set_format('torch')

        # Generate embeddings
        input_sequences = task_dataset[model.model_input]

        # This task should be small (# seqs) - so duplicating the data should be fine
        # but I want to test the caching mechanism here
        with HDF5CachedList(
            self.config.cache_dir / f'{model.config.name}_{self.config.name}.h5'
        ) as model_outputs:
            model_outputs = model.generate_model_outputs(
                input_sequences,
                model_outputs,
                return_input_ids=True,
                return_logits=True,
                return_embeddings=True,
            )

            #### Check average pooled embeddings ####
            mean_pool_embeddings = [model_output for model_output in model_outputs]  # noqa: C416
            mean_pool_transform = find_transformation(
                model.model_input, model.model_encoding, 'sequence'
            )

            for transform in mean_pool_transform:
                mean_pool_embeddings = transform.apply(mean_pool_embeddings)

            # Check if the embeddings are of the right shape
            if all(
                mean_pool_embeddings[0].embedding.shape == embedding.embedding.shape
                for embedding in mean_pool_embeddings
            ):
                sanity_metric.evaluate(True)
            else:
                sanity_metric.evaluate(False)
                logger.warning(
                    'Mean pooled embeddings are not of the right shape. Exiting.'
                )
                sys.exit()

            #### Check AA/NA character level embeddings ####
            char_level_embeddings = [model_output for model_output in model_outputs]  # noqa: C416
            char_level = (
                'aminoacid' if model.model_input == 'aminoacid' else 'nucleotide'
            )
            char_level_transform = find_transformation(
                model.model_input, model.model_encoding, char_level
            )
            for transform in char_level_transform:
                char_level_embeddings = transform.apply(char_level_embeddings)

            # Check if the embeddings are of the right shape
            if all(
                model_output.embedding.shape[0] == len(sequence)
                for model_output, sequence in zip(
                    char_level_embeddings, input_sequences, strict=True
                )
            ):
                sanity_metric.evaluate(True)
            else:
                sanity_metric.evaluate(False)
                logger.warning(
                    'Character level embeddings are not of the right shape. Exiting.'
                )
                sys.exit()

        return [sanity_metric]


# Create a dictionary to map the task config to the task for registration
sanity_tasks = {SanityConfig: Sanity}
