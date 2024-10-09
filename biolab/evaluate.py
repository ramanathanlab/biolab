"""Evaluation entrypoint for the benchmarking pipeline."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from pydantic import Field
from pydantic.functional_validators import model_validator

import biolab.metrics
import biolab.modeling

# Trigger registry population, even though we don't use it is necessary
import biolab.tasks  # noqa: F401
from biolab import model_registry
from biolab import task_registry
from biolab.api.config import BaseConfig
from biolab.api.logging import logger
from biolab.api.modeling import LM
from biolab.modeling import ModelConfigTypes
from biolab.tasks import TaskConfigTypes


class EvalConfig(BaseConfig):
    """Configuration for the benchmarking pipeline."""

    # Might also be a list of configs?
    lm_config: ModelConfigTypes

    task_configs: list[TaskConfigTypes]

    # General evaluation settings
    # Results output directory
    output_dir: Path = Field(
        default_factory=lambda: Path(
            f"results-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    )
    # Cache dir for intermediate results (different from model cache dir -
    # this is where the model is downloaded)
    cache_dir: Path = None

    @model_validator(mode='after')
    def set_cache_dir(self):
        """Set the cache directory to be within the output directory if not provided."""
        # Set cache_dir to be within output_dir if not explicitly provided
        if self.cache_dir is None:
            self.cache_dir = Path(self.output_dir) / 'cache'

        return self


def setup_evaluations(eval_config: EvalConfig):
    """Setup environment for the evaluations."""
    eval_config.output_dir.mkdir(parents=True, exist_ok=True)


def evaluate_task(task_config: TaskConfigTypes, model: LM):
    """Evaluate a task given a configuration and a model."""
    # Find the task class and config class
    task_cls_info = task_registry.get(task_config.name)
    if task_cls_info is None:
        logger.debug(f'Task {task_config.name} not found in registry')
        logger.debug(f'Available tasks:\n\t{task_registry._registry.keys()}')
        raise ValueError(f'Task {task_config.name} not found in registry')

    task_cls = task_cls_info['class']
    task = task_cls(task_config)
    logger.info(f'Setup {task.config.name}')

    # Run the evaluation
    task.evaluate(model)


def evaluate(eval_config: EvalConfig):
    """Evaluate the models on the tasks."""
    logger.info(f'{eval_config.lm_config}')

    # Get model and tokenizer
    model_cls_info = model_registry.get(eval_config.lm_config.name)
    model_cls = model_cls_info['class']

    model = model_cls(eval_config.lm_config)

    logger.info(f'Setup {model.config.name}')

    # Iterate over tasks and evaluate
    for task_config in eval_config.task_configs:
        evaluate_task(task_config, model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--config', required=True, help='Path to the evaluation config file'
    )
    args = parser.parse_args()

    config = EvalConfig.from_yaml(args.config)
    evaluate(config)
