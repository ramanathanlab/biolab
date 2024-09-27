"""Evaluation entrypoint for the benchmarking pipeline."""

from __future__ import annotations

import os
from argparse import ArgumentParser

import biolab.metrics
import biolab.modeling

# Trigger registry population, even though we don't use it is neccessary
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


def setup_evaluations(eval_config: EvalConfig):
    """Setup environment for the evaluations."""
    # TODO: setup output directories and caching here
    os.environ['HF_DATASETS_CACHE'] = (
        '/nfs/lambda_stor_01/homes/khippe/github/biolab/datasets_cache'
    )
    os.environ['HF_DATASETS_IN_MEMORY_MAX_SIZE'] = '64424509440'  # 60GB
    os.environ['HF_DATASETS_DISABLE_PROGRESS_BARS'] = 'true'


def evaluate_task(task_config: TaskConfigTypes, model: LM):
    """Evaluate a task given a configuration and a model."""
    # Find the task class and config class
    task_cls_info = task_registry.get(task_config.name)
    if task_cls_info is None:
        logger.warning(f'Task {task_config.name} not found in registry')
        logger.warning(f'Available tasks:\n\t{task_registry._registry.keys()}')

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
