"""Evaluation entrypoint for the benchmarking pipeline."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from pathlib import Path

from parsl.concurrent import ParslPoolExecutor
from pydantic import Field
from pydantic import model_validator

from biolab.api.config import BaseConfig
from biolab.api.logging import logger
from biolab.api.metric import MetricCollection
from biolab.distribution import ParslConfigTypes
from biolab.modeling import ModelConfigTypes
from biolab.tasks import TaskConfigTypes


class EvalConfig(BaseConfig):
    """Configuration for the benchmarking pipeline."""

    # TODO: Add support for multiple configs
    lm_config: ModelConfigTypes

    task_configs: list[TaskConfigTypes]

    # For distributed evaluation using parsl
    parsl_config: ParslConfigTypes | None = None

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
    eval_config.cache_dir.mkdir(parents=True, exist_ok=True)

    # Inject output/cache dirs into the task configs
    # TODO: is there a better/more idiomatic way to 'push down'
    # global settings (like output_dir) to nested objects?
    for task_config in eval_config.task_configs:
        task_config.output_dir = eval_config.output_dir
        task_config.cache_dir = eval_config.cache_dir

    # Dump the original config for reproducibility
    eval_config.write_yaml(eval_config.output_dir / 'config.yaml')


def evaluate_task(task_config: TaskConfigTypes, model_config: ModelConfigTypes):
    """Evaluate a task given a configuration and a model."""
    from biolab.api.logging import logger
    from biolab.modeling import model_registry
    from biolab.tasks import task_registry

    # Get model and tokenizer
    model_cls = model_registry.get(model_config.__class__)
    if model_cls is None:
        logger.debug(f'Model {model_config.__class__} not found in registry')
        logger.debug(f'Available models:\n\t{model_registry.keys()}')
        raise ValueError(f'Model {model_config.__class__} not found in registry')

    model = model_cls(model_config)

    logger.info(f'Setup {model.config.name}')

    # Find the task class and config class
    task_cls = task_registry.get(task_config.__class__)
    if task_cls is None:
        logger.debug(f'Task {task_config.__class__} not found in registry')
        logger.debug(f'Available tasks:\n\t{task_registry.keys()}')
        raise ValueError(f'Task {task_config.__class__} not found in registry')

    task = task_cls(task_config)
    logger.info(f'Setup {task.config.name}')

    # Run the evaluation and get metrics
    metrics: MetricCollection = task.evaluate(model)

    # Save metrics and report
    metric_save_path = (
        task_config.output_dir / f'{model.config.name}_{task_config.name}.metrics'
    )
    metrics.save(metric_save_path)
    for metric in metrics:
        logger.info(metric.report())


def evaluate(eval_config: EvalConfig):
    """Evaluate the models on the tasks."""
    setup_evaluations(eval_config)
    logger.info(f'{eval_config.lm_config}')

    if eval_config.parsl_config is not None:
        # Initialize Parsl
        logger.info('Initializing Parsl')
        parsl_run_dir = eval_config.output_dir / 'parsl'
        parsl_config = eval_config.parsl_config.get_config(parsl_run_dir)

        evaluate_function = partial(evaluate_task, model_config=eval_config.lm_config)
        with ParslPoolExecutor(parsl_config) as pool:
            # Submit tasks to be executed
            list(pool.map(evaluate_function, eval_config.task_configs))
    else:
        # Evaluate tasks sequentially
        for task_config in eval_config.task_configs:
            evaluate_task(task_config, model_config=eval_config.lm_config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--config', required=True, help='Path to the evaluation config file'
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel('DEBUG')

    config = EvalConfig.from_yaml(args.config)
    evaluate(config)
