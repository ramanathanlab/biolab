from __future__ import annotations
from argparse import ArgumentParser

# Trigger registry population, even though we don't use it is neccessary
import biolab.tasks  # noqa: F401
import biolab.modeling  # noqa: F401
import biolab.metrics  # noqa: F401
from biolab import model_registry, task_registry
from biolab.modeling import ModelConfigTypes
from biolab.tasks import TaskConfigTypes
from biolab.api.logging import logger
from biolab.api.config import BaseConfig
from biolab.api.modeling import LM


class EvalConfig(BaseConfig):
    """Configuration for the benchmarking pipeline"""

    # Might also be a list of configs?
    lm_config: ModelConfigTypes

    task_configs: list[TaskConfigTypes]


def evaluate_task(task_config: TaskConfigTypes, model: LM):
    # Find the task class and config class
    task_cls_info = task_registry.get(task_config.name)
    if task_cls_info is None:
        logger.warn(f"Task {task_config.name} not found in registry")
        logger.warn("Available tasks:")
        logger.warn(f"\t{task_registry._registry.keys()}")

    task_cls = task_cls_info["class"]
    task = task_cls(task_config)
    logger.info(f"Setup {task.config.name}")

    # Run the evaluation
    task.evaluate(model)


def evaluate(eval_config: EvalConfig):
    """Evaluate the models on the tasks"""
    logger.info(f"{eval_config.lm_config}")

    # Get model and tokenizer
    model_cls_info = model_registry.get(eval_config.lm_config.name)
    model_cls = model_cls_info["class"]

    model = model_cls(eval_config.lm_config)

    logger.info(f"Setup {model.config.name}")

    # Iterate over tasks and evaluate
    for task_config in eval_config.task_configs:
        evaluate_task(task_config, model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Path to the evaluation config file"
    )
    args = parser.parse_args()

    config = EvalConfig.from_yaml(args.config)
    evaluate(config)
