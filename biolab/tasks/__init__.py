"""Implementation of tasks for Biolab."""

from __future__ import annotations

from typing import Union

from biolab.api.logging import logger

from .calm_tasks import calm_tasks
from .dna_classification import dna_classification_tasks
from .evo_tasks import evo_tasks
from .flip import flip_tasks
from .gc_content import gc_content_tasks
from .gue import gue_tasks
from .patric_secondary_structure_classification import (
    patric_secondary_structure_classification_tasks,
)
from .sanity import sanity_tasks

# Registry of tasks - append when adding new tasks
task_registry = {
    **calm_tasks,
    **dna_classification_tasks,
    **evo_tasks,
    **flip_tasks,
    **gc_content_tasks,
    **gue_tasks,
    **patric_secondary_structure_classification_tasks,
    **sanity_tasks,
}

# TODO: consider explicitly defining this so that we get rid of the
# pydantic warning (that is currently silenced)
TaskConfigTypes = Union[*task_registry.keys(),]


# Utility for getting and instantiating a task from a config
def get_task(task_config: TaskConfigTypes):
    """Get a task instance from a config."""
    # Find the task class and config class
    task_cls = task_registry.get(task_config.__class__)
    if task_cls is None:
        logger.debug(f'Task {task_config.__class__} not found in registry')
        logger.debug(f'Available tasks:\n\t{task_registry.keys()}')
        raise ValueError(f'Task {task_config.__class__} not found in registry')

    return task_cls(task_config)
