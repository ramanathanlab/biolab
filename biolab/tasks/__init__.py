"""Implementation of tasks for Biolab."""

from __future__ import annotations

from typing import Union

from .calm_tasks import calm_tasks
from .dna_classification import dna_classification_tasks
from .flip import flip_tasks
from .gc_content import gc_content_tasks
from .gue import gue_tasks
from .patric_secondary_structure_classification import (
    patric_secondary_structure_classification_tasks,
)
from .sanity import sanity_tasks

task_registry = {
    **calm_tasks,
    **dna_classification_tasks,
    **flip_tasks,
    **gc_content_tasks,
    **gue_tasks,
    **patric_secondary_structure_classification_tasks,
    **sanity_tasks,
}

# TODO: consider explicitly defining this so that we get rid of the
# pydantic warning (that is currently silenced)
TaskConfigTypes = Union[*task_registry.keys(),]
