from typing import Union

from . import core  # noqa: F401
from biolab.api.registry import import_submodules
from biolab import task_registry

# Dynamically import submodules to trigger registration of tasks
import_submodules(__name__)

TaskConfigTypes = Union[
    tuple(elem["config"] for elem in task_registry._registry.values())  # noqa: F821
]
