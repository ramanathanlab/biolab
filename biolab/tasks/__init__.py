from __future__ import annotations  # noqa: D104

from typing import Union

from biolab import task_registry
from biolab.api.registry import import_submodules

from . import core

# Dynamically import submodules to trigger registration of tasks
import_submodules(__name__)

TaskConfigTypes = Union[  # noqa: UP007
    tuple(elem['config'] for elem in task_registry._registry.values())
]
