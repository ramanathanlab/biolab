"""Module for transformer inference."""

# This ensures that the models submodule is imported
# and triggers the decorators
from __future__ import annotations

from typing import Union

from biolab import model_registry
from biolab.api.registry import import_submodules

from .models import *  # noqa: F403
from .transforms import *  # noqa: F403

# Dynamically import all submodules to trigger registration of models, transforms
import_submodules(__name__)

ModelConfigTypes = Union[  # noqa: UP007
    tuple(elem['config'] for elem in model_registry._registry.values())
]
