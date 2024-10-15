"""biolab package root."""

from __future__ import annotations

import importlib.metadata as importlib_metadata

__version__ = importlib_metadata.version('biolab')

# Define global registries
from biolab.api.registry import CoupledRegistry
from biolab.api.registry import Registry

model_registry = CoupledRegistry()
task_registry = CoupledRegistry()
metric_registry = Registry()
