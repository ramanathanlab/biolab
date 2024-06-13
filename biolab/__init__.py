"""biolab package."""

from __future__ import annotations
import importlib.metadata as importlib_metadata

__version__ = importlib_metadata.version("biolab")

# Define global registries
from biolab.api.registry import Registry, CoupledRegistry  # noqa: F401

model_registry = CoupledRegistry()
task_registry = CoupledRegistry()
transform_registry = Registry()
metric_registry = Registry()
