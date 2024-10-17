"""Test task for GC content on MDH sequences."""

from __future__ import annotations

from typing import Literal

from biolab import task_registry
from biolab.tasks.core.sequence_embedding import SequenceTask
from biolab.tasks.core.sequence_embedding import SequenceTaskConfig


class GCContentConfig(SequenceTaskConfig):
    """Configuration for MDH GC content. (Debug task)."""

    # Name of the task
    name: Literal['GCContent'] = 'GCContent'
    # embedding transformation
    task_type: Literal['regression'] = 'regression'
    # Metrics to measure TODO: should be choice of literals
    metrics: list[str] = ['mse', 'r2']


@task_registry.register(config=GCContentConfig)
class GCContent(SequenceTask):
    """GC content from MDH."""

    resolution: str = 'sequence'
