"""Test task for GC content on MDH sequences."""

from __future__ import annotations

from typing import Literal

from biolab.tasks.core.embedding_task import EmbeddingTask
from biolab.tasks.core.embedding_task import EmbeddingTaskConfig


class GCContentConfig(EmbeddingTaskConfig):
    """Configuration for MDH GC content. (Debug task)."""

    # Name of the task
    name: Literal['GCContent'] = 'GCContent'
    # embedding transformation
    task_type: Literal['regression'] = 'regression'
    # Metrics to measure
    metrics: list[str] = ['mse', 'r2']


class GCContent(EmbeddingTask):
    """GC content from MDH."""

    resolution: str = 'sequence'


# Associate the task config with the task class for explicit registration
gc_content_tasks = {
    GCContentConfig: GCContent,
}
