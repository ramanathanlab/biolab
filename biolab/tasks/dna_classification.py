"""BV-BRC task DNA classification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from biolab import task_registry
from biolab.tasks.core.sequence_embedding import SequenceTask
from biolab.tasks.core.sequence_embedding import SequenceTaskConfig


class DNAClassificationConfig(SequenceTaskConfig):
    """Configuration for the DNA classification task."""

    # Name of the task
    name: Literal['DNAClassification'] = 'DNAClassification'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure TODO: should be choice of literals
    metrics: list[str] = Field(default_factory=lambda: ['accuracy', 'f1'])


@task_registry.register(config=DNAClassificationConfig)
class DNAClassification(SequenceTask):
    """DNA classification task."""

    resolution: str = 'sequence'
