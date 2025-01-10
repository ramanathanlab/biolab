"""BV-BRC task DNA classification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from biolab.tasks.core.embedding_task import EmbeddingTask
from biolab.tasks.core.embedding_task import EmbeddingTaskConfig


class DNAClassificationConfig(EmbeddingTaskConfig):
    """Configuration for the DNA classification task."""

    # Name of the task
    name: Literal['DNAClassification'] = 'DNAClassification'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = Field(default_factory=lambda: ['accuracy', 'f1'])


class DNAClassification(EmbeddingTask):
    """DNA classification task."""

    resolution: str = 'sequence'


# Associate the task config with the task class for explicit registration
dna_classification_tasks = {
    DNAClassificationConfig: DNAClassification,
}
