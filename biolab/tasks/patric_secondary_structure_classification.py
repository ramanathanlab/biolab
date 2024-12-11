"""Secondary structure on PATRIC proteins dataset."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

# from biolab import task_registry
from biolab.tasks.core.char_embedding import CharTask
from biolab.tasks.core.char_embedding import CharTaskConfig


class PatricSecondaryStructureClassificationConfig(CharTaskConfig):
    """Configuration for the PATRIC secondary structure classification task."""

    # Name of the task
    name: Literal['PatricSecondaryStructureClassification'] = (
        'PatricSecondaryStructureClassification'
    )
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = Field(default_factory=lambda: ['accuracy', 'f1'])

    # Truncate the ends of the sequences as DSSP does not output labels for these
    truncate_end: bool = True


# @task_registry.register(config_class=PatricSecondaryStructureClassificationConfig)
class PatricSecondaryStructureClassification(CharTask):
    """Patric secondary structure prediction classification."""

    # Specify this is an amino acid level task
    resolution: str = 'aminoacid'


patric_secondary_structure_classification_configs = [
    PatricSecondaryStructureClassificationConfig
]
patric_secondary_structure_classification_tasks = {
    PatricSecondaryStructureClassificationConfig: PatricSecondaryStructureClassification
}
