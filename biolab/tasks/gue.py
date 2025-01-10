"""Implementation of the GUE tasks from DNABert2."""

from __future__ import annotations

from typing import Literal

from biolab.tasks.core.embedding_task import EmbeddingTask
from biolab.tasks.core.embedding_task import EmbeddingTaskConfig


class GUEEMPConfig(EmbeddingTaskConfig):
    """Configuration for the Epigenetic Marker Prediction classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUEEMP'] = 'GUEEMP'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['accuracy', 'f1']


class GUEEMP(EmbeddingTask):
    """Epigenetic marker prediction task from DNABert2.

    https://arxiv.org/pdf/2306.15006
    """

    resolution: str = 'sequence'


class GUEHumanTranscriptionFactorConfig(EmbeddingTaskConfig):
    """Config for the GUE Human Transcription Factor classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUEHumanTranscriptionFactor'] = 'GUEHumanTranscriptionFactor'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['accuracy', 'f1']


class GUEHumanTranscriptionFactor(EmbeddingTask):
    """GUE Human Transcription Factor classification task."""

    resolution: str = 'sequence'


class GUEMouseTranscriptionFactorConfig(EmbeddingTaskConfig):
    """Config for the GUE Mouse Transcription Factor classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUEMouseTranscriptionFactor'] = 'GUEMouseTranscriptionFactor'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['accuracy', 'f1']


class GUEMouseTranscriptionFactor(EmbeddingTask):
    """GUE Mouse Transcription Factor classification task."""

    resolution: str = 'sequence'


class GUECovidVariantClassificationConfig(EmbeddingTaskConfig):
    """Configuration for the COVID variant classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUECovidVariantClassification'] = 'GUECovidVariantClassification'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['accuracy', 'f1']


class GUECovidVariantClassification(EmbeddingTask):
    """COVID variant prediction task from DNABert2.

    https://arxiv.org/pdf/2306.15006
    """

    resolution: str = 'sequence'


class GUEPromoterDetectionConfig(EmbeddingTaskConfig):
    """Configuration for the PromoterDetection classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUEPromoterDetection'] = 'GUEPromoterDetection'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['accuracy', 'f1']


class GUEPromoterDetection(EmbeddingTask):
    """Promoter detection prediction task from DNABert2.

    https://arxiv.org/pdf/2306.15006
    """

    resolution: str = 'sequence'


class GUECorePromoterDetectionConfig(EmbeddingTaskConfig):
    """Configuration for the PromoterDetection classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUECorePromoterDetection'] = 'GUECorePromoterDetection'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['accuracy', 'f1']


class GUECorePromoterDetection(EmbeddingTask):
    """Core promoter detection prediction task from DNABert2.

    https://arxiv.org/pdf/2306.15006
    """

    resolution: str = 'sequence'


class GUESpliceSiteDetectionConfig(EmbeddingTaskConfig):
    """Configuration for the splice site detection classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUESpliceSiteDetection'] = 'GUESpliceSiteDetection'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['accuracy', 'f1']


class GUESpliceSiteDetection(EmbeddingTask):
    """Splice site detection prediction task from DNABert2.

    https://arxiv.org/pdf/2306.15006
    """

    resolution: str = 'sequence'


# Create a mapping of the task config to the task class for registry
gue_tasks = {
    GUEEMPConfig: GUEEMP,
    GUEHumanTranscriptionFactorConfig: GUEHumanTranscriptionFactor,
    GUEMouseTranscriptionFactorConfig: GUEMouseTranscriptionFactor,
    GUECovidVariantClassificationConfig: GUECovidVariantClassification,
    GUEPromoterDetectionConfig: GUEPromoterDetection,
    GUECorePromoterDetectionConfig: GUECorePromoterDetection,
    GUESpliceSiteDetectionConfig: GUESpliceSiteDetection,
}
