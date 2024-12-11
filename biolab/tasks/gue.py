"""Implementation of the GUE tasks from DNABert2."""

from __future__ import annotations

from typing import Literal

# from biolab import task_registry
from biolab.tasks.core.sequence_embedding import SequenceTask
from biolab.tasks.core.sequence_embedding import SequenceTaskConfig


class GUEEMPConfig(SequenceTaskConfig):
    """Configuration for the Epigenetic Marker Prediction classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUEEMP'] = 'GUEEMP'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['accuracy', 'f1']


# @task_registry.register(config_class=GUEEMPConfig)
class GUEEMP(SequenceTask):
    """Epigenetic marker prediction task from DNABert2.

    https://arxiv.org/pdf/2306.15006
    """

    resolution: str = 'sequence'


class GUEHumanTranscriptionFactorConfig(SequenceTaskConfig):
    """Config for the GUE Human Transcription Factor classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUEHumanTranscriptionFactor'] = 'GUEHumanTranscriptionFactor'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['accuracy', 'f1']


# @task_registry.register(config_class=GUEHumanTranscriptionFactorConfig)
class GUEHumanTranscriptionFactor(SequenceTask):
    """GUE Human Transcription Factor classification task."""

    resolution: str = 'sequence'


class GUEMouseTranscriptionFactorConfig(SequenceTaskConfig):
    """Config for the GUE Mouse Transcription Factor classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUEMouseTranscriptionFactor'] = 'GUEMouseTranscriptionFactor'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['accuracy', 'f1']


# @task_registry.register(config_class=GUEMouseTranscriptionFactorConfig)
class GUEMouseTranscriptionFactor(SequenceTask):
    """GUE Mouse Transcription Factor classification task."""

    resolution: str = 'sequence'


class GUECovidVariantClassificationConfig(SequenceTaskConfig):
    """Configuration for the COVID variant classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUECovidVariantClassification'] = 'GUECovidVariantClassification'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['accuracy', 'f1']


# @task_registry.register(config_class=GUECovidVariantClassificationConfig)
class GUECovidVariantClassification(SequenceTask):
    """COVID variant prediction task from DNABert2.

    https://arxiv.org/pdf/2306.15006
    """

    resolution: str = 'sequence'


class GUEPromoterDetectionConfig(SequenceTaskConfig):
    """Configuration for the PromoterDetection classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUEPromoterDetection'] = 'GUEPromoterDetection'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['accuracy', 'f1']


# @task_registry.register(config_class=GUEPromoterDetectionConfig)
class GUEPromoterDetection(SequenceTask):
    """Promoter detection prediction task from DNABert2.

    https://arxiv.org/pdf/2306.15006
    """

    resolution: str = 'sequence'


class GUECorePromoterDetectionConfig(SequenceTaskConfig):
    """Configuration for the PromoterDetection classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUECorePromoterDetection'] = 'GUECorePromoterDetection'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['accuracy', 'f1']


# @task_registry.register(config_class=GUECorePromoterDetectionConfig)
class GUECorePromoterDetection(SequenceTask):
    """Core promoter detection prediction task from DNABert2.

    https://arxiv.org/pdf/2306.15006
    """

    resolution: str = 'sequence'


class GUESpliceSiteDetectionConfig(SequenceTaskConfig):
    """Configuration for the splice site detection classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUESpliceSiteDetection'] = 'GUESpliceSiteDetection'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['accuracy', 'f1']


# @task_registry.register(config_class=GUESpliceSiteDetectionConfig)
class GUESpliceSiteDetection(SequenceTask):
    """Splice site detection prediction task from DNABert2.

    https://arxiv.org/pdf/2306.15006
    """

    resolution: str = 'sequence'


gue_configs = [
    GUEEMPConfig,
    GUEHumanTranscriptionFactorConfig,
    GUEMouseTranscriptionFactorConfig,
    GUECovidVariantClassificationConfig,
    GUEPromoterDetectionConfig,
    GUECorePromoterDetectionConfig,
    GUESpliceSiteDetectionConfig,
]
gue_tasks = {
    GUEEMPConfig: GUEEMP,
    GUEHumanTranscriptionFactorConfig: GUEHumanTranscriptionFactor,
    GUEMouseTranscriptionFactorConfig: GUEMouseTranscriptionFactor,
    GUECovidVariantClassificationConfig: GUECovidVariantClassification,
    GUEPromoterDetectionConfig: GUEPromoterDetection,
    GUECorePromoterDetectionConfig: GUECorePromoterDetection,
    GUESpliceSiteDetectionConfig: GUESpliceSiteDetection,
}
