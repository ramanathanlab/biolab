from __future__ import annotations  # noqa: D100

from typing import Literal

from biolab import task_registry
from biolab.tasks.core.sequence import SequenceTask
from biolab.tasks.core.sequence import SequenceTaskConfig


class GUEEMPConfig(SequenceTaskConfig):
    """Configuration for the Epigenetic Marker Prediction classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUEEMP'] = 'GUEEMP'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure TODO: should be choice of literals
    metrics: list[str] = ['accuracy', 'f1']


@task_registry.register(config=GUEEMPConfig)
class GUEEMP(SequenceTask):
    """Epigenetic marker prediction task from DNABert2.

    https://arxiv.org/pdf/2306.15006
    """

    resolution: str = 'sequence'


class GUEHumanTranscriptionFactorConfig(SequenceTaskConfig):
    """Config for the GUE Human Transcription Factor classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUEHumanTransriptionFactor'] = 'GUEHumanTransriptionFactor'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure TODO: should be choice of literals
    metrics: list[str] = ['accuracy', 'f1']


@task_registry.register(config=GUEHumanTranscriptionFactorConfig)
class GUEHumanTranscriptionFactor(SequenceTask):
    """GUE Human Transcription Factor classification task."""

    resolution: str = 'sequence'


class GUEMouseTranscriptionFactorConfig(SequenceTaskConfig):
    """Config for the GUE Mouse Transcription Factor classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUEMouseTransriptionFactor'] = 'GUEMouseTransriptionFactor'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure TODO: should be choice of literals
    metrics: list[str] = ['accuracy', 'f1']


@task_registry.register(config=GUEMouseTranscriptionFactorConfig)
class GUEMouseTranscriptionFactor(SequenceTask):
    """GUE Mouse Transcription Factor classification task."""

    resolution: str = 'sequence'


class GUECovidVariantClassificationConfig(SequenceTaskConfig):
    """Configuration for the PromoterDetection classification task."""

    # Name of the task to be set by subclass
    name: Literal['CovidVariantClassification'] = 'CovidVariantClassification'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure TODO: should be choice of literals
    metrics: list[str] = ['accuracy', 'f1']


@task_registry.register(config=GUECovidVariantClassificationConfig)
class GUECovidVariantClassification(SequenceTask):
    """Epigenetic marker prediction task from DNABert2.

    https://arxiv.org/pdf/2306.15006
    """

    resolution: str = 'sequence'


class GUEPromoterDetectionConfig(SequenceTaskConfig):
    """Configuration for the PromoterDetection classification task."""

    # Name of the task to be set by subclass
    name: Literal['PromoterDetection'] = 'PromoterDetection'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure TODO: should be choice of literals
    metrics: list[str] = ['accuracy', 'f1']


@task_registry.register(config=GUEPromoterDetectionConfig)
class GUEPromoterDetection(SequenceTask):
    """Epigenetic marker prediction task from DNABert2.

    https://arxiv.org/pdf/2306.15006
    """

    resolution: str = 'sequence'


class GUECorePromoterDetectionConfig(SequenceTaskConfig):
    """Configuration for the PromoterDetection classification task."""

    # Name of the task to be set by subclass
    name: Literal['CorePromoterDetection'] = 'CorePromoterDetection'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure TODO: should be choice of literals
    metrics: list[str] = ['accuracy', 'f1']


@task_registry.register(config=GUECorePromoterDetectionConfig)
class GUECorePromoterDetection(SequenceTask):
    """Epigenetic marker prediction task from DNABert2.

    https://arxiv.org/pdf/2306.15006
    """

    resolution: str = 'sequence'


class GUESpliceSiteDetectionConfig(SequenceTaskConfig):
    """Configuration for the PromoterDetection classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUESpliceSiteDetection'] = 'GUESpliceSiteDetection'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure TODO: should be choice of literals
    metrics: list[str] = ['accuracy', 'f1']


@task_registry.register(config=GUESpliceSiteDetectionConfig)
class GUESpliceSiteDetection(SequenceTask):
    """Epigenetic marker prediction task from DNABert2.

    https://arxiv.org/pdf/2306.15006
    """

    resolution: str = 'sequence'
