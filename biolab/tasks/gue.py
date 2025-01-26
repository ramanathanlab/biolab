"""Implementation of the GUE tasks from DNABert2.

Tasks present:

Task                               Metric   Datasets                  Train / Dev / Test
--------------------------------------------------------------------------------------------
Core Promoter Detection            mcc     tata                      4904 / 613 / 613
                                           notata                    42452 / 5307 / 5307
                                           all                       47356 / 5920 / 5920

Promoter Detection                 mcc     tata                      4904 / 613 / 613
                                           notata                    42452 / 5307 / 5307
                                           all                       47356 / 5920 / 5920

Transcription Factor Prediction    mcc     wgEncodeEH000552 (0)      32378 / 1000 / 1000
(Human)                                    wgEncodeEH000606 (1)      30672 / 1000 / 1000
                                           wgEncodeEH001546 (2)      19000 / 1000 / 1000
                                           wgEncodeEH001776 (3)      27294 / 1000 / 1000
                                           wgEncodeEH002829 (4)      19000 / 1000 / 1000

Splice Site Prediction             mcc      reconstructed            36496 / 4562 / 4562

Transcription Factor Prediction    mcc     Ch12Nrf2Iggrab   (0)      6478 / 810 / 810
(Mouse)                                    Ch12Znf384hpa004051Iggrab 53952 / 6745 / 6745
                                           MelJundIggrab    (2)      2620 / 328 / 328
                                           MelMafkDm2p5dStd (3)      1904 / 239 / 239
                                           MelNelfeIggrab   (4)      15064 / 1883 / 1883

Epigenetic Marks Prediction        mcc      H3                       11971 / 1497 / 1497
                                           H3K14ac                   26483 / 3305 / 3305
                                           H3K36me3                  27904 / 3488 / 3488
                                           H3K4me1                   25341 / 3168 / 3168
                                           H3K4me2                   24545 / 3069 / 3069
                                           H3K4me3                   29439 / 3680 / 3680
                                           H3K79me3                  23069 / 2883 / 2884
                                           H3K9ac                    22224 / 2779 / 2779
                                           H4                        11679 / 1461 / 1461
                                           H4ac                      27275 / 3410 / 3410

Covid Variant Classification       f1       Covid                    77669 / 7000 / 7000

Paper: https://arxiv.org/pdf/2306.15006
"""

from __future__ import annotations

from typing import Literal

import datasets
from datasets import DatasetDict
from pydantic import field_serializer
from pydantic import model_validator

from biolab.api.logging import logger
from biolab.api.metric import Metric
from biolab.api.metric import MetricCollection
from biolab.api.modeling import HDF5CachedList
from biolab.api.modeling import LM
from biolab.api.task import Task
from biolab.api.task import TaskConfig
from biolab.metrics import get_and_instantiate_metric
from biolab.tasks.core.downstream import get_downstream_model
from biolab.tasks.core.embedding_task import EmbeddingTaskConfig
from biolab.tasks.core.utils import find_transformation
from biolab.tasks.core.utils import limit_training_samples


class GUETaskConfig(EmbeddingTaskConfig):
    """Base configuration for GUE tasks."""

    # Split of the task
    subset: Literal['']

    @model_validator(mode='after')
    def update_task_name(self):
        """Update the task name to have the subset name.

        This needs to be done post-init so we can successfully instantiate the task,
        but before the results are saved so that we can differentiate the subsets of
        of the same task.
        """
        self.name = f'{self.name}-{self.subset}'
        return self

    @field_serializer('name', check_fields=False, when_used='json')
    def serialize_name(self, name: str):
        """Serialize the task name to remove the split name.

        This allows us to dump the model config and reload appropriately.
        """
        return name.replace(f'-{self.subset}', '')


class GUETask(Task):
    """Base implementation for tasks in the GUE benchmark.

    The main differences here are the data loading and splitting, the core
    embedding and transformation logic is unchanged.

    All GUE tasks are sequence level tasks.
    """

    resolution: str = 'sequence'

    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def evaluate(self, model: LM) -> list[Metric]:
        """Evaluate a GUE character (aminoacid) level task."""
        logger.info(f'Evaluating subset: {self.config.subset}')

        # Load the dataset
        # NOTE: Originally I set format to numpy, but it prohibits multi-dimension
        # arrays being concatenated into the downstream dataset, removing it does
        # not seem to cause issues.
        task_dataset: datasets.Dataset = datasets.load_from_disk(
            self.config.dataset_name_or_path
        )

        # Filter out any sequences that are non-needed
        if 'set' in task_dataset.column_names:
            task_dataset = task_dataset.filter(lambda x: x['set'] is not None)

        # TODO: does limiting samples not faithfully represent the original task?
        if self.config.max_samples is not None:
            task_dataset = limit_training_samples(
                task_dataset,
                self.config.max_samples,
                model.model_input,
                target_col='set',  # for GUE tasks, want to balance on train/test set
                continuous=False,
            )

        # Generate embeddings
        logger.info(
            f'Generating {model.model_input} embeddings ({len(task_dataset):,})'
        )
        input_sequences = task_dataset[model.model_input]

        cache_file = (
            self.config.cache_dir / f'{model.config.name}_{self.config.name}.h5'
        )
        with HDF5CachedList(cache_file, mode='w') as model_outputs:
            model_outputs = model.generate_model_outputs(
                input_sequences, model_outputs, return_embeddings=True
            )

            # find and instantiate an output transform object
            transforms = find_transformation(
                model.model_input, model.model_encoding, self.resolution
            )
            logger.info(
                f'Found transformation {[transform.name for transform in transforms]}'
            )
            # Apply the transformations
            for transform in transforms:
                logger.info(f'Applying {transform.name} transformation')
                model_outputs.map(
                    transform.apply_h5,
                    sequences=input_sequences,
                    tokenizer=model.tokenizer,
                )

            # Create the downstream modeling dataset
            # Sequence-level embeddings don't require any transformation of
            # task_dataset, so we can just concatenate the embeddings to the dataset
            # TODO: this might run out of memory for large datasets, think about how
            # to pull this from a generator, currently we get an issue about not
            # being able to pickle the dataset
            embed_dict = {
                'transformed': [output.embedding for output in model_outputs],
            }
            modeling_dataset = datasets.Dataset.from_dict(embed_dict)
            modeling_dataset = datasets.concatenate_datasets(
                [
                    task_dataset,
                    modeling_dataset,
                ],
                axis=1,
            )

            # manually set the train test split based on the 'set' column
            modeling_dataset = DatasetDict(
                {
                    'train': modeling_dataset.filter(lambda x: x['set'] == 'train'),
                    'test': modeling_dataset.filter(lambda x: x['set'] == 'test'),
                }
            )
            logger.debug(modeling_dataset)

            # Setup metrics to pass to downstream prediction model and run modeling
            metrics = MetricCollection(
                [get_and_instantiate_metric(metric) for metric in self.config.metrics]
            )

            # Evaluate with appropriate model
            downstream_modeling = get_downstream_model(
                self.config.task_type, self.config.downstream_model
            )
            metrics = downstream_modeling(
                task_dataset=modeling_dataset,
                input_col='transformed',
                target_col=self.config.target_col,
                metrics=metrics,
                k_fold=0,
            )

        # Cleanup the cache files if they have been created by this task
        task_dataset.cleanup_cache_files()

        return metrics


# Already manually split into train/test
class GUECorePromoterDetectionConfig(GUETaskConfig):
    """Configuration for the human core PromoterDetection classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUECorePromoterDetection'] = 'GUECorePromoterDetection'
    # Subset of dataset being tested
    subset: Literal['tata', 'notata', 'all']
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['mcc']


class GUECorePromoterDetection(GUETask):
    """Core promoter detection prediction task from DNABert2."""


# Already manually split into train/test
class GUEPromoterDetectionConfig(GUETaskConfig):
    """Configuration for the human PromoterDetection classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUEPromoterDetection'] = 'GUEPromoterDetection'
    # Subset of dataset being tested
    subset: Literal['tata', 'notata', 'all']
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['mcc']


class GUEPromoterDetection(GUETask):
    """Promoter detection prediction task from DNABert2."""


# Subsets 0..4 are different sources, not k-fold splits
class GUEHumanTranscriptionFactorConfig(GUETaskConfig):
    """Config for the GUE Human Transcription Factor classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUEHumanTranscriptionFactor'] = 'GUEHumanTranscriptionFactor'
    # Subset of dataset being tested
    subset: Literal['0', '1', '2', '3', '4']
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['mcc']


class GUEHumanTranscriptionFactor(GUETask):
    """GUE Human Transcription Factor classification task."""


# Reconstructed is only subset
class GUESpliceSiteDetectionConfig(GUETaskConfig):
    """Configuration for the splice site detection classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUESpliceSiteDetection'] = 'GUESpliceSiteDetection'
    # Subset of dataset being tested
    subset: Literal['reconstructed'] = 'reconstructed'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['mcc']


class GUESpliceSiteDetection(GUETask):
    """Splice site detection prediction task from DNABert2."""


# Subsets 0..4 are different sources, not k-fold splits
class GUEMouseTranscriptionFactorConfig(GUETaskConfig):
    """Config for the GUE Mouse Transcription Factor classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUEMouseTranscriptionFactor'] = 'GUEMouseTranscriptionFactor'
    # Subset of dataset being tested
    subset: Literal['0', '1', '2', '3', '4']
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['mcc']


class GUEMouseTranscriptionFactor(GUETask):
    """GUE Mouse Transcription Factor classification task."""


# All subsets are unique data sources
class GUEEMPConfig(GUETaskConfig):
    """Configuration for the Epigenetic Marker Prediction classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUEEMP'] = 'GUEEMP'
    # Subset of dataset being tested
    subset: Literal[
        'H3',
        'H3K14ac',
        'H3K36me3',
        'H3K4me1',
        'H3K4me2',
        'H3K4me3',
        'H3K79me3',
        'H3K9ac',
        'H4',
        'H4ac',
    ]
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['mcc']


class GUEEMP(GUETask):
    """Epigenetic marker prediction task from DNABert2."""


# Only one split to this
class GUECovidVariantClassificationConfig(GUETaskConfig):
    """Configuration for the COVID variant classification task."""

    # Name of the task to be set by subclass
    name: Literal['GUECovidVariantClassification'] = 'GUECovidVariantClassification'
    # Subset of dataset being tested
    subset: Literal['covid'] = 'covid'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = ['f1']


class GUECovidVariantClassification(GUETask):
    """COVID variant prediction task from DNABert2."""


# Data sources for species classification and enhancer promoter interaction is not
# immediately provided from the google drive


# Create a mapping of the task config to the task class for registry
gue_tasks = {
    GUECorePromoterDetectionConfig: GUECorePromoterDetection,
    GUEPromoterDetectionConfig: GUEPromoterDetection,
    GUEHumanTranscriptionFactorConfig: GUEHumanTranscriptionFactor,
    GUESpliceSiteDetectionConfig: GUESpliceSiteDetection,
    GUEMouseTranscriptionFactorConfig: GUEMouseTranscriptionFactor,
    GUEEMPConfig: GUEEMP,
    GUECovidVariantClassificationConfig: GUECovidVariantClassification,
}
