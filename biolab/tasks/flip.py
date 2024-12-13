"""Semi-faithful implementation of FLIP.

https://benchmark.protein.properties/home

Tasks
------

- [X] AAV
- [X] GB1
- [X] Meltome
- [X] SCL
- [ ] Bind (multi-class classification, not implemented yet)
- [X] Single Amino Acid Variant
- [X] Conservation


# TODO: normalize regression values??
"""

from __future__ import annotations

from typing import Literal

import datasets
from datasets import DatasetDict
from pydantic import model_validator
from pydantic import field_serializer

from biolab.api.logging import logger
from biolab.api.metric import Metric
from biolab.api.metric import MetricCollection
from biolab.api.modeling import HDF5CachedList
from biolab.api.modeling import LM
from biolab.api.task import Task
from biolab.api.task import TaskConfig
from biolab.metrics import get_and_instantiate_metric
from biolab.tasks.core.downstream.classification import sklearn_svc
from biolab.tasks.core.downstream.regression import sklearn_svr
from biolab.tasks.core.utils import find_transformation
from biolab.tasks.core.utils import flatten_to_token_level
from biolab.tasks.core.utils import limit_training_samples


class FLIPTaskConfig(TaskConfig):
    """Base configuration for FLIP tasks."""

    @model_validator(mode='after')
    def update_task_name(self):
        """Update the task name to have the split name.

        This needs to be done post-init so we can successfully instantiate the task,
        but before the results are saved so that we can differentiate the splits of
        of the same task.
        """
        self.name = f'{self.name}-{self.split}'
        return self

    @field_serializer('name', check_fields=False, when_used='json')
    def serialize_name(self, name: str):
        """Serialize the task name to remove the split name.


        This allows us to dump the model config and reload appropriately."""
        return name.replace(f'-{self.split}', '')


class FLIPTask(Task):
    """Base implementation for tasks in the FLIP benchmark.

    The main differences here are the data loading and splitting, the core
    embedding and transformation logic is unchanged.

    This implementation handles _BOTH_ character level and sequence level tasks.
    """

    resolution: str

    def __init__(self, config: TaskConfig):
        super().__init__(config)
        assert hasattr(self, 'resolution'), 'Resolution must be set in the subclass'

    def evaluate(self, model: LM) -> list[Metric]:
        """Evaluate a FLIP character (aminoacid) level task."""
        char_level = self.resolution in ['aminoacid', 'nucleotide']
        logger.info(f'Task resolution: {self.resolution} (char level: {char_level})')
        logger.info(f'Evaluating split {self.config.split}')

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
        # Limit training samples if specified. Tricky here because in char level tasks
        # we both need to limit the number of sequences we do inference on
        # and the number of samples used in downstream modeling.
        char_level_limit = char_level and 'max_sequences' in self.config.model_fields
        sequence_level_limit = (
            not char_level and 'max_samples' in self.config.model_fields
        )
        if char_level_limit and not sequence_level_limit:
            task_dataset = limit_training_samples(
                task_dataset,
                self.config.max_sequences,
                model.model_input,
                target_col='set',  # for flip tasks, want to balance on training set
                continuous=False,
            )
        elif sequence_level_limit and not char_level_limit:
            task_dataset = limit_training_samples(
                task_dataset,
                self.config.max_samples,
                model.model_input,
                target_col='set',  # for flip tasks, want to balance on training set
                continuous=False,
            )
        else:
            if 'max_sequences' in self.config.model_fields:
                logger.error(
                    'Logic error: could not limit samples using `max_sequences`'
                )
            if 'max_samples' in self.config.model_fields:
                logger.error('Logic error: could not limit samples using `max_samples`')

        # Generate embeddings
        logger.info(
            f'Generating {model.model_input} embeddings ({len(task_dataset):,})'
        )
        input_sequences = task_dataset[model.model_input]

        cache_file = (
            self.config.cache_dir / f'{model.config.name}_{self.config.name}.h5'
        )
        with HDF5CachedList(cache_file, mode='w') as model_outputs:
            model_outputs = model.generate_embeddings(input_sequences, model_outputs)

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
                model_outputs.map(transform.apply_h5)

            # Create the downstream modeling dataset
            if self.resolution == 'sequence':
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
            # TODO: pull this into enums, its hardcoded around the codebase
            elif self.resolution in ['aminoacid', 'nucleotide']:
                # Flatten the dataset to the token level for downstream modeling
                modeling_dataset = flatten_to_token_level(
                    task_dataset,
                    model_outputs,
                    truncate_end=False,  # FLIP tasks never need to exclude end
                )

                # TODO: limiting samples here might be necessary as we can't really do an SVC huge
                # token level datasets - think about how to make a compromise here
                if (
                    'max_samples' in self.config.model_fields
                    and self.config.max_samples
                ):
                    modeling_dataset = limit_training_samples(
                        modeling_dataset,
                        self.config.max_samples,
                        model.model_input,
                        target_col='set',  # for flip tasks, want to balance on 'set'
                        continuous=False,
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
            if self.config.task_type == 'regression':
                metrics = sklearn_svr(
                    modeling_dataset,
                    'transformed',
                    self.config.target_col,
                    metrics,
                    k_fold=0,
                )
            elif self.config.task_type == 'classification':
                metrics = sklearn_svc(
                    modeling_dataset,
                    'transformed',
                    self.config.target_col,
                    metrics,
                    k_fold=0,
                )

        # Cleanup the cache files if they have been created by this task
        task_dataset.cleanup_cache_files()

        return metrics


class FLIPGB1Config(FLIPTaskConfig):
    """Configuration for the FLIP-GB1 task."""

    name: Literal['FLIP-gb1'] = 'FLIP-gb1'
    metrics: list[str] = ['rmse', 'pearson', 'spearman']
    task_type: Literal['regression'] = 'regression'

    split: Literal['low_vs_high', 'one_vs_rest', 'two_vs_rest', 'three_vs_rest']

    target_col: str = 'label'
    max_samples: int | None = None


# @task_registry.register(config_class=FLIPGB1Config)
class FLIPGB1(FLIPTask):
    """Implementation of the GB1 epistatic mutational landscape prediction."""

    # TODO: this can also be a character level task
    resolution: str = 'sequence'


class FLIPAAVConfig(FLIPTaskConfig):
    """Configuration for the FLIP-GB1 task."""

    name: Literal['FLIP-aav'] = 'FLIP-aav'
    metrics: list[str] = ['rmse', 'pearson', 'spearman']
    task_type: Literal['regression'] = 'regression'

    split: Literal[
        'des_mut',
        'mut_des',
        'low_vs_high',
        'one_vs_many',
        'two_vs_many',
        'seven_vs_many',
        'sampled',
    ]

    target_col: str = 'label'

    max_samples: int | None = None


# @task_registry.register(config_class=FLIPAAVConfig)
class FLIPAAV(FLIPTask):
    """Implementation FLIP task on fitness of designed and mutated AAV sequences."""

    # TODO: this can also be a character level task
    resolution: str = 'sequence'


class FLIPConservationConfig(FLIPTaskConfig):
    """Configuration for the FLIP conservation task.

    Goal is to predict the 'conservation score' of the residues of a protein sequence.
    """

    name: Literal['FLIP-conservation'] = 'FLIP-conservation'
    metrics: list[str] = ['accuracy', 'f1']
    task_type: Literal['classification'] = 'classification'

    split: Literal['sampled'] = 'sampled'

    target_col: str = 'label'
    max_sequences: int | None = None
    max_samples: int | None = None


# @task_registry.register(config_class=FLIPConservationConfig)
class FLIPConservation(FLIPTask):
    """Implementation of the FLIP conservation task."""

    resolution: str = 'aminoacid'


class FLIPMeltomeConfig(FLIPTaskConfig):
    """Configuration for the FLIP meltome task.

    Goal is to predict the melting point.
    """

    name: Literal['FLIP-meltome'] = 'FLIP-meltome'
    metrics: list[str] = ['rmse', 'pearson', 'spearman']
    task_type: Literal['regression'] = 'regression'

    split: Literal['mixed_split', 'human', 'human_cell']

    target_col: str = 'label'
    max_samples: int | None = None


# @task_registry.register(config_class=FLIPMeltomeConfig)
class FLIPMeltome(FLIPTask):
    """Implementation of the FLIP meltome prediction task."""

    resolution: str = 'sequence'


class FLIPSCLConfig(FLIPTaskConfig):
    """Configuration for the FLIP subcellular location task."""

    name: Literal['FLIP-SCL'] = 'FLIP-SCL'
    metrics: list[str] = ['accuracy', 'f1']
    task_type: Literal['classification'] = 'classification'

    split: Literal[
        'human_hard',
        'mixed_hard',
        'mixed_soft',
        'human_soft',
        'mixed_vs_human_2',
        'balanced',
    ]

    target_col: str = 'label'
    max_samples: int | None = None


# @task_registry.register(config_class=FLIPSCLConfig)
class FLIPSCL(FLIPTask):
    """Implementation of the FLIP subcellular location prediction task."""

    resolution: str = 'sequence'


class FLIPSecondaryStructureConfig(FLIPTaskConfig):
    """Configuration for the FLIP secondary structure prediction task.

    Goal is to predict the 3 class secondary structure of a protein sequence.
    """

    name: Literal['FLIP-secondary-structure'] = 'FLIP-secondary-structure'
    metrics: list[str] = ['accuracy', 'f1']
    task_type: Literal['classification'] = 'classification'

    split: Literal['sampled'] = 'sampled'

    target_col: str = 'label'
    max_sequences: int | None = None
    max_samples: int | None = None


# @task_registry.register(config_class=FLIPSecondaryStructureConfig)
class FLIPSecondaryStructure(FLIPTask):
    """Implementation of the FLIP econdary structure prediction task."""

    resolution: str = 'aminoacid'


class FLIPSAVConfig(FLIPTaskConfig):
    """Configuration for the FLIP single amino acid variant effect prediction."""

    name: Literal['FLIP-SAV'] = 'FLIP-SAV'
    metrics: list[str] = ['accuracy', 'f1']
    task_type: Literal['classification'] = 'classification'

    split: Literal['human', 'mixed', 'only_savs']

    target_col: str = 'label'
    max_samples: int | None = None


# TODO: feels like this should also have a character level implementation?
# @task_registry.register(config_class=FLIPSAVConfig)
class FLIPSAV(FLIPTask):
    """Implementation of the FLIP single amino acid variant effect prediction task."""

    resolution: str = 'sequence'


flip_configs = [
    FLIPGB1Config,
    FLIPAAVConfig,
    FLIPConservationConfig,
    FLIPMeltomeConfig,
    FLIPSCLConfig,
    FLIPSecondaryStructureConfig,
    FLIPSAVConfig,
]
flip_tasks = {
    FLIPGB1Config: FLIPGB1,
    FLIPAAVConfig: FLIPAAV,
    FLIPConservationConfig: FLIPConservation,
    FLIPMeltomeConfig: FLIPMeltome,
    FLIPSCLConfig: FLIPSCL,
    FLIPSecondaryStructureConfig: FLIPSecondaryStructure,
    FLIPSAVConfig: FLIPSAV,
}
