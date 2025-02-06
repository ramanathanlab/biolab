"""Tasks for CaLM."""

from __future__ import annotations

import shutil
import subprocess
import sys
from abc import ABC
from pathlib import Path
from typing import Literal

import datasets
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from pydantic import Field
from pydantic import field_serializer
from pydantic import model_validator

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    from typing import Self
else:  # pragma: <3.11 cover
    from typing_extensions import Self

from biolab.api.logging import logger
from biolab.api.metric import Metric
from biolab.api.metric import MetricCollection
from biolab.api.modeling import HDF5CachedList
from biolab.api.modeling import LM
from biolab.api.task import DownstreamModel
from biolab.api.task import Task
from biolab.metrics import get_and_instantiate_metric
from biolab.tasks.core.downstream import get_downstream_model
from biolab.tasks.core.embedding_task import EmbeddingTask
from biolab.tasks.core.embedding_task import EmbeddingTaskConfig
from biolab.tasks.core.utils import find_transformation


def _write_sequence_dset(
    sequences: list[Seq], labels: list[float | str], output_path: Path
):
    """Write a sequence dataset to disk."""
    # Check that the sequences and labels are the same length
    if len(sequences) != len(labels):
        raise ValueError('Sequences and labels must be the same length')

    # Setup the data fields
    data = {
        'dna': [str(x).upper() for x in sequences],
        'aminoacid': [str(x.translate()) for x in sequences],
        'label': labels,
    }

    # Create the dataset
    dset = datasets.Dataset.from_dict(data)

    # Save the dataset to disk
    dset.save_to_disk(output_path)


def _setup_calm_task(
    csv_path: Path, output_path: Path, sequence_col: str, label_col: str
):
    """Process the CaLM meltome CSV file."""
    # Load the CSV file and create sequences and labels
    df = pd.read_csv(csv_path, header=0)
    sequences = [Seq(seq) for seq in df[sequence_col]]
    labels = df[label_col].values.tolist()

    # Write the dataset to disk
    _write_sequence_dset(sequences, labels, output_path)


def _build_species_dataset(
    train_species_data: list[Path], test_species_data: list[Path], output_path: Path
):
    """Build a species classification dataset."""

    def create_entries_from_file(fasta_path, training_partition):
        species = fasta_path.stem
        sequences = list(SeqIO.parse(fasta_path, 'fasta'))

        entries = []
        for seq in sequences:
            entry = {
                'dna': str(seq.seq),
                'aminoacid': str(seq.seq.translate()),
                'species': species,
                'set': training_partition,
            }
            entries.append(entry)

        return entries

    output_data = []
    for species_file in train_species_data:
        output_data.extend(create_entries_from_file(species_file, 'train'))

    for species_file in test_species_data:
        output_data.extend(create_entries_from_file(species_file, 'test'))

    # Create the dataset
    dset = datasets.Dataset.from_list(output_data)
    dset.save_to_disk(output_path)


def download_calm_tasks(download_dir: Path) -> None:
    """Download and process the CaLM tasks."""
    # Check if the CaLM repository has been cloned
    if not (download_dir / 'CaLM').exists():
        # The CaLM repository path
        repo_path = 'https://github.com/oxpig/CaLM.git'

        # Clone the repository
        subprocess.run(['git', 'clone', repo_path], check=True, cwd=download_dir)

    # Set the data path
    calm_data_root = download_dir / 'CaLM' / 'data'

    # Process the meltome task data
    input_path = calm_data_root / 'meltome' / 'meltome_data.csv'
    output_path = download_dir / 'CaLM-Meltome'
    _setup_calm_task(input_path, output_path, 'sequence', 'melting_temperature')

    # Process the solubility task data
    input_path = calm_data_root / 'solubility' / 'solubility_data.csv'
    output_path = download_dir / 'CaLM-Solubility'
    _setup_calm_task(input_path, output_path, 'cds', 'solubility')

    # Process the localization task data
    input_path = calm_data_root / 'localization' / 'localization_data.csv'
    output_path = download_dir / 'CaLM-Localization'
    label_fields = [
        'Cell membrane',
        'Cytoplasm',
        'Endoplasmic reticulum',
        'Extracellular',
        'Golgi apparatus',
        'Lysosome/Vacuole',
        'Membrane',
        'Mitochondrion',
        'Nucleus',
        'Peroxisome',
        'Plastid',
    ]
    _setup_calm_task(input_path, output_path, 'Sequence', label_fields)

    # There are multiple sub splits for protein abundance
    for csv_file in calm_data_root.glob('protein_abundance/*.csv'):
        output_path = download_dir / f'CaLM-ProteinAbundance-{csv_file.stem}'
        _setup_calm_task(csv_file, output_path, 'cds', 'abundance')

    # There are multiple sub-splits for transcript abundance
    for csv_file in calm_data_root.glob('transcript_abundance/*.csv'):
        output_path = download_dir / f'CaLM-TranscriptAbundance-{csv_file.stem}'
        _setup_calm_task(csv_file, output_path, 'cds', 'logtpm')

    # Species classification needs its own special handling
    # Training is done on the test set of the validation data
    # and testing is done on the test set of the test data
    output_path = download_dir / 'CaLM-SpeciesClassification'
    train_species_data = list(calm_data_root.glob('species/validation/*.fasta'))
    test_species_data = list(calm_data_root.glob('species/test/*.fasta'))
    _build_species_dataset(train_species_data, test_species_data, output_path)

    # Remove the cloned repository
    shutil.rmtree(download_dir / 'CaLM')


class CaLMTaskConfig(EmbeddingTaskConfig, ABC):
    """Configuration for CaLM tasks."""

    # Placeholder for the task name (to be set by subclasses)
    name: Literal[''] = ''
    # Subset of task to use (only species, protein abundance, transcript abundance)
    subset: Literal[''] | None = None

    metrics: list[str] = Field(
        default=['mse', 'r2', 'pearson'], description='Metrics to measure'
    )
    max_samples: int | None = Field(
        default=None, description='Whether to limit the number of training samples'
    )
    target_col: str = Field(default='label', description='Target column in the dataset')

    # TODO: Consider moving this to a more general location
    download_dir: Path = Field(
        default=Path.home() / '.biolab' / 'data',
        description='Directory to download data',
    )

    # TODO: We can remove this if dataset_name_or_path is refactored
    # to be compatible with auto downloads (this is so users don't have
    # to manually set the data path)
    dataset_name_or_path: str = ''

    @model_validator(mode='after')
    def update_task_name(self):
        """Update the task name to have the subset name.

        This needs to be done post-init so we can successfully instantiate the task,
        but before the results are saved so that we can differentiate the splits of
        of the same task.
        """
        if self.subset:
            self.name = f'{self.name}-{self.subset}'
        return self

    @model_validator(mode='after')
    def download(self) -> Self:
        """Download the CaLM data."""
        # Create the download directory
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Set the dataset name or path if not provided
        if not self.dataset_name_or_path:
            self.dataset_name_or_path = str(self.download_dir / self.name)

        # Download the data
        if not Path(self.dataset_name_or_path).exists():
            logger.info('Downloading CaLM data')
            download_calm_tasks(self.download_dir)

        return self

    @field_serializer('name', check_fields=False, when_used='json')
    def serialize_name(self, name: str):
        """Serialize the task name to remove the subset name.

        This allows us to dump the model config and reload appropriately.
        """
        if self.subset:
            return name.replace(f'-{self.subset}', '')
        return name


class CaLMMeltomeConfig(CaLMTaskConfig):
    """Configuration for CaLM meltome task."""

    name: Literal['CaLM-Meltome'] = 'CaLM-Meltome'
    task_type: Literal['regression'] = 'regression'


class CaLMMeltome(EmbeddingTask):
    """CaLM meltome (melting temperature) task."""

    resolution: str = 'sequence'


class CaLMSolubilityConfig(CaLMTaskConfig):
    """Configuration for CaLM solubility task."""

    name: Literal['CaLM-Solubility'] = 'CaLM-Solubility'
    task_type: Literal['regression'] = 'regression'


class CaLMSolubility(EmbeddingTask):
    """CaLM solubility task."""

    resolution: str = 'sequence'


class CaLMLocalizationConfig(CaLMTaskConfig):
    """Configuration for CaLM localization task."""

    name: Literal['CaLM-Localization'] = 'CaLM-Localization'
    task_type: Literal['multi-label-classification'] = 'multi-label-classification'
    metrics: list[str] = ['accuracy', 'f1']


class CaLMLocalization(EmbeddingTask):
    """CaLM localization task."""

    resolution: str = 'sequence'


class CaLMProteinAbundanceConfig(CaLMTaskConfig):
    """Configuration for protein abundance prediction task."""

    name: Literal['CaLM-ProteinAbundance'] = 'CaLM-ProteinAbundance'
    subset: Literal[
        'athaliana',
        'dmelanogaster',
        'ecoli',
        'hsapiens',
        'hvolcanii',
        'ppastoris',
        'scerevisiae',
    ]
    task_type: Literal['regression'] = 'regression'
    metrics: list[str] = ['mse', 'r2', 'pearson']


class CaLMProteinAbundance(EmbeddingTask):
    """CaLM protein abundance prediction task."""

    resolution: str = 'sequence'


class CaLMTranscriptAbundanceConfig(CaLMTaskConfig):
    """Configuration for protein abundance prediction task."""

    name: Literal['CaLM-TranscriptAbundance'] = 'CaLM-TranscriptAbundance'
    subset: Literal[
        'athaliana',
        'dmelanogaster',
        'ecoli',
        'hsapiens',
        'hvolcanii',
        'ppastoris',
        'scerevisiae',
    ]
    task_type: Literal['regression'] = 'regression'
    metrics: list[str] = ['mse', 'r2', 'pearson']


class CaLMTranscriptAbundance(EmbeddingTask):
    """CaLM transcript abundance prediction task."""

    resolution: str = 'sequence'


class CaLMSpeciesClassificationConfig(CaLMTaskConfig):
    """Configuration for species classification task."""

    name: Literal['CaLM-SpeciesClassification'] = 'CaLM-SpeciesClassification'
    task_type: Literal['classification'] = 'classification'
    metrics: list[str] = ['accuracy', 'f1']

    species: list[str] = Field(
        default=[
            'athaliana',
            'dmelanogaster',
            'ecoli',
            'hsapiens',
            'hvolcanii',
            'ppastoris',
            'scerevisiae',
        ],
        description='Species fields in the dataset',
    )
    target_col: str = 'species'


class CaLMSpeciesClassification(EmbeddingTask):
    """CaLM species classification task."""

    resolution: str = 'sequence'

    def __init__(self, config: CaLMSpeciesClassificationConfig):
        super().__init__(config)

    def evaluate(
        self, model: LM, cache_dir: Path
    ) -> tuple[dict[str, DownstreamModel | None], list[Metric]]:
        """Evaluate the species classification task."""
        char_level = self.resolution in ['aminoacid', 'nucleotide']
        logger.info(f'Task resolution: {self.resolution} (char level: {char_level})')
        # Load the dataset
        task_dataset: datasets.Dataset = datasets.load_from_disk(
            self.config.dataset_name_or_path
        )

        # Special case for species classification we need to get embeddings
        # for each type of species, average them, then perform classification
        # doing a k-nearest center
        species_centers = {}
        for species in self.config.species:
            species_dataset = task_dataset.filter(
                lambda x: x['species'] == species and x['set'] == 'train'  # noqa: B023
            )
            input_sequences = species_dataset[model.model_input]
            species_cache_file = (
                cache_dir
                / f'{model.config.name}_{self.config.name}-{species}-embeddings.hdf5'
            )
            with HDF5CachedList(species_cache_file, mode='w') as model_outputs:
                logger.info(
                    f'Generating {model.model_input} embeddings '
                    f'({len(input_sequences):,})'
                )
                model_outputs = model.generate_model_outputs(
                    input_sequences, model_outputs, return_embeddings=True
                )

                # find and instantiate an output transform object
                transforms = find_transformation(
                    model.model_input, model.model_encoding, self.resolution
                )
                logger.info(
                    f'Found transformation {[tran.name for tran in transforms]}'
                )
                # Apply the transformations
                for transform in transforms:
                    logger.info(f'Applying {transform.name} transformation')
                    # TODO: Test the new input_ids return
                    model_outputs.map(
                        transform.apply_h5,
                        sequences=input_sequences,
                        tokenizer=model.tokenizer,
                    )

                species_center_embedding = np.array(
                    [model_output.embedding for model_output in model_outputs]
                )
                species_centers[species] = species_center_embedding.mean(axis=0)

        # Now we need to evaluate the test set
        test_dataset = task_dataset.filter(lambda x: x['set'] == 'test')
        input_sequences = test_dataset[model.model_input]

        # Create the cache file
        cache_file = cache_dir / f'{model.config.name}_{self.config.name}.h5'
        with HDF5CachedList(cache_file, mode='w') as model_outputs:
            logger.info(
                f'Generating {model.model_input} embeddings ({len(input_sequences):,})'
            )
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

            labels = test_dataset[self.config.target_col]
            encoded_labels = [self.config.species.index(label) for label in labels]

            # Now we need to classify the species
            predictions = [
                self.determine_species(model_output.embedding, species_centers)
                for model_output in model_outputs
            ]
            encoded_predictions = [
                self.config.species.index(label) for label in predictions
            ]
            # Calculate the metrics
            metrics = MetricCollection(
                [get_and_instantiate_metric(metric) for metric in self.config.metrics]
            )
            for metric in metrics:
                metric.evaluate(
                    predicted=encoded_predictions, labels=encoded_labels, train=False
                )

        # Return expects a dict of DownstreamModels, but we don't have any,
        # so return a dictionary with empty 'default' field and the metrics
        return {'default': None}, metrics

    def determine_species(
        self, query: np.ndarray, species_centers: dict[str, np.ndarray]
    ) -> str:
        """Calculate cosine similarity of query to the centers and choose the max."""
        distances = {}
        query_norm = np.linalg.norm(query)

        for species, center in species_centers.items():
            center_norm = np.linalg.norm(center)
            distances[species] = np.dot(query, center) / (query_norm * center_norm)

        return max(distances, key=distances.get)


class SpeciesClassificationModelingConfig(CaLMTaskConfig):
    """Configuration for species classification task using a embedding based model."""

    name: Literal['SpeciesClassificationModeling'] = 'SpeciesClassificationModeling'
    task_type: Literal['classification'] = 'classification'
    metrics: list[str] = ['accuracy', 'f1']

    species: list[str] = Field(
        default=[
            'athaliana',
            'dmelanogaster',
            'ecoli',
            'hsapiens',
            'hvolcanii',
            'ppastoris',
            'scerevisiae',
        ],
        description='Species fields in the dataset',
    )
    target_col: str = 'species'

    @model_validator(mode='after')
    def download(self) -> Self:
        """Overload the model validator to look for the species classification data."""
        # Create the download directory
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Set the dataset name or path if not provided
        if (
            not self.dataset_name_or_path
            or 'CaLM-SpeciesClassification' not in self.dataset_name_or_path
        ):
            self.dataset_name_or_path = str(
                self.download_dir / 'CaLM-SpeciesClassification'
            )

        # Download the data
        if not Path(self.dataset_name_or_path).exists():
            logger.info('Downloading CaLM data')
            download_calm_tasks(self.download_dir)

        return self


class SpeciesClassificationModeling(Task):
    """Species classification with downstream model."""

    resolution: str = 'sequence'

    def __init__(self, config: EmbeddingTaskConfig):
        super().__init__(config)
        assert hasattr(self, 'resolution'), 'Resolution must be set in the subclass'

    def evaluate(
        self, model: LM, cache_dir: Path
    ) -> tuple[dict[str, DownstreamModel | None], list[Metric]]:
        """Evaluate a FLIP character (aminoacid) level task."""
        logger.info(f'Task resolution: {self.resolution}')

        # Load the dataset
        task_dataset: datasets.Dataset = datasets.load_from_disk(
            self.config.dataset_name_or_path
        )

        # Filter out any sequences that are non-needed
        if 'set' in task_dataset.column_names:
            task_dataset = task_dataset.filter(lambda x: x['set'] is not None)

        # NOTE: Normally we can limit the samples but there are so few we will skip this

        # Generate embeddings
        logger.info(
            f'Generating {model.model_input} embeddings ({len(task_dataset):,})'
        )
        input_sequences = task_dataset[model.model_input]

        cache_file = cache_dir / f'{model.config.name}_{self.config.name}.h5'
        with HDF5CachedList(cache_file, mode='w') as model_outputs:
            logger.info(
                f'Generating {model.model_input} embeddings ({len(task_dataset):,})'
            )
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
            modeling_dataset = datasets.DatasetDict(
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
            downstream_models, metrics = downstream_modeling(
                task_dataset=modeling_dataset,
                input_col='transformed',
                target_col=self.config.target_col,
                metrics=metrics,
                k_fold=0,
            )

        # Cleanup the cache files if they have been created by this task
        task_dataset.cleanup_cache_files()

        return downstream_models, metrics


# Define tasks and configurations
calm_tasks = {
    CaLMMeltomeConfig: CaLMMeltome,
    CaLMSolubilityConfig: CaLMSolubility,
    CaLMLocalizationConfig: CaLMLocalization,
    CaLMProteinAbundanceConfig: CaLMProteinAbundance,
    CaLMTranscriptAbundanceConfig: CaLMTranscriptAbundance,
    CaLMSpeciesClassificationConfig: CaLMSpeciesClassification,
    SpeciesClassificationModelingConfig: SpeciesClassificationModeling,
}
