"""Tasks for CaLM."""

from __future__ import annotations

import shutil
import subprocess
import sys
from abc import ABC
from pathlib import Path
from typing import Literal

import datasets
import pandas as pd
from Bio.Seq import Seq
from pydantic import Field
from pydantic import field_serializer
from pydantic import model_validator

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    from typing import Self
else:  # pragma: <3.11 cover
    from typing_extensions import Self

# from biolab import task_registry
from biolab.api.logging import logger
from biolab.tasks.core.embedding_task import EmbeddingTask
from biolab.tasks.core.embedding_task import EmbeddingTaskConfig


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


# Define tasks and configurations
calm_tasks = {
    CaLMMeltomeConfig: CaLMMeltome,
    CaLMSolubilityConfig: CaLMSolubility,
    CaLMLocalizationConfig: CaLMLocalization,
    CaLMProteinAbundanceConfig: CaLMProteinAbundance,
    CaLMTranscriptAbundanceConfig: CaLMTranscriptAbundance,
}
