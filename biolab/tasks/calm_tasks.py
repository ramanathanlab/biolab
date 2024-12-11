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
from pydantic import model_validator

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    from typing import Self
else:  # pragma: <3.11 cover
    from typing_extensions import Self

# from biolab import task_registry
from biolab.api.logging import logger
from biolab.tasks.core.sequence_embedding import SequenceTask
from biolab.tasks.core.sequence_embedding import SequenceTaskConfig


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
    labels = df[label_col].to_list()

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

    # Remove the cloned repository
    shutil.rmtree(download_dir / 'CaLM')


class CaLMTaskConfig(SequenceTaskConfig, ABC):
    """Configuration for CaLM tasks."""

    # Placeholder for the task name (to be set by subclasses)
    name: Literal[''] = ''

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
    def download(self) -> Self:
        """Download the CaLM data."""
        # Create the download directory
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Set the dataset name or path
        self.dataset_name_or_path = str(self.download_dir / self.name)

        # Download the data
        if not Path(self.dataset_name_or_path).exists():
            logger.info('Downloading CaLM data')
            download_calm_tasks(self.download_dir)

        return self


class CaLMMeltomeConfig(CaLMTaskConfig):
    """Configuration for CaLM meltome task."""

    name: Literal['CaLM-Meltome'] = 'CaLM-Meltome'
    task_type: Literal['regression'] = 'regression'


# @task_registry.register(config_class=CaLMMeltomeConfig)
class CaLMMeltome(SequenceTask):
    """CaLM meltome (melting temperature) task."""


class CaLMSolubilityConfig(CaLMTaskConfig):
    """Configuration for CaLM solubility task."""

    name: Literal['CaLM-Solubility'] = 'CaLM-Solubility'
    task_type: Literal['regression'] = 'regression'


# @task_registry.register(config_class=CaLMSolubilityConfig)
class CaLMSolubility(SequenceTask):
    """CaLM solubility task."""

    resolution: str = 'sequence'


# Define tasks and configurations
calm_configs = [CaLMMeltomeConfig, CaLMSolubilityConfig]
calm_tasks = {
    CaLMMeltomeConfig: CaLMMeltome,
    CaLMSolubilityConfig: CaLMSolubility,
}
