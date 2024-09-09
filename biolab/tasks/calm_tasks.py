"""Tasks for CaLM."""

from __future__ import annotations

import os
import subprocess
import sys
from abc import ABC
from pathlib import Path
from typing import Literal

import datasets
import pandas as pd
from Bio.Seq import Seq
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    from typing import Self
else:  # pragma: <3.11 cover
    from typing_extensions import Self

from biolab import metric_registry
from biolab import task_registry
from biolab.api.logging import logger
from biolab.api.modeling import LM
from biolab.api.task import Task
from biolab.api.task import TaskConfig
from biolab.tasks.core.regression import sklearn_svr
from biolab.tasks.core.utils import find_transformation
from biolab.tasks.core.utils import limit_training_samples


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
    # The CaLM repository path
    repo_path = 'https://github.com/oxpig/CaLM.git'

    # Clone the repository
    subprocess.run(['git', 'clone', repo_path], check=True, cwd=download_dir)

    # Set the data path
    calm_data_root = download_dir / 'CaLM' / 'data'

    # Process the meltome task data
    input_path = calm_data_root / 'meltome' / 'meltome_data.csv'
    output_path = download_dir / CaLMMeltomeConfig.name
    _setup_calm_task(input_path, output_path, 'sequence', 'melting_temperature')

    # Process the solubility task data
    input_path = calm_data_root / 'solubility' / 'solubility_data.csv'
    output_path = download_dir / CaLMSolubilityConfig.name
    _setup_calm_task(input_path, output_path, 'sequence', 'solubility')

    # Remove the cloned repository
    os.rmdir(download_dir / 'CaLM')


class CaLMTaskConfig(TaskConfig, ABC):
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

    # TODO: Consider moving this and the validator to a more general location
    download_dir: Path = Field(
        default=Path.home() / '.biolab' / 'data',
        description='Directory to download data',
    )

    # TODO: We can remove this if dataset_name_or_path is refactored
    # to be compatible with auto downloads (this is so users don't have
    # to manually set the data path)
    dataset_name_or_path: str = ''

    @model_validator(mode='after')
    def _set_dataset(self) -> Self:
        self.dataset_name_or_path = str(self.download_dir / self.name)
        return self

    @field_validator('download_dir')
    @classmethod
    def create_download_dir(cls, v: Path) -> Path:
        """Create the download directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode='after')
    def download(self) -> Self:
        """Download the CaLM data."""
        download_calm_tasks(self.download_dir)
        return self


# TODO: add caching to task (way to store some results/models/intermediates)
class CaLMTask(Task, ABC):
    """Melting temperature prediction from diverse sequences."""

    resolution: str = 'sequence'

    def __init__(self, config: CaLMTaskConfig):
        self.config = config

    def evaluate(self, model: LM):
        """Evaluate a given model on GC content task."""
        # Load the dataset
        task_dataset = datasets.load_from_disk(self.config.dataset_name_or_path)
        task_dataset.set_format('torch')
        logger.info('Loaded dataset')

        # Limit the number of training samples if specified
        if self.config.max_samples:
            task_dataset = limit_training_samples(
                task_dataset,
                self.config.max_samples,
                model.model_input,
                self.config.target_col,
                continuous=True,
            )

        # Generate embeddings
        logger.info(f'Generating {model.model_input} embeddings')
        input_sequences = task_dataset[model.model_input]
        model_outputs = model.generate_embeddings(input_sequences)

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
            model_outputs = transform.apply(
                model_outputs, sequences=input_sequences, tokenizer=model.tokenizer
            )

        embed_dict = {
            'transformed': [output.embedding for output in model_outputs],
        }
        task_dataset = datasets.concatenate_datasets(
            [task_dataset, datasets.Dataset.from_dict(embed_dict)], axis=1
        )

        # Setup metrics to pass to regressor
        metrics = [metric_registry.get(metric)() for metric in self.config.metrics]
        metrics = sklearn_svr(
            task_dataset, 'transformed', self.config.target_col, metrics
        )

        for metric in metrics:
            logger.info(f'Metric: {metric.__class__.__name__}\tValue: {metric.result}')


class CaLMMeltomeConfig(CaLMTaskConfig):
    """Configuration for CaLM meltome task."""

    name: Literal['CaLM-Meltome'] = 'CaLM-Meltome'


@task_registry.register(config=CaLMMeltomeConfig)
class CaLMMeltome(CaLMTask):
    """CaLM meltome (melting temperature) task."""


class CaLMSolubilityConfig(CaLMTaskConfig):
    """Configuration for CaLM solubility task."""

    name: Literal['CaLM-Solubility'] = 'CaLM-Solubility'


@task_registry.register(config=CaLMSolubilityConfig)
class CaLMSolubility(CaLMTask):
    """CaLM solubility task."""
