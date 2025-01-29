"""Implementation of LM specific tasks from Evo 1.5."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import datasets
import torch
from pydantic import field_serializer
from pydantic import model_validator

from biolab.api.logging import logger
from biolab.api.metric import Metric
from biolab.api.metric import MetricCollection
from biolab.api.modeling import HDF5CachedList
from biolab.api.modeling import LM
from biolab.api.modeling import SequenceModelOutput
from biolab.api.task import DownstreamModel
from biolab.api.task import Task
from biolab.api.task import TaskConfig
from biolab.metrics import get_and_instantiate_metric


class EvoTaskConfig(TaskConfig):
    """Base configuration for GUE tasks."""

    # Split of the task
    subset: Literal['']
    # TODO: most tasks are under 100k sequences, but think about if we need a max_samples field

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


class EvoDMSConfig(EvoTaskConfig):
    """Configuration for DMS prediction tasks from Evo."""

    name: Literal['Evo-DMS'] = 'Evo-DMS'
    task_type: Literal['sequence-likelihood'] = 'sequence-likelihood'
    logit_reduction: Literal['mean', 'sum'] = 'mean'

    subset: Literal[
        'andreasson_2020',
        'BLAT_ECOLX_Firnberg_2014',
        'BLAT_ECOLX_Jacquier_2013',
        'BRCA1_Findlay_2018',
        'CBS_Sun_2020',
        'CCDB_ECOLI_Adkar_2012',
        'domingo_2018',
        'ECOLI_THERM_Tsuboyama_2023',
        'GDI1_Silverstein_2022',
        'guy_2014',
        'hayden_2011',
        'IF1_ECOLI_Kelsic_2016',
        'kobori_2016',
        'P53_Giacomelli_2018',
        'P53_Kotler_2018',
        'PDE3A_Garvie_2021',
        'pitt_2010',
        'RNC_ECOLI_Weeks_2023',
        'zhang_2009',
    ]

    metrics: list[str] = ['r2', 'pearson', 'spearman']


# TODO how can I implement logits more effectively?
class EvoDMS(Task):
    """Implementation of DMS prediction tasks from Evo.

    This implementation uses logits as downstream modeling input.
    """

    resolution = 'sequence'  # won't be used, but required for downstream tasks

    def __init__(self, config: EvoTaskConfig):
        super().__init__(config)
        self.subset = config.subset

    def evaluate(
        self, model: LM, cache_dir: Path
    ) -> tuple[dict[str, DownstreamModel | None], list[Metric]]:
        """Evaluate a model on the DMS prediction task."""
        logger.info(f'Evaluating subset: {self.subset}')

        # Load the dataset
        # NOTE: Originally I set format to numpy, but it prohibits multi-dimension
        # arrays being concatenated into the downstream dataset, removing it does
        # not seem to cause issues.
        task_dataset: datasets.Dataset = datasets.load_from_disk(
            self.config.dataset_name_or_path
        )

        # Score sequence log likelihoods
        logger.info(f'Scoreing {model.model_input} sequences ({len(task_dataset):,})')
        input_sequences = task_dataset[model.model_input]

        cache_file = cache_dir / f'{model.config.name}_{self.config.name}.h5'
        with HDF5CachedList(cache_file, mode='w') as model_outputs:
            model_outputs = model.generate_model_outputs(
                input_sequences,
                model_outputs,
                return_input_ids=True,
                return_logits=True,
            )

            # Calculate the sequence likelihood
            sequence_scores = [
                self.sequence_likelihood(model_output, self.config.logit_reduction)
                for model_output in model_outputs
            ]

            # Setup metrics
            metrics = MetricCollection(
                [get_and_instantiate_metric(metric) for metric in self.config.metrics]
            )

            # Evaluate the logits on the task
            for metric in metrics:
                metric.evaluate(sequence_scores, task_dataset['label'], train=False)

        # Return expects a dict of DownstreamModels, but we don't have any,
        # so return a dictionary with empty 'default' field
        return {'default': None}, metrics

    def sequence_likelihood(
        self, model_output: SequenceModelOutput, logit_reduction: str
    ) -> SequenceModelOutput:
        """Return the sequence likelihood from the logits and input_ids."""
        assert hasattr(model_output, 'logits'), 'Model output does not have logits'
        assert hasattr(
            model_output, 'input_ids'
        ), 'Model output does not have input_ids'

        # make logits, input_ids torch tensor for easier manipulation
        logits = torch.tensor(model_output.logits)
        input_ids = torch.tensor(model_output.input_ids)

        # Calculate the sequence likelihood
        softmax_logprobs = torch.log_softmax(logits, dim=-1)

        # Get the log likelihood of the correct token
        token_scores = torch.gather(
            softmax_logprobs,
            dim=1,
            index=input_ids.unsqueeze(-1),
        ).squeeze(-1)

        # Reduce the log likelihoods to a single scalar
        if logit_reduction == 'mean':
            logit_reduction = torch.mean
        elif logit_reduction == 'sum':
            logit_reduction = torch.sum

        return logit_reduction(token_scores)


# Define tasks and configurations
evo_tasks = {EvoDMSConfig: EvoDMS}
