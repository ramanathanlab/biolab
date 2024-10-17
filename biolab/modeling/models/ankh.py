"""Implementation of the AnkH model."""

from __future__ import annotations

from typing import Any
from typing import Literal

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import PreTrainedTokenizer

from biolab import model_registry
from biolab.api.logging import logger
from biolab.api.modeling import HDF5CachedList
from biolab.api.modeling import LM
from biolab.api.modeling import LMConfig
from biolab.api.modeling import SequenceModelOutput
from biolab.modeling.utils.data import sequences_to_dataset


class AnkhConfig(LMConfig):
    """AnkH configuration."""

    name: Literal['Ankh'] = 'Ankh'
    # Model id or path to load the model
    size: str = 'base'
    # path to HF cache if download needed
    cache_dir: str | None = None


@model_registry.register(config=AnkhConfig)
class Ankh(LM):
    """AnkH wrapper class."""

    model_input: str = 'aminoacid'
    model_encoding: str = 'char'

    def __init__(self, config: AnkhConfig) -> None:
        """Initialize the Ankh model. Requires `pip install ankh`."""
        import os

        import ankh

        os.environ['HF_HOME'] = config.cache_dir

        # Load the model and tokenizer
        if config.size.lower() == 'large':
            model, tokenizer = ankh.load_large_model()
        else:
            model, tokenizer = ankh.load_base_model()

        # Set the model to evaluation mode
        model.eval()

        # Load the model onto the device
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
        model.to(device)
        # Set persistent attributes
        self.config = config
        self.model = model
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """HF Tokenizer object."""
        return self._tokenizer

    @property
    def tokenizer_config(self) -> dict[str, Any]:
        """Tokenizer configuration options."""
        return (
            self.config.tokenizer_config.model_dump()
            if self.config.tokenizer_config
            else {}
        )

    @property
    def dataloader_config(self) -> dict[str, Any]:
        """Dataloader configuration options."""
        return (
            self.config.dataloader_config.model_dump()
            if self.config.dataloader_config
            else {}
        )

    # TODO: might not actually need this
    @property
    def device(self) -> torch.device:
        """Torch device the model is placed on."""
        return self.model.device

    def generate_embeddings(
        self, sequences: list[str], model_outputs: HDF5CachedList | None = None
    ) -> list[SequenceModelOutput]:
        """Generate embeddings and logits for sequence input."""

        # Tokenize the dataset
        def tokenize_input(examples):
            seqs = [list(s) for s in examples['sequences']]
            return self.tokenizer(
                seqs,
                add_special_tokens=True,
                is_split_into_words=True,
                **self.tokenizer_config,
            )

        modeling_dataset = sequences_to_dataset(sequences)
        modeling_dataset = modeling_dataset.map(
            tokenize_input,
            batched=True,
            remove_columns=['sequences'],
        ).with_format('torch')

        # turn into dataloader and grab dset info
        dataloader = DataLoader(modeling_dataset, **self.dataloader_config)

        # Generate embeddings
        if model_outputs is None:
            model_outputs: list[SequenceModelOutput] = []
        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[logger]):
                for batch in tqdm(dataloader, desc='Generating embeddings'):
                    outputs = self.model(
                        batch['input_ids'].to(self.device),
                        batch['attention_mask'].to(self.device),
                        output_hidden_states=True,
                    )

                    # Get the sequence lengths, only eos token, remove last
                    seq_lengths = batch['attention_mask'].sum(axis=1) - 1

                    # Get the last hidden state
                    last_hidden_state = outputs.last_hidden_state
                    # Move the outputs to the CPU
                    embedding = last_hidden_state.cpu().detach().numpy()
                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        # Only an EOS token, removed by subtracting 1 from attn length
                        trimmed_embedding = embedding[i, :seq_len]

                        # Create the output object and append to list
                        output = SequenceModelOutput(embedding=trimmed_embedding)
                        model_outputs.append(output)

        return model_outputs

    def generate_sequences(self, input: list[str]) -> Dataset:
        """Generate sequences from one or more input prompts."""
        raise NotImplementedError
