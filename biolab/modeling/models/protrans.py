"""Implementation of ProtTrans model."""

from __future__ import annotations

from typing import Any
from typing import Literal

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import PreTrainedTokenizer

from biolab.api.logging import logger
from biolab.api.modeling import HDF5CachedList
from biolab.api.modeling import LM
from biolab.api.modeling import LMConfig
from biolab.api.modeling import SequenceModelOutput


class ProtTransConfig(LMConfig):
    """ProtTrans configuration."""

    name: Literal['ProtTrans'] = 'ProtTrans'
    # Model id or path to load the model
    pretrained_model_name_or_path: str
    # path to HF cache if download needed
    cache_dir: str | None = None
    # Half precision
    half_precision: bool = False


class ProtTrans(LM):
    """ProtTrans wrapper class."""

    model_input: str = 'aminoacid'
    model_encoding: str = 'char'

    def __init__(self, config: ProtTransConfig) -> None:
        """Initialize the ProtTrans model."""
        from transformers import T5EncoderModel
        from transformers import T5Tokenizer

        # Load tokenizer
        tokenizer = T5Tokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
            do_lower_case=False,
            cache_dir=config.cache_dir,
        )

        # Load model
        model = T5EncoderModel.from_pretrained(
            config.pretrained_model_name_or_path, cache_dir=config.cache_dir
        )

        # Convert the model to half precision
        if config.half_precision:
            model.half()

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
        """Access tokenizer object."""
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

    @property
    def device(self) -> torch.device:
        """Torch device of current model."""
        return self.model.device

    def generate_embeddings(
        self, sequences: list[str], model_outputs: HDF5CachedList | None = None
    ) -> list[SequenceModelOutput]:
        """Generate embeddings and logits for sequence input."""

        # Tokenize the dataset
        def tokenize_input(examples):
            seqs = [' '.join(list(sequence)) for sequence in examples['sequences']]
            return self.tokenizer(
                seqs,
                add_special_tokens=True,
                **self.tokenizer_config,
            )

        modeling_input = {'sequences': sequences}
        modeling_dataset = Dataset.from_dict(modeling_input)
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

                        # Create the output object
                        output = SequenceModelOutput(embedding=trimmed_embedding)
                        model_outputs.append(output)

        return model_outputs

    def generate_sequences(self, input: list[str]) -> list[SequenceModelOutput]:
        """Generate sequences from one or more input prompts."""
        raise NotImplementedError


protrans_models = {
    ProtTransConfig: ProtTrans,
}
