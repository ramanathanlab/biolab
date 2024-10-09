from __future__ import annotations  # noqa: D100

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


class EvoConfig(LMConfig):
    """Configuration for Evo."""

    name: Literal['Evo'] = 'Evo'
    # Model id or path to load the model
    pretrained_model_name_or_path: str
    # Model context length
    context_length: int = 8000
    # path to HF cache if download needed
    cache_dir: str | None = None
    # Use the model in half precision
    half_precision: bool = False
    # Set the model to evaluation mode
    eval_mode: bool = True


@model_registry.register(config=EvoConfig)
class Evo(LM):
    """Wrapper for Evo."""

    model_input: str = 'dna'
    model_encoding: str = 'char'

    def __init__(self, config: EvoConfig) -> None:
        """Initialize Evo (striped hyena)."""
        import os

        import torch
        from evo import Evo

        # Set context length if mismatched, assume globally set length is truth
        if config.tokenizer_config.max_length != config.context_length:
            config.tokenizer_config.max_length = config.context_length

        if config.cache_dir:
            os.environ['HF_HOME'] = config.cache_dir

        # Grab the model constructors
        evo_model = Evo(config.pretrained_model_name_or_path)

        # Instant
        model, tokenizer = evo_model.model, evo_model.tokenizer

        # Set the model to evaluation mode
        if config.eval_mode:
            model.eval()

        # Load the model onto the device
        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
        model.to(self._device)

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

    @property
    def device(self) -> torch.device:
        """Torch device the model is placed on."""
        return self._device

    def generate_embeddings(
        self, sequences: list[str], model_outputs: HDF5CachedList
    ) -> list[SequenceModelOutput]:
        """Generate embeddings and logits for sequence input."""
        from evo.scoring import prepare_batch

        # Temporarily replace the unembed function to get embeddings from the model
        # We must do this to get the embeddings from the model.
        # See: https://github.com/evo-design/evo/issues/32
        from torch import nn

        class CustomEmbedding(nn.Module):
            def unembed(self, u):
                return u

        original_unembed = self.model.unembed
        self.model.unembed = CustomEmbedding()

        # Tokenize the dataset
        def tokenize_input(examples):
            input_ids, seq_lenghts = prepare_batch(
                examples['sequences'], self.tokenizer, prepend_bos=False, device='cpu'
            )
            return {
                'input_ids': input_ids,
                'attention_mask': input_ids != self.tokenizer.pad_id,
            }

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
                    hidden_states, _ = self.model(batch['input_ids'].to(self._device))

                    # Get the sequence lengths (no bos/eos in evo model)
                    seq_lengths = batch['attention_mask'].sum(axis=1)

                    embedding = hidden_states.half().cpu().detach().numpy()

                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        # Remove the cls token and the padding
                        trimmed_embedding = embedding[i, 1:seq_len, :]

                        # Create the output object
                        output = SequenceModelOutput(
                            logits=None, embedding=trimmed_embedding
                        )
                        model_outputs.append(output)

        # Reset the unembed function
        self.model.unembed = original_unembed

        return model_outputs

    def generate_sequences(self, input: list[str]) -> list[SequenceModelOutput]:
        """Generate sequences from one or more input prompts."""
        raise NotImplementedError
