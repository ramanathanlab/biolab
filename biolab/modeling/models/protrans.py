"""Implementation of ProtTrans model."""

from __future__ import annotations

from typing import Any
from typing import Literal

import numpy as np
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

    # Specify which embedding layer to use
    # there are n+1 layers in the model, 1 input layer,
    # and n transformer layers
    embedding_layer: int = -1


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

    def generate_model_outputs(
        self,
        sequences: list[str],
        model_outputs: HDF5CachedList | None = None,
        return_input_ids: bool = True,
        return_logits: bool = False,
        return_embeddings: bool = False,
        return_attention_maps: bool = False,
    ) -> list[SequenceModelOutput]:
        """Generate embeddings, logits, attention masks for sequence input."""

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
                        output_hidden_states=return_embeddings,
                        output_attentions=return_attention_maps,
                    )

                    # Get the sequence lengths, only eos token, remove last
                    seq_lengths = batch['attention_mask'].sum(axis=1) - 1

                    if return_embeddings:
                        # Get the last hidden state
                        hidden_state = outputs.hidden_states[
                            self.config.embedding_layer
                        ]
                        # Move the outputs to the CPU
                        embedding = hidden_state.cpu().detach().numpy()
                    else:
                        embedding = None

                    if return_attention_maps:
                        attention_maps = [
                            layer_attn.cpu().detach().numpy()
                            for layer_attn in outputs.attentions
                        ]
                        attention_maps = np.stack(attention_maps, axis=1)
                    else:
                        attention_maps = None

                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        seq_input_ids = None
                        seq_logits = None
                        seq_embedding = None
                        seq_attention_maps = None

                        # Only an EOS token, removed by subtracting 1 from attn length
                        if return_input_ids:
                            seq_input_ids = batch['input_ids'][i, :seq_len]
                        if return_logits:
                            # TODO: look at model implementation for logits
                            seq_logits = None
                        if return_embeddings:
                            seq_embedding = embedding[i, :seq_len]
                        if return_attention_maps:
                            # Attention maps are shape (B, L, H, T, T)
                            seq_attention_maps = attention_maps[
                                i, :, :seq_len, :seq_len
                            ]

                        output_fields = {
                            'input_ids': seq_input_ids,
                            'logits': seq_logits,
                            'embedding': seq_embedding,
                            'attention_maps': seq_attention_maps,
                        }

                        # Create the output object
                        model_outputs.append(SequenceModelOutput(**output_fields))

        return model_outputs

    def generate_sequences(self, input: list[str]) -> list[SequenceModelOutput]:
        """Generate sequences from one or more input prompts."""
        raise NotImplementedError


protrans_models = {
    ProtTransConfig: ProtTrans,
}
