"""Implementation of ProtGPT2 model."""

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


class ProtGPT2Config(LMConfig):
    """ProtGPT2 Configuration."""

    name: Literal['ProtGPT2'] = 'ProtGPT2'
    # Model id or path to load the model
    pretrained_model_name_or_path: str = 'nferruz/ProtGPT2'
    # path to HF cache if download needed
    cache_dir: str | None = None

    # Specify the embedding layer to use
    # there are n+1 layers in the model, 1 input layer,
    # and n transformer layers
    embedding_layer: int = -1


class ProtGPT2(LM):
    """ProtGPT2 wrapper model."""

    model_input: str = 'aminoacid'
    model_encoding: str = 'bpe'

    def __init__(self, config: ProtGPT2Config) -> None:
        """Initialize the Nucleotide transformer."""
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer

        model_kwargs = {}
        if config.cache_dir:
            model_kwargs['cache_dir'] = config.cache_dir

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        # Hack to allow for mixed-length inputs in a batch.
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
            **model_kwargs,
        )

        # Convert the model to half precision
        model = model.half()

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
            return self.tokenizer(examples['sequences'], **self.tokenizer_config)

        # Have to manually set padding to false because we con only fit a single
        # sequence in the model at the same time
        # self.config.tokenizer_config.padding = False
        modeling_input = {'sequences': sequences}
        modeling_dataset = Dataset.from_dict(modeling_input)
        modeling_dataset = modeling_dataset.map(
            tokenize_input,
            batched=True,
            remove_columns=['sequences'],
        ).with_format('torch')

        # turn into dataloader and grab dset info
        # logger.info("Manually setting batch size to 1 (due to model constraints)")
        # self.config.dataloader_config.batch_size = 1
        dataloader = DataLoader(modeling_dataset, **self.dataloader_config)

        # Generate embeddings
        if model_outputs is None:
            model_outputs: list[SequenceModelOutput] = []
        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[logger]):
                for batch in tqdm(dataloader, desc='Generating embeddings'):
                    input_ids = batch['input_ids'].to(self.model.device)
                    outputs = self.model(
                        input_ids,
                        labels=input_ids,
                        attention_mask=batch['attention_mask'].to(self.model.device),
                        output_hidden_states=return_embeddings,
                        output_attentions=return_attention_maps,
                    )

                    # Get the sequence lengths (no apparent bos/eos in ProtGPT2)
                    seq_lengths = batch['attention_mask'].sum(axis=1)

                    logits = outputs.logits.cpu().detach().numpy()
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

                        # Model does not have special tokens, only remove padding
                        if return_input_ids:
                            seq_input_ids = (
                                input_ids[i, :seq_len].cpu().detach().numpy()
                            )
                        if return_logits:
                            seq_logits = logits[i, :seq_len, :]
                        if return_embeddings:
                            seq_embedding = embedding[i, :seq_len, :]
                        if return_attention_maps:
                            # Attention maps are of shape (B, L, H, T, T)
                            seq_attention_maps = attention_maps[
                                i, :, :, :seq_len, :seq_len
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


protgpt2_models = {
    ProtGPT2Config: ProtGPT2,
}
