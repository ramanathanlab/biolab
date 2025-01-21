"""Implementations of ESM like models trained from scratch on RefSeq."""

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


class RefSeqESMConfig(LMConfig):
    """RefSeqGenSLM configuration."""

    name: Literal['RefSeqESM'] = 'RefSeqESM'
    # Model id or path to load the model
    pretrained_model_name_or_path: str
    # If tokenizer is needed, allow for path to tokenizer
    tokenizer_path: str | None = None
    # Whether to load from a CausalLM model or MaskedLM model
    # (some refseq models are trained on Llama models, they typically have BOS and EOS)
    is_causal: bool = False
    # remove bos token (true for refseq HF pretraining, is cls token)
    remove_bos_token: bool = True
    # remove eos token (false for refseq HF pretraining)
    remove_eos_token: bool = False
    # path to HF cache if download needed
    cache_dir: str | None = None
    # Use the model in half precision
    half_precision: bool = False


class RefSeqESM(LM):
    """RefSeqGenSLM wrapper model.

    This wrapper basically assumes the ESM backbone is used for the model.
    """

    model_input: str = 'aminoacid'
    model_encoding: str = 'char'

    def __init__(self, config: RefSeqESMConfig) -> None:
        """Initialize the Nucleotide transformer."""
        from transformers import AutoModelForCausalLM
        from transformers import AutoModelForMaskedLM
        from transformers import AutoTokenizer

        model_kwargs = {}
        if config.cache_dir:
            model_kwargs['cache_dir'] = config.cache_dir

            # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path
            if config.tokenizer_path
            else config.pretrained_model_name_or_path,
            trust_remote_code=True,
            cache_dir=config.cache_dir,
        )

        # Load model
        if config.is_causal:
            model_cls = AutoModelForCausalLM
        else:
            model_cls = AutoModelForMaskedLM

        model = model_cls.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
            **model_kwargs,
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
            # insert whitespaces between characters of the sequences
            examples['sequences'] = [' '.join(seq) for seq in examples['sequences']]
            return self.tokenizer(examples['sequences'], **self.tokenizer_config)

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

        start_offset = 1 if self.config.remove_bos_token else 0
        end_offset = 1 if self.config.remove_eos_token else 0
        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[logger]):
                for batch in tqdm(dataloader, desc='Generating embeddings'):
                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    outputs = self.model(
                        **batch, output_hidden_states=return_embeddings
                    )

                    # Get the sequence lengths  bos/eos in esm model, remove last token)
                    seq_lengths = batch['attention_mask'].sum(axis=1) - end_offset

                    logits = outputs.logits.cpu().detach().numpy()
                    if return_embeddings:
                        # Get the last hidden state
                        last_hidden_state = outputs.hidden_states[-1]
                        # Move the outputs to the CPU
                        embedding = last_hidden_state.cpu().detach().numpy()
                    else:
                        embedding = None

                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        seq_input_ids = None
                        seq_logits = None
                        seq_embedding = None
                        seq_attention_maps = None

                        # Remove the bos token and the padding
                        if return_input_ids:
                            seq_input_ids = (
                                batch['input_ids'][i, start_offset:seq_len]
                                .cpu()
                                .detach()
                                .numpy()
                            )
                        if return_logits:
                            seq_logits = logits[i, start_offset:seq_len, :]
                        if return_embeddings:
                            seq_embedding = embedding[i, start_offset:seq_len, :]
                        if return_attention_maps:
                            # TODO: look at model implementation for attention maps
                            seq_attention_maps = None

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


class RefSeqLlamaMLMConfig(LMConfig):
    """Llama with MLM configuration from Azton's GenSLM2 codebase."""

    name: Literal['RefSeqLlamaMLM'] = 'RefSeqLlamaMLM'
    # Model id or path to load the model
    pretrained_model_name_or_path: str
    # If tokenizer is needed, allow for path to tokenizer
    tokenizer_path: str | None = None
    # remove bos token (true for refseq HF pretraining, is cls token)
    remove_bos_token: bool = True
    # remove eos token (false for refseq HF pretraining)
    remove_eos_token: bool = True
    # path to HF cache if download needed
    cache_dir: str | None = None
    # Use the model in half precision
    half_precision: bool = True


class RefSeqLlamaMLM(LM):
    """RefSeq Llama for MLM wrapper model.

    This uses Azton's GenSLM2 codebase to for the model definition.
    """

    model_input: str = 'aminoacid'
    model_encoding: str = 'char'

    def __init__(self, config: RefSeqESMConfig) -> None:
        """Initialize the Nucleotide transformer."""
        from transformers import AutoTokenizer

        from .utils.modeling_genslm2 import LlamaForCausalLM as GenSLM2

        model_kwargs = {}
        if config.cache_dir:
            model_kwargs['cache_dir'] = config.cache_dir

            # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path
            if config.tokenizer_path
            else config.pretrained_model_name_or_path,
            trust_remote_code=True,
            cache_dir=config.cache_dir,
        )

        # Load model
        model = GenSLM2.from_pretrained(
            config.pretrained_model_name_or_path,
            **model_kwargs,
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
            # insert whitespaces between characters of the sequences
            examples['sequences'] = [' '.join(seq) for seq in examples['sequences']]
            return self.tokenizer(examples['sequences'], **self.tokenizer_config)

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

        start_offset = 1 if self.config.remove_bos_token else 0
        end_offset = 1 if self.config.remove_eos_token else 0
        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[logger]):
                for batch in tqdm(dataloader, desc='Generating embeddings'):
                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    outputs = self.model(**batch, output_hidden_states=True)

                    # Get the sequence lengths  bos/eos in esm model, remove last token)
                    seq_lengths = batch['attention_mask'].sum(axis=1) - end_offset

                    logits = outputs.logits.cpu().detach().numpy()
                    if return_embeddings:
                        # Get the last hidden state
                        last_hidden_state = outputs.hidden_states[-1]
                        # Move the outputs to the CPU
                        embedding = last_hidden_state.cpu().detach().numpy()
                    else:
                        embedding = None

                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        seq_input_ids = None
                        seq_logits = None
                        seq_embedding = None
                        seq_attention_maps = None

                        # Remove the bos token and the padding
                        if return_input_ids:
                            seq_input_ids = (
                                batch['input_ids'][i, start_offset:seq_len]
                                .cpu()
                                .detach()
                                .numpy()
                            )
                        if return_logits:
                            seq_logits = logits[i, start_offset:seq_len, :]
                        if return_embeddings:
                            seq_embedding = embedding[i, start_offset:seq_len, :]
                        if return_attention_maps:
                            # NOTE: the forward does return attention heads
                            # TODO: look at model implementation for attention maps
                            seq_attention_maps = None

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


genslm2_models = {
    RefSeqESMConfig: RefSeqESM,
    RefSeqLlamaMLMConfig: RefSeqLlamaMLM,
}
