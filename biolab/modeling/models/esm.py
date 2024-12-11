"""Implementations of ESM(2/3) models."""

from __future__ import annotations

from typing import Any
from typing import Literal

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import PreTrainedTokenizer

# from biolab import model_registry
from biolab.api.logging import logger
from biolab.api.modeling import HDF5CachedList
from biolab.api.modeling import LM
from biolab.api.modeling import LMConfig
from biolab.api.modeling import SequenceModelOutput


class ESMConfig(LMConfig):
    """ESM configuration."""

    name: Literal['ESM'] = 'ESM'
    # Model id or path to load the model
    pretrained_model_name_or_path: str
    # path to HF cache if download needed
    cache_dir: str | None = None
    # Use the model in half precision
    half_precision: bool = False


# @model_registry.register(config=ESMConfig)
class ESM(LM):
    """ESM2 wrapper model."""

    model_input: str = 'aminoacid'
    model_encoding: str = 'char'

    def __init__(self, config: ESMConfig) -> None:
        """Initialize the Nucleotide transformer."""
        from transformers import AutoModelForMaskedLM
        from transformers import AutoTokenizer

        model_kwargs = {}
        if config.cache_dir:
            model_kwargs['cache_dir'] = config.cache_dir

            # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
            cache_dir=config.cache_dir,
        )

        # Load model
        model = AutoModelForMaskedLM.from_pretrained(
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

    def generate_embeddings(
        self, sequences: list[str], model_outputs: HDF5CachedList | None = None
    ) -> list[SequenceModelOutput]:
        """Generate embeddings and logits for sequence input."""

        # Tokenize the dataset
        def tokenize_input(examples):
            return self.tokenizer(examples['sequences'], **self.tokenizer_config)

        modeling_input = {'sequences': sequences}
        modeling_dataset = Dataset.from_dict(modeling_input)
        modeling_dataset = modeling_dataset.map(
            tokenize_input,
            batched=True,
            remove_columns=['sequences'],
        ).with_format('torch')
        logger.info('Tokenized dataset.')

        # turn into dataloader and grab dset info
        dataloader = DataLoader(modeling_dataset, **self.dataloader_config)

        # Generate embeddings
        if model_outputs is None:
            model_outputs: list[SequenceModelOutput] = []
        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[logger]):
                for batch in tqdm(dataloader, desc='Generating embeddings'):
                    # Get the sequence lengths  bos/eos in esm model, remove last token)
                    # before moving to device
                    seq_lengths = batch['attention_mask'].sum(axis=1) - 1

                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    outputs = self.model(**batch, output_hidden_states=True)

                    # Get the last hidden state
                    last_hidden_state = outputs.hidden_states[-1]

                    # Move the outputs to the CPU
                    logits = outputs.logits.cpu().detach().numpy()
                    embedding = last_hidden_state.cpu().detach().numpy()

                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        # Remove the bos token and the padding
                        logit = logits[i, 1:seq_len, :]
                        trimmed_embedding = embedding[i, 1:seq_len, :]

                        # Create the output object
                        output = SequenceModelOutput(
                            logits=logit, embedding=trimmed_embedding
                        )
                        model_outputs.append(output)

        return model_outputs

    def generate_sequences(self, input: list[str]) -> list[SequenceModelOutput]:
        """Generate sequences from one or more input prompts."""
        raise NotImplementedError


class ESM3Config(LMConfig):
    """ESM3 configuration."""

    name: Literal['ESM3'] = 'ESM3'
    # Model id or path to load the model
    pretrained_model_name_or_path: str = 'esm3_sm_open_v1'
    # HF token with read only access to ESM3
    hf_token: str
    # path to HF cache if download needed
    cache_dir: str | None = None
    # Use the model in half precision
    half_precision: bool = False


# @model_registry.register(config=ESM3Config)
class ESM3(LM):
    """ESM3 wrapper module."""

    model_input = 'aminoacid'
    model_encoding = 'char'

    def __init__(self, config: ESM3Config) -> None:
        import os

        os.environ['HF_HOME'] = config.cache_dir
        os.environ['HF_TOKEN'] = config.hf_token
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        from esm.models.esm3 import ESM3
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        # Load the model
        model = ESM3.from_pretrained(config.pretrained_model_name_or_path)

        # Set the model to evaluation mode
        model.eval()

        # Load the model onto the device
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
        model.to(device)

        # Load the tokenizer
        tokenizer = EsmSequenceTokenizer()

        # Set persistent attributes
        self.config = config
        self.model = model
        self._tokenizer = tokenizer
        self._device = device

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
        self, sequences: list[str], model_outputs: HDF5CachedList | None = None
    ) -> list[SequenceModelOutput]:
        """Generate embeddings and logits for sequence input."""

        # Tokenize the dataset
        def tokenize_input(examples):
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
        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[logger]):
                for batch in tqdm(dataloader, desc='Generating embeddings'):
                    # The model takes lots of types of inputs in different tracks
                    # Until we can support non-sequence types the only thing
                    # needed is input_ids
                    outputs = self.model(
                        sequence_tokens=batch['input_ids'].to(self.device)
                    )

                    # Get the sequence lengths  bos/eos in esm model, remove last token)
                    seq_lengths = batch['attention_mask'].sum(axis=1) - 1

                    # Get the last hidden state
                    last_hidden_state = outputs.embeddings

                    # Move the outputs to the CPU
                    embedding = last_hidden_state.cpu().detach().numpy()

                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        # Remove the bos token and the padding
                        trimmed_embedding = embedding[i, 1:seq_len, :]

                        # Create the output object
                        output = SequenceModelOutput(embedding=trimmed_embedding)
                        model_outputs.append(output)

        return model_outputs

    def generate_sequences(self, input: list[str]) -> list[SequenceModelOutput]:
        """Generate sequences from one or more input prompts."""
        raise NotImplementedError


class ESMCConfig(LMConfig):
    """ESMC configuration."""

    name: Literal['ESMC'] = 'ESMC'
    # Model id or path to load the model
    pretrained_model_name_or_path: str = 'esmc_300m'
    # path to HF cache if download needed
    cache_dir: str | None = None


# @model_registry.register(config=ESMCConfig)
class ESMC(LM):
    """ESMC wrapper module."""

    model_input = 'aminoacid'
    model_encoding = 'char'

    def __init__(self, config: ESMCConfig) -> None:
        import os

        from esm.models.esmc import ESMC
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        # Set the cache directory as its not exposed by the esm api
        os.environ['HF_HOME'] = config.cache_dir

        # Load the model
        model = ESMC.from_pretrained(config.pretrained_model_name_or_path)

        # Set the model to evaluation mode
        model.eval()

        # Load the model onto the device
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
        model.to(device)

        # Load the tokenizer
        tokenizer = EsmSequenceTokenizer()

        # Set persistent attributes
        self.config = config
        self.model = model
        self._tokenizer = tokenizer
        self._device = device

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
        self, sequences: list[str], model_outputs: HDF5CachedList | None = None
    ) -> list[SequenceModelOutput]:
        """Generate embeddings and logits for sequence input."""

        # Tokenize the dataset
        def tokenize_input(examples):
            return self.tokenizer(examples['sequences'], **self.tokenizer_config)

        modeling_input = {'sequences': sequences}
        modeling_dataset = Dataset.from_dict(modeling_input)
        modeling_dataset = modeling_dataset.map(
            tokenize_input,
            batched=True,
            remove_columns=['sequences'],
        ).with_format('torch')
        logger.info('Tokenized dataset')

        # turn into dataloader and grab dset info
        dataloader = DataLoader(modeling_dataset, **self.dataloader_config)

        # Generate embeddings
        if model_outputs is None:
            model_outputs: list[SequenceModelOutput] = []
        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[logger]):
                for batch in tqdm(dataloader, desc='Generating embeddings'):
                    # The model takes lots of types of inputs in different tracks
                    # Until we can support non-sequence types the only thing
                    # needed is input_ids
                    outputs = self.model(
                        sequence_tokens=batch['input_ids'].to(self.device)
                    )

                    # Get the sequence lengths  bos/eos in esm model, remove last token)
                    seq_lengths = batch['attention_mask'].sum(axis=1) - 1

                    # Get the last hidden state
                    last_hidden_state = outputs.embeddings

                    # Move the outputs to the CPU
                    # Cast to float16 since model is in bfloat16
                    # (this could lead to loss of precision?)
                    logits = (
                        outputs.sequence_logits.detach().cpu().to(torch.float16).numpy()
                    )
                    embedding = (
                        last_hidden_state.detach().cpu().to(torch.float16).numpy()
                    )

                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        # Remove the bos token and the padding
                        trimmed_logits = logits[i, 1:seq_len, :]
                        trimmed_embedding = embedding[i, 1:seq_len, :]

                        # Create the output object
                        output = SequenceModelOutput(
                            logits=trimmed_logits, embedding=trimmed_embedding
                        )
                        model_outputs.append(output)

        return model_outputs

    def generate_sequences(self, input: list[str]) -> list[SequenceModelOutput]:
        """Generate sequences from one or more input prompts."""
        raise NotImplementedError


esm_models = {
    ESMConfig: ESM,
    ESM3Config: ESM3,
    ESMCConfig: ESMC,
}
