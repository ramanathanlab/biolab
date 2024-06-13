"""Sane defaults for the AutoModel class."""

from typing import Literal, Optional, Any

from biolab.api.lm import LM, LMConfig
from biolab import model_registry

import torch
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput


class HFAutoModelConfig(LMConfig):
    """Config for the transformers AutoModel class."""

    # The name of the encoder
    name: Literal["HFAutoModel"] = "HFAutoModel"  # type: ignore[assignment]
    # path to HF cache if download needed
    cache_dir: Optional[str] = None
    # Model id or path to load the model
    pretrained_model_name_or_path: str
    # Tokenizer id or path (if different from model)
    tokenizer_name_or_path: Optional[str] = None
    # Use the model in half precision
    half_precision: bool = False
    # Set the model to evaluation mode
    eval_mode: bool = True


@model_registry.register(config=HFAutoModelConfig)
class HFAutoModel(LM):
    def __init__(self, config: HFAutoModelConfig) -> None:
        """Initialize the AutoModel class."""
        from transformers import AutoModel, AutoTokenizer

        model_kwargs = {}
        if config.cache_dir:
            model_kwargs["cache_dir"] = config.cache_dir

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            (
                config.pretrained_model_name_or_path
                if config.tokenizer_name_or_path is None
                else config.tokenizer_name_or_path
            ),
            trust_remote_code=True,
        )

        # Load model
        model = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
            **model_kwargs,
        )

        # Convert the model to half precision
        if config.half_precision:
            model.half()

        # Set the model to evaluation mode
        if config.eval_mode:
            model.eval()

        # Load the model onto the device
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        model.to(device)

        # Set persistent attributes
        self.config = config
        self.model = model
        self._tokenizer = tokenizer

    @property
    # TODO: could potentially need a full class to deal with templating
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer of the encoder."""
        return self._tokenizer

    @property
    def tokenizer_config(self) -> dict[str, Any]:
        """Get the tokenizer configuration"""
        return (
            self.config.tokenizer_config.model_dump()
            if self.config.tokenizer_config
            else {}
        )

    @property
    def dataloader_config(self) -> dict[str, Any]:
        """Get the dataloader configuration"""
        return (
            self.config.dataloader_config.model_dump()
            if self.config.dataloader_config
            else {}
        )

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the encoder."""
        return self.model.dtype

    @property
    def device(self) -> torch.device:
        """Get the device of the encoder."""
        return self.model.device

    @property
    def embedding_size(self) -> int:
        """Get the embedding size of the encoder."""
        return self.model.config.hidden_size

    def embed(self, batch_encoding: BatchEncoding) -> BaseModelOutput:
        """Embed the sequence.

        Parameters
        ----------
        batch_encoding : BatchEncoding
            The batch encoding of the sequence (containing the input_ids,
            attention_mask, and token_type_ids).

        Returns
        -------
        BaseModelOutput
            The embeddings of the sequence extracted from the last hidden state
            (shape: [num_sequences, sequence_length, embedding_size])
        """
        # Most basic forward pass, enable returning hidden states
        return self.model(**batch_encoding, output_hidden_states=True)

    # TODO: figure out for a standard auto model
    def generate(self, batch_encoding: BatchEncoding) -> BaseModelOutput:
        """Generate the sequence.

        Parameters
        ----------
        batch_encoding : BatchEncoding
            The batch encoding of the sequence (containing the input_ids,
            attention_mask, and token_type_ids).

        Returns
        -------
        BaseModelOutput
            Container for model outputs, includes logits
                logits shape: [num_sequences, sequence_length, vocab_size]
        """
        raise NotImplementedError


class HFAutoModelForMaskedLMConfig(HFAutoModelConfig):
    """Config for the transformers BERT style AutoModel class."""

    # The name of the encoder
    name: Literal["HFAutoModelForMaskedLM"] = "HFAutoModelForMaskedLM"  # type: ignore[assignment] # noqa: E501


@model_registry.register(config=HFAutoModelForMaskedLMConfig)
class HFAutoModelForMaskedLM(HFAutoModel):
    def __init__(self, config: HFAutoModelForMaskedLMConfig) -> None:
        """Initialize the AutoModel class."""
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        model_kwargs = {}
        if config.cache_dir:
            model_kwargs["cache_dir"] = config.cache_dir

            # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            (
                config.pretrained_model_name_or_path
                if config.tokenizer_name_or_path is None
                else config.tokenizer_name_or_path
            ),
            trust_remote_code=True,
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
        if config.eval_mode:
            model.eval()

        # Load the model onto the device
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        model.to(device)

        # Set persistent attributes
        self.config = config
        self.model = model
        self._tokenizer = tokenizer


class HFAutoModelForCausalLMConfig(HFAutoModelConfig):
    """Config for the transformers BERT style AutoModel class."""

    # The name of the encoder
    name: Literal["HFAutoModelForCausalLM"] = "HFAutoModelForCausalLM"  # type: ignore[assignment] # noqa: E501


@model_registry.register(config=HFAutoModelForCausalLMConfig)
class HFAutoModelForCausalLM(HFAutoModel):
    def __init__(self, config: HFAutoModelForCausalLMConfig) -> None:
        """Initialize the AutoModel class."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs = {}
        if config.cache_dir:
            model_kwargs["cache_dir"] = config.cache_dir

            # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            (
                config.pretrained_model_name_or_path
                if config.tokenizer_name_or_path is None
                else config.tokenizer_name_or_path
            ),
            trust_remote_code=True,
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
            **model_kwargs,
        )

        # Convert the model to half precision
        if config.half_precision:
            model.half()

        # Set the model to evaluation mode
        if config.eval_mode:
            model.eval()

        # Load the model onto the device
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        model.to(device)

        # Set persistent attributes
        self.config = config
        self.model = model
        self._tokenizer = tokenizer
