from typing import Protocol, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from biolab.api.config import BaseConfig


class TorchDataloaderConfig(BaseConfig):
    """Config for the torch DataLoader class."""

    # The batch size
    batch_size: int = 1
    # Whether to shuffle the data
    shuffle: bool = False
    # The drop last parameter
    drop_last: bool = False
    # pin memory for faster GPU transfer
    pin_memory: bool = True


class TokenizerConfig(BaseConfig):
    """Config for tokenizer encode arguments."""

    # Padding strategy
    padding: str = "max_length"
    # Truncation strategy
    truncation: bool = True
    # Maximum length of the sequence
    max_length: int = 1024
    # Return type of the tokenizer
    return_tensors: str = "pt"


class LMConfig(BaseConfig):
    """Base configuration class for a language model."""

    # Tokenizer encode configuration options
    tokenizer_config: Optional[TokenizerConfig] = None
    # dataloader config for the forward passes
    dataloader_config: Optional[TorchDataloaderConfig] = None


@dataclass
class SequenceModelOutput:
    """Container for outputs of a biology sequence model."""

    sequences: Optional[list[str]] = field(
        default=None, metadata={"description": "Generated sequences."}
    )

    logits: Optional[np.ndarray] = field(
        default=None,
        metadata={
            "description": "The logits of the sequence "
            "(shape: [sequence_length, vocab_size])."
        },
    )
    embeddings: Optional[np.ndarray] = field(
        default=None,
        metadata={
            "description": "The sequence embeddings "
            "(shape: [sequence_length, embedding_size])."
        },
    )
    attention_maps: Optional[np.ndarray] = field(
        default=None,
        metadata={
            "description": "The attention maps of the sequence "
            "(shape: [num_heads, sequence_length, sequence_length])."
        },
    )


class LM(Protocol):

    model_input: str
    model_resolution: str

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer of the encoder."""
        ...

    @property
    def tokenizer_config(self) -> dict[str, Any]:
        """Get the tokenizer configuration"""
        ...

    @property
    def dataloader_config(self) -> dict[str, Any]:
        """Get the dataloader configuration"""
        ...

    @property
    def device(self):
        """Accelerator object model is running on"""
        ...

    @property
    def dtype(self):
        """Data type of model"""
        ...

    def generate_embeddings(self, input: list[str]) -> list[SequenceModelOutput]:
        """Embed a batch of sequences"""
        ...

    def generate_sequences(self, input: list[str]) -> list[SequenceModelOutput]:
        """Generate sequences from one or more input prompts"""
        ...


# TODO: does this belong here? or in a separate module?
# TODO currently we rely on a `name` attribute in task config,
# figure out how to enforce this from the protocol
# TODO: the 'super resolution' pooler requires a tokenizer, how do I couple this?
class Transform(ABC):
    """Base class for a transformation."""

    @abstractmethod
    def apply(self, input: list[SequenceModelOutput]) -> list[Any]:
        """Transform outputs from a sequence model."""
        ...
