from __future__ import annotations  # noqa: D100

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Protocol

import datasets
import numpy as np

from biolab.api.config import BaseConfig

# TODO: prompt config? (things like temp, top_k, etc)


class TorchDataloaderConfig(BaseConfig):
    """Config for the torch DataLoader class."""

    # Batch size
    batch_size: int = 1
    # Whether to shuffle the data
    shuffle: bool = False
    # If dataset size // batch_size is not an integer, drop the last incomplete batch
    drop_last: bool = False
    # Pin memory for faster GPU transfer
    pin_memory: bool = True


class TokenizerConfig(BaseConfig):
    """Config for tokenizer encode arguments."""

    # Padding strategy
    padding: str | bool = 'max_length'
    # Truncation strategy
    truncation: str | bool = True
    # Maximum length of the sequence
    max_length: int = 1024
    # Return type of the tokenizer
    return_tensors: str = 'pt'


class LMConfig(BaseConfig):
    """Base configuration class for a language model."""

    # Tokenizer encode configuration options
    tokenizer_config: TokenizerConfig = field(default_factory=TokenizerConfig)
    # dataloader config for the forward passes
    dataloader_config: TorchDataloaderConfig = field(
        default_factory=TorchDataloaderConfig
    )


@dataclass
class SequenceModelOutput:
    """Container for outputs of a biology sequence model.

    This behaves minimally like a dictionary so that it can be used in a dataset.
    """

    # TODO: might need to store a prompt too?
    sequence: str | None = field(
        default=None, metadata={'description': 'Generated sequence.'}
    )

    logits: np.ndarray | None = field(
        default=None,
        metadata={
            'description': 'The logits of the sequence '
            '(shape: [sequence_length, vocab_size]).'
        },
    )
    embedding: np.ndarray | None = field(
        default=None,
        metadata={
            'description': 'The sequence embeddings '
            '(shape: [sequence_length, embedding_size]).'
        },
    )
    attention_maps: np.ndarray | None = field(
        default=None,
        metadata={
            'description': 'The attention maps of the sequence '
            '(shape: [num_heads, sequence_length, sequence_length]).'
        },
    )

    def __getitem__(self, key: str) -> Any:
        """Get the value of an attribute if it's not None."""
        value = getattr(self, key)
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        """Iterate over the non-None attributes of the class."""
        return (k for k, v in self.__dict__.items() if v is not None)

    def __len__(self) -> int:
        """Get the number of non-None attributes in the class."""
        return sum(1 for v in self.__dict__.values() if v is not None)

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value of an attribute with a default value."""
        value = getattr(self, key, default)
        return value if value is not None else default


class LM(Protocol):
    """Interface for a general protein language model."""

    model_input: str
    model_encoding: str

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer of the encoder."""
        ...

    @property
    def tokenizer_config(self) -> dict[str, Any]:
        """Get the tokenizer configuration."""
        ...

    @property
    def dataloader_config(self) -> dict[str, Any]:
        """Get the dataloader configuration."""
        ...

    @property
    def device(self):
        """Accelerator object model is running on."""
        ...

    def generate_embeddings(self, input: list[str]) -> datasets.Dataset:
        """Embed a batch of sequences."""
        ...

    def generate_sequences(self, input: list[str]) -> datasets.Dataset:
        """Generate sequences from one or more input prompts."""
        ...


class Transform(ABC):
    """Base class for a transformation."""

    @staticmethod
    @abstractmethod
    def apply(input: list[SequenceModelOutput], **kwargs) -> list[SequenceModelOutput]:
        """Transform outputs from a sequence model."""
        ...
