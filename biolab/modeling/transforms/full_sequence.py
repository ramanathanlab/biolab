"""Full sequence transform for transformer models."""

from __future__ import annotations

from typing import Any

from biolab.api.modeling import SequenceModelOutput
from biolab.api.modeling import Transform


class FullSequence(Transform):
    """Return dense representation of the hidden states of a transformer model."""

    name = 'full_sequence'
    resolution: str = 'token'

    @staticmethod
    def apply(inputs: list[SequenceModelOutput], **kwargs) -> list[SequenceModelOutput]:
        """Return the dense embeddings for the full sequence.

        Parameters
        ----------
        input : torch.Tensor
            The hidden states to pool (B, SeqLen, HiddenDim).

        Returns
        -------
        list[torch.Tensor]
            The pooled embeddings (B, SeqLen, HiddenDim).
        """
        return inputs

    @staticmethod
    def apply_h5(model_output: SequenceModelOutput, **kwargs) -> SequenceModelOutput:
        """Return the dense embeddings for the full sequence.

        Parameters
        ----------
        input : SequenceModelOutput
            The hidden states to pool (B, SeqLen, HiddenDim).

        Returns
        -------
        SequenceModelOutput
            The pooled embeddings (B, SeqLen, HiddenDim).
        """
        return model_output

    @staticmethod
    def apply_hf(examples: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Return the dense embeddings for the full sequence.

        This is for use with datasets.Dataset.map().

        Parameters
        ----------
        examples : dict[str, Any]
            Dict of model outputs, generally working with 'embedding'.

        Returns
        -------
        dict[str, Any]
            Dict of model outputs with no transformation applied to the embeddings.
        """
        return examples
