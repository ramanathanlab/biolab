from __future__ import annotations  # noqa D100

import torch

from biolab.api.modeling import SequenceModelOutput
from biolab.api.modeling import Transform


# TODO: this transform implies embeddings, either make this more clear
# or make it more general
class FullSequence(Transform):
    """Return desnse representation of the hidden states of a transformer model."""

    name = 'full_sequence'
    resolution: str = 'token'

    @staticmethod
    def apply(inputs: list[SequenceModelOutput], **kwargs) -> list[torch.Tensor]:
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
