from __future__ import annotations

import numpy as np

from biolab.api.modeling import Transform, SequenceModelOutput


# TODO: this transform implies embeddings, either make this more clear
# or make it more general
class AveragePool(Transform):
    """Average pool the hidden states of a transformer model."""

    name: str = "average_pool"
    resolution: str = "sequence"

    @staticmethod
    def apply(inputs: list[SequenceModelOutput], **kwargs) -> list[SequenceModelOutput]:
        """Average pool the hidden states using the attention mask.

        Parameters
        ----------
        input : torch.Tensor
            The hidden states to pool (B, SeqLen, HiddenDim).
        attention_mask : torch.Tensor
            The attention mask for the hidden states (B, SeqLen).

        Returns
        -------
        torch.Tensor
            The pooled embeddings (B, HiddenDim).
        """
        for model_out in inputs:
            model_out.embedding = model_out.embedding.mean(axis=1)

        return inputs
