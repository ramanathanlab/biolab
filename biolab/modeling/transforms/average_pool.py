from __future__ import annotations

import numpy as np

from biolab.api.modeling import Transform, SequenceModelOutput
from biolab import transform_registry


# TODO: this transform implies embeddings, either make this more clear
# or make it more general
# TODO: figure out how to avoid duplicating "name" with the name field of the transform
@transform_registry.register(name="average_pool")
class AveragePool(Transform):
    """Average pool the hidden states of a transformer model."""

    name: str = "average_pool"
    resolution: str = "sequence"

    # TODO: could also be np?
    @staticmethod
    def apply(inputs: list[SequenceModelOutput], **kwargs) -> list[np.ndarray]:
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
        return [model_out.embeddings.mean(axis=0) for model_out in inputs]
