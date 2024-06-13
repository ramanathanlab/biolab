from __future__ import annotations

import torch

from biolab.api.lm import Transform
from biolab import transform_registry


# TODO: figure out how to avoid duplicating "name" with the name field of the transform
@transform_registry.register(name="average_pool")
class AveragePool(Transform):
    """Average pool the hidden states of a transformer model."""

    name = "average_pool"

    @staticmethod
    def apply(
        input: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
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
        # Get the sequence lengths
        seq_lengths = attention_mask.sum(axis=1)

        # Set the attention mask to 0 for start and end tokens
        attention_mask[:, 0] = 0
        attention_mask[:, seq_lengths - 1] = 0

        # Create a mask for the pooling operation (B, SeqLen, HiddenDim)
        pool_mask = attention_mask.unsqueeze(-1).expand(input.shape)

        # Sum the embeddings over the sequence length (use the mask to avoid
        # pad, start, and stop tokens)
        sum_embeds = torch.sum(input * pool_mask, 1)

        # Avoid division by zero for zero length sequences by clamping
        sum_mask = torch.clamp(pool_mask.sum(1), min=1e-9)

        # Compute mean pooled embeddings for each sequence
        return sum_embeds / sum_mask
