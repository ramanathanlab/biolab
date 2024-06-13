import torch

from biolab.api.lm import Transform
from biolab import transform_registry


@transform_registry.register(name="full_sequence")
class FullSequence(Transform):
    """Return desnse representation of the hidden states of a transformer model."""

    name = "full_sequence"

    def apply(
        self,
        input: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the dense embeddings for the full sequence.

        Parameters
        ----------
        input : torch.Tensor
            The hidden states to pool (B, SeqLen, HiddenDim).
        attention_mask : torch.Tensor
            The attention mask for the hidden states (B, SeqLen).

        Returns
        -------
        torch.Tensor
            The pooled embeddings (B, SeqLen, HiddenDim).
        """
        return input
