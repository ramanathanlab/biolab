import torch

from biolab.api.modeling import Transform, SequenceModelOutput
from biolab import transform_registry


# TODO: this transform implies embeddings, either make this more clear
# or make it more general
@transform_registry.register(name="full_sequence")
class FullSequence(Transform):
    """Return desnse representation of the hidden states of a transformer model."""

    name = "full_sequence"

    def apply(self, input: list[SequenceModelOutput]) -> torch.Tensor:
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
        return [elem.embeddings for elem in input]
