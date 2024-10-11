from __future__ import annotations  # noqa D100

from typing import Any


from biolab.api.modeling import SequenceModelOutput
from biolab.api.modeling import Transform


# TODO: this transform implies embeddings, either make this more clear
# or make it more general
class AveragePool(Transform):
    """Average pool the hidden states of a transformer model."""

    name: str = 'average_pool'
    resolution: str = 'sequence'

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
            model_out.embedding = model_out.embedding.mean(axis=0)

        return inputs

    @staticmethod
    def apply_h5(model_output: SequenceModelOutput, **kwargs) -> SequenceModelOutput:
        """Average pool the hidden states using the attention mask.

        Parameters
        ----------
        input : SequenceModelOutput
            The hidden states to pool (B, SeqLen, HiddenDim).
        attention_mask : torch.Tensor
            The attention mask for the hidden states (B, SeqLen).

        Returns
        -------
        SequenceModelOutput
            The pooled embeddings (B, HiddenDim).
        """
        model_output.embedding = model_output.embedding.mean(axis=0)

        return model_output

    @staticmethod
    def apply_hf(examples: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Average pool the hidden states using the attention mask.

        This is for use with datasets.Dataset.map().

        Parameters
        ----------
        input : dict[str, Any]
            The hidden states to pool (B, SeqLen, HiddenDim).
            attention_mask : torch.Tensor
                The attention mask for the hidden states (B, SeqLen).

        Returns
        -------
        dict[str, Any]
            The pooled embeddings (B, HiddenDim).
        """
        examples['embedding'] = [elem.mean(axis=0) for elem in examples['embedding']]

        return examples
