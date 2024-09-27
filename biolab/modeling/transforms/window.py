from __future__ import annotations  # noqa: D100
from typing import Any

import numpy as np
from tqdm import tqdm

from biolab.api.modeling import SequenceModelOutput
from biolab.api.modeling import Transform


# TODO: this transform implies embeddings, either make this more clear
# or make it more general
class Window3(Transform):
    """Windowed embeddings to shape (num_tokens//3, hidden_dim)."""

    name: str = '3_window'
    resolution: str = '3mer'

    @staticmethod
    def apply(inputs: list[SequenceModelOutput], **kwargs) -> list[SequenceModelOutput]:
        """Windowed embeddings to shape (num_tokens//3, hidden_dim).

        Parameters
        ----------
        input : list[SequenceModelOutput]
            SequenceModelOutput objects to pool embeddings over.
        window_size : int
            The size of the window to pool over, passed in as keyword argument.

        Returns
        -------
        List[SequenceModelOutput]
            Returns the input embeddings averaged over the window size in a
            SequenceModelOutput object.
        """
        # TODO: this is now a lot of params getting passed by kwargs - think about streamlining
        window_size = kwargs.get('window_size', 3)

        for model_out in tqdm(inputs, desc='Transform'):
            # Find output length, if not divisible by window size, add one to
            # capture the remainder
            output_size = model_out.embedding.shape[0] // window_size
            if model_out.embedding.shape[0] % window_size != 0:
                output_size += 1
            windowed_emb = np.zeros((output_size, model_out['embedding'].shape[1]))
            # Average over the window size
            for window_i, token_i in enumerate(
                range(0, model_out['embedding'].shape[0], window_size)
            ):
                windowed_emb[window_i] = model_out['embedding'][
                    token_i : token_i + window_size
                ].mean(axis=0)
            # Update the embedding
            model_out['embedding'] = windowed_emb

        return inputs

    @staticmethod
    def apply_hf(examples: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Window embeddings to create an output with shape (num_tokens//3, hidden_dim).

        This is for use with datasets.Dataset.map().

        Parameters
        ----------
        input : dict[str, Any]
            Dict of model outputs, generally working with 'embedding'.
        window_size : int
            The size of the window to pool over, passed in as keyword argument.

        Returns
        -------
        dict[str, Any]
            Returns the input embeddings averaged over the window_size in a dict.
        """
        # TODO: lots of params getting passed by kwargs - think about streamlining
        window_size = kwargs.get('window_size', 3)

        for i in range(len(examples['embedding'])):
            embedding = examples['embedding'][i]
            # Find output length, if not divisible by window size, add one for remainder
            output_size = embedding.shape[0] // window_size
            if embedding.shape[0] % window_size != 0:
                output_size += 1
            windowed_emb = np.zeros((output_size, embedding.shape[1]))
            # Average over the window size
            for window_i, token_i in enumerate(
                range(0, embedding.shape[0], window_size)
            ):
                windowed_emb[window_i] = embedding[
                    token_i : token_i + window_size
                ].mean(axis=0)
            # Update the embedding
            examples['embedding'][i] = windowed_emb

        return examples
