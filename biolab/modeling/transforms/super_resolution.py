from __future__ import annotations  # noqa D100
from typing import Any

from itertools import repeat

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from biolab.api.logging import logger
from biolab.api.modeling import SequenceModelOutput
from biolab.api.modeling import Transform


# TODO: this transform implies embeddings, either make this more clear
# or make it more general
class SuperResolution(Transform):
    """Up-sample the hidden states of a transformer model.

    This is to recover the original sequence lenth (in chars) of an input sequence
    regardless of the tokenization scheme of the model.
    """

    name: str = 'super_resolution'
    resolution: str = 'char'

    @staticmethod
    def apply(inputs: list[SequenceModelOutput], **kwargs) -> list[np.ndarray]:
        """Run the 'super resolution transformation on a set of embeddings.

        Parameters
        ----------
        inputs : list[SequenceModelOutput]
            Modeloutput to pool.
        sequences: str
            list of sequences, sent in as kwargs. Used to get the single char level
            representation out of a coarser input representation.
        tokenizer : PreTrainedTokenizerFast
            Tokenizer sent in as a kwarg. Neccesary to get the figure out how many chars
            are inside of a token.

        Returns
        -------
        torch.Tensor
            The pooled embeddings (B, HiddenDim).
        """
        sequences: list[str] = kwargs.get('sequences')
        tokenizer: PreTrainedTokenizerFast = kwargs.get('tokenizer')

        assert sequences is not None, 'Sequences must be provided as a kwarg'
        assert tokenizer is not None, 'Tokenizer must be provided as a kwarg'

        tokenized_seqs = [tokenizer.tokenize(seq) for seq in sequences]
        for model_input, tokenized_seq in tqdm(
            zip(inputs, tokenized_seqs, strict=False), desc='Transform'
        ):
            # Iterate over each token and take convex combination of window
            # around the token.
            super_res_emb = SuperResolution.super_resolution(
                model_input['embedding'], tokenized_seq
            )
            model_input['embedding'] = super_res_emb

        return inputs

    @staticmethod
    def apply_hf(examples: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Average pool the hidden states using the attention mask.

        Parameters
        ----------
        inputs : list[SequenceModelOutput]
            Modeloutput to pool.
        sequences: str
            list of sequences, sent in as kwargs. Used to get the single char level
            representation out of a coarser input representation.
        tokenizer : PreTrainedTokenizerFast
            Tokenizer sent in as a kwarg. Neccesary to get the figure out how many chars
            are inside of a token.

        Returns
        -------
        torch.Tensor
            The pooled embeddings (B, HiddenDim).
        """
        sequences: list[str] = kwargs.get('sequences')
        tokenizer: PreTrainedTokenizerFast = kwargs.get('tokenizer')

        assert sequences is not None, 'Sequences must be provided as a kwarg'
        assert tokenizer is not None, 'Tokenizer must be provided as a kwarg'

        tokenized_seqs = [tokenizer.tokenize(seq) for seq in sequences]
        for i, tokenized_seq in enumerate(tokenized_seqs):
            # Iterate over each token and take convex combination of window
            # around the token.
            super_res_emb = SuperResolution.super_resolution(
                examples['embedding'][i], tokenized_seq
            )
            examples['embedding'][i] = super_res_emb

        return examples

    @staticmethod
    def super_resolution(embedding, tokens, window_size=None):
        """On a single embedding expand to char level embedding."""
        # Determine location of each token in the sequence
        char_locations = []
        for i, token in enumerate(tokens):
            char_locations.extend(list(repeat(i, len(token))))

        # Determine the maximum token length if window_size is not provided
        # window size is the number of tokens to consider on either side of the center
        # TODO: see if this can be shorter (//2? ... might provide enough coverage)
        if window_size is None:
            window_size = max(len(token) for token in tokens) // 2 + 1

        _, hidden_size = embedding.shape
        seq_length = len(''.join(tokens))

        # Initialize the output tensor
        super_res_embedding = np.zeros((seq_length, hidden_size))

        total_window_size = window_size * 2 + 1
        for char_loc in range(seq_length):
            # Initialize the window embedding
            window_embedding = np.zeros((total_window_size, hidden_size))

            # Fill the window embedding with the embedding of the tokens in the window
            for idx in range(total_window_size):
                # Determine the location of the residue in the sequence
                residue_location = char_loc - window_size + idx
                if residue_location < 0 or residue_location >= seq_length:
                    continue
                # Determine the location of the residue in the embeddings
                emb_idx = char_locations[residue_location]
                # TODO: figure out if I can silence this if it happens once, becuase if it happens once
                # It will raise for every position afterwords (potentially hundreds...)
                if emb_idx > embedding.shape[0] - 1:
                    logger.warning(
                        'Embedding shorter than tokenized sequence, '
                        f'skipping char locations {residue_location}-{seq_length}'
                    )
                    break
                window_embedding[idx] = embedding[emb_idx, :]

            # Mean pool the windowed embedding for the char level representation
            super_res_embedding[char_loc, :] = window_embedding.mean(axis=0)

        return super_res_embedding
