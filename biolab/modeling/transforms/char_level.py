from __future__ import annotations
from itertools import repeat

from transformers import PreTrainedTokenizerFast
import numpy as np

from biolab.api.modeling import Transform, SequenceModelOutput
from biolab import transform_registry


# TODO: this transform implies embeddings, either make this more clear
# or make it more general
# TODO: figure out how to avoid duplicating "name" with the name field of the transform
@transform_registry.register(name="char_level")
class CharLevel(Transform):
    """Average pool the hidden states of a transformer model."""

    name: str = "char_level"
    resolution: str = "char"

    @staticmethod
    def apply(inputs: list[SequenceModelOutput], **kwargs) -> list[np.ndarray]:
        """Average pool the hidden states using the attention mask.

        Parameters
        ----------
        inputs : list[SequenceModelOutput]
            Modeloutput to pool.
        sequences: str
            list of sequences, sent in as kwargs. Used to get the single char level representation out of a
            coarser input representation.
        tokenizer : PreTrainedTokenizerFast
            Tokenizer sent in as a kwarg. Neccesary to get the single char level representation out of a

        Returns
        -------
        torch.Tensor
            The pooled embeddings (B, HiddenDim).
        """

        sequences: list[str] = kwargs.get("sequences")
        tokenizer: PreTrainedTokenizerFast = kwargs.get("tokenizer")

        assert sequences is not None, "Sequences must be provided as a kwarg"
        assert tokenizer is not None, "Tokenizer must be provided as a kwarg"

        tokenized_seqs = [tokenizer.tokenize(seq) for seq in sequences]
        char_level_representations = []
        for model_input, tokenized_seq in zip(inputs, tokenized_seqs):
            # Iterate over each token and take convex combination of window around the token
            super_res_emb = CharLevel.super_resolution(
                model_input.embeddings, tokenized_seq
            )
            char_level_representations.append(super_res_emb)

        return char_level_representations

    @staticmethod
    def super_resolution(embeddings, tokens, window_size=None):
        # Determine location of each token in the sequence
        char_locations = []
        for i, token in enumerate(tokens):
            char_locations.extend(list(repeat(i, len(token))))

        # Determine the maximum token length if window_size is not provided
        # window size is the number of tokens to consider on either side of the current token
        if window_size is None:
            window_size = max(len(token) for token in tokens) + 1

        num_tokens, hidden_size = embeddings.shape
        seq_length = len("".join(tokens))

        # Initialize the output tensor
        super_res_embeddings = np.zeros((seq_length, hidden_size))

        total_window_size = window_size * 2 + 1
        for char_loc in range(seq_length):
            # Initialize the window embedding
            window_embedding = np.zeros((total_window_size, hidden_size))

            # Fill the window embedding with the embeddings of the tokens in the window
            for idx in range(total_window_size):
                # Determine the location of the residue in the sequence
                residue_location = char_loc - window_size + idx
                if residue_location < 0 or residue_location >= seq_length:
                    continue
                # Determine the location of the residue in the embeddigns
                emb_idx = char_locations[residue_location]
                window_embedding[idx] = embeddings[emb_idx, :]

            # Mean pool the windowed embedding for the char level representation
            super_res_embeddings[char_loc, :] = window_embedding.mean(axis=0)

        return super_res_embeddings
