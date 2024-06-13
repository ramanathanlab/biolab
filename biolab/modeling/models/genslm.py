from typing import Literal, Any

from biolab.api.lm import LM, LMConfig
from biolab import model_registry

import torch
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput


class GenSLMConfig(LMConfig):
    """Config for GenSLM."""

    # The name of the encoder
    name: Literal["GenSLM"] = "GenSLM"  # type: ignore[assignment]
    # Original HF config json path
    architecture_json: str
    # Tokenizer json path
    tokenizer_json: str
    # Path to the model weights
    weight_path: str
    # Use the model in half precision
    half_precision: bool = False
    # Set the model to evaluation mode
    eval_mode: bool = True


@model_registry.register(config=GenSLMConfig)
class GenSLM(LM):
    """Wrapper class for original GenSLM model."""

    def __init__(self, config: GenSLMConfig):
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            PreTrainedTokenizerFast,
        )
        from tokenizers import Tokenizer

        # Initialize the tokenizer
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_file(config.tokenizer_json)
        )
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Setup + load the model
        base_config = AutoConfig.from_pretrained(config.architecture_json)
        model = AutoModelForCausalLM.from_config(base_config)

        ptl_checkpoint = torch.load(config.weight_path, map_location="cpu")
        model.load_state_dict(ptl_checkpoint["state_dict"], strict=False)

        if config.half_precision:
            model = model.half()

        if config.eval_mode:
            model = model.eval()

        # Load the model onto the device
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        model.to(device)

        self.config = config
        self.model = model
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    @property
    def tokenizer_config(self) -> dict[str, Any]:
        """Get the tokenizer configuration"""
        return (
            self.config.tokenizer_config.model_dump()
            if self.config.tokenizer_config
            else {}
        )

    @property
    def dataloader_config(self) -> dict[str, Any]:
        """Get the dataloader configuration"""
        return (
            self.config.dataloader_config.model_dump()
            if self.config.dataloader_config
            else {}
        )

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the encoder."""
        return self.model.dtype

    @property
    def device(self) -> torch.device:
        """Get the device of the encoder."""
        return self.model.device

    @property
    def embedding_size(self) -> int:
        """Get the embedding size of the encoder."""
        return self.model.config.hidden_size

    def embed(self, batch_encoding: BatchEncoding) -> BaseModelOutput:
        """Embed the sequence.

        Parameters
        ----------
        batch_encoding : BatchEncoding
            The batch encoding of the sequence (containing the input_ids,
            attention_mask, and token_type_ids).

        Returns
        -------
        BaseModelOutput
            The embeddings of the sequence extracted from the last hidden state
            (shape: [num_sequences, sequence_length, embedding_size])
        """
        # The GPTNeoX model does not have all the fields of BatchEncoding, 
        # so we just pass the input_ids and attention_mask
        return self.model(
            input_ids=batch_encoding["input_ids"],
            labels=batch_encoding["input_ids"],
            attention_mask=batch_encoding["attention_mask"],
            output_hidden_states=True,
        )

    # TODO: figure out generation for GenSLM
    def generate(self, batch_encoding: BatchEncoding) -> BaseModelOutput:
        """Generate the sequence.

        Parameters
        ----------
        batch_encoding : BatchEncoding
            The batch encoding of the sequence (containing the input_ids,
            attention_mask, and token_type_ids).

        Returns
        -------
        BaseModelOutput
            Container for model outputs, includes logits
                logits shape: [num_sequences, sequence_length, vocab_size]
        """
        ...
