"""Implementation of the GenSLM model."""

from __future__ import annotations

from typing import Any
from typing import Literal

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import PreTrainedTokenizer

from biolab.api.logging import logger
from biolab.api.modeling import HDF5CachedList
from biolab.api.modeling import LM
from biolab.api.modeling import LMConfig
from biolab.api.modeling import SequenceModelOutput


class GenSLMConfig(LMConfig):
    """Config for GenSLM."""

    # The name of the encoder
    name: Literal['GenSLM'] = 'GenSLM'  # type: ignore[assignment]
    # Original HF config json path
    architecture_json: str
    # Tokenizer json path
    tokenizer_json: str
    # Path to the model weights
    weight_path: str
    # Use the model in half precision
    half_precision: bool = False


class GenSLM(LM):
    """Wrapper class for original GenSLM model."""

    model_input: str = 'dna'
    model_encoding: str = '3mer'

    def __init__(self, config: GenSLMConfig):
        from tokenizers import Tokenizer
        from transformers import AutoConfig
        from transformers import AutoModelForCausalLM
        from transformers import PreTrainedTokenizerFast

        # Initialize the tokenizer
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_file(config.tokenizer_json)
        )
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Setup + load the model
        base_config = AutoConfig.from_pretrained(config.architecture_json)
        model = AutoModelForCausalLM.from_config(base_config)

        ptl_checkpoint = torch.load(config.weight_path, map_location='cpu')
        model.load_state_dict(ptl_checkpoint['state_dict'], strict=False)

        # Convert the model to half precision
        if config.half_precision:
            model = model.half()

        # Set the model to evaluation mode
        model = model.eval()

        # Load the model onto the device
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
        model.to(device)

        self.config = config
        self.model = model
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """HF Tokenizer object."""
        return self._tokenizer

    @property
    def tokenizer_config(self) -> dict[str, Any]:
        """Get the tokenizer configuration."""
        return (
            self.config.tokenizer_config.model_dump()
            if self.config.tokenizer_config
            else {}
        )

    @property
    def dataloader_config(self) -> dict[str, Any]:
        """Get the dataloader configuration."""
        return (
            self.config.dataloader_config.model_dump()
            if self.config.dataloader_config
            else {}
        )

    @property
    def device(self) -> torch.device:
        """Get the device of the encoder."""
        return self.model.device

    def generate_model_outputs(
        self,
        sequences: list[str],
        model_outputs: HDF5CachedList | None = None,
        return_input_ids: bool = True,
        return_logits: bool = False,
        return_embeddings: bool = False,
        return_attention_maps: bool = False,
    ) -> list[SequenceModelOutput]:
        """Generate embeddings, logits, attention masks for sequence input."""

        # Tokenize the dataset
        # Needed to insert blank space every 3 tokens as required by tokenizer
        def group_by_kmer(seq: str, kmer: int = 3) -> str:
            return ' '.join(seq[i : i + kmer] for i in range(0, len(seq), kmer)).upper()

        def tokenize_input(examples):
            return self.tokenizer(examples['sequences'], **self.tokenizer_config)

        modeling_input = {'sequences': [group_by_kmer(s) for s in sequences]}
        modeling_dataset = Dataset.from_dict(modeling_input)
        modeling_dataset = modeling_dataset.map(
            tokenize_input,
            batched=True,
            remove_columns=['sequences'],
        ).with_format('torch')

        # turn into dataloader and grab dset info
        dataloader = DataLoader(modeling_dataset, **self.dataloader_config)

        # Generate embeddings
        if model_outputs is None:
            model_outputs: list[SequenceModelOutput] = []
        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[logger]):
                for batch in tqdm(dataloader, desc='Generating embeddings'):
                    outputs = self.model(
                        batch['input_ids'].to(self.device),
                        batch['attention_mask'].to(self.device),
                        output_hidden_states=True,
                    )
                    # Get the sequence lengths (no bos/eos in NT model)
                    seq_lengths = batch['attention_mask'].sum(axis=1)

                    logits = outputs.logits.cpu().detach().numpy()
                    if return_embeddings:
                        # Get the last hidden state
                        last_hidden_state = outputs.hidden_states[-1]
                        # Move the outputs to the CPU
                        embedding = last_hidden_state.cpu().detach().numpy()
                    else:
                        embedding = None

                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        seq_input_ids = None
                        seq_logits = None
                        seq_embedding = None
                        seq_attention_maps = None

                        # Remove the padding
                        if return_input_ids:
                            seq_input_ids = (
                                batch['input_ids'][i, :seq_len].cpu().detach().numpy()
                            )
                        if return_logits:
                            seq_logits = logits[i, :seq_len, :]
                        if return_embeddings:
                            seq_embedding = embedding[i, :seq_len, :]
                        if return_attention_maps:
                            # TODO: look at model implementation for attention maps
                            seq_attention_maps = None

                        output_fields = {
                            'input_ids': seq_input_ids,
                            'logits': seq_logits,
                            'embedding': seq_embedding,
                            'attention_maps': seq_attention_maps,
                        }

                        # Create the output object
                        model_outputs.append(SequenceModelOutput(**output_fields))

        return model_outputs

    def generate_sequences(self, input: list[str]) -> list[SequenceModelOutput]:
        """Generate sequences from one or more input prompts."""
        raise NotImplementedError


genslm_models = {
    GenSLMConfig: GenSLM,
}
