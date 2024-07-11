from typing import Literal, Optional, Any

from biolab.api.modeling import LM, LMConfig, SequenceModelOutput
from biolab import model_registry
from biolab.api.logging import logger

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


class DNABERT2Config(LMConfig):
    """Config for DNABERT."""

    name: Literal["DNABERT2"] = "DNABERT2"
    # Model id or path to load the model
    pretrained_model_name_or_path: str = "zhihan1996/DNABERT-2-117M"
    # path to HF cache if download needed
    cache_dir: Optional[str] = None
    # Set the model to evaluation mode
    eval_mode: bool = True


@model_registry.register(config=DNABERT2Config)
class DNABERT2(LM):

    model_input: str = "dna"
    model_encoding: str = "bpe"

    def __init__(self, config: DNABERT2Config) -> None:
        """Initialize the DNABERT."""
        # The version of triton used by the original authors no longer works. Default
        # to the transformers library attention for this specifc model only
        # TODO: test if this bungles loading other models in the same session
        import sys

        triton_module = sys.modules.get("triton")
        sys.modules["triton"] = None
        from transformers.models.bert.configuration_bert import BertConfig
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        sys.modules["triton"] = triton_module

        model_kwargs = {}
        if config.cache_dir:
            model_kwargs["cache_dir"] = config.cache_dir

            # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

        # Load model
        model_cfg = BertConfig.from_pretrained(config.pretrained_model_name_or_path)
        model = AutoModelForMaskedLM.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
            config=model_cfg,
            **model_kwargs,
        )

        # Set the model to evaluation mode
        if config.eval_mode:
            model.eval()

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
        return (
            self.config.tokenizer_config.model_dump()
            if self.config.tokenizer_config
            else {}
        )

    @property
    def dataloader_config(self) -> dict[str, Any]:
        return (
            self.config.dataloader_config.model_dump()
            if self.config.dataloader_config
            else {}
        )

    def generate_embeddings(self, sequences: list[str]) -> SequenceModelOutput:
        """Generate embeddings and logits for sequence input."""

        # Tokenize the dataset
        # TODO: remove column specifier, is this a property of the LM?
        def tokenize_input(examples):
            return self.tokenizer(examples["sequences"], **self.tokenizer_config)

        modeling_input = {"sequences": sequences}
        modeling_dataset = Dataset.from_dict(modeling_input)
        modeling_dataset = modeling_dataset.map(
            tokenize_input,
            batched=True,
            remove_columns=["sequences"],
        ).with_format("torch")

        # turn into dataloader and grab dset info
        dataloader = DataLoader(modeling_dataset, **self.dataloader_config)

        # Generate embeddings
        model_outputs: list[SequenceModelOutput] = []
        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[logger]):
                for batch in tqdm(dataloader, desc="Generating embeddings"):
                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    outputs = self.model(**batch, output_hidden_states=True)

                    # Get the sequence lengths (no bos/eos in DNABERT model)
                    seq_lengths = batch["attention_mask"].sum(axis=1)

                    # Get the last hidden state (the only hidden state for this model)
                    last_hidden_state = outputs.hidden_states

                    # Move the outputs to the CPU
                    logits = outputs.logits.cpu().detach().numpy()
                    embedding = last_hidden_state.cpu().detach().numpy()

                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        # Remove the the padding
                        logit = logits[i, :seq_len, :]
                        trimmed_embedding = embedding[i, :seq_len, :]

                        # Create the output object
                        output = SequenceModelOutput(
                            logits=logit, embedding=trimmed_embedding
                        )
                        model_outputs.append(output)

        return model_outputs

    def generate_sequences(self, input: list[str]) -> list[SequenceModelOutput]:
        """Generate sequences from one or more input prompts"""
        raise NotImplementedError
