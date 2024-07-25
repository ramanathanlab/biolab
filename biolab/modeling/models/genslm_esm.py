from typing import Literal, Any

from biolab.api.modeling import LM, LMConfig, SequenceModelOutput
from biolab import model_registry
from biolab.api.logging import logger

import torch
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


class GenSLMESMConfig(LMConfig):
    """Config for GenSLMESM contrastive (joint AA-Codon) model."""

    # The name of the encoder
    name: Literal["GenSLM-ESM"] = "GenSLM-ESM"  # type: ignore[assignment]
    # Eval dna or amino acids
    evaluate_dna: bool = True
    # HF checkpoint path
    checkpoint_path: str
    # Tokenizer checkpoint path
    tokenizer_path: str
    # Use the model in half precision
    half_precision: bool = False
    # Set the model to evaluation mode
    eval_mode: bool = True


@model_registry.register(name="GenSLM-ESM", config=GenSLMESMConfig)
class GenSLMESM(LM):
    """Wrapper class for GenSLM-ESM joint AA-Codon models."""

    model_input: str = "dna"
    model_encoding: str = "3mer"

    def __init__(self, config: GenSLMESMConfig):
        from transformers import EsmTokenizer
        from genslm_esm.modeling_esm_v3 import EsmForContrastiveMaskedLM

        # Set token model input and encoding:
        if config.evaluate_dna:
            self.model_input = "dna"
            self.model_encoding = "3mer"
        else:  # Evaluating amino acids
            self.model_input = "aminoacid"
            self.model_encoding = "char"

        # Load model
        model = EsmForContrastiveMaskedLM.from_pretrained(config.checkpoint_path)
        # Load the tokenizer
        tokenizer = EsmTokenizer.from_pretrained(config.tokenizer_path)

        # Convert the model to half precision
        if config.half_precision:
            model.half()

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
    # TODO: could potentially need a full class to deal with templating
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer of the encoder."""
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
    def device(self) -> torch.device:
        """Get the device of the encoder."""
        return self.model.device

    def generate_embeddings(self, sequences: list[str]) -> list[SequenceModelOutput]:
        """Generate embeddings and logits for sequence input."""
        from genslm_esm.dataset import (
            FastaAminoAcidDataset,
            FastaDataset,
            GenSLMColatorForLanguageModeling,
        )

        if self.model_input == "aminoacid":
            return_codon = False
            return_aminoacid = True
            dataset = FastaAminoAcidDataset(sequences=sequences)

        elif self.model_input == "dna":
            return_codon = True
            return_aminoacid = False

            dataset = FastaDataset(
                sequences=sequences,
                return_codon=return_codon,
                return_aminoacid=return_aminoacid,
            )

        data_collator = GenSLMColatorForLanguageModeling(
            return_codon=return_codon,
            return_aminoacid=return_aminoacid,
            tokenizer=self.tokenizer,
            train_mode=False,
        )

        dataloader = DataLoader(
            dataset, collate_fn=data_collator, **self.dataloader_config
        )

        # Tell the model whether it should process codons or amino acids
        self.model.config.compute_aminoacid_loss = return_aminoacid
        self.model.config.compute_codon_loss = return_codon

        model_outputs: list[SequenceModelOutput] = []
        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[logger]):
                for batch in tqdm(dataloader, desc="Generating embeddings"):
                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    outputs = self.model(**batch, output_hidden_states=True)

                    # Get the sequence lengths - 1 for the BOS token
                    seq_lengths = batch["attention_mask"].sum(axis=1) - 1

                    # Get the last hidden state
                    last_hidden_state = outputs.hidden_states[-1]

                    # Move the outputs to the CPU
                    logits = outputs.logits.cpu().detach().numpy()
                    embedding = last_hidden_state.cpu().detach().numpy()

                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        # Remove the BOS/EOS token and the padding
                        logit = logits[i, 1:seq_len, :]
                        trimmed_embedding = embedding[i, 1:seq_len, :]

                        # Create the output object
                        output = SequenceModelOutput(
                            logits=logit, embedding=trimmed_embedding
                        )
                        model_outputs.append(output)

        return model_outputs

    def generate_sequences(self, input: list[str]) -> list[SequenceModelOutput]:
        """Generate sequences from one or more input prompts"""
        raise NotImplementedError
