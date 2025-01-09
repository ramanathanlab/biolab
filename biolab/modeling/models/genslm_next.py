"""Implementation of first (production) generation contrastive models."""

from __future__ import annotations

import os
from typing import Any
from typing import Literal

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import PreTrainedTokenizer

from biolab.api.logging import logger
from biolab.api.modeling import HDF5CachedList
from biolab.api.modeling import LM
from biolab.api.modeling import LMConfig
from biolab.api.modeling import SequenceModelOutput


class GenSLMESMCConfig(LMConfig):
    """Config for GenSLMESMC contrastive (joint AA-Codon) model."""

    # The name of the encoder
    name: Literal['GenSLM-ESMC'] = 'GenSLM-ESMC'  # type: ignore[assignment]
    # Pretrained model name or path
    pretrained_model_name_or_path: str
    # Eval dna or amino acids
    evaluate_dna: bool = True
    # path to HF cache if download needed
    cache_dir: str | None = None


class GenSLMESMC(LM):
    """Wrapper class for GenSLM-ESMC joint AA-Codon models."""

    model_input: str = 'dna'
    model_encoding: str = '3mer'

    def __init__(self, config: GenSLMESMCConfig):
        from .utils.modeling_esmc import EsmCForContrastiveMaskedLM

        # Set the cache directory as its not exposed by the esm api
        os.environ['HF_HOME'] = config.cache_dir

        # Set token model input and encoding:
        if config.evaluate_dna:
            self.model_input = 'dna'
            self.model_encoding = '3mer'
        else:  # Evaluating amino acids
            self.model_input = 'aminoacid'
            self.model_encoding = 'char'

        # Load model (this also loads the tokenizer found in the checkpoint)
        model = EsmCForContrastiveMaskedLM.from_pretrained(
            config.pretrained_model_name_or_path
        )
        model.transformer.to(torch.bfloat16)  # Convert to bfloat16

        # Set the model to evaluation mode
        model.eval()

        self.config = config
        self.model = model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer of the encoder."""
        return self.model.transformer.tokenizer

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
        return self.model.transformer.device

    def generate_embeddings(
        self, sequences: list[str], model_outputs: HDF5CachedList | None = None
    ) -> list[SequenceModelOutput]:
        """Generate embeddings and logits for sequence input."""
        from genslm_esm.dataset import FastaAminoAcidDataset
        from genslm_esm.dataset import FastaDataset
        from genslm_esm.dataset import GenSLMColatorForLanguageModeling

        if self.model_input == 'aminoacid':
            return_codon = False
            return_aminoacid = True
            dataset = FastaAminoAcidDataset(
                sequences=[' '.join(seq) for seq in sequences]
            )

        elif self.model_input == 'dna':
            return_codon = True
            return_aminoacid = False
            # TODO: Unsure if these need to be split into 3-mers
            dataset = FastaDataset(
                sequences=[
                    ' '.join(seq[i : i + 3] for i in range(0, len(seq), 3))
                    for seq in sequences
                ],
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

        if model_outputs is None:
            model_outputs: list[SequenceModelOutput] = []
        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[logger]):
                for batch in tqdm(dataloader, desc='Generating embeddings'):
                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    outputs = self.model(**batch, output_hidden_states=True)

                    # Get the sequence lengths - 1 for the BOS token
                    seq_lengths = batch['attention_mask'].sum(axis=1) - 1

                    # Get the last hidden state
                    last_hidden_state = outputs.hidden_states[-1]

                    # Move the outputs to the CPU
                    logits = outputs.logits.cpu().detach().to(torch.float16).numpy()
                    embedding = (
                        last_hidden_state.cpu().detach().to(torch.float16).numpy()
                    )
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
        """Generate sequences from one or more input prompts."""
        raise NotImplementedError


class GenSLMESMConfig(LMConfig):
    """Config for GenSLMESM contrastive (joint AA-Codon) model."""

    # The name of the encoder
    name: Literal['GenSLM-ESM'] = 'GenSLM-ESM'  # type: ignore[assignment]
    # Eval dna or amino acids
    evaluate_dna: bool = True
    # Pretrained model name or path (HF format)
    pretrained_model_name_or_path: str
    # Tokenizer checkpoint path
    tokenizer_path: str
    # Use the model in half precision
    half_precision: bool = False


class GenSLMESM(LM):
    """Wrapper class for GenSLM-ESM joint AA-Codon models."""

    model_input: str = 'dna'
    model_encoding: str = '3mer'

    def __init__(self, config: GenSLMESMConfig):
        from genslm_esm.modeling_esm_v3 import EsmForContrastiveMaskedLM
        from transformers import EsmTokenizer

        # Set token model input and encoding:
        if config.evaluate_dna:
            self.model_input = 'dna'
            self.model_encoding = '3mer'
        else:  # Evaluating amino acids
            self.model_input = 'aminoacid'
            self.model_encoding = 'char'

        # Load model
        model = EsmForContrastiveMaskedLM.from_pretrained(
            config.pretrained_model_name_or_path
        )
        # Load the tokenizer
        tokenizer = EsmTokenizer.from_pretrained(config.tokenizer_path)

        # Convert the model to half precision
        if config.half_precision:
            model.half()

        # Set the model to evaluation mode
        model.eval()

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
        """Get the tokenizer of the encoder."""
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

    def generate_embeddings(
        self, sequences: list[str], model_outputs: HDF5CachedList | None = None
    ) -> list[SequenceModelOutput]:
        """Generate embeddings and logits for sequence input."""
        from genslm_esm.dataset import FastaAminoAcidDataset
        from genslm_esm.dataset import FastaDataset
        from genslm_esm.dataset import GenSLMColatorForLanguageModeling

        if self.model_input == 'aminoacid':
            return_codon = False
            return_aminoacid = True
            dataset = FastaAminoAcidDataset(sequences=sequences)

        elif self.model_input == 'dna':
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

        if model_outputs is None:
            model_outputs: list[SequenceModelOutput] = []
        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[logger]):
                for batch in tqdm(dataloader, desc='Generating embeddings'):
                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    outputs = self.model(**batch, output_hidden_states=True)

                    # Get the sequence lengths - 1 for the BOS token
                    seq_lengths = batch['attention_mask'].sum(axis=1) - 1

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
        """Generate sequences from one or more input prompts."""
        raise NotImplementedError


genslmnext_models = {
    GenSLMESMCConfig: GenSLMESMC,
    GenSLMESMConfig: GenSLMESM,
}
