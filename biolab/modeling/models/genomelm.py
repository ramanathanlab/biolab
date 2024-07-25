from typing import Literal, Optional, Any
import json

from biolab.api.modeling import LM, LMConfig, SequenceModelOutput
from biolab import model_registry
from biolab.api.logging import logger

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


class GenomeLMConfig(LMConfig):

    name: Literal["GenomeLM"] = "GenomeLM"
    # Model id or path to load the model
    pretrained_model_name_or_path: str
    # Model context length
    context_length: int = 2048
    # Path to tokenizer file
    tokenizer_path: str
    # path to HF cache if download needed
    cache_dir: Optional[str] = None
    # Use the model in half precision
    half_precision: bool = False
    # Set the model to evaluation mode
    eval_mode: bool = True


@model_registry.register(config=GenomeLMConfig)
class GenomeLM(LM):

    model_input: str = "dna"
    model_encoding: str = "bpe"

    def __init__(self, config: GenomeLMConfig) -> None:
        """This is geared for the long context Llama style models."""
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.processors import TemplateProcessing

        model_kwargs = {}
        if config.cache_dir:
            model_kwargs["cache_dir"] = config.cache_dir

        # Load tokenizer
        t = Tokenizer.from_file(config.tokenizer_path)
        t.post_processor = TemplateProcessing(
            single="$A [EOS]",
            special_tokens=[("[EOS]", t.token_to_id("[EOS]"))],
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=t,
            unk_token="[UNK]",
            cls_token="[CLS]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            mask_token="[MASK]",
        )

        # Set context length if mismatched, assume globally set length is truth
        if config.tokenizer_config.max_length != config.context_length:
            config.tokenizer_config.max_length = config.context_length

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
            **model_kwargs,
        )

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

        # Set persistent attributes
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

    # TODO: might not actually need this
    @property
    def device(self) -> torch.device:
        return self.model.device

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
                    outputs = self.model(
                        batch["input_ids"].to(self.model.device),
                        batch["attention_mask"].to(self.model.device),
                        output_hidden_states=True,
                    )

                    # Get the sequence lengths (subtract eos)
                    seq_lengths = batch["attention_mask"].sum(axis=1) - 1

                    # Get the last hidden state
                    last_hidden_state = outputs.hidden_states[-1]

                    # Move the outputs to the CPU
                    logits = outputs.logits.cpu().detach().numpy()
                    embedding = last_hidden_state.cpu().detach().numpy()

                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        # Remove the EOS token (no bos token in this model)
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


class GenomeLMRawConfig(LMConfig):
    """Original genomelm config relying on the OG package"""

    name: Literal["GenomeLMRaw"] = "GenomeLMRaw"
    # Model id or path to load the model
    pt_weights: str
    # Model context length
    hparam_file: str
    # Path to tokenizer file
    tokenizer_file: str
    # Sliding window tokenization
    sliding_window: bool = False
    # Define kmer size
    kmer_size: int = 3
    # Use the model in half precision
    half_precision: bool = False
    # Set the model to evaluation mode
    eval_mode: bool = True


@model_registry.register(config=GenomeLMRawConfig)
class GenomeLMRaw(LM):
    model_input: str = "dna"
    model_encoding: str = "3mer"

    def __init__(self, config: GenomeLMRawConfig) -> None:
        import os

        # Guard for making sure the model is loaded on rank 0
        os.environ["PMI_RANK"] = "0"
        from transformers import PreTrainedTokenizerFast
        from genomelm.models.minBERT import (
            BERT,
            BERTSelfAttention,
            BERTMLP,
            MinBERTConfig,
            BERTBlock,
        )
        from genomelm.util.arguments import dotdict

        with open(config.hparam_file, "r") as f:
            hpars = dotdict(json.load(f))

        args = dotdict({})
        for k, v in hpars.items():
            args[k] = v

        args.sliding_window = config.sliding_window

        model_config = MinBERTConfig(
            block_size=(
                args["seq_length"] if not args.memory_tokens else args.memory_chunk_size
            ),  # configured in tokenizer to match GPT-3
            vocab_size=args["vocab_size"],
            n_layer=args["num_layers"],
            n_head=args["num_heads"],
            n_embd=args["embed_dim"],
            n_hidden=args["hidden_dim"],
            dropout=args["dropout"],
            bias=False,
            memory_tokens=args.memory_tokens,
            n_memory_tokens=args.n_memory_tokens,
            mask_memory=True,
            alibi=args.alibi_pos_emb,
            rotary=args.rotary_pos_emb,
        )
        model = BERT(model_config)

        checkpoint = torch.load(config.pt_weights, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint)
        logger.info("Loaded model weights")

        # Convert the model to half precision
        # if config.half_precision:
        #     model.half()

        # Set the model to evaluation mode
        if config.eval_mode:
            model.eval()

        # Load the model onto the device
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        model.to(device)

        # Load the tokenizer
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(config.tokenizer_file))
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})

        # Set persistent attributes
        self.config = config
        self.model = model
        self._tokenizer = tokenizer
        self._device = device

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

    # TODO: might not actually need this
    @property
    def device(self) -> torch.device:
        return self._device

    def generate_embeddings(self, sequences: list[str]) -> SequenceModelOutput:
        """Generate embeddings and logits for sequence input."""

        # Tokenize the dataset

        def split_by_kmer(sequence, k, window=False):
            """
            Returns a string of substrings of length k from the given sequence,
            with a window shift of 1 if specified.
            """
            sequence = sequence.upper()
            window = k if not window else 1
            return " ".join(
                sequence[i : i + k] for i in range(0, len(sequence) - k + 1, window)
            )

        def tokenize_input(examples, k=3, max_length=3072, sliding_window=False):
            seqs = [
                split_by_kmer(s, k=k, window=sliding_window)
                for s in examples["sequences"]
            ]
            return self.tokenizer(seqs, **self.tokenizer_config)

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
                    batch["label_ids"] = batch["input_ids"].clone()
                    batch["label_ids"] = batch["label_ids"].half()
                    batch["input_ids"] = batch["input_ids"].int()
                    batch["attention_mask"] = batch["attention_mask"].float()
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        labels=batch["label_ids"],
                        attention_mask=batch["attention_mask"],
                        output_hidden_states=True,
                    )

                    # Get the sequence lengths (subtract eos)
                    seq_lengths = batch["attention_mask"].int().sum(axis=1) - 1

                    # Get the last hidden state
                    last_hidden_state = outputs.hidden_states[-1]

                    # Move the outputs to the CPU
                    logits = outputs.logits.cpu().detach().numpy()
                    embedding = last_hidden_state.cpu().detach().numpy()

                    # breakpoint()
                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        # Remove the EOS/BOS token
                        logit = logits[i, :seq_len, :]
                        trimmed_embedding = embedding[i, 1:seq_len, :]

                        # Create the output object
                        output = SequenceModelOutput(
                            logits=logit, embedding=trimmed_embedding
                        )
                        model_outputs.append(output)
        # breakpoint()
        return model_outputs

    def generate_sequences(self, input: list[str]) -> list[SequenceModelOutput]:
        """Generate sequences from one or more input prompts"""
        raise NotImplementedError
