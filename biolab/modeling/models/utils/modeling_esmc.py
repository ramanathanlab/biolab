"""Adapted PyTorch ESM model."""
# braceal 08/31/23: modified the original ESM Huggingface model to add a contrastive loss head.
# braceal 01/20/24: modified the implementation to separate lm_head's for codons and amino acids.
from mpi4py import MPI
from typing import Optional, Tuple, Union
from copy import deepcopy
import torch
import os
import json
import safetensors
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import (
                    logging, 
                    EsmTokenizer, 
                    PreTrainedModel, 
                    PretrainedConfig
)
from transformers.models.esm.configuration_esm import EsmConfig
from transformers.models.esm.modeling_esm import (
    EsmPooler,
    MaskedLMOutput,
)
from esm.models.esmc import ESMC
from esm.pretrained import register_local_model, ESMC_300M_202412, ESMC_600M_202412
from esm.layers.regression_head import RegressionHead

from genslm_esm.dataset import translation_table

logger = logging.get_logger(__name__)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# TODO: Only used for a type hint
@dataclass
class ContrastiveEsmConfig(PretrainedConfig):
    """Add contrastive loss parameters to the ESM config."""
    # ensure any string is utf-8
    def __init__(self, 
                    model_name: str  = "esmc_300m",
                    base_model_path: str = None,
                    d_model: int = 960,
                    n_heads: int = 15,
                    n_layers: int = 30,
                    tokenizer_name_or_path: str = None,
                    compute_aminoacid_loss: bool = True,
                    compute_codon_loss: bool = False,
                    compute_contrastive_loss: bool = False,
                    contrastive_temperature: float = 0.1,
                    contrastive_pooler: str = "mean",
                    **kwargs):
        super().__init__(**kwargs)

        self.model_name = model_name
        self.base_model_path = base_model_path
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.compute_aminoacid_loss = compute_aminoacid_loss
        self.compute_codon_loss = compute_codon_loss
        self.compute_contrastive_loss = compute_contrastive_loss
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_pooler = contrastive_pooler
        for key, value in kwargs.items():
            setattr(self, key, value)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float) -> None:
        """Contrastive loss for SimCLR.

        Reference: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html#SimCLR

        Parameters
        ----------
        temperature: float
            Determines how peaked the distribution. Since many similarity
            metrics are bounded, the temperature parameter allows us to
            balance the influence of many dissimilar image patches versus
            one similar patch.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # NOTE: z.shape == (batch_size, hidden_size)
        # TODO: Can we cache the pos_mask calculation with lru_cache?
        batch_size = z.shape[0]
        z_all = [torch.empty_like(z) for _ in range(size)]
        # gather all the tensors 
        torch.distributed.allgather(z_all, z)
        # replace local rank information so gradients propagate
        z_all[rank] = z
        # concatenate all the tensors on batch dimension
        z = torch.cat(z_all, dim=0)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        cos_sim.masked_fill_(self_mask, -65504)
        # Find positive example -> batch_size // 2 away from the original example
        pos_mask = self_mask.roll(shifts=batch_size // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        return nll


class MeanPooler(nn.Module):
    """Reduces the sequence embeddings (batch_size, seq_length, hidden_size)
    to a single embedding (batch_size, hidden_size) by averaging."""

    def __init__(self, config) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The average over sequence length gives even weighting to each sequence position
        return x.mean(dim=1)


class FirstPooler(EsmPooler):
    """Reduces the sequence embeddings (batch_size, seq_length, hidden_size)
    to a single embedding (batch_size, hidden_size) by taking the first hidden state."""


POOLER_DISPATCH = {"mean": MeanPooler, "first": FirstPooler}


class EsmContrastiveProjectionHead(nn.Module):
    def __init__(self, config: ContrastiveEsmConfig) -> None:
        super().__init__()
        # The projection representions z are trained to become invariant to
        # many gene/protein specific features

        # We use a different projection head for codons and amino acids
        # since, by default, the embeddings fall into different subspaces.
        self.codon_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.d_model // 4),
        )
        self.aminoacid_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.d_model // 4),
        )

        self.loss_fn = ContrastiveLoss(temperature=config.contrastive_temperature)
        self.pooler = POOLER_DISPATCH[config.contrastive_pooler](config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assumes that the codon embeddings are the first half of the tensor
        # and the aminoacid embeddings are the second half.

        # Pool the sequence embeddings to get a single embedding per sequence
        x = self.pooler(x)  # (batch_size, hidden_size)

        # Collect the codon and aminoacid embeddings separately
        # These have shape (batch_size // 2, hidden_size)
        half_batch_size = x.shape[0] // 2
        codon_embed = x[:half_batch_size]
        aminoacid_embed = x[half_batch_size:]

        # Project the embeddings into a lower dimensional space
        # These have shape (batch_size // 2, projection_size)
        z_codon = self.codon_projection(codon_embed)
        z_aminoacid = self.aminoacid_projection(aminoacid_embed)

        # Concatenate the codon and aminoacid embeddings
        # This has shape (batch_size, projection_size)
        z = torch.cat([z_codon, z_aminoacid], dim=0)

        # Compute the contrastive loss following SimCLR
        return self.loss_fn(z)


class EsmCForContrastiveMaskedLM(PreTrainedModel):
    def __init__(
        self,
        config: ContrastiveEsmConfig,
    ) -> None:
        super().__init__(config)
        self.set_config(config)
        if '300m' in str(config.model_name.lower()):
            model_builder = ESMC_300M_202412
        elif '600m' in str(config.model_name.lower()):
            model_builder = ESMC_600M_202412
        else:
            raise ValueError("Model name must contain '300m' or '600m' referring to the ESMC model size")        
        register_local_model(config.base_model_path, model_builder)
        self.transformer = ESMC.from_pretrained(config.base_model_path)
        # has AA embed/decode named
        #   self.transformer.embed
        #   self.sequence_head

        try:
            self.transformer.tokenizer = EsmTokenizer.from_pretrained(config.tokenizer_name_or_path)
        except:
            pass
        # Inject contrastive loss parameters into the config
        # Note: The config settings will override the init settings. This
        # is so that checkpoints saved on the first training run will be
        # properly initialized when the checkpoint is loaded. If this is
        # not done, then it will default to the settings passed in the init.
        # Note: the saved checkpoint configs contain these settings, but the
        # original ESM configs do not.
        if not hasattr(config, "compute_aminoacid_loss"):
            config.compute_aminoacid_loss = compute_aminoacid_loss
        if not hasattr(config, "compute_codon_loss"):
            config.compute_codon_loss = compute_codon_loss
        if not hasattr(config, "compute_contrastive_loss"):
            config.compute_contrastive_loss = compute_contrastive_loss
        if not hasattr(config, "contrastive_temperature"):
            config.contrastive_temperature = contrastive_temperature
        if not hasattr(config, "contrastive_pooler"):
            config.contrastive_pooler = contrastive_pooler

        # Only used if compute_contrastive_loss is True
        self.add_contrastive_loss_head(config)

        # Only used if compute_codon_loss is True. Make a new config with the
        # same parameters as the original config but with a different vocab size
        # (the number of codons 64 + special tokens)
        codon_config = EsmConfig(**config.to_dict())
        codon_config.vocab_size = 69
        self.add_codon_lm_head(codon_config)

        # Need to initialize the lm_head for amino acids with the
        # correct vocab size in order to load weights for inference properly
        # (since the vocab size in config is the combined vocab).
        # Note: lm_head is also modified if update_model_weights is called.
        # ESMC uses regression head for this task.
        amino_config = EsmConfig(**config.to_dict())
        amino_config.vocab_size = 33
        self.add_aminoacid_lm_head(amino_config)
        # base tokenizer has vocab of 102? in ESMC; includes only peptides and some special tokens though
        self.config.vocab_size = codon_config.vocab_size + amino_config.vocab_size
        # These are the entries in vocab we *really* dont want to change
        self.config.amino_vocab_size = amino_config.vocab_size
        # Initialize weights and apply final processing
        self.post_init()

        # print(self)
        # exit()
    def post_init(self):
        super().post_init()
        device = next(self.transformer.transformer.parameters()).device
        self.transformer.to(device)
        # set all parameters to bf16
        self.validate_dtype()

    def save_pretrained(self, save_directory: str, *args, **kwargs) -> None:
        is_main_process = True
        if "is_main_process" in kwargs:
            is_main_process = kwargs.pop("is_main_process")
        if is_main_process:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory, exist_ok=True)
            with open(os.path.join(save_directory, "config.json"), "w") as f:
                json.dump(self.get_config(), f)
            # write model with safetensors
            torch.save(self.state_dict(), os.path.join(save_directory, "model.pt"))
            # override save_pretrained to make sure we save the tokenizer also
            self.transformer.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(self, model_name_or_path: str, *model_args, **kwargs):
        # Load the model from the base class
        config_file = os.path.join(model_name_or_path, "config.json")
        with open(config_file, "r") as f:
             config_dict = json.load(f)
        config = ContrastiveEsmConfig(**config_dict)

        # yes... initiating this way isnt the best--but these models arent that large. Who cares (for now) TODO: fix this
        model = EsmCForContrastiveMaskedLM(config)
        model.update_model_weights(model.transformer.tokenizer)
        # load via safetensors
        state_dict = torch.load(os.path.join(model_name_or_path, "model.pt"))
        model.load_state_dict(state_dict)
        # Load the tokenizer
        model.transformer.tokenizer = EsmTokenizer.from_pretrained(model_name_or_path)
        return model

    def set_config(self, config: ContrastiveEsmConfig):
        self.config_file = config
    def get_config(self):
        print(self.config_file)
        print(self.config_file.to_dict())
        return self.config_file.to_dict()
    def add_contrastive_loss_head(self, config: ContrastiveEsmConfig):
        self.transformer.contrastive_head = EsmContrastiveProjectionHead(config)

    def add_codon_lm_head(self, config: EsmConfig):
        self.transformer.codon_lm_head = RegressionHead(config.d_model, config.vocab_size)
    
    def add_aminoacid_lm_head(self, config: EsmConfig):
        self.transformer.lm_head = RegressionHead(config.d_model, config.vocab_size)

    def get_output_embeddings(self):
        # We override weight tieing since the new token embedding matrix
        # has a different vocab size from the amino acid lm_head.decoder
        # with the addition of the new codon tokens. It would also be
        # challenging to tie weights between the token embedding matrix
        # and the codon_lm_head.decoder (since it would also need to share
        # weights for the special tokens which would be tied to the amino
        # acid lm_head.decoder as well.
        return None
        # return self.lm_head.decoder

    @torch.no_grad()
    def update_model_weights(self, tokenizer: EsmTokenizer) -> None:
        # If the tokenizer has the same number of tokens as the model, then
        # the model weights are already updated and we can return early.
        new_vocab_size = len(tokenizer)
        orig_aa_vocab_size = self.config.amino_vocab_size
        if new_vocab_size == self.config.vocab_size:
            return
        # if rank == 0: print(f"Expanding vocab from {self.config.vocab_size} to {new_vocab_size}")

        logger.warning(
            "Resizing token embedding layer from {} to {} and initializing codon_lm_head with amino acid lm_head".format(
                self.config.vocab_size, new_vocab_size
            )
        )

        # Get the original token embedding matrix (TEM)
        original_tem = self.transformer.embed.weight.clone()
        # Get the original amino acid lm head
        original_lm_head = deepcopy(self.transformer.sequence_head)

        # Inject new vocabulary (modifies config and TEM)
        self.resize_token_embeddings(new_vocab_size)
        assert self.transformer.embed.weight.shape[0] == new_vocab_size
        assert torch.equal(self.transformer.embed.weight[: orig_aa_vocab_size], 
                            original_tem[: orig_aa_vocab_size])
        # Get a reference to the new TEM matrix
        new_tem = self.transformer.embed.weight

        # Set each of the new codon representations equal to their corresponding
        # amino acid representations (note that amino acid representations are
        # unchanged by vocabulary expansion)
        # Here we are looping over the entire vocab, but we are only using the
        # vocab elements which are codons, so we name the variables accordingly.
        for codon, codon_id in tokenizer.get_vocab().items():
            # if rank == 0: print(f"{codon=} ::: {codon_id=}")
            # Note: the translation table maps stop codons to "" and the get
            # function maps the special tokens and amino acids to "" as well.
            aminoacid = translation_table.get(codon, "")
            if aminoacid:
                aminoacid_id = tokenizer._token_to_id[aminoacid]
                # The new TEM matrix aminoacid represenations are the same
                # as the original TEM (they are preserved during resizing)
                # if rank == 0: print(f"{aminoacid=} ::: {aminoacid_id=}")
                # if rank == 0: print(f"setting {codon=}, {codon_id} to {aminoacid=}, {aminoacid_id}")
                new_tem[codon_id] = new_tem[aminoacid_id].clone()
                assert torch.equal(original_tem[aminoacid_id], new_tem[codon_id])

        # Check that the TEM matrix was updated correctly
        assert torch.equal(original_tem[: orig_aa_vocab_size], new_tem[: orig_aa_vocab_size])
        assert torch.equal(self.transformer.embed.weight, new_tem)

        # Now that the TEM is updated, we also need to update the lm heads
        # for amino acids and codons.

        # Resizing vocab changes the original aminoacid lm_head, we want to
        # change it back since we have a separate lm_head for the codon vocabulary.
        ## New resize actually didnt change the AA head because I'm lazy 12-2024 AIW
        self.lm_head = deepcopy(original_lm_head).to(original_lm_head[0].weight.device)
        copied_params = dict(self.lm_head.named_parameters())
        # Check that the original amino acid lm_head was reloaded correctly
        for n, p in original_lm_head.named_parameters():
            assert torch.equal(p, copied_params[n]), f"{n=} was not copied correctly"
        
        # We initialize the codon_lm_head dense and layer_norm layers
        # with the vocab-size invariant amino acid counterparts
        self.transformer.codon_lm_head[0].weight.data = self.lm_head[0].weight.data.clone()
        self.transformer.codon_lm_head[0].bias.data = self.lm_head[0].bias.data.clone()
        self.transformer.codon_lm_head[2].weight.data = self.lm_head[2].weight.data.clone()
        self.transformer.codon_lm_head[2].bias.data = self.lm_head[2].bias.data.clone()

        # Check that the weights have been updated properly
        assert torch.equal(self.transformer.codon_lm_head[0].weight, self.lm_head[0].weight)
        assert torch.equal(self.transformer.codon_lm_head[0].bias, self.lm_head[0].bias)
        assert torch.equal(
            self.transformer.codon_lm_head[2].weight, self.lm_head[2].weight
        )
        assert torch.equal(
            self.transformer.codon_lm_head[2].bias, self.lm_head[2].bias
        )
        assert id(self.transformer.codon_lm_head[0].weight) != id(self.lm_head[0].weight)
        assert id(self.transformer.codon_lm_head[2].weight) != id(self.lm_head[2].weight)

        # Get the original aminoacid lm_head decoder weights
        aminoacid_decoder = original_lm_head[3].weight
        aminoacid_bias = original_lm_head[3].bias

        # Get references to the codon_lm_head components to change
        codon_decoder = self.transformer.codon_lm_head[3].weight
        codon_bias = self.transformer.codon_lm_head[3].bias
        codon_decoder.to(aminoacid_decoder.device)
        codon_bias.to(aminoacid_bias.device)
        # Since the vocabulary is combined, there is a total vocab size of 97
        # (amino acid + codons + special). However, the codon_lm_head decoder
        # and bias only has an output size of 69 (codon + special). Our goals
        # are to (1) initialize the weights relevant to special tokens to be
        # identical to the original amino acid lm_head weights, and (2) to
        # initialize the weights for each codon to the weights of the translated
        # amino acid in the original lm head. Here the enumeration is taken over
        # the 69 codon/special tokens. The first 5 tokens are the special tokens
        # which are set to the first 5 dimensions of the codon_decoder/bias.
        for codon_idx, codon in enumerate(tokenizer.added_tokens_decoder.values()):
            aminoacid = translation_table.get(str(codon), "")
            # The codon could be a special token (<cls>, <pad>, <eos>, <unk>, <mask>)
            if aminoacid or codon.special:
                # Get the id of the aminoacid/special token corresponding to the current codon
                aminoacid_id = tokenizer._token_to_id[
                    str(codon) if codon.special else aminoacid
                ]
                # Update the codon_decoder/bias with the corresponding aminoacid weights
                codon_decoder[codon_idx] = aminoacid_decoder[aminoacid_id].clone()
                codon_bias[codon_idx] = aminoacid_bias[aminoacid_id].clone()
                codon_decoder.to(aminoacid_decoder.device)
                codon_bias.to(aminoacid_bias.device)
                # Check that the update was successfull
                assert torch.equal(
                    codon_decoder[codon_idx].cpu(), aminoacid_decoder[aminoacid_id].cpu()
                )
                assert torch.equal(codon_bias[codon_idx].cpu(), aminoacid_bias[aminoacid_id].cpu())
                assert torch.equal(codon_decoder, self.transformer.codon_lm_head[3].weight)
                assert torch.equal(codon_bias, self.transformer.codon_lm_head[3].bias)

    # have to make sure everything is on the same device and type
    def validate_dtype(self):
        for name, param in self.named_parameters():
            if param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)


    def get_input_embeddings(self):
        return self.transformer.embed

    def set_input_embeddings(self, new_embeddings):
        self.transformer.embed = new_embeddings

    def resize_token_embeddings(self, new_num_tokens: int):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # Update the lm_head if it exists
        # here we manually update this information AIW
        # if hasattr(self, 'lm_head'):
        #     old_lm_head = self.lm_head
        #     new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
        #     self.lm_head = new_lm_head

        return self.get_input_embeddings()

    def _get_resized_embeddings(self, old_embeddings: nn.Embedding, new_num_tokens: int) -> nn.Embedding:
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device)

        # Copy the weights from the old embeddings to the new embeddings
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        return new_embeddings

    def _get_resized_lm_head(self, old_lm_head: nn.Linear, new_num_tokens: int) -> nn.Linear:
        old_num_tokens, old_hidden_dim = old_lm_head.weight.size()
        new_lm_head = nn.Linear(old_hidden_dim, new_num_tokens, bias=False)
        new_lm_head.to(old_lm_head.weight.device)

        # Copy the weights from the old lm_head to the new lm_head
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]

        return new_lm_head


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        codon_input_ids: Optional[torch.LongTensor] = None,
        codon_attention_mask: Optional[torch.Tensor] = None,
        codon_labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Note: We needed to pass the codon input_ids and attention_mask separately
        # since the distrubted sampler shuffles the codon and amino acid sequences
        # and there was no way to guarantee that the codon and amino acid sequences
        # would be index properly for the contrastive loss.

        # Pack the amino acid and codon input_ids and attention_mask into a single tensor
        # This avoids two separate forward passes through the model but effectively doubles
        # the batch size.
        if input_ids is not None and codon_input_ids is not None:
            input_ids = torch.cat([codon_input_ids, input_ids], dim=0)
        if attention_mask is not None and codon_attention_mask is not None:
            attention_mask = torch.cat([codon_attention_mask, attention_mask], dim=0)

        # Note: During inference we can pass either the amino acid or codon sequences
        # but not both. If we pass both, then the model will try to stack the amino acid
        # and codon input ids and attention masks as shown above, and this may not be
        # the desired behavior.
        #if codon_input_ids is not None and input_ids is None:
        #    input_ids = codon_input_ids
        #    assert (
        #        codon_attention_mask is not None
        #    ), "Please provide codon_attention_mask"
        #    attention_mask = codon_attention_mask
        #    self.config.compute_codon_loss = True
        #    self.config.compute_aminoacid_loss = False

        #elif input_ids is not None and codon_input_ids is None:
        #    assert (
        #        attention_mask is not None
        #    ), "Please provide attention_mask for amino acid sequences"
        #    self.config.compute_codon_loss = False
        #    self.config.compute_aminoacid_loss = True

        # ESMC doesnt have the standard HF interface...
        outputs = self.transformer(
            sequence_tokens = input_ids,
            sequence_id=attention_mask,
        )
        sequence_output = outputs.hidden_states[-1]  # (batch_size, seq_length, hidden_size)

        # Compute the logits / prediction scores for each head
        if self.config.compute_aminoacid_loss and self.config.compute_codon_loss:
            # Split the sequence output into codon and amino acid embeddings
            # These have shape (batch_size // 2, seq_length, hidden_size)
            half_batch_size = sequence_output.shape[0] // 2
            codon_embed = sequence_output[:half_batch_size]
            amino_embed = sequence_output[half_batch_size:]
            codon_prediction_scores = self.transformer.codon_lm_head(codon_embed)
            amino_prediction_scores = self.lm_head(amino_embed)
            # The prediction scores have different vocab sizes, so we can't concatenate them.
            # Instead, we return the aminoacid scores (during inference, set either
            # compute_aminoacid_loss or compute_codon_loss to False)
            prediction_scores = amino_prediction_scores
        elif self.config.compute_aminoacid_loss:
            prediction_scores = self.lm_head(sequence_output)
        elif self.config.compute_codon_loss:
            prediction_scores = self.transformer.codon_lm_head(sequence_output)
        else:
            raise ValueError(
                "Either compute_aminoacid_loss or compute_codon_loss must be True"
            )

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            # Compute the masked language modeling loss for each head
            if self.config.compute_aminoacid_loss and self.config.compute_codon_loss:
                # Split the labels into codon and amino acid labels
                # These have shape (batch_size // 2, seq_length)
                # half_batch_size = labels.shape[0] // 2
                # codon_labels = labels[:half_batch_size]
                # aminoacid_labels = labels[half_batch_size:]

                # Compute the masked language modeling loss for each head
                aminoacid_masked_lm_loss = loss_fct(
                    amino_prediction_scores.view(-1, amino_prediction_scores.shape[-1]),
                    labels.view(-1),  # The labels store the amino acid labels
                )
                codon_masked_lm_loss = loss_fct(
                    codon_prediction_scores.view(-1, codon_prediction_scores.shape[-1]),
                    codon_labels.view(-1),
                )
                # Add the two losses together
                masked_lm_loss = (codon_masked_lm_loss + aminoacid_masked_lm_loss) / 2.0

            else:
                # # (-1, vocab_size) is the shape of the prediction scores
                masked_lm_loss = loss_fct(
                    prediction_scores.view(-1, prediction_scores.shape[-1]),
                    labels.view(-1),
                )

        # Custom logic to compute a contrastive loss between Codons and Amino Acid embeddings.
        # Everything else in this function is the same as the base class implementation.
        if labels is not None and self.config.compute_contrastive_loss:
            # Compute the contrastive loss following SimCLR and add it to the masked language modeling loss
            masked_lm_loss += self.transformer.contrastive_head(sequence_output)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )
    
if __name__ == '__main__':
    from esm.pretrained import register_local_model, ESMC_300M_202412
    from dataset import GenSLMColatorForLanguageModeling
    model_path = "/lus/eagle/projects/FoundEpidem/azton/esmc_models/esmc_300m"
    model_name = 'ESMC_300M'
    tokenizer_path = "/lus/eagle/projects/CVD-Mol-AI/braceal/src/genslm-esm/tokenizer_esm_genslm"

    config = ContrastiveEsmConfig(model_name=model_name,
                                base_model_path=model_path,
                                tokenizer_name_or_path=tokenizer_path, 
                                compute_contrastive_loss=True, 
                                compute_aminoacid_loss=True, 
                                compute_codon_loss=True)
    # encode any string in config as utf-8
    model = EsmCForContrastiveMaskedLM(config)
    print(model)
    model.update_model_weights(model.transformer.tokenizer)
    model.save_pretrained('./TestESMCSave_AA', )

    model = None
    model = EsmCForContrastiveMaskedLM.from_pretrained('./TestESMCSave_AA')
    print('Reloaded model:')
    print(model)

    collator = GenSLMColatorForLanguageModeling(return_codon=True,
                                                return_aminoacid=True,
                                                tokenizer=model.transformer.tokenizer,
                                                mlm=True,
                                                train_mode=True,
                                                mlm_probability=0.15,
                                            )
    test_input = [{
        'aminoacid': 'MKVLPQETVRIG',  # 12 residues
        'codon': 'ATGAAGGTACTACCACAAGAAACTGTAAGAATTGGA'  # 36 nucleotides
        },
        {
        'aminoacid': 'MDKTHIRLSVDNPFAKL',  # 16 residues
        'codon': 'ATGGACAAAACACATATTCGACTATCTGTTGACAATCCATTTGCAAAACTA'  # 48 nucleotides
        }
    ]
    test_input = collator(test_input)
    print(test_input)
    outputs = model(**test_input)
    print(outputs.loss)