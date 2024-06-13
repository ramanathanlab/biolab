from transformers.modeling_outputs import ModelOutput
from datasets import Dataset, concatenate_datasets

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from biolab.api.lm import LM, Transform
from biolab.api.logging import logger

# TODO: set this higher up, should not be here i don't think
torch.random.manual_seed(42)


# TODO: wrap model outputs otherwise you get Masked/CausalLMOutput?
# TODO: caching for the dataset?
# TODO: what about tokenization on the fly? that would allow for genslm-esm
def generate_embeddings(
    model: LM, dataset: Dataset, transform: Transform
) -> ModelOutput:
    # Tokenize the dataset
    # TODO: remove column specifier, is this a property of the LM?
    def tokenize_dna(examples):
        tokenize_kwargs = model.tokenizer_config
        return model.tokenizer(examples["input_dna"], **tokenize_kwargs)

    modeling_dataset = dataset.map(
        tokenize_dna,
        batched=True,
        remove_columns=dataset.column_names,
    ).with_format("torch")

    # turn into dataloader and grab dset info
    dataloader = DataLoader(modeling_dataset, **model.dataloader_config)
    num_embeddings = len(modeling_dataset)
    batch_size = model.dataloader_config.get("batch_size", 1)

    # generate embeddings
    all_embeddings = torch.empty(
        (num_embeddings, model.embedding_size),
        dtype=model.dtype,
    )

    idx = 0
    with torch.no_grad():
        with logging_redirect_tqdm(loggers=[logger]):
            for batch in tqdm(dataloader, desc="Generating embeddings"):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model.embed(batch)

                # TODO: need hook for last hidden state
                last_hidden_state = outputs.hidden_states[-1]

                # average pool the embeddings
                pooled = transform.apply(last_hidden_state, batch["attention_mask"])
                pooled = pooled.cpu()
                # insert into all_embeddings
                # TODO: consider iteratively pushing these to the dataset for
                # more efficient use of memory
                all_embeddings[idx : idx + batch_size, :] = pooled  # noqa: E203

                # Increment the output buffer index by the batch size
                idx += batch_size

    # Create temp dataset for faster concatenation (faster than adding col via numpy)
    embs_ds = Dataset.from_dict({transform.name: all_embeddings.numpy()})
    return concatenate_datasets([dataset, embs_ds], axis=1).with_format("torch")
