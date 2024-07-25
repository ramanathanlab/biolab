from pathlib import Path
from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
from tqdm import tqdm

from genslm.utils import read_fasta, Sequence
from transformers import EsmForMaskedLM, EsmTokenizer
from genslm_esm.modeling_esm_v3 import EsmForContrastiveMaskedLM
from genslm_esm.dataset import (
    FastaAminoAcidDataset,
    FastaDataset,
    GenSLMColatorForLanguageModeling,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def group_by_kmer(seq: str, k: int = 3):
    return " ".join(seq[i : i + k] for i in range(0, len(seq), k)).upper()


def generate_embeddings(
    model,
    tokenizer,
    fasta_path: Path,
    return_codon: bool,
    return_aminoacid: bool,
    batch_size: int,
    fasta_contains_aminoacid: bool = False
) -> List[np.ndarray]:

    if fasta_contains_aminoacid:
        dataset = FastaAminoAcidDataset(file_path=fasta_path)
    else:
        dataset = FastaDataset(
            file_path=fasta_path,
            return_codon=return_codon,
            return_aminoacid=return_aminoacid,
        )

    data_collator = GenSLMColatorForLanguageModeling(
        return_codon=return_codon,
        return_aminoacid=return_aminoacid,
        tokenizer=tokenizer,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
    )
    print("Tokenized dataset")

    embeddings = []
    model.to(DEVICE)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(model.device)
            outputs = model(**batch, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            seq_lengths = batch.attention_mask.sum(axis=1)
            for seq_len, elem in zip(seq_lengths, last_hidden_states):
                embedding = elem[1:seq_len - 1, :].cpu().numpy()
                embeddings.append(embedding)

    return embeddings


def save_h5(embeddings, ouput_path: Path):
    h5_kwargs = {
        # "compression": "gzip",
        # "compression_opts": 4, Compression is too slow for current impl
        # "fletcher32": True,
    }
    with h5py.File(ouput_path, "w") as f:
        group = f.create_group("embeddings")
        for i, embedding in enumerate(embeddings):
            group.create_dataset(str(i), data=embedding, **h5_kwargs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        help="Optional if tokenizer is not part of checkpoint_path",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=1026)
    parser.add_argument("--dna_embeddings", action="store_true")
    parser.add_argument("--protein_embeddings", action="store_true")
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Save as npy file, defualts to saving as a h5 file with embeddings as datasets",
    )

    args = parser.parse_args()

    # setup
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Load model
    model = EsmForContrastiveMaskedLM.from_pretrained(args.checkpoint_path)
    # Load the tokenizer
    tokenizer = EsmTokenizer.from_pretrained(args.tokenizer_path)
    print(f"Loaded model from {args.checkpoint_path}")

    # Load sequences
    base_path = Path(
        "/lambda_stor/homes/khippe/genslm_foundation/downstream_evaluation/evaluation_pipeline_data"
    )
    input_ffn_data = {
        "mdh_tsne": base_path / "mdh_tsne" / "mdh_phylogeny_seqs.fasta",
        "PGF_00007092": base_path / "family_discrimination" / "PGF_00007092.ffn",
        "PGF_00008864": base_path / "family_discrimination" / "PGF_00008864.ffn",
        "dna_classification": base_path
        / "dna_classification"
        / "dna_classification_sequences.fasta",
        "secondary_structure_classification": base_path
        / "secondary_structure_classification"
        / "struct_sequences.ffn",
        "isotropy": base_path / "pgfam_missing_codon" / "small_dataset.fasta",
    }

    input_aa_data = {
        "mdh_tsne": base_path / "mdh_tsne" / "aminoacid_mdh_phylogeny_seqs.fasta",
        "PGF_00007092": base_path / "family_discrimination" / "PGF_00007092.faa",
        "PGF_00008864": base_path / "family_discrimination" / "PGF_00008864.faa",
        "dna_classification": base_path
        / "dna_classification"
        / "esm-dna_classification_sequences_aminoacids.fasta",
        "secondary_structure_classification": base_path
        / "secondary_structure_classification"
        / "aminoacid_struct_sequences.fasta",
        "isotropy": base_path / "pgfam_missing_codon" / "aminoacid_small_dataset.fasta",
    }

    if args.dna_embeddings:
        for task_name, task_seq_path in input_ffn_data.items():
            print(f"Generating (dna) embeddings for {task_name}")
            # Generate embeddings
            embeddings = generate_embeddings(
                model,
                tokenizer,
                task_seq_path,
                return_codon=True,
                return_aminoacid=False,
                fasta_contains_aminoacid=False,
                batch_size=args.batch_size,
            )
            # Save embeddings
            if args.save_npy:
                np.save(args.output_dir / f"{task_name}_dna_embeddings.npy", embeddings)
            else:
                save_h5(embeddings, args.output_dir / f"{task_name}_dna_embeddings.h5")

    if args.protein_embeddings:
        for task_name, task_seq_path in input_aa_data.items():
            print(f"Generating (aa) embeddings for {task_name}")
            sequences = [seq.sequence for seq in read_fasta(task_seq_path)]
            # Generate embeddings
            embeddings = generate_embeddings(
                model,
                tokenizer,
                task_seq_path,
                return_codon=False,
                return_aminoacid=True,
                fasta_contains_aminoacid=True,
                batch_size=args.batch_size,
            )
            # Save embeddings
            if args.save_npy:
                np.save(args.output_dir / f"{task_name}_aa_embeddings.npy", embeddings)
            else:
                save_h5(embeddings, args.output_dir / f"{task_name}_aa_embeddings.h5")
