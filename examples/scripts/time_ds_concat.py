import timeit
import datasets
import torch
from datasets import Dataset

setup_add_column = """
import datasets
import torch
dset = datasets.load_from_disk("./dna_classification")
embds = torch.rand(len(dset), 512)
"""

setup_concatenate_datasets = """
import datasets
import torch
from datasets import Dataset
dset = datasets.load_from_disk("./small_mdh_gc")
embds = torch.rand(len(dset), 512)
"""

stmt_add_column = """
dset.add_column("embeddings", embds.tolist())
"""

stmt_concatenate_datasets = """
embs_ds = Dataset.from_dict({"mean_pooled_embeddings": embds.numpy()})
ds = datasets.concatenate_datasets([dset, embs_ds], axis=1)
"""

add_column_time = timeit.timeit(stmt=stmt_add_column, setup=setup_add_column, number=10)
concatenate_time = timeit.timeit(stmt=stmt_concatenate_datasets, setup=setup_concatenate_datasets, number=10)

print("Time taken for add_column method:", add_column_time)
print("Time taken for concatenate_datasets method:", concatenate_time)

# Time taken for add_column method: 369.4749341889983
# Time taken for concatenate_datasets method: 0.0829016660572961
