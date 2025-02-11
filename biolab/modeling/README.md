# Models

| Model                 | Input       | Tokenization | Logits                       | Embedding | Attention | Layer Specific Embeddings |
|-----------------------|-------------|--------------|------------------------------|-----------|-----------|---------------------------|
| Ankh                  | Amino acid  | Character    | No (need decoder)            | Yes       | No        | Yes                       |
| CaLM                  | DNA         | 3-mer        | Yes                          | Yes       | Yes       | Yes                       |
| DNABert               | DNA         | BPE          | Yes                          | Yes       | No        | No                        |
| ESM2                  | Amino Acid  | Character    | Yes                          | Yes       | Yes       | Yes                       |
| ESM3                  | Amino Acid* | Character    | No (not implemented yet)     | Yes       | No        | No                        |
| ESMC                  | Amino Acid  | Character    | Yes                          | Yes       | No        | No                        |
| EVO                   | DNA         | Character    | No (clashing with embedding) | Yes       | N/A       | ?                         |
| GenaLM                | DNA         | BPE          | Yes                          | Yes       | Yes       | Yes                       |
| GenSLM                | DNA         | 3-mer        | Yes                          | Yes       | Yes       | Yes                       |
| NucleotideTransformer | DNA         | 6-mer        | Yes                          | Yes       | Yes       | Yes                       |
| ProtGPT2              | Amino Acid  | BPE          | Yes                          | Yes       | Yes       | Yes                       |
| ProtTrans             | Amino Acid  | Character    | No (need decoder)            | Yes       | Yes       | Yes                       |
