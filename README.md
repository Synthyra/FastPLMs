# FastPLMs

FastPLMs is an open-source effort to increase the efficiency of pretrained protein language models, switching out native attention implementations for Flash or Flex attention.

All models can be loaded from Huggingface 🤗 transformers via `AutoModel`, this repository does not need to be cloned for most use cases.

## Supported models
The currently supported models can be found [here](https://huggingface.co/collections/Synthyra/pretrained-plms-675351ecc050f63baedd77de).

## Suggestions
Have suggestions, comments, or requests? Found a bug? Open a GitHub issue and we'll respond soon.

## Low code embedding of protein datasets
```python
embeddings = model.embed_dataset(
    sequences=sequences, # list of protein strings
    batch_size=16, # embedding batch size
    max_len=2048, # truncate to max_len
    full_embeddings=True, # return residue-wise embeddings
    full_precision=False, # store as float32
    pooling_type='mean', # use mean pooling if protein-wise embeddings
    num_workers=0, # data loading num workers
    sql=False, # return dictionary of sequences and embeddings
)

_ = model.embed_dataset(
    sequences=sequences, # list of protein strings
    batch_size=16, # embedding batch size
    max_len=2048, # truncate to max_len
    full_embeddings=True, # return residue-wise embeddings
    full_precision=False, # store as float32
    pooling_type='mean', # use mean pooling if protein-wise embeddings
    num_workers=0, # data loading num workers
    sql=True, # store sequences in local SQL database
    sql_db_path='embeddings.db', # path to .db file of choice
)
```

### Note
The ANKH implementation is in progress. It is functional but is still currently native attention.
