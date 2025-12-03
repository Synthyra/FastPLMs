# FastPLMs

<img width="2816" height="1536" alt="Gemini_Generated_Image_5bmmdc5bmmdc5bmm" src="https://github.com/user-attachments/assets/ffaf84b6-9970-40fd-aa31-1b314d6ca146" />

FastPLMs is an open-source effort to increase the efficiency of pretrained protein language models, switching out native attention implementations for Flash or Flex attention.

All models can be loaded from Huggingface 🤗 transformers via `AutoModel`, this repository does not need to be cloned for most use cases.

## Supported models
The currently supported models can be found [here](https://huggingface.co/collections/Synthyra/pretrained-plms-675351ecc050f63baedd77de).

## Suggestions
Have suggestions, comments, or requests? Found a bug? Open a GitHub issue and we'll respond soon.

## Embed entire datasets with no new code
To embed a list of protein sequences **fast**, just call embed_dataset. Sequences are sorted to reduce padding tokens, so the initial progress bar estimation is usually much longer than the actual time it will take.

Example:
```python
embedding_dict = model.embed_dataset(
    sequences=[
        'MALWMRLLPLLALLALWGPDPAAA', ... # list of protein sequences
    ],
    batch_size=2, # adjust for your GPU memory
    max_len=512, # adjust for your needs
    full_embeddings=False, # if True, no pooling is performed
    embed_dtype=torch.float32, # cast to what dtype you want
    pooling_type=['mean', 'cls'], # more than one pooling type will be concatenated together
    num_workers=0, # if you have many cpu cores, we find that num_workers = 4 is fast for large datasets
    sql=False, # if True, embeddings will be stored in SQLite database
    sql_db_path='embeddings.db',
    save=True, # if True, embeddings will be saved as a .pth file
    save_path='embeddings.pth',
)
# embedding_dict is a dictionary mapping sequences to their embeddings as tensors for .pth or numpy arrays for sql
```

```
model.embed_dataset()
Args:
    sequences: List of protein sequences
    batch_size: Batch size for processing
    max_len: Maximum sequence length
    full_embeddings: Whether to return full residue-wise (True) embeddings or pooled (False)
    pooling_type: Type of pooling ('mean' or 'cls')
    num_workers: Number of workers for data loading, 0 for the main process
    sql: Whether to store embeddings in SQLite database - will be stored in float32
    sql_db_path: Path to SQLite database
    
Returns:
    Dictionary mapping sequences to embeddings, or None if sql=True

Note:
    - If sql=True, embeddings can only be stored in float32
    - sql is ideal if you need to stream a very large dataset for training in real-time
    - save=True is ideal if you can store the entire embedding dictionary in RAM
    - sql will be used if it is True and save is True or False
    - If your sql database or .pth file is already present, they will be scanned first for already embedded sequences
    - Sequences will be truncated to max_len and sorted by length in descending order for faster processing
```

## Upcoming releases
A Fast version of ANKH is in progress. It is functional but is still currently native attention, we are waiting for bias gradient support in [FlexAttention](https://pytorch.org/blog/flexattention/).
