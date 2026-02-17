# FastPLMs

<img width="2816" height="1536" alt="Gemini_Generated_Image_5bmmdc5bmmdc5bmm" src="https://github.com/user-attachments/assets/ffaf84b6-9970-40fd-aa31-1b314d6ca146" />

FastPLMs is an open-source effort to increase the efficiency of pretrained protein language models, switching out native attention implementations for Flash or Flex attention.

All models can be loaded from Huggingface 🤗 transformers via `AutoModel`, this repository does not need to be cloned for most use cases.

## Attention backend defaults
`sdpa` is now the default attention backend for `ESM++` and `FastESM2` for stability across PyTorch versions.

If you want Flex Attention, set `attn_backend="flex"` in the model config before loading the model:

```python
import torch
from transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained("Synthyra/ESMplusplus_small", trust_remote_code=True)
config.attn_backend = "flex"
model = AutoModel.from_pretrained("Synthyra/ESMplusplus_small", config=config, trust_remote_code=True)
```

For throughput and memory efficiency, `torch.compile(...)` is strongly recommended, especially with Flex Attention:

```python
model = torch.compile(model)
```

If your environment has a compiler regression, keep `attn_backend="sdpa"` and run without compile or with a safer backend such as `aot_eager`.

## Supported models
The currently supported models can be found [here](https://huggingface.co/collections/Synthyra/pretrained-plms-675351ecc050f63baedd77de).

## Testing suite

The testing workflow is now CLI-first under `test_scripts/` with clean, structured outputs.

### Main test entrypoints

- Compliance and correctness checks:
  - `py -m test_scripts.run_compliance`
- Boltz2 compliance vs pip boltz reference:
  - `py -m test_scripts.run_boltz2_compliance`
- Embedding mixin behavior checks:
  - `py -m test_scripts.run_embedding`
- Throughput and memory benchmarks:
  - `py -m test_scripts.run_throughput`
- Run everything in one command:
  - `py -m test_scripts.run_all`

By default, each suite runs one representative checkpoint per family (`E1`, `ESM2`, `ESMplusplus`).

### Common options

- Run full checkpoint coverage:
  - `--full-models`
- Restrict to specific families:
  - `--families e1 esm2 esmplusplus`
- Select device and dtype:
  - `--device auto|cpu|cuda`
  - `--dtype auto|float32|float16|bfloat16`
- Set a custom output directory:
  - `--output-dir <path>`
- Quick wiring check without loading models:
  - `--dry-run`

### Output artifacts

Each suite writes professional artifacts to:

- Default: `test_scripts/results/<timestamp>/<suite>/`
- Files:
  - `metrics.json` (full structured metrics)
  - `metrics.csv` (tabular summary)
  - `summary.txt` (human-readable pass/fail summary)
  - `*.png` plots saved at 300 dpi

### Useful examples

- Full run with all model checkpoints:
  - `py -m test_scripts.run_all --full-models`
- Throughput benchmark on CUDA:
  - `py -m test_scripts.run_throughput --device cuda --lengths 64,128,256 --batch-sizes 1,2,4`
- Embedding validation for ESM2 only:
  - `py -m test_scripts.run_embedding --families esm2`
- Compliance checks with output directory override:
  - `py -m test_scripts.run_compliance --output-dir test_scripts/results/manual_compliance`

### Docker-first testing

Build the image:

- `docker build -t fastplms-test -f Dockerfile .`

Run tests inside the container from your checked-out repo:

- `docker run --rm --gpus all -it -v ${PWD}:/workspace fastplms-test python ...`

Inside the container (`/workspace`):

- Boltz2 compliance (3 sequences, 3 recycles, 200 diffusion steps, 1 sample):
  - `python -m test_scripts.run_boltz2_compliance --device cuda --dtype float32 --seed 42 --num-sequences 3 --recycling-steps 3 --num-sampling-steps 200 --diffusion-samples 1 --pass-coord-metric aligned --write-cif-artifacts`
- Full suite (including Boltz2 compliance):
  - `python -m test_scripts.run_all --device cuda --compliance-dtype float32`

Boltz2 compliance writes per-sequence CIF artifacts for both predictions under:
- `test_scripts/results/<timestamp>/boltz2_compliance/structures/seq_<idx>/ours_seq<idx>.cif`
- `test_scripts/results/<timestamp>/boltz2_compliance/structures/seq_<idx>/ref_seq<idx>.cif`

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
