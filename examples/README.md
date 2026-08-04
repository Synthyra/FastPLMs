# FastPLMs runnable examples

The examples are executable entry points. They do not make performance or
biological-validity claims. Curated offline examples use local Hugging Face
artifacts built from the manifest. They set both Hub offline variables, pass
`local_files_only=True`, and do not download models, tokenizers, kernels, or
runtime assets.

## Dependencies and platform requirements

FastPLMs 1.0 requires Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13.
Published models include their runtime source in the Hugging Face repository.
Install core dependencies, then load the model with
`trust_remote_code=True`:

```bash
python -m pip install \
  "torch>=2.13,<2.14" \
  "transformers>=5.13,<5.14"
```

These examples run from a source checkout. Install the required profile. Then
run them with `PYTHONPATH=src`. The CPU validation profile supports portable
examples:

```bash
uv pip install \
  -r requirements/profiles/cpu-validation.in \
  -c requirements/constraints/validation.txt \
  --torch-backend cpu
```

Core sequence examples accept `--device cpu|cuda[:index]` and
`--dtype float32|bfloat16`. The default is portable CPU FP32. Structure examples
require `requirements/features/structure.in`, verified runtime assets, and CUDA
for the published execution contract. Binder design uses
`requirements/profiles/binder.in`. FlashAttention requires
`requirements/features/flash.in`, compatible CUDA hardware, a populated pinned
kernel cache, and BF16. Fine-tuning uses `requirements/features/train.in`.
Add `requirements/features/reporting.in` only for plots and statistical reports.

For ESMFold2, select an artifact by its conditioning contract. The full
`ESMFold2` and `ESMFold2-Experimental-Cutoff2025` checkpoints have 48 folding
blocks and support optional MSA conditioning. Fast and experimental Fast
checkpoints have 24 folding blocks. They are optimized for single-sequence
inference and reject MSA-derived inputs. See Biohub
[Appendix A.2.1](https://biohub.ai/papers/esm_protein.pdf). Fast supports
supported multichain and multimolecule requests, but each protein chain uses
single-sequence mode.

## Prepare an offline artifact

Build and validate the local artifact before you disconnect the network:

```bash
PYTHONPATH=src python -m tools.artifacts.build \
  esm2_8m /cache/fast-snapshot \
  --tokenizer-dir /cache/official-tokenizer-snapshot \
  --output-root dist/hub
```

For command help, use `--help`. These are representative portable commands:

```bash
PYTHONPATH=src python examples/artifact_loading.py dist/hub/ESM2-8M --auto-class AutoModel
PYTHONPATH=src python examples/embedding_and_retrieval.py dist/hub/ESM2-8M \
  --sequence MSTNPKPQRKTKRNT --device cpu --dtype float32
PYTHONPATH=src python examples/attention_switching.py dist/hub/ESM2-8M \
  --backend sdpa --device cpu --dtype float32
PYTHONPATH=src python examples/task_heads.py dist/hub/ESM2-8M \
  --attn-backend eager --device cpu --dtype float32
```

The structure-preparation example makes an MSA-conditioned request. Use a full
ESMFold2 artifact:

```bash
PYTHONPATH=src python examples/structure_preparation.py \
  esmfold2 dist/hub/ESMFold2 --device cuda:0
```

Use Flex for a compiled path on the current GH200/aarch64 validation target:

```bash
PYTHONPATH=src python examples/attention_switching.py dist/hub/ESM2-8M \
  --backend flex_attention --device cuda:0 --dtype bfloat16
```

The CLI has explicit FlashAttention 2 and 3 choices for supported family and
platform combinations with a populated pinned kernel cache. The locked
GH200/aarch64 environment has no expected Flash kernels. Use SDPA or Flex on
that target. Do not build an unpinned Flash kernel from source. FlashAttention
2 results are historical exact-environment evidence. FlashAttention 3 is
supported but is unavailable on the locked target.

## Example inventory and evidence boundary

| Workflow | Example | Demonstrated contract | Boundary |
| --- | --- | --- | --- |
| Offline AutoClass loading | [`artifact_loading.py`](artifact_loading.py) | Load any advertised AutoClass from a local artifact | Loading only; forward/loss/save-reload are CPU contract tests |
| MLM, contacts, and task heads | [`task_heads.py`](task_heads.py) | ESM2 masked-residue scoring, trained contact head, sequence and token classification loss | Sequence and token classifiers use base weights + untrained task head unless a separately fine-tuned head is supplied |
| Ordered embeddings and retrieval | [`embedding_and_retrieval.py`](embedding_and_retrieval.py) | Repeated sequences or FASTA, mean/std pooling, safetensors or SQLite, duplicate-preserving SQLite retrieval | Full-residue, all-layer, mapping, generator, and other poolers remain shared-API examples/tests |
| Attention switching | [`attention_switching.py`](attention_switching.py) | Eager, SDPA, Flex, explicit Flash requirements, warning-emitting masked eager fallback without configuration mutation | Not a parity or throughput benchmark; the current GH200/aarch64 lock has no expected Flash kernels |
| ANKH stack selection | [`ankh_embeddings.py`](ankh_embeddings.py) | Encoder final/all layers, decoder layer with explicit prompt, deterministic seq2seq generation | The offline example accepts a validated local artifact and loads both views, so budget device memory accordingly |
| Diffusion and multimodal generation | [`generation.py`](generation.py) | Seeded DPLM, DPLM2, and conditioned ESM3 generation | One representative deterministic strategy per family |
| E1 RAG | [`e1_rag.py`](e1_rag.py) | Local A3M retrieval, ordered duplicate records, shared persistence | No remote MSA search or network fallback |
| Test-time training | [`ttt.py`](ttt.py) | Seeded update, atomic save, reset, local reload | Output must be absent and outside the source artifact |
| Structure preparation | [`structure_preparation.py`](structure_preparation.py) | Typed ESMFold2 multimolecule/MSA/modification/bond/distogram input, pocket rejection, seeded ESMFold/Boltz helpers | The MSA branch requires a full 48-block ESMFold2 variant; Fast variants reject MSA-derived inputs; tiny preparation and helper contracts are not full folding parity |
| Fine-tuning | [`fine_tuning.py`](fine_tuning.py) | ESM2 classification/regression, LoRA or full tuning, eager/SDPA/Flex selection, immutable inputs, atomic verified final artifact | LoRA is the demonstrated PEFT method; Flash training requires a separate explicit BF16 CUDA policy; other PEFT methods are not claimed by this example |
| Binder design | [`binder_design_fastplms.py`](binder_design_fastplms.py) | Differentiable ESMFold2/ESM++ optimization and critic consensus | Research prioritization only; no experimental binding claim |

The generated [capability-to-evidence manifest](../docs/generated/capability_evidence.md)
maps each curated example to required CPU, feature, structure, nightly, or
compliance evidence. If a capability is absent from this table, an example does
not imply that support only because its model class exists.

## Embedding coverage matrix

| Surface | Runnable CLI coverage | Where the remaining contract is shown |
| --- | --- | --- |
| Repeated sequence list | `--sequence` may be repeated | `embedding_and_retrieval.py` |
| FASTA streaming | `--fasta` | `embedding_and_retrieval.py` |
| Insertion-ordered mapping | Not a CLI encoding | [Embedding API](../docs/embedding_api.md) and CPU contracts |
| One-shot generator | Not a CLI encoding | [Embedding API](../docs/embedding_api.md) and CPU contracts |
| In-memory output | Omit `--output` | `embedding_and_retrieval.py` |
| Safetensors write/reopen | `--format safetensors` | Runnable example plus persistence CPU contracts |
| SQLite write/read-only filtered retrieval | `--output PATH --format sqlite --select-id ID` | Runnable example and duplicate-order CPU contracts; other `--select-id` combinations fail before loading |
| Mean and standard-deviation pooling | Always demonstrated together | `embedding_and_retrieval.py` |
| Full-residue and all-layer tensors | Not exposed by this compact CLI | [Embedding API](../docs/embedding_api.md) and ANKH example |
| Other declared poolers | Not exposed by this compact CLI | [Embedding API](../docs/embedding_api.md) and CPU contracts |

## Network and output policy

`fine_tuning.py` and `binder_design_fastplms.py` are checkpoint workflows. They
are not part of the fully offline example gate. Their shipped remote defaults
are pinned automatically. Custom remote model or dataset sources reject missing,
branch, and tag revisions. Populate each snapshot before a network-isolated run.
Local fine-tuning dataset directories must use layouts accepted by
`datasets.load_dataset`. Arbitrary `Dataset.save_to_disk()` trees are not
accepted.

Fine-tuning writes a separate task-specific child below `--output-dir`. It
records requested and effective attention backends. Binder design rejects an
existing output directory. It writes `run_manifest.json` atomically last. If
this file is absent, the run is incomplete. Keep the complete directory for
reproducibility.

The CPU gate runs CLI wiring and dependency-free preparation with small local
artifacts. Full checkpoints, optimized kernels, GPU parity, structure
prediction, and throughput are in the feature, nightly, compliance, structure,
and benchmark tiers.
