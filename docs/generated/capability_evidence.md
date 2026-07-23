<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Capability-to-evidence manifest

This manifest maps every advertised FastPLMs 1.0 capability to its user
documentation, runnable example, and required validation tier. It is a
coverage contract, not a statement that an unreported run passed. The exact
checkpoint list and family declarations come from `src/fastplms/models.toml`.

The Example column links a curated CLI when that interface exposes the whole
capability. Programmatic-only forms instead link their runnable CPU contract so
the manifest does not imply broader CLI coverage than the example provides.

## Frozen ESMC release evidence

**Status: pending.** Default documentation generation never discovers or
trusts reports implicitly. Release rendering requires an explicitly selected,
complete schema-v3 set of exactly 30 records on one exact GH200/aarch64
target: 18 measured eager, SDPA, and Flex records plus 12 structured
FlashAttention 2/3 unavailable records across three checkpoints and two
immutable sequence panels.
The set must also carry the final candidate/reference image identities,
dependency lock, installed inventory, and official source attestations.
A partial, stale, malformed, self-digest-invalid, or cross-device set fails
closed and cannot replace this status.

### Locked oracle package compatibility exception

The frozen oracle lock permits exactly one nonzero `pip check` diagnostic:
`nvidia-cusparselt-cu13 0.8.1 is not supported on this platform`. It applies only to
`nvidia-cusparselt-cu13==0.8.1` on
`NVIDIA GH200 480GB` / `linux` /
`aarch64`. The vendor filename tag is
`py3-none-manylinux2014_aarch64`, while the wheel metadata declares
`py3-none-manylinux2014_sbsa`. The exact wheel is
`nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl` with SHA-256 `4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f`.
FastPLMs accepts this vendor metadata mismatch only after the lock, installed
inventory, wheel bytes, metadata tag, and target identity all match. The wheel
is not rewritten (`validated-vendor-metadata-exception-no-wheel-rewrite`). Any additional diagnostic or
identity drift fails closed.

## Executable evidence selectors

Only the selectors below are claimed. Their scopes are intentionally narrower
than a whole family or validation tier. A tier appearing on another row does not
automatically apply to the capability in this row.

| Selector | Tier/job | Executable target | Scope |
| --- | --- | --- | --- |
| `cpu:autoclass-runtime` | `cpu_contract` | `tests/cpu/test_autoclass_evidence_matrix.py::test_autoclass_runtime_evidence_matrix_exactly_matches_all_37_entries`<br>`tests/cpu/test_autoclass_evidence_matrix.py::test_autoclass_runtime_evidence_targets_are_collected_cpu_tests` | Every family-level AutoClass entry and its explicit tiny runtime contracts. |
| `artifact:checkpoint-autoclasses` | `artifact` | `tests/release/test_published_automodel.py::test_local_artifact_offline_autoclass_parity` | Every advertised AutoClass for every built checkpoint, grouped by checkpoint. |
| `compliance:sequence-primary-head` | `compliance` | `tests/parity/test_native_results.py::test_native_exact_checkpoint_contract`<br>`tests/parity/test_native_results.py::test_native_every_checkpoint_bf16_inference` | The official-parity head only: AutoModel for ANKH or a family without MaskedLM; otherwise AutoModelForMaskedLM. |
| `compliance:ankh-seq2seq` | `compliance` | `tests/parity/test_native_results.py::test_native_ankh_explicit_decoder_prompt_generation` | ANKH AutoModelForSeq2SeqLM explicit-prompt generation only. |
| `compliance:structure-automodel` | `compliance` | `tests/structure/test_esmfold_folding_compliance.py`<br>`tests/structure/test_esmfold2_folding_compliance.py` | ESMFold and ESMFold2 AutoModel folding paths only. |
| `benchmark:claim-eligible-primary-head` | `benchmark` | `benchmarks/suite.py::benchmark_cases[claim_eligible=True]` | The benchmark-selected head for representative sequence checkpoints and ESMFold2 projection cases; startup and embedding cases are excluded. |
| `cpu:attention-contracts` | `cpu_contract` | `tests/cpu/test_attention_contracts.py` | Portable dispatch, masks, fallback, fake FA2/FA3, ESMC Flex/FA3, and eager/SDPA gradient contracts. |
| `nightly:sequence-backends` | `nightly` | `tests/integration/test_backend_consistency.py` | Current GH200 eager, SDPA, and Flex forward/backward paths. Flash kernels are not downloaded, built, or executed in the current locked environment. |
| `historical:fa2-focused` | `historical` | `tools/remote/run.py::_kernel_capability_preflight` | Policy records prior real FlashAttention 2 focused execution, but the immutable execution report is not bundled in this repository and no current GH200 numerical claim is inferred from it. |
| `compliance:flash-unavailable-gh200` | `compliance` | `tests/parity/test_native_results.py::test_esmc_bf16_calibration_and_biological_holdout` | Complete report-bound FA2/FA3 unavailability records and fail-closed dispatch on the frozen release environment. |
| `compliance:deep-backends` | `compliance` | `tests/parity/test_native_results.py::test_native_representatives_all_backends` | Every advertised backend on the pinned deep sequence representative per family. |
| `benchmark:claim-eligible-backends` | `benchmark` | `benchmarks/suite.py::benchmark_cases[claim_eligible=True]` | Backends emitted by claim-eligible sequence and ESMFold2 benchmark cases. |
| `cpu:embedding-contracts` | `cpu_contract` | `tests/cpu/test_embedding_contracts.py` | Ordered inputs, biological masking, pooling, streaming, and persistence. |
| `cpu:e1-embeddings` | `cpu_contract` | `tests/cpu/test_e1_contracts.py` | E1 raw-sequence and MSA embedding persistence. |
| `feature:e1-rag` | `feature` | `tests/integration/test_e1_rag.py` | E1 retrieval, MSA preparation, cache, scoring, and embedding flows. |
| `cpu:ankh-contracts` | `cpu_contract` | `tests/cpu/test_ankh_contracts.py` | ANKH encoder and explicit-decoder embeddings, layers, masks, and T5 views. |
| `cpu:generation-contracts` | `cpu_contract` | `tests/cpu/test_generation_contracts.py` | Tiny deterministic DPLM, DPLM2, and ESM3 generation contracts. |
| `feature:generation` | `feature` | `tests/integration/test_dplm_generation.py`<br>`tests/integration/test_esm3.py` | DPLM, DPLM2, and ESM3 generation behavior in the feature suite. |
| `cpu:peft` | `cpu_contract` | `tests/cpu/test_peft_contracts.py` | Real initializer, collators, one optimizer step, and adapter/classifier reload. |
| `nightly:peft` | `nightly` | `tests/unit/test_fine_tuning_example.py` | Fine-tuning example contracts in the nightly feature job. |
| `cpu:ttt` | `cpu_contract` | `tests/cpu/test_ttt_contracts.py` | Seeded TTT initialization, update, reset, save, reload, and family isolation. |
| `feature:ttt` | `feature` | `tests/integration/test_ttt.py` | TTT integration behavior in the feature suite. |
| `cpu:structure-contracts` | `cpu_contract` | `tests/cpu/test_structure_contracts.py` | Tiny injected structure cores, public outputs, save/reload, and binder batching. |
| `structure:public-contracts` | `structure` | `tests/structure/test_structure_public_helpers.py` | Seeded Boltz helper, linker masking, real features, losses, and binder gradients. |
| `structure:full-suite` | `structure` | `tests/structure` | The declared GPU structure suite for folding and preparation behavior. |
| `feature:binder` | `feature` | `tests/integration/test_binder_design.py` | Seeded binder workflow, atom padding, critic ranking, and traceability. |
| `cpu:artifact-example` | `cpu_contract` | `tests/cpu/test_documentation_contracts.py::test_artifact_loading_example_executes_local_only_autoconfig` | The offline local-artifact example with AutoConfig. |
| `cpu:task-head-example` | `cpu_contract` | `tests/cpu/test_documentation_contracts.py::test_task_head_example_executes_all_advertised_heads_offline` | Offline ESM2 masked-LM scoring, contacts, sequence classification, and token classification through the documented example. |

## Curated offline example execution

Every curated example is routed to the exact collected CPU test nodes below.
These tests run under the required offline `cpu_contract` gate.

| Example | Tier | Exact executable CPU node |
| --- | --- | --- |
| [`embedding_and_retrieval.py`](../../examples/embedding_and_retrieval.py) | `cpu_contract` | `tests/cpu/test_documentation_contracts.py::test_embedding_and_retrieval_example_executes_with_ordered_sqlite` |
| [`attention_switching.py`](../../examples/attention_switching.py) | `cpu_contract` | `tests/cpu/test_documentation_contracts.py::test_attention_switching_main_executes_optimized_and_masked_fallback` |
| [`ankh_embeddings.py`](../../examples/ankh_embeddings.py) | `cpu_contract` | `tests/cpu/test_documentation_contracts.py::test_ankh_embedding_example_executes_encoder_and_decoder_layers` |
| [`generation.py`](../../examples/generation.py) | `cpu_contract` | `tests/cpu/test_documentation_contracts.py::test_generation_example_executes_seeded_dplm_branch_offline` |
| [`generation.py`](../../examples/generation.py) | `cpu_contract` | `tests/cpu/test_documentation_contracts.py::test_generation_example_executes_seeded_dplm2_branch_offline` |
| [`generation.py`](../../examples/generation.py) | `cpu_contract` | `tests/cpu/test_documentation_contracts.py::test_generation_example_executes_seeded_esm3_trace` |
| [`e1_rag.py`](../../examples/e1_rag.py) | `cpu_contract` | `tests/cpu/test_documentation_contracts.py::test_e1_rag_example_executes_local_msa_and_shared_persistence` |
| [`ttt.py`](../../examples/ttt.py) | `cpu_contract` | `tests/cpu/test_documentation_contracts.py::test_ttt_example_executes_seeded_adapt_save_and_reset` |
| [`structure_preparation.py`](../../examples/structure_preparation.py) | `cpu_contract` | `tests/cpu/test_documentation_contracts.py::test_structure_preparation_example_executes_each_public_branch` |
| [`artifact_loading.py`](../../examples/artifact_loading.py) | `cpu_contract` | `tests/cpu/test_documentation_contracts.py::test_artifact_loading_example_executes_local_only_autoconfig` |
| [`task_heads.py`](../../examples/task_heads.py) | `cpu_contract` | `tests/cpu/test_documentation_contracts.py::test_task_head_example_executes_all_advertised_heads_offline` |
| [`fine_tuning.py`](../../examples/fine_tuning.py) | `cpu_contract` | `tests/cpu/test_peft_contracts.py::test_fine_tuning_main_wires_both_tasks_without_external_io` |
| [`fine_tuning.py`](../../examples/fine_tuning.py) | `cpu_contract` | `tests/cpu/test_peft_contracts.py::test_shipped_collators_create_tokenizer_aware_sequence_and_pair_batches` |
| [`fine_tuning.py`](../../examples/fine_tuning.py) | `cpu_contract` | `tests/cpu/test_peft_contracts.py::test_shipped_initializer_drives_one_peft_step_and_atomic_final_reload` |
| [`binder_design_fastplms.py`](../../examples/binder_design_fastplms.py) | `cpu_contract` | `tests/cpu/test_structure_contracts.py::test_public_binder_workflow_pads_heterogeneous_prepared_atoms_without_truncation` |
| [`binder_design_fastplms.py`](../../examples/binder_design_fastplms.py) | `cpu_contract` | `tests/cpu/test_structure_contracts.py::test_binder_example_main_wires_explicit_offline_cli_arguments` |
| [`binder_design_fastplms.py`](../../examples/binder_design_fastplms.py) | `cpu_contract` | `tests/cpu/test_structure_contracts.py::test_binder_structure_loss_is_finite_and_differentiable` |

## Families and AutoClasses

| Family | Tokenizer mode | AutoClass | Weight status | Guide | Family workflow and runnable entry-point contract | Required evidence |
| --- | --- | --- | --- | --- | --- | --- |
| `esm2` | `tokenizer` | `AutoConfig` | `FastPLMs extension` | [guide](../models.md#esm2) | [family workflow](../../examples/embedding_and_retrieval.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `esm2` | `tokenizer` | `AutoModel` | `pretrained` | [guide](../models.md#esm2) | [family workflow](../../examples/embedding_and_retrieval.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `esm2` | `tokenizer` | `AutoModelForMaskedLM` | `pretrained` | [guide](../models.md#esm2) | [family workflow](../../examples/task_heads.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses`, `cpu:task-head-example`, `compliance:sequence-primary-head`, `benchmark:claim-eligible-primary-head` |
| `esm2` | `tokenizer` | `AutoModelForSequenceClassification` | `base weights + untrained task head` | [guide](../models.md#esm2) | [family workflow](../../examples/task_heads.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses`, `cpu:task-head-example` |
| `esm2` | `tokenizer` | `AutoModelForTokenClassification` | `base weights + untrained task head` | [guide](../models.md#esm2) | [family workflow](../../examples/task_heads.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses`, `cpu:task-head-example` |
| `esm_plusplus` | `tokenizer` | `AutoConfig` | `FastPLMs extension` | [guide](../models.md#esm-and-esmc) | [family workflow](../../examples/attention_switching.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `esm_plusplus` | `tokenizer` | `AutoModel` | `pretrained` | [guide](../models.md#esm-and-esmc) | [family workflow](../../examples/attention_switching.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `esm_plusplus` | `tokenizer` | `AutoModelForMaskedLM` | `pretrained` | [guide](../models.md#esm-and-esmc) | [family workflow](../../examples/attention_switching.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses`, `compliance:sequence-primary-head`, `benchmark:claim-eligible-primary-head` |
| `esm3` | `tokenizer` | `AutoConfig` | `FastPLMs extension` | [guide](../models.md#esm3) | [family workflow](../../examples/generation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `esm3` | `tokenizer` | `AutoModel` | `pretrained` | [guide](../models.md#esm3) | [family workflow](../../examples/generation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses`, `compliance:sequence-primary-head`, `benchmark:claim-eligible-primary-head` |
| `e1` | `sequence` | `AutoConfig` | `FastPLMs extension` | [guide](../models.md#e1) | [family workflow](../../examples/e1_rag.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `e1` | `sequence` | `AutoModel` | `pretrained` | [guide](../models.md#e1) | [family workflow](../../examples/e1_rag.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `e1` | `sequence` | `AutoModelForMaskedLM` | `pretrained` | [guide](../models.md#e1) | [family workflow](../../examples/e1_rag.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses`, `compliance:sequence-primary-head`, `benchmark:claim-eligible-primary-head` |
| `e1` | `sequence` | `AutoModelForSequenceClassification` | `base weights + untrained task head` | [guide](../models.md#e1) | [family workflow](../../examples/e1_rag.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `e1` | `sequence` | `AutoModelForTokenClassification` | `base weights + untrained task head` | [guide](../models.md#e1) | [family workflow](../../examples/e1_rag.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `dplm` | `tokenizer` | `AutoConfig` | `FastPLMs extension` | [guide](../models.md#dplm) | [family workflow](../../examples/generation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `dplm` | `tokenizer` | `AutoModel` | `pretrained` | [guide](../models.md#dplm) | [family workflow](../../examples/generation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `dplm` | `tokenizer` | `AutoModelForMaskedLM` | `pretrained` | [guide](../models.md#dplm) | [family workflow](../../examples/generation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses`, `compliance:sequence-primary-head`, `benchmark:claim-eligible-primary-head` |
| `dplm` | `tokenizer` | `AutoModelForSequenceClassification` | `base weights + untrained task head` | [guide](../models.md#dplm) | [family workflow](../../examples/generation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `dplm` | `tokenizer` | `AutoModelForTokenClassification` | `base weights + untrained task head` | [guide](../models.md#dplm) | [family workflow](../../examples/generation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `dplm2` | `tokenizer` | `AutoConfig` | `FastPLMs extension` | [guide](../models.md#dplm2) | [family workflow](../../examples/generation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `dplm2` | `tokenizer` | `AutoModel` | `pretrained` | [guide](../models.md#dplm2) | [family workflow](../../examples/generation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `dplm2` | `tokenizer` | `AutoModelForMaskedLM` | `pretrained` | [guide](../models.md#dplm2) | [family workflow](../../examples/generation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses`, `compliance:sequence-primary-head`, `benchmark:claim-eligible-primary-head` |
| `dplm2` | `tokenizer` | `AutoModelForSequenceClassification` | `base weights + untrained task head` | [guide](../models.md#dplm2) | [family workflow](../../examples/generation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `dplm2` | `tokenizer` | `AutoModelForTokenClassification` | `base weights + untrained task head` | [guide](../models.md#dplm2) | [family workflow](../../examples/generation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `ankh` | `tokenizer` | `AutoConfig` | `FastPLMs extension` | [guide](../models.md#ankh) | [family workflow](../../examples/ankh_embeddings.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `ankh` | `tokenizer` | `AutoModel` | `pretrained` | [guide](../models.md#ankh) | [family workflow](../../examples/ankh_embeddings.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses`, `compliance:sequence-primary-head`, `benchmark:claim-eligible-primary-head` |
| `ankh` | `tokenizer` | `AutoModelForMaskedLM` | `FastPLMs extension` | [guide](../models.md#ankh) | [family workflow](../../examples/ankh_embeddings.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `ankh` | `tokenizer` | `AutoModelForSeq2SeqLM` | `pretrained` | [guide](../models.md#ankh) | [family workflow](../../examples/ankh_embeddings.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses`, `compliance:ankh-seq2seq` |
| `ankh` | `tokenizer` | `AutoModelForSequenceClassification` | `base weights + untrained task head` | [guide](../models.md#ankh) | [family workflow](../../examples/ankh_embeddings.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `ankh` | `tokenizer` | `AutoModelForTokenClassification` | `base weights + untrained task head` | [guide](../models.md#ankh) | [family workflow](../../examples/ankh_embeddings.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `boltz2` | `structure` | `AutoConfig` | `FastPLMs extension` | [guide](../models.md#boltz2) | [family workflow](../../examples/structure_preparation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `boltz2` | `structure` | `AutoModel` | `pretrained` | [guide](../models.md#boltz2) | [family workflow](../../examples/structure_preparation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `esmfold` | `structure` | `AutoConfig` | `FastPLMs extension` | [guide](../models.md#esmfold) | [family workflow](../../examples/structure_preparation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `esmfold` | `structure` | `AutoModel` | `pretrained` | [guide](../models.md#esmfold) | [family workflow](../../examples/structure_preparation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses`, `compliance:structure-automodel` |
| `esmfold2` | `structure` | `AutoConfig` | `FastPLMs extension` | [guide](../esmfold2.md) | [family workflow](../../examples/structure_preparation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses` |
| `esmfold2` | `structure` | `AutoModel` | `pretrained` | [guide](../esmfold2.md) | [family workflow](../../examples/structure_preparation.py); [runnable AutoClass contract](../../tests/cpu/test_autoclass_evidence_matrix.py) | `cpu:autoclass-runtime`, `artifact:checkpoint-autoclasses`, `compliance:structure-automodel`, `benchmark:claim-eligible-primary-head` |

## Attention backends

| Backend | Advertising families | Guide | Example | Required evidence |
| --- | --- | --- | --- | --- |
| `eager` | `ankh`, `boltz2`, `dplm`, `esm2`, `esm3`, `esm_plusplus`, `esmfold`, `esmfold2` | [guide](../attention_backends.md) | [example](../../examples/attention_switching.py) | `cpu:attention-contracts`, `nightly:sequence-backends`, `compliance:deep-backends`, `benchmark:claim-eligible-backends` |
| `flash_attention_2` | `esm2`, `esm_plusplus` | [guide](../attention_backends.md) | [example](../../examples/attention_switching.py) | `cpu:attention-contracts`, `historical:fa2-focused`, `compliance:flash-unavailable-gh200` |
| `flash_attention_3` | `dplm`, `esm2`, `esm_plusplus` | [guide](../attention_backends.md) | [example](../../examples/attention_switching.py) | `cpu:attention-contracts`, `compliance:flash-unavailable-gh200` |
| `flex_attention` | `dplm`, `e1`, `esm2`, `esm3`, `esm_plusplus`, `esmfold`, `esmfold2` | [guide](../attention_backends.md) | [example](../../examples/attention_switching.py) | `cpu:attention-contracts`, `nightly:sequence-backends`, `compliance:deep-backends`, `benchmark:claim-eligible-backends` |
| `sdpa` | `ankh`, `dplm`, `dplm2`, `e1`, `esm2`, `esm3`, `esm_plusplus`, `esmfold`, `esmfold2` | [guide](../attention_backends.md) | [example](../../examples/attention_switching.py) | `cpu:attention-contracts`, `nightly:sequence-backends`, `compliance:deep-backends`, `benchmark:claim-eligible-backends` |

## Input, embedding, and storage contracts

| Capability | Guide | Example | Required evidence |
| --- | --- | --- | --- |
| Sequence list or streaming FASTA | [embedding API](../embedding_api.md) | [embedding and retrieval](../../examples/embedding_and_retrieval.py) | `cpu:embedding-contracts` |
| Ordered mapping or one-shot generator | [embedding API](../embedding_api.md) | [runnable API contracts](../../tests/cpu/test_embedding_contracts.py) | `cpu:embedding-contracts` |
| Biological-residue `max_length`, bounded token windows, and stable order | [embedding API](../embedding_api.md#bounded-streaming-and-length-policy) | [runnable API contracts](../../tests/cpu/test_embedding_contracts.py) | `cpu:embedding-contracts` |
| Mean and standard-deviation pooling | [embedding API](../embedding_api.md#pooling) | [embedding and retrieval](../../examples/embedding_and_retrieval.py) | `cpu:embedding-contracts` |
| Max/norm/median/variance/CLS/PARTI pooling | [embedding API](../embedding_api.md#pooling) | [runnable pooler contract](../../tests/cpu/test_embedding_contracts.py) | `cpu:embedding-contracts` |
| Full-residue and all-selected-layer output | [embedding API](../embedding_api.md#full-residue-embeddings) | [ANKH layers](../../examples/ankh_embeddings.py) | `cpu:embedding-contracts`, `cpu:ankh-contracts` |
| Transactional sharded safetensors and exact resume | [embedding API](../embedding_api.md#safetensors-storage) | [embedding and retrieval](../../examples/embedding_and_retrieval.py) | `cpu:embedding-contracts` |
| Read-only SQLite and ordered duplicate-preserving filters | [embedding API](../embedding_api.md#sqlite-streaming-retrieval-and-resume) | [embedding and retrieval](../../examples/embedding_and_retrieval.py) | `cpu:embedding-contracts` |
| Legacy SQLite conversion without pickle deserialization | [embedding API](../embedding_api.md#sqlite-streaming-retrieval-and-resume) | [runnable converter contract](../../tests/cpu/test_embedding_contracts.py) | `cpu:embedding-contracts` |
| E1 raw-sequence and MSA-aware ordered embeddings | [E1 guide](../models.md#e1) | [E1 RAG](../../examples/e1_rag.py) | `cpu:e1-embeddings`, `feature:e1-rag` |
| ANKH encoder/explicit-decoder hidden-state selection | [ANKH guide](../models.md#ankh) | [ANKH layers](../../examples/ankh_embeddings.py) | `cpu:ankh-contracts` |

## Generation and adaptation contracts

| Capability | Guide | Example | Required evidence |
| --- | --- | --- | --- |
| ESM2 pretrained masked-LM scoring and contact prediction | [ESM2](../models.md#esm2) | [task heads](../../examples/task_heads.py) | `cpu:task-head-example`, `cpu:autoclass-runtime` |
| ESM2 sequence/token classification with explicitly untrained task heads | [ESM2](../models.md#esm2) | [task heads](../../examples/task_heads.py) | `cpu:task-head-example`, `cpu:autoclass-runtime` |
| DPLM amino-acid diffusion generation | [DPLM](../models.md#dplm) | [generation](../../examples/generation.py) | `cpu:generation-contracts`, `feature:generation` |
| DPLM2 modality-aware sequence/structure co-generation | [DPLM2](../models.md#dplm2) | [generation](../../examples/generation.py) | `cpu:generation-contracts`, `feature:generation` |
| ESM3 multimodal-conditioned generation | [ESM3](../models.md#esm3) | [generation](../../examples/generation.py) | `cpu:generation-contracts`, `feature:generation` |
| ANKH task-prompted sequence-to-sequence generation | [ANKH](../models.md#ankh) | [ANKH embeddings and generation](../../examples/ankh_embeddings.py) | `cpu:ankh-contracts`, `compliance:ankh-seq2seq` |
| Trainer/PEFT LoRA with immutable inputs and verified save/reload | [fine-tuning](../finetuning.md) | [fine-tuning](../../examples/fine_tuning.py) | `cpu:peft`, `nightly:peft` |
| Seeded TTT adapter initialize/update/reset/save/reload | [TTT](../ttt.md) | [TTT](../../examples/ttt.py) | `cpu:ttt`, `feature:ttt` |

## Structure contracts

| Capability | Guide | Example | Required evidence |
| --- | --- | --- | --- |
| ESMFold single-chain folding and multimer-linker confidence masking | [models](../models.md#esmfold) | [structure preparation](../../examples/structure_preparation.py) | `cpu:structure-contracts`, `structure:public-contracts`, `structure:full-suite`, `compliance:structure-automodel` |
| Seed-scoped Boltz2 protein helper and BF16 execution policy | [Boltz2](../models.md#boltz2) | [structure preparation](../../examples/structure_preparation.py) | `cpu:structure-contracts`, `structure:public-contracts`, `structure:full-suite` |
| Atom-dense binder optimization and critic reporting | [binder design](../binder_design.md) | [binder design](../../examples/binder_design_fastplms.py) | `cpu:structure-contracts`, `structure:public-contracts`, `feature:binder` |
| Offline local artifact AutoClass loading | [artifacts](../artifacts.md) | [artifact loading](../../examples/artifact_loading.py) | `cpu:artifact-example`, `artifact:checkpoint-autoclasses` |
| `esmfold2` 48-block full ESMFold2: single-sequence or optional MSA-conditioned protein inputs, typed complexes, ligands, nucleic acids, modifications, bonds, and distograms; pocket requests fail closed | [ESMFold2](../esmfold2.md) | [structure preparation](../../examples/structure_preparation.py) | `cpu:structure-contracts`, `structure:full-suite`, `compliance:structure-automodel` |
| `esmfold2_fast` 24-block Fast ESMFold2: inference-optimized single-sequence conditioning with typed multichain and multimolecule inputs; every protein must have `msa=None` and MSA inputs fail closed | [ESMFold2](../esmfold2.md) | [structure preparation](../../examples/structure_preparation.py) | `cpu:structure-contracts`, `structure:full-suite`, `compliance:structure-automodel` |
| `esmfold2_experimental_cutoff2025` 48-block full ESMFold2: single-sequence or optional MSA-conditioned protein inputs, typed complexes, ligands, nucleic acids, modifications, bonds, and distograms; pocket requests fail closed | [ESMFold2](../esmfold2.md) | [structure preparation](../../examples/structure_preparation.py) | `cpu:structure-contracts`, `structure:full-suite`, `compliance:structure-automodel` |
| `esmfold2_experimental_fast_cutoff2025` 24-block Fast ESMFold2: inference-optimized single-sequence conditioning with typed multichain and multimolecule inputs; every protein must have `msa=None` and MSA inputs fail closed | [ESMFold2](../esmfold2.md) | [structure preparation](../../examples/structure_preparation.py) | `cpu:structure-contracts`, `structure:full-suite`, `compliance:structure-automodel` |

Release evidence must name the exact head, checkpoint and runtime revisions,
tokenizer identity, backend, dtype, hardware, sequence or structure panel,
seed, environment, and input hash. Missing evidence remains visibly pending;
it must not be replaced by a synthetic benchmark number or an inferred claim.
