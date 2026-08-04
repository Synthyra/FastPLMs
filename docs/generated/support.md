<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Model support

This file is generated from `src/fastplms/models.toml`. A listed capability is
selectable. Strict-parity exceptions are documented in the checkpoint cards.

## Family interfaces

| Family | Architecture | Checkpoints | Public input | AutoClasses | Tokenizer class |
| --- | --- | ---: | --- | --- | --- |
| `esm2` | ESM2 | 5 | Amino-acid sequences tokenized to residue IDs | `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification` | `n/a` |
| `esm_plusplus` | ESMC | 3 | Amino-acid sequences tokenized to residue IDs | `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM` | `n/a` |
| `esm3` | ESM3 | 1 | Sequence, structure, and function tracks prepared through the multimodal helpers | `AutoConfig`, `AutoModel` | `n/a` |
| `e1` | E1 | 3 | Raw amino-acid sequences prepared by the native E1 adapter | `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification` | `n/a` |
| `dplm` | DPLM | 3 | Amino-acid sequences tokenized to masked or partially masked residue IDs | `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification` | `n/a` |
| `dplm2` | DPLM2 | 3 | Tokenized amino-acid and structure tracks with explicit modality boundaries | `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification` | `fastplms.models.dplm2.tokenization_dplm2.DPLM2Tokenizer` |
| `ankh` | ANKH | 5 | Amino-acid sequences tokenized for encoder or sequence-to-sequence use | `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSeq2SeqLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification` | `n/a` |
| `boltz2` | Boltz2 | 1 | Raw amino-acid sequences through the convenience API, or prepared model features | `AutoConfig`, `AutoModel` | `n/a` |
| `esmfold` | ESMFold | 1 | Raw amino-acid sequences through folding helpers, or prepared residue tensors | `AutoConfig`, `AutoModel` | `n/a` |
| `esmfold2` | ESMFold2 | 4 | Raw amino-acid sequences or typed molecular-complex specifications; low-level forward accepts prepared feature tensors | `AutoConfig`, `AutoModel` | `n/a` |

## AutoClass weight status

`pretrained` means the advertised head is present in the checkpoint. `base weights + untrained task head` means the task head must be trained before use. `FastPLMs extension` is an integration or head that is not an official pretrained ANKH capability.

| Family | AutoClass | Weight status |
| --- | --- | --- |
| `esm2` | `AutoConfig` | `FastPLMs extension` |
| `esm2` | `AutoModel` | `pretrained` |
| `esm2` | `AutoModelForMaskedLM` | `pretrained` |
| `esm2` | `AutoModelForSequenceClassification` | `base weights + untrained task head` |
| `esm2` | `AutoModelForTokenClassification` | `base weights + untrained task head` |
| `esm_plusplus` | `AutoConfig` | `FastPLMs extension` |
| `esm_plusplus` | `AutoModel` | `pretrained` |
| `esm_plusplus` | `AutoModelForMaskedLM` | `pretrained` |
| `esm3` | `AutoConfig` | `FastPLMs extension` |
| `esm3` | `AutoModel` | `pretrained` |
| `e1` | `AutoConfig` | `FastPLMs extension` |
| `e1` | `AutoModel` | `pretrained` |
| `e1` | `AutoModelForMaskedLM` | `pretrained` |
| `e1` | `AutoModelForSequenceClassification` | `base weights + untrained task head` |
| `e1` | `AutoModelForTokenClassification` | `base weights + untrained task head` |
| `dplm` | `AutoConfig` | `FastPLMs extension` |
| `dplm` | `AutoModel` | `pretrained` |
| `dplm` | `AutoModelForMaskedLM` | `pretrained` |
| `dplm` | `AutoModelForSequenceClassification` | `base weights + untrained task head` |
| `dplm` | `AutoModelForTokenClassification` | `base weights + untrained task head` |
| `dplm2` | `AutoConfig` | `FastPLMs extension` |
| `dplm2` | `AutoModel` | `pretrained` |
| `dplm2` | `AutoModelForMaskedLM` | `pretrained` |
| `dplm2` | `AutoModelForSequenceClassification` | `base weights + untrained task head` |
| `dplm2` | `AutoModelForTokenClassification` | `base weights + untrained task head` |
| `ankh` | `AutoConfig` | `FastPLMs extension` |
| `ankh` | `AutoModel` | `pretrained` |
| `ankh` | `AutoModelForMaskedLM` | `FastPLMs extension` |
| `ankh` | `AutoModelForSeq2SeqLM` | `pretrained` |
| `ankh` | `AutoModelForSequenceClassification` | `base weights + untrained task head` |
| `ankh` | `AutoModelForTokenClassification` | `base weights + untrained task head` |
| `boltz2` | `AutoConfig` | `FastPLMs extension` |
| `boltz2` | `AutoModel` | `pretrained` |
| `esmfold` | `AutoConfig` | `FastPLMs extension` |
| `esmfold` | `AutoModel` | `pretrained` |
| `esmfold2` | `AutoConfig` | `FastPLMs extension` |
| `esmfold2` | `AutoModel` | `pretrained` |

## Family execution

| Family | Attention | Precision | BF16 execution | Extra | Reference |
| --- | --- | --- | --- | --- | --- |
| `esm2` | `eager`, `sdpa`, `flex_attention`, `flash_attention_2`, `flash_attention_3` | `default` | `fp32_parameters_autocast` | `core` | `reference-esm2` |
| `esm_plusplus` | `eager`, `sdpa`, `flex_attention`, `flash_attention_2`, `flash_attention_3` | `default`, `fp8` (experimental) | `static_parameters` | `core` | `reference-biohub-esm` |
| `esm3` | `eager`, `sdpa`, `flex_attention` | `default` | `fp32_parameters_autocast` | `core` | `reference-biohub-esm` |
| `e1` | `sdpa`, `flex_attention` | `default` | `static_parameters` | `core` | `reference-e1` |
| `dplm` | `eager`, `sdpa`, `flex_attention`, `flash_attention_3` | `default` | `fp32_parameters_autocast` | `core` | `reference-dplm` |
| `dplm2` | `sdpa` | `default` | `fp32_parameters_autocast` | `core` | `reference-dplm` |
| `ankh` | `eager`, `sdpa` | `default` | `static_parameters` | `core` | `reference-ankh` |
| `boltz2` | `eager` | `default` | `fp32_parameters_autocast` | `structure` | `reference-boltz2` |
| `esmfold` | `eager`, `sdpa`, `flex_attention` | `default` | `fp32_parameters_autocast` | `structure` | `reference-esmfold` |
| `esmfold2` | `eager`, `sdpa`, `flex_attention` | `auto`, `fp32`, `bf16`, `fp8` (experimental) | `fp32_parameters_autocast` | `structure` | `reference-esmfold2` |

## Family release contracts

| Family | Checkpoint terms | Hub license | Weight publication | Tiers |
| --- | --- | --- | --- | --- |
| `esm2` | MIT | `mit` | manifest policy | `check`, `compliance`, `feature`, `artifact`, `benchmark` |
| `esm_plusplus` | MIT | `mit` | manifest policy | `check`, `compliance`, `feature`, `artifact`, `benchmark` |
| `esm3` | MIT | `mit` | manifest policy | `check`, `compliance`, `feature`, `artifact`, `benchmark` |
| `e1` | Profluent-E1-Agreement | `other` ([Profluent-E1 Clickthrough License Agreement](https://github.com/Profluent-AI/E1/blob/main/LICENSE)) | manifest policy | `check`, `compliance`, `feature`, `artifact`, `benchmark` |
| `dplm` | Apache-2.0 | `apache-2.0` | manifest policy | `check`, `compliance`, `feature`, `artifact`, `benchmark` |
| `dplm2` | Apache-2.0 | `apache-2.0` | manifest policy | `check`, `compliance`, `feature`, `artifact`, `benchmark` |
| `ankh` | CC-BY-NC-SA-4.0 | `cc-by-nc-sa-4.0` | manifest policy | `check`, `compliance`, `feature`, `artifact`, `benchmark` |
| `boltz2` | MIT | `mit` | manifest policy | `structure`, `artifact`, `benchmark` |
| `esmfold` | MIT | `mit` | manifest policy | `check`, `compliance`, `structure`, `feature`, `artifact`, `benchmark` |
| `esmfold2` | MIT | `mit` | manifest policy | `check`, `compliance`, `structure`, `feature`, `artifact`, `benchmark` |

## Runtime assets

| ID | Family | Repository | Path | SHA-256 | Size | License | Trust boundary | Offline behavior |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| `esmfold2_ccd` | `esmfold2` | `biohub/ESMFold2` | `ccd.pkl` | `9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5` | 417306584 | `MIT` | `hash_pinned_pickle` | `requires_cached_verified_file` |

## Checkpoints

| ID | Family | Size | FastPLMs checkpoint | Official checkpoint | Artifact source | State transform | Generation contract | MSA conditioning | Unresolved files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `esm2_8m` | `esm2` | `small` | [Synthyra/ESM2-8M](https://huggingface.co/Synthyra/ESM2-8M) | [facebook/esm2_t6_8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D) | `fast` | `esm2_hf_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esm2_35m` | `esm2` | `small` | [Synthyra/ESM2-35M](https://huggingface.co/Synthyra/ESM2-35M) | [facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D) | `fast` | `esm2_hf_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esm2_150m` | `esm2` | `medium` | [Synthyra/ESM2-150M](https://huggingface.co/Synthyra/ESM2-150M) | [facebook/esm2_t30_150M_UR50D](https://huggingface.co/facebook/esm2_t30_150M_UR50D) | `fast` | `esm2_hf_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esm2_650m` | `esm2` | `large` | [Synthyra/ESM2-650M](https://huggingface.co/Synthyra/ESM2-650M) | [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D) | `fast` | `esm2_hf_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esm2_3b` | `esm2` | `xlarge` | [Synthyra/ESM2-3B](https://huggingface.co/Synthyra/ESM2-3B) | [facebook/esm2_t36_3B_UR50D](https://huggingface.co/facebook/esm2_t36_3B_UR50D) | `fast` | `esm2_hf_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esmc_small` | `esm_plusplus` | `medium` | [Synthyra/ESMplusplus_small](https://huggingface.co/Synthyra/ESMplusplus_small) | [biohub/ESMC-300M](https://huggingface.co/biohub/ESMC-300M) | `fast` | `esmc_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esmc_large` | `esm_plusplus` | `large` | [Synthyra/ESMplusplus_large](https://huggingface.co/Synthyra/ESMplusplus_large) | [biohub/ESMC-600M](https://huggingface.co/biohub/ESMC-600M) | `fast` | `esmc_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esmc_6b` | `esm_plusplus` | `xlarge` | [Synthyra/ESMplusplus_6B](https://huggingface.co/Synthyra/ESMplusplus_6B) | [biohub/ESMC-6B](https://huggingface.co/biohub/ESMC-6B) | `fast` | `esmc_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esm3_small` | `esm3` | `large` | [Synthyra/ESM3_small](https://huggingface.co/Synthyra/ESM3_small) | [biohub/esm3-sm-open-v1](https://huggingface.co/biohub/esm3-sm-open-v1) | `fast` | `esm3_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `e1_150m` | `e1` | `small` | [Synthyra/Profluent-E1-150M](https://huggingface.co/Synthyra/Profluent-E1-150M) | [Profluent-Bio/E1-150m](https://huggingface.co/Profluent-Bio/E1-150m) | `fast` | `e1_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `e1_300m` | `e1` | `medium` | [Synthyra/Profluent-E1-300M](https://huggingface.co/Synthyra/Profluent-E1-300M) | [Profluent-Bio/E1-300m](https://huggingface.co/Profluent-Bio/E1-300m) | `fast` | `e1_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `e1_600m` | `e1` | `large` | [Synthyra/Profluent-E1-600M](https://huggingface.co/Synthyra/Profluent-E1-600M) | [Profluent-Bio/E1-600m](https://huggingface.co/Profluent-Bio/E1-600m) | `fast` | `e1_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `dplm_150m` | `dplm` | `small` | [Synthyra/DPLM-150M](https://huggingface.co/Synthyra/DPLM-150M) | [airkingbd/dplm_150m](https://huggingface.co/airkingbd/dplm_150m) | `fast` | `dplm_to_fastplms_v1` | `required` | not applicable | 0 |
| `dplm_650m` | `dplm` | `large` | [Synthyra/DPLM-650M](https://huggingface.co/Synthyra/DPLM-650M) | [airkingbd/dplm_650m](https://huggingface.co/airkingbd/dplm_650m) | `fast` | `dplm_to_fastplms_v1` | `required` | not applicable | 0 |
| `dplm_3b` | `dplm` | `xlarge` | [Synthyra/DPLM-3B](https://huggingface.co/Synthyra/DPLM-3B) | [airkingbd/dplm_3b](https://huggingface.co/airkingbd/dplm_3b) | `fast` | `dplm_to_fastplms_v1` | `required` | not applicable | 0 |
| `dplm2_150m` | `dplm2` | `small` | [Synthyra/DPLM2-150M](https://huggingface.co/Synthyra/DPLM2-150M) | [airkingbd/dplm2_150m](https://huggingface.co/airkingbd/dplm2_150m) | `official` | `dplm2_to_fastplms_v1` | `required` | not applicable | 0 |
| `dplm2_650m` | `dplm2` | `large` | [Synthyra/DPLM2-650M](https://huggingface.co/Synthyra/DPLM2-650M) | [airkingbd/dplm2_650m](https://huggingface.co/airkingbd/dplm2_650m) | `official` | `dplm2_to_fastplms_v1` | `required` | not applicable | 0 |
| `dplm2_3b` | `dplm2` | `xlarge` | [Synthyra/DPLM2-3B](https://huggingface.co/Synthyra/DPLM2-3B) | [airkingbd/dplm2_3b](https://huggingface.co/airkingbd/dplm2_3b) | `official` | `dplm2_to_fastplms_v1` | `official_unavailable` | not applicable | 0 |
| `ankh_base` | `ankh` | `medium` | [Synthyra/ANKH_base](https://huggingface.co/Synthyra/ANKH_base) | [ElnaggarLab/ankh-base](https://huggingface.co/ElnaggarLab/ankh-base) | `official` | `ankh_t5_to_fastplms_v1` | `required` | not applicable | 0 |
| `ankh_large` | `ankh` | `large` | [Synthyra/ANKH_large](https://huggingface.co/Synthyra/ANKH_large) | [ElnaggarLab/ankh-large](https://huggingface.co/ElnaggarLab/ankh-large) | `official` | `ankh_t5_to_fastplms_v1` | `required` | not applicable | 0 |
| `ankh2_large` | `ankh` | `large` | [Synthyra/ANKH2_large](https://huggingface.co/Synthyra/ANKH2_large) | [ElnaggarLab/ankh2-ext2](https://huggingface.co/ElnaggarLab/ankh2-ext2) | `official` | `ankh_t5_to_fastplms_v1` | `required` | not applicable | 0 |
| `ankh3_large` | `ankh` | `large` | [Synthyra/ANKH3_large](https://huggingface.co/Synthyra/ANKH3_large) | [ElnaggarLab/ankh3-large](https://huggingface.co/ElnaggarLab/ankh3-large) | `official` | `ankh_t5_to_fastplms_v1` | `required` | not applicable | 0 |
| `ankh3_xl` | `ankh` | `xlarge` | [Synthyra/ANKH3_xl](https://huggingface.co/Synthyra/ANKH3_xl) | [ElnaggarLab/ankh3-xl](https://huggingface.co/ElnaggarLab/ankh3-xl) | `official` | `ankh_t5_to_fastplms_v1` | `required` | not applicable | 0 |
| `boltz2` | `boltz2` | `structure` | [Synthyra/Boltz2](https://huggingface.co/Synthyra/Boltz2) | [boltz-community/boltz-2](https://huggingface.co/boltz-community/boltz-2) | `fast` | `boltz2_inference_core_v1` | `not_applicable` | not applicable | 0 |
| `esmfold` | `esmfold` | `structure` | [Synthyra/FastESMFold](https://huggingface.co/Synthyra/FastESMFold) | [facebook/esmfold_v1](https://huggingface.co/facebook/esmfold_v1) | `fast` | `esmfold_meta_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esmfold2` | `esmfold2` | `structure` | [Synthyra/ESMFold2](https://huggingface.co/Synthyra/ESMFold2) | [biohub/ESMFold2](https://huggingface.co/biohub/ESMFold2) | `fast` | `identity` | `not_applicable` | `optional` (full checkpoint) | 0 |
| `esmfold2_fast` | `esmfold2` | `structure` | [Synthyra/ESMFold2-Fast](https://huggingface.co/Synthyra/ESMFold2-Fast) | [biohub/ESMFold2-Fast](https://huggingface.co/biohub/ESMFold2-Fast) | `fast` | `identity` | `not_applicable` | `none` (Fast; MSA inputs rejected) | 0 |
| `esmfold2_experimental_cutoff2025` | `esmfold2` | `structure` | [Synthyra/ESMFold2-Experimental-Cutoff2025](https://huggingface.co/Synthyra/ESMFold2-Experimental-Cutoff2025) | [biohub/ESMFold2-Experimental-Cutoff2025](https://huggingface.co/biohub/ESMFold2-Experimental-Cutoff2025) | `fast` | `identity` | `not_applicable` | `optional` (full checkpoint) | 0 |
| `esmfold2_experimental_fast_cutoff2025` | `esmfold2` | `structure` | [Synthyra/ESMFold2-Experimental-Fast-Cutoff2025](https://huggingface.co/Synthyra/ESMFold2-Experimental-Fast-Cutoff2025) | [biohub/ESMFold2-Experimental-Fast-Cutoff2025](https://huggingface.co/biohub/ESMFold2-Experimental-Fast-Cutoff2025) | `fast` | `identity` | `not_applicable` | `none` (Fast; MSA inputs rejected) | 0 |

A nonzero unresolved-file count blocks release. It is not permission to
omit that file from checkpoint, tokenizer, artifact, or compliance checks.
