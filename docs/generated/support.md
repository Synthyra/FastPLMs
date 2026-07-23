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
| `esm_plusplus` | `eager`, `sdpa`, `flex_attention`, `flash_attention_2`, `flash_attention_3` | `default` | `static_parameters` | `core` | `reference-biohub-esm` |
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
| `e1` | Profluent-E1-Agreement | `other` ([Profluent-E1 Clickthrough License Agreement](https://github.com/Profluent-AI/E1/blob/bfd2620a602248499f3d2583d85a7ecddf0b6e02/LICENSE)) | manifest policy | `check`, `compliance`, `feature`, `artifact`, `benchmark` |
| `dplm` | Apache-2.0 | `apache-2.0` | manifest policy | `check`, `compliance`, `feature`, `artifact`, `benchmark` |
| `dplm2` | Apache-2.0 | `apache-2.0` | manifest policy | `check`, `compliance`, `feature`, `artifact`, `benchmark` |
| `ankh` | CC-BY-NC-SA-4.0 | `cc-by-nc-sa-4.0` | complete checkpoint required | `check`, `compliance`, `feature`, `artifact`, `benchmark` |
| `boltz2` | MIT | `mit` | manifest policy | `structure`, `artifact`, `benchmark` |
| `esmfold` | MIT | `mit` | manifest policy | `check`, `compliance`, `structure`, `feature`, `artifact`, `benchmark` |
| `esmfold2` | MIT | `mit` | manifest policy | `check`, `compliance`, `structure`, `feature`, `artifact`, `benchmark` |

## Runtime assets

| ID | Family | Repository revision | Path | SHA-256 | Size | License | Trust boundary | Offline behavior |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| `esmfold2_ccd` | `esmfold2` | `biohub/ESMFold2@1ebf0e3481a5184eb6171d40615c79e384b48796` | `ccd.pkl` | `9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5` | 417306584 | `MIT` | `hash_pinned_pickle` | `requires_cached_verified_file` |

## Checkpoints

| ID | Family | Size | FastPLMs checkpoint | Official checkpoint | Artifact source | State transform | Generation contract | MSA conditioning | Unresolved files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `esm2_8m` | `esm2` | `small` | [Synthyra/ESM2-8M](https://huggingface.co/Synthyra/ESM2-8M/tree/185ecbd45665d050a8dae326d91886d330c5f9d0) | [facebook/esm2_t6_8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D/tree/c731040fcd8d73dceaa04b0a8e6329b345b0f5df) | `fast` | `esm2_hf_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esm2_35m` | `esm2` | `small` | [Synthyra/ESM2-35M](https://huggingface.co/Synthyra/ESM2-35M/tree/37ab9f56b41e365b3bd9e25d6fefe9150fd910f0) | [facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D/tree/6fbf070e65b0b7291e7bbcd451118c216cff79d8) | `fast` | `esm2_hf_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esm2_150m` | `esm2` | `medium` | [Synthyra/ESM2-150M](https://huggingface.co/Synthyra/ESM2-150M/tree/979e0880dfc9e0c0080839b83d9d2dc05b92786a) | [facebook/esm2_t30_150M_UR50D](https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/a695f6045e2e32885fa60af20c13cb35398ce30c) | `fast` | `esm2_hf_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esm2_650m` | `esm2` | `large` | [Synthyra/ESM2-650M](https://huggingface.co/Synthyra/ESM2-650M/tree/ca0718a5d52b80d5c60dd76860e55e061a95fb0a) | [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/08e4846e537177426273712802403f7ba8261b6c) | `fast` | `esm2_hf_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esm2_3b` | `esm2` | `xlarge` | [Synthyra/ESM2-3B](https://huggingface.co/Synthyra/ESM2-3B/tree/ff89d0180f414ab9c677219a25da79bf09185456) | [facebook/esm2_t36_3B_UR50D](https://huggingface.co/facebook/esm2_t36_3B_UR50D/tree/476b639933c8baad5ad09a60ac1a87f987b656fc) | `fast` | `esm2_hf_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esmc_small` | `esm_plusplus` | `medium` | [Synthyra/ESMplusplus_small](https://huggingface.co/Synthyra/ESMplusplus_small/tree/46c5f7d562e47d4c14165b424c71ab7db008e6fb) | [biohub/ESMC-300M](https://huggingface.co/biohub/ESMC-300M/tree/a59b831785f907e96e6a246b1d142bfb76df31ee) | `fast` | `esmc_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esmc_large` | `esm_plusplus` | `large` | [Synthyra/ESMplusplus_large](https://huggingface.co/Synthyra/ESMplusplus_large/tree/f813401638b3fddab09748aec1ad2bf537aa4208) | [biohub/ESMC-600M](https://huggingface.co/biohub/ESMC-600M/tree/a7e82012c83126b9eedb055fea9fa84b6c02f094) | `fast` | `esmc_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esmc_6b` | `esm_plusplus` | `xlarge` | [Synthyra/ESMplusplus_6B](https://huggingface.co/Synthyra/ESMplusplus_6B/tree/0d579cce3b0f09efa6b3baddf6cc3fd8c9b616c8) | [biohub/ESMC-6B](https://huggingface.co/biohub/ESMC-6B/tree/45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a) | `fast` | `esmc_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esm3_small` | `esm3` | `large` | [Synthyra/ESM3_small](https://huggingface.co/Synthyra/ESM3_small/tree/7ddb5a740f9e5f93933eb6410c0ee8684bc63ec1) | [biohub/esm3-sm-open-v1](https://huggingface.co/biohub/esm3-sm-open-v1/tree/47f0545b2b6daf26a93439a3cd610f4f7f3d5478) | `fast` | `esm3_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `e1_150m` | `e1` | `small` | [Synthyra/Profluent-E1-150M](https://huggingface.co/Synthyra/Profluent-E1-150M/tree/7c5f3bbf697226a2e0900db7a100f9201774a907) | [Profluent-Bio/E1-150m](https://huggingface.co/Profluent-Bio/E1-150m/tree/c4dbfe827e4aa6ed7f95eaef50dc1e084f4d77dc) | `fast` | `e1_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `e1_300m` | `e1` | `medium` | [Synthyra/Profluent-E1-300M](https://huggingface.co/Synthyra/Profluent-E1-300M/tree/5ef52c0ad2ae2578f40622696b763523810e8e26) | [Profluent-Bio/E1-300m](https://huggingface.co/Profluent-Bio/E1-300m/tree/5a2871c587eadbcc9237bc686ea45e5b4d28dfb3) | `fast` | `e1_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `e1_600m` | `e1` | `large` | [Synthyra/Profluent-E1-600M](https://huggingface.co/Synthyra/Profluent-E1-600M/tree/6c8bf0ec83b0e0178677c528b101efffd0677742) | [Profluent-Bio/E1-600m](https://huggingface.co/Profluent-Bio/E1-600m/tree/52d959fb87a609d15cf223a485127b29ed5c382a) | `fast` | `e1_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `dplm_150m` | `dplm` | `small` | [Synthyra/DPLM-150M](https://huggingface.co/Synthyra/DPLM-150M/tree/90ba742754151a774f3b7ed580170d0a76b3e69d) | [airkingbd/dplm_150m](https://huggingface.co/airkingbd/dplm_150m/tree/49b7125a5d28c6418fcc2f3c4fe799352ac1488b) | `fast` | `dplm_to_fastplms_v1` | `required` | not applicable | 0 |
| `dplm_650m` | `dplm` | `large` | [Synthyra/DPLM-650M](https://huggingface.co/Synthyra/DPLM-650M/tree/05dc16d97c5c028aed924c9ed681cee4ab609760) | [airkingbd/dplm_650m](https://huggingface.co/airkingbd/dplm_650m/tree/7a7e651baa667d094aba05e9dc1cf52a3332110a) | `fast` | `dplm_to_fastplms_v1` | `required` | not applicable | 0 |
| `dplm_3b` | `dplm` | `xlarge` | [Synthyra/DPLM-3B](https://huggingface.co/Synthyra/DPLM-3B/tree/7d764dd3d70ecf1ac0e64693de64a0064aacac65) | [airkingbd/dplm_3b](https://huggingface.co/airkingbd/dplm_3b/tree/53849d4a7fe944ae0b9cf2bbc0d2cc0054795b51) | `fast` | `dplm_to_fastplms_v1` | `required` | not applicable | 0 |
| `dplm2_150m` | `dplm2` | `small` | [Synthyra/DPLM2-150M](https://huggingface.co/Synthyra/DPLM2-150M/tree/182745b8dc5661f898481a4fa60a7af9d53385c4) | [airkingbd/dplm2_150m](https://huggingface.co/airkingbd/dplm2_150m/tree/3451d984d06497f835ed49634bd68c9dfb54d730) | `official` | `dplm2_to_fastplms_v1` | `required` | not applicable | 0 |
| `dplm2_650m` | `dplm2` | `large` | [Synthyra/DPLM2-650M](https://huggingface.co/Synthyra/DPLM2-650M/tree/b9d8527a9473a54954fa2764f590b9ea1b435bb2) | [airkingbd/dplm2_650m](https://huggingface.co/airkingbd/dplm2_650m/tree/0bc69b644976c6680ab7e26669854d1979e8876e) | `official` | `dplm2_to_fastplms_v1` | `required` | not applicable | 0 |
| `dplm2_3b` | `dplm2` | `xlarge` | [Synthyra/DPLM2-3B](https://huggingface.co/Synthyra/DPLM2-3B/tree/2a63babe8848abf5233d31bd55891dff8285fc50) | [airkingbd/dplm2_3b](https://huggingface.co/airkingbd/dplm2_3b/tree/9e77567926f98d1b997ea9131a8eeb035b9bf827) | `official` | `dplm2_to_fastplms_v1` | `official_unavailable` | not applicable | 0 |
| `ankh_base` | `ankh` | `medium` | [Synthyra/ANKH_base](https://huggingface.co/Synthyra/ANKH_base/tree/7ec329aae8e3e174bf22a1eb9e0e9fcc12b53092) | [ElnaggarLab/ankh-base](https://huggingface.co/ElnaggarLab/ankh-base/tree/d99cb6b966530dfc2ae96bc69d9255c2a07308b0) | `official` | `ankh_t5_to_fastplms_v1` | `required` | not applicable | 0 |
| `ankh_large` | `ankh` | `large` | [Synthyra/ANKH_large](https://huggingface.co/Synthyra/ANKH_large/tree/3be3df34140f49dc4e65bd1f247e3ce819e7fc59) | [ElnaggarLab/ankh-large](https://huggingface.co/ElnaggarLab/ankh-large/tree/74b371dbfa3ee0a05d32ae74df0c2e0b82d6b9a6) | `official` | `ankh_t5_to_fastplms_v1` | `required` | not applicable | 0 |
| `ankh2_large` | `ankh` | `large` | [Synthyra/ANKH2_large](https://huggingface.co/Synthyra/ANKH2_large/tree/392de5ed52bbfd73b45f545e378aaebcff096d0e) | [ElnaggarLab/ankh2-ext2](https://huggingface.co/ElnaggarLab/ankh2-ext2/tree/aa9b9fa72288c47d9f618ce80c011e24b54e17a8) | `official` | `ankh_t5_to_fastplms_v1` | `required` | not applicable | 0 |
| `ankh3_large` | `ankh` | `large` | [Synthyra/ANKH3_large](https://huggingface.co/Synthyra/ANKH3_large/tree/53600f175f328f986f43e55ca8ceb14935d337a4) | [ElnaggarLab/ankh3-large](https://huggingface.co/ElnaggarLab/ankh3-large/tree/2be091622e8a393f0ef21735070084123c874b6e) | `official` | `ankh_t5_to_fastplms_v1` | `required` | not applicable | 0 |
| `ankh3_xl` | `ankh` | `xlarge` | [Synthyra/ANKH3_xl](https://huggingface.co/Synthyra/ANKH3_xl/tree/3cbf2c22c4f7d67bf0bfcbdcd500f41723e91d29) | [ElnaggarLab/ankh3-xl](https://huggingface.co/ElnaggarLab/ankh3-xl/tree/e00113df5c95ef71df7ea3f5a73d56bd00e473a4) | `official` | `ankh_t5_to_fastplms_v1` | `required` | not applicable | 0 |
| `boltz2` | `boltz2` | `structure` | [Synthyra/Boltz2](https://huggingface.co/Synthyra/Boltz2/tree/3b148fc5efea109c065ec82ba8683d024de7134e) | [boltz-community/boltz-2](https://huggingface.co/boltz-community/boltz-2/tree/6fdef46d763fee7fbb83ca5501ccceff43b85607) | `fast` | `boltz2_inference_core_v1` | `not_applicable` | not applicable | 0 |
| `esmfold` | `esmfold` | `structure` | [Synthyra/FastESMFold](https://huggingface.co/Synthyra/FastESMFold/tree/b88c8cb50d19b2cf7ab4fee4b0a61f5e02da7823) | [facebook/esmfold_v1](https://huggingface.co/facebook/esmfold_v1/tree/75a3841ee059df2bf4d56688166c8fb459ddd97a) | `fast` | `esmfold_meta_to_fastplms_v1` | `not_applicable` | not applicable | 0 |
| `esmfold2` | `esmfold2` | `structure` | [Synthyra/ESMFold2](https://huggingface.co/Synthyra/ESMFold2/tree/cd5a0927cec585a778d983b99a8db23d2e9b281e) | [biohub/ESMFold2](https://huggingface.co/biohub/ESMFold2/tree/1ebf0e3481a5184eb6171d40615c79e384b48796) | `fast` | `identity` | `not_applicable` | `optional` (full checkpoint) | 0 |
| `esmfold2_fast` | `esmfold2` | `structure` | [Synthyra/ESMFold2-Fast](https://huggingface.co/Synthyra/ESMFold2-Fast/tree/407875bfcaa42552bfcb25acd67ee1888b790170) | [biohub/ESMFold2-Fast](https://huggingface.co/biohub/ESMFold2-Fast/tree/b28d8ace5e05e61e5bec1e6820cfd3e221819d12) | `fast` | `identity` | `not_applicable` | `none` (Fast; MSA inputs rejected) | 0 |
| `esmfold2_experimental_cutoff2025` | `esmfold2` | `structure` | [Synthyra/ESMFold2-Experimental-Cutoff2025](https://huggingface.co/Synthyra/ESMFold2-Experimental-Cutoff2025/tree/632ff4a9e68f1de78ee956a613267bdcdb5b354d) | [biohub/ESMFold2-Experimental-Cutoff2025](https://huggingface.co/biohub/ESMFold2-Experimental-Cutoff2025/tree/56f94f5c1069ecde17512c96928850518340d287) | `fast` | `identity` | `not_applicable` | `optional` (full checkpoint) | 0 |
| `esmfold2_experimental_fast_cutoff2025` | `esmfold2` | `structure` | [Synthyra/ESMFold2-Experimental-Fast-Cutoff2025](https://huggingface.co/Synthyra/ESMFold2-Experimental-Fast-Cutoff2025/tree/8f022c2514a6c32692aaca078a8391d6bc6c4bac) | [biohub/ESMFold2-Experimental-Fast-Cutoff2025](https://huggingface.co/biohub/ESMFold2-Experimental-Fast-Cutoff2025/tree/74b88548bf19688b8727432db0d698cb2e1d8783) | `fast` | `identity` | `not_applicable` | `none` (Fast; MSA inputs rejected) | 0 |

A nonzero unresolved-file count blocks release. It is not permission to
omit that file from checkpoint, tokenizer, artifact, or compliance checks.
