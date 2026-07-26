# Third-party license inventory

This directory is the distributable legal inventory. Verbatim upstream files
are copied from the pinned repositories under `vendor/upstream/`. Supplemental
provenance and modified-file notices are maintained here because they describe
the FastPLMs distribution rather than an upstream repository.

Keep distributable license files in this directory rather than beside runtime
modules under `src/fastplms/`. Model-specific license summaries belong in the
generated cards under `model_cards/`; the canonical legal texts and notices
listed here control.

`src/fastplms/models.toml` records the SHA-256 digest of every legal file
required by a distributable model artifact. Artifact construction verifies the
declared upstream and distribution copies before writing output. Missing or
changed declared content is a release error.

Digests use the UTF-8 text stored by Git with LF line endings. Validation
normalizes CRLF or LF checkouts to that canonical representation, and artifact
construction always writes LF. No other whitespace or content is normalized.

## Required inventory

| Source | Distribution file | Purpose |
|---|---|---|
| FastPLMs | root `LICENSE` | Apache-2.0 project code license with checkpoint caveat |
| FastPLMs | root `THIRD_PARTY_NOTICES.md` | Consolidated attribution and redistribution notice |
| ANKH | `ankh/LICENSE.md` | Verbatim CC BY-NC-SA 4.0 terms |
| Biohub ESM | `biohub-esm/LICENSE.md` | Verbatim MIT license |
| Biohub ESM | `biohub-esm/THIRD_PARTY_NOTICE.md` | Verbatim Biohub third-party notices |
| Biohub Transformers | `biohub-transformers/LICENSE` | Verbatim Apache-2.0 license |
| Boltz | `boltz/LICENSE` | Verbatim MIT license |
| DPLM | `dplm/LICENSE` | Verbatim Apache-2.0 license |
| DPLM | `dplm/PROVENANCE.md` | Pinned license scope for DPLM1 and DPLM2 checkpoint weights |
| Profluent-E1 | `e1/LICENSE` | Verbatim Profluent-E1 agreement |
| Profluent-E1 | `e1/ATTRIBUTION` | Verbatim attribution guidelines |
| Profluent-E1 | `e1/NOTICE` | Verbatim required notice |
| Profluent-E1 | `e1/Apache-2.0.txt` | Complete Apache-2.0 text for the model code |
| Profluent-E1 | `e1/BSD-3-Clause.txt` | Complete BSD-3-Clause text for the FlashAttention-derived utility identified upstream |
| Profluent-E1 | `e1/MODIFICATIONS.md` | FastPLMs modified-file notice and conversion identifier |
| Meta ESM | `fair-esm/LICENSE` | Verbatim MIT license |
| Meta ESM | `fair-esm/PROVENANCE.md` | Pinned revision and parity-oracle boundary |
| OpenFold | `openfold/LICENSE` | Verbatim Apache-2.0 license |
| OpenFold | `openfold/MODIFICATIONS.md` | FastPLMs modified-file notice |
| OpenFold | `openfold/PROVENANCE.md` | Pinned revision and parity-oracle boundary |
| ProteinTTT | `protein-ttt/LICENSE` | Verbatim MIT license |
| ProteinTTT | `protein-ttt/PROVENANCE.md` | Pinned revision and optional-workflow boundary |

## Reference-only notices

Some legal records apply only to isolated parity environments and are not
copied into model artifacts. `dllogger/PROVENANCE.md` records the pinned
DLLogger dependency and installed-license check used by the ESMFold reference
image. FastPLMs production code does not import or distribute DLLogger.

The pinned E1 repository contains its agreement, attribution guidelines, and
notice, but does not include standalone Apache-2.0 or BSD-3-Clause files. The
complete standard texts are included here. The BSD notice follows the official
E1 source header that identifies `flash_attention_utils.py` as adapted from
Dao-AILab FlashAttention under BSD-3-Clause.

FastPLMs does not enforce ANKH use restrictions in software. The pinned DPLM
repository's Apache-2.0 `LICENSE` and README scope the official release to the
pretrained DPLM1 and DPLM2 weights. The immutable evidence and FastPLMs
conversion boundary are recorded in `dplm/PROVENANCE.md`.
