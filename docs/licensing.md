# Licensing and attribution

FastPLMs source is distributed under the Apache License 2.0. Model checkpoints,
official source repositories, and copied third-party components retain their own
terms. The model manifest records code and checkpoint licenses separately.

Each family also records Hugging Face model-card metadata separately from its
human-readable checkpoint terms. Standard checkpoints use the Hub identifiers
`mit`, `apache-2.0`, or `cc-by-nc-sa-4.0`. E1 uses `other` together with the
name and immutable source link for the Profluent-E1 Clickthrough License
Agreement. Missing, mismatched, or free-form Hub identifiers block generation
and artifact validation.

`LICENSES/` contains distributable copies of required legal texts.
`THIRD_PARTY_NOTICES.md` maps each model family and component to its source,
revision, terms, modifications, and attribution. Release validation compares
these files with the canonical pinned upstream files by hash.

## ANKH

ANKH implementations and mirrored weights are retained under CC BY-NC-SA 4.0.
Artifacts and model cards display those terms prominently. FastPLMs does not
implement a runtime restriction or decide whether a particular use satisfies
the license.

## E1

E1 retains the upstream agreement, `ATTRIBUTION`, `NOTICE`, Apache and BSD
texts, modified-file notices, and documentation attribution. Relevant launches
display `Profluent-E1`. Redistribution and use remain subject to the upstream
agreement; review `LICENSES/e1/` before use.

## DPLM

The project records Apache-2.0 as the DPLM1 checkpoint-license assumption. The
official checkpoint card does not provide an independent checkpoint-license
statement, so this is documented as a project assumption rather than an
upstream assertion.

## Biohub, Boltz, Meta, OpenFold, and ProteinTTT

Biohub MIT and Apache notices, including `THIRD_PARTY_NOTICE`, are retained for
ESM++, ESM3, and ESMFold2. Boltz MIT terms, Meta ESM and ESMFold notices,
OpenFold notices, and ProteinTTT provenance are included where their code or
derived behavior is distributed.

## Artifact and container rules

An artifact or reference container fails validation when a required source
revision, license, notice, attribution, modified-file record, or conversion
record is absent. Reference images receive only the legal files relevant to
their corresponding upstream. Runtime images contain no official submodules or
checkpoint weights.

This guide summarizes repository policy and is not legal advice. The complete
texts in `LICENSES/` and the original upstream sources control.
