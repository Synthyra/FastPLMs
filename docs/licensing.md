# Licensing and attribution

FastPLMs source is distributed under the Apache License 2.0. Model checkpoints,
official source repositories, and copied third-party components have their own
terms. The model manifest records code and checkpoint licenses separately.

Each family records Hugging Face model-card metadata separately from its
human-readable checkpoint terms. Standard checkpoints use the Hub identifiers
`mit`, `apache-2.0`, or `cc-by-nc-sa-4.0`. E1 uses `other` with the name and
source link for its clickthrough agreement. DPLM1 and DPLM2 use
`apache-2.0`. Missing or mismatched identifiers block generation and artifact
validation.

`LICENSES/` contains distributable copies of required legal texts.
`THIRD_PARTY_NOTICES.md` maps each model family and component to its source,
revision, terms, modifications, and attribution. Release validation compares
these files to the canonical pinned upstream files by hash.

## ANKH

ANKH implementations and mirrored weights are retained under CC BY-NC-SA 4.0.
Artifacts and model cards show those terms prominently. FastPLMs does not
implement a runtime restriction or decide if a use satisfies the license.

## E1

E1 retains the upstream agreement, `ATTRIBUTION`, `NOTICE`, Apache and BSD
texts, modified-file notices, and documentation attribution. Relevant launches
display `Profluent-E1`. Redistribution and use are subject to the upstream
agreement. Review `LICENSES/e1/` before use.

## DPLM

The ByteDance DPLM repository carries an
[Apache-2.0 license](https://github.com/bytedance/dplm/blob/main/LICENSE)
and its [official README](https://github.com/bytedance/dplm/blob/main/README.md#overview)
defines the repository release as including pretrained weights for both DPLM1
and DPLM2. FastPLMs records both checkpoint families as Apache-2.0,
with Hub metadata `license: apache-2.0`,
`weights_license_status="resolved"`, and `redistributable=true`.

`LICENSES/dplm/` contains the verbatim license and immutable evidence record.
Complete publication fails closed unless the artifact, legal inventory,
state-parity evidence, and atomic Hub preflight pass.

## Biohub, Boltz, Meta, OpenFold, and ProteinTTT

Biohub MIT and Apache notices, including `THIRD_PARTY_NOTICE`, are retained for
ESM++, ESM3, and ESMFold2. The ESM++ hidden-state sparse autoencoder is a
FastPLMs implementation of the published Biohub SAE contract, held to exact
agreement with the pinned Biohub source by test. Biohub retains ownership of
the SAE weights; FastPLMs reads their published repositories and redistributes
neither those weights nor Biohub SAE code. Boltz MIT terms, Meta ESM and ESMFold notices,
OpenFold notices, and ProteinTTT source records are included where their code or
derived behavior is distributed.

ESMFold2 additionally uses a 417,306,584-byte `ccd.pkl` runtime asset under MIT
terms. The manifest pins repository, immutable revision, path, size, and
SHA-256. Because it is pickle, validated deserialization is an explicit trust
boundary. The loader rejects user and `cache_dir` symlinks, except for the exact
manifest snapshot link resolving into its repository's contained blob
directory. It copies into a private loader-owned temporary snapshot, verifies
that snapshot's size and hash, and unpickles only those verified bytes. Offline
runs require the exact cache object and never fetch a substitute.

## Artifact and container rules

An artifact or reference container fails validation when a required source
revision, license, notice, attribution, modified-file record, or conversion
record is absent. Reference images receive only the legal files relevant to
their corresponding upstream. Runtime images contain no official submodules or
checkpoint weights.

This guide summarizes repository policy and is not legal advice. The complete
texts in `LICENSES/` and the original upstream sources control.
