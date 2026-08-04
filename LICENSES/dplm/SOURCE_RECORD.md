# DPLM checkpoint license provenance

FastPLMs uses the ByteDance DPLM repository at immutable revision
`8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d` as the official source for both
DPLM1 and DPLM2.

At that revision:

- the repository contains the complete [Apache License 2.0](https://github.com/bytedance/dplm/blob/8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d/LICENSE); and
- the [official README](https://github.com/bytedance/dplm/blob/8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d/README.md#overview)
  defines the repository release as including the pretrained weights for the
  DPLM family, specifically DPLM1 and DPLM2, alongside training and inference
  implementations.

FastPLMs therefore records the official DPLM1 and DPLM2 checkpoint weights as
Apache-2.0. Converted Synthyra checkpoints retain that license and include the
verbatim upstream `LICENSE`. The deterministic conversion identifiers are
`dplm_to_fastplms_v1` and `dplm2_to_fastplms_v1`; neither adds restrictions to
the upstream terms.

Complete publication is permitted only after the ordinary FastPLMs artifact,
state-parity, legal-inventory, and atomic-publication checks pass. This record
does not change the terms of third-party training data or downstream outputs.
