# Third-party notices

FastPLMs implements interfaces and checkpoint mappings for independently
released protein models. The pinned repositories under `vendor/upstream/` are
parity oracles. Production code does not import them, and runtime images do not
contain them.

This notice is informational and is not legal advice. A checkpoint license can
differ from the license covering its source implementation. The typed inventory
in `src/fastplms/models.toml` and the verbatim files under `LICENSES/` are the
distribution record.

## ANKH

The pinned ANKH implementation and the mirrored ANKH checkpoints are identified
as CC BY-NC-SA 4.0. FastPLMs displays those terms but does not enforce them in
software. Users are responsible for determining whether their use and
redistribution comply. The complete text is in `LICENSES/ankh/LICENSE.md`.

## Profluent-E1

Profluent identifies its E1 model code as Apache-2.0. The E1 weights and full
release are subject to the Profluent-E1 Clickthrough License Agreement and the
incorporated attribution requirements. Any E1 distribution must retain all of
the following files:

- `LICENSES/e1/LICENSE`, the Profluent-E1 agreement
- `LICENSES/e1/ATTRIBUTION`, the attribution guidelines
- `LICENSES/e1/NOTICE`, the required notice
- `LICENSES/e1/Apache-2.0.txt`, the code license
- `LICENSES/e1/BSD-3-Clause.txt`, covering the FlashAttention-derived padding
  utility identified by the official E1 source
- `LICENSES/e1/MODIFICATIONS.md`, the FastPLMs modified-file notice

The exact text `Profluent-E1` must remain prominently displayed in E1
documentation and at each launch of an executable E1 workflow, as required by
the upstream attribution guidelines. Certain commercial outputs, including
specified pharmaceutical and target-related outputs, can require the separate
`Built with Profluent-E1` statement described in `ATTRIBUTION`.

## DPLM

The pinned ByteDance DPLM repository is Apache-2.0. Its
[README](https://github.com/bytedance/dplm/blob/8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d/README.md#overview)
explicitly defines the repository release as including pretrained DPLM1 and
DPLM2 weights, and the same revision carries the complete
[Apache-2.0 license](https://github.com/bytedance/dplm/blob/8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d/LICENSE).
FastPLMs records both checkpoint families as Apache-2.0 and distributes the
verbatim license plus `LICENSES/dplm/PROVENANCE.md`. Converted weights retain
those terms and remain subject to the ordinary artifact and publication gates.

## Biohub

The pinned Biohub ESM implementation is MIT and includes a separate
`THIRD_PARTY_NOTICE.md`; both files are distributed under
`LICENSES/biohub-esm/`. The pinned Biohub Transformers fork is Apache-2.0, with
its complete text under `LICENSES/biohub-transformers/`.

## Boltz

The pinned Boltz source is MIT. The verbatim notice is in
`LICENSES/boltz/LICENSE`.

## Meta ESM and OpenFold

The pinned Meta ESM source is MIT. The pinned OpenFold source is Apache-2.0.
Their verbatim texts and revision-specific provenance notices are under
`LICENSES/fair-esm/` and `LICENSES/openfold/`.

The native H100 ESMFold reference image applies the tracked
`docker/constraints/openfold-sm90.patch` to the copied OpenFold `setup.py`.
This build-only change restricts the CUDA extension to `sm90` and selects the
C++17 standard required by the reference PyTorch version. It leaves the pinned
submodule, extension source, model classes, checkpoint data, and public API
unchanged. The complete modified-file record is in
`LICENSES/openfold/MODIFICATIONS.md`.

The isolated reference image also includes Apache-2.0 PyTorch Lightning,
TorchMetrics, Lightning Utilities, and NVIDIA DLLogger. Their exact versions or
revision are pinned in `docker/constraints/esmfold.txt`; OpenFold imports them
eagerly, and FastPLMs production code does not depend on them. DLLogger's exact
source identity and installed-license handling are recorded in
`LICENSES/dllogger/PROVENANCE.md`.

## ProteinTTT

The optional test-time training workflow is validated against the pinned
ProteinTTT repository under its MIT license. Its verbatim license and
revision-specific provenance are under `LICENSES/protein-ttt/`.

## Conversion and packaging record

For every supported family, `src/fastplms/models.toml` records an immutable
official checkpoint revision, an immutable FastPLMs checkpoint revision, file
digests, a named state transformation, and a mechanism-level conversion record.
Generated artifacts reproduce that record in `provenance.json`. A release or
artifact build must fail when a required file identity, legal text, attribution
notice, modified-file notice, upstream revision, or conversion record is absent
or differs from its manifest digest.
