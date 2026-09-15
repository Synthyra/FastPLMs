# ESMFold2-300 single-protein validation

## Scope and status

This report records one isolated reference-versus-candidate comparison for the
experimental `ESMFold2-300` mirror. The case uses one 56-residue protein, seed
17, one diffusion sample, three recycling loops, and 15 diffusion sampling
steps. The comparison passed its declared checks. It is a single-protein case,
not the full structure benchmark. The mirrors were published at revisions
`a38a62ae930d157484b331c2bf4241684573adba` (300M) and
`71c67d0b2b73dc245ea7c3cc0d0476439a882d08` (600M). No inference result is
available for `ESMFold2-600`.

The experimental config disables the confidence head and MSA features. The
comparison therefore checks structure and representation outputs without
pLDDT, pTM, iPTM, or PAE fields.

## Reproduction details

The run used the repository's [validation Dockerfile](../../docker/esmfold2-validation.Dockerfile)
and the `tools.validation.esmfold2_small` producer and comparator. Reference
and candidate producers ran in separate processes with the following settings:

```text
sequence: MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE
seed: 17
num_loops: 3
num_diffusion_samples: 1
num_sampling_steps: 15
chunk_size: 32
attention_backend: sdpa
backbone_compute_dtype: bfloat16
folding_parameter_dtype: float32
confidence_head_enabled: false
```

The commands were:

```bash
python -m tools.validation.esmfold2_small produce \
  --producer reference \
  --fold-snapshot artifacts/esmfold2-small/fold300 \
  --backbone-snapshot artifacts/esmfold2-small/official-backbone300 \
  --output artifacts/esmfold2-small/reference

python -m tools.validation.esmfold2_small produce \
  --producer candidate \
  --fold-snapshot artifacts/esmfold2-small/fold300 \
  --backbone-snapshot artifacts/esmfold2-small/standard300 \
  --output artifacts/esmfold2-small/candidate

python -m tools.validation.esmfold2_small compare \
  --reference artifacts/esmfold2-small/reference \
  --candidate artifacts/esmfold2-small/candidate \
  --output artifacts/esmfold2-small/comparison.json
```

To rebuild the pinned reference backbone layout from the native 300M snapshot,
run:

```bash
PYTHONPATH=src python -m tools.conversion.esmc_native \
  --native-snapshot artifacts/esmfold2-small/backbone300 \
  --output artifacts/esmfold2-small/official-backbone300-rebuilt
```

The reference process used `biohub/ESMC-300M-1500000` with Transformers
4.57.6. The candidate used `Synthyra/ESMplusplus_small` with Transformers
5.13.0. Both used Python 3.12.14 and Torch 2.13.0+cu130 with CUDA runtime
13.0 on an NVIDIA GeForce RTX 4070 Laptop GPU (compute capability 8.9, 8 GiB
reported device memory). Peak allocated memory was 1,454,067,200 bytes for
the reference and 1,438,637,056 bytes for the candidate.

## Results

The comparator reported `status: passed` and no failures. All 23 feature
tensors and the initial diffusion noise tensor, 24 exact tensors in total,
were equal. Selected metrics are:

| Output | Result |
| --- | ---: |
| C-alpha RMSD | 0.0774813518 |
| C-alpha lDDT | 1.0 |
| Hidden-state pooled cosine minimum | 1.0000007153 |
| Hidden-state relative L2 | 7.1661839e-7 |
| Projection pooled cosine minimum | 0.9999999404 |
| Projection relative L2 | 0.0018453832 |
| Distogram-logit relative L2 | 0.0097329058 |
| Sample-atom-coordinate relative L2 | 1.1643044949 |

The comparator's hard geometry checks passed, while the coordinate and
distogram values remain numerically different. These results support the
declared single-protein comparison only. They do not establish full
benchmark parity, broad biological validity, or equivalence of the native and
standard FP32 backbone weights. The native checkpoint stores BF16-rounded
values promoted to FP32; the standard ESM++ checkpoint retains its original
FP32 values.

The reloaded 300M Hub artifact produced the same passed comparison and metrics.
This confirms the files-only artifact reload for the reported case; it does not
expand the case into a full benchmark.

Automatic loading and inference from the published 300M repository also
passed. The run used checkpoint revision
`a38a62ae930d157484b331c2bf4241684573adba` and runtime revision
`d28d60c7ebe32c793d7e5387b258df3f2b3d093d`. The raw result is recorded in
[published loading evidence](esmfold2_300_published_loading.json).

## Evidence files

The 300M model-card quick start also completed with two protein chains and
`verbose=True`. Its CIF contains chains A and B, 250 atoms with finite
coordinates, and `?` for unknown confidence values. This checks the example
and export path, without claiming complex-prediction accuracy. See the
[quick-start result](esmfold2_300_quickstart.json).

The raw comparison and metadata are tracked beside this report:

- [comparison JSON](esmfold2_small_300_comparison.json)
- [reference metadata](esmfold2_small_300_reference_metadata.json)
- [candidate metadata](esmfold2_small_300_candidate_metadata.json)
- [artifact reload comparison](esmfold2_300_artifact_reload.json)
- [published artifact evidence](esmfold2_small_publication.json)
- [published loading evidence](esmfold2_300_published_loading.json)
