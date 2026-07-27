# ESMC-6B weights-only evidence archive

This is a pre-synthesis evidence index. It documents the full analysis output without converting correlations or weight geometry into biological or causal claims.

## Scope

- ESMC blocks: 0 through 79; block 50 produces state 51 and block 51 consumes it.
- Inputs or activations: none.
- Baseline evidence directory: `C:\Users\lhall\Desktop\Research\FastPLMs\artifacts\analysis\esmc_weight_geometry_results_20260725`.
- Raw checkpoint: `C:\tmp\esmc_weight_geometry_cache\biohub--ESMC-6B--45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a`.
- Completed stages: catalog, derived, raw, normalization, subspaces, statistics, figures, notebook.
- Scalar tables: CSV. Large vectors, bases, spectra, pairwise matrices, and neuron-level arrays: compressed NPZ.
- Numeric conventions: zero-based blocks, one-based produced states, FP64 accumulation where stated, deterministic randomized SVD for new singular vectors.

## CSV tables

| File | Rows | Columns | SHA-256 |
| --- | ---: | ---: | --- |
| `adjacent_raw_matrix_changes.csv` | 553 | 49 | `8da0abefc8d077bce5973328f75b0c1325170f5e124a0241c8efe881ce29ffd2` |
| `adjacent_spectral_transitions.csv` | 1,659 | 9 | `6079c14554d8615894d77712bf3f009d7d0ac6001e2b2536a734cfcc41a9af70` |
| `adjacent_subspace_turnover.csv` | 3,871 | 13 | `e7b681debce00b6399fb004e8e591db1465322bc3f36ac73ba30e8eedd470767` |
| `array_catalog.csv` | 5,061 | 10 | `a8806a60b0d6e901d3ea40784eb15551ef4f6e35f9fe61d54d409c559fe930e4` |
| `artifact_inventory.csv` | 596 | 4 | `f014b525a9166e2ebbe4d1eccea6d0fe19eb79089d8ace074b1dad887aaac253` |
| `attention_head_distribution_summary.csv` | 80 | 640 | `d4a5f742efdae384c9cf89b253a1a066779ab09cd573b9dfafe67a38d2d9e530` |
| `attention_head_permutation_geometry.csv` | 79 | 13 | `fedf687f138f711625b01920a0f040db2413721076f40c123c64bffb456c3af3` |
| `compression_pareto.csv` | 3,920 | 11 | `37da075012386e17a5ad828c77a03caaffd8f3ef0a838f6e3a4db6a5492179de` |
| `data_quality_checks.csv` | 42 | 5 | `913838e08f174800b6faeddbaf31dc1b5c965d8dab4a1a965d333151a6505c27` |
| `derived_layer_anomalies.csv` | 364,080 | 18 | `8c7ccc9334f654912e367fc708423d5680ba384b619dfd7885364e35f273f425` |
| `derived_layer_metrics_long.csv` | 438,706 | 6 | `efd47380f84dc76d49c823c4d29ab73376808ab91b4fec9b3baae8d9ff5d5298` |
| `derived_mixer_correlations.csv` | 91,020 | 14 | `9e155aa67d70ef2eaf88a644343e0294b94b04078dbef09b6ec2d5dbe6280205` |
| `ffn_neuron_summary.csv` | 80 | 154 | `3e4c78ffcea2de905938a7dcc3c38096bfb80c66c9209ab1bea9e576d542b94a` |
| `field_dictionary.csv` | 335 | 9 | `1f6b614386f38879d0ef49331ab13770db0e186e70b5a014a92549d92ab3d9df` |
| `layer_omnibus_balanced.csv` | 800 | 8 | `29a0bb7166d7b7722e6fc265d3144f902b6acf8a745a7a9958ce194171634092` |
| `layer_omnibus_scores.csv` | 800 | 11 | `815c815bc2fb3dd63a9345998b84a9409d2558efc5796406ddbcaf400debe988` |
| `mixer_profile_pairwise.csv` | 6 | 9 | `cccb303dffec9cc220c199797931f5df8e3b3448638c31f513c44b24ba906880` |
| `mixer_profile_summary.csv` | 4 | 15 | `0c1b3b4e1ce5a8e24a72f011cb5b294b52c00a55c2c457229c672fac2b67bfdd` |
| `normalization_adjacent_changes.csv` | 474 | 8 | `c73aa3a44688780a6ac1e7b2cda685ba0a64b95552115c7423481bf4f1a5a2aa` |
| `normalization_channel_depth_metrics.csv` | 15,360 | 13 | `7e3ec264efb38e515be8b49a09da4faacb87532b6d2786e2c8301dd53a7cc711` |
| `normalization_cross_role.csv` | 1,200 | 6 | `7a0fc9ddf2b28237c7a83135df350186ce9c549edbe502f0d58a32f890f4b938` |
| `normalization_vector_statistics.csv` | 480 | 23 | `b418395367a3bd424878e9ea1877f856f9e3c7c798e40dc6d77f09de8404ee07` |
| `output_data_quality_checks.csv` | 64 | 7 | `7b362f0f31856a38fb7a19fbc47a25a86fc493018bb4c1d2ce6e2682839a0329` |
| `projection_alignment_randomized_svd.csv` | 13,440 | 15 | `6fd483574c60efb35249d7ace9ce08b465994bcbc7b93d58d5d9d7cabacef48b` |
| `projection_checkpoint_pairwise.csv` | 72 | 14 | `e8112eb9f1718fb92e501822d0373dd7ea25d84dff2edab3ececfd3c4705c6a9` |
| `projection_checkpoint_statistics.csv` | 16 | 70 | `eaf152db8cd26f13500f7bee7b31dfac25868ba1abb09d28b00cbdfd152df41c` |
| `randomized_svd_sensitivity.csv` | 168 | 17 | `1d231c9813646a201f5b4f9c89c362c5b69aa67d3e93f8d9135fa118469634ac` |
| `raw_tensor_statistics.csv` | 560 | 97 | `a3400b695f023fd833448260d3c74a2bac72392c9211162555ffee1b664cd1d4` |
| `spectral_shape_metrics.csv` | 1,680 | 59 | `18a4ebb2ba4e4d93c0460fbe646fc34d5d5531e7186f1588793ae405777e2021` |
| `trajectory_eigenspectra.csv` | 560 | 5 | `3efd6dd815b85e5b861c05caf8bbe392fb0b09a7dc41c2734f65fff29d770a47` |
| `trajectory_local_geometry.csv` | 560 | 25 | `c2c54c190c1b3a5b632d44b55e6bbc55be8bd684203030b6f5b3e33b6f0f1438` |
| `within_block_subspace_geometry.csv` | 5,040 | 14 | `af52c82910f6c1f6a7bd5c26650230fc99a9b5d8ca74b16620db43abaf02951b` |

## Large arrays

See `output_array_catalog.csv` for every NPZ member, shape, dtype, element count, non-finite count, extrema, definition, and containing artifact. The original checkpoint is not duplicated into the result directory.

## Field definitions

See `output_field_dictionary.csv` for every CSV field and its dtype, row count, null count, distinct count, numeric range, and definition. See `field_dictionary.csv` and `array_catalog.csv` for the frozen baseline bundle.

## Interpretation boundary

These files support weight-space hypotheses only. Approximate top singular vectors are explicitly labeled by their randomized-SVD diagnostics. Four ESMFold2 checkpoints are highly related checkpoints and therefore consistency checks, not four independent experiments. No result here establishes effects on P@L, folding accuracy, perplexity, or biological information retention.
