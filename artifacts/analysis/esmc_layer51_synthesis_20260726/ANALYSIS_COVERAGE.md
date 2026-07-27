# Analysis coverage ledger

| Lens | Status | Preserved outputs | Evidence boundary |
| --- | --- | --- | --- |
| Exact uncentered operator spectra | complete | baseline spectra NPZ, tensor metrics | Full singular values, no vectors |
| Centered row and column PCA | complete | baseline spectra NPZ, tensor metrics | Weight-neuron geometry only |
| Spectral concentration, gaps, tails, slopes, knees | complete | spectral shape CSV | Descriptive shape diagnostics |
| Low-rank reconstruction | complete | tensor metrics | Parameter error only |
| INT8, INT4, unstructured and 2:4 reconstruction | complete | compression and Pareto CSV | No performance claim |
| Raw value distributions | complete after raw stage | raw tensor statistics | Sampled higher moments are labeled |
| Row and column norm concentration | complete after raw stage | raw tensor statistics | Full vector norms |
| Row and column coherence | complete after raw stage | raw tensor statistics | Deterministic 256-vector diagnostic |
| Singular-vector localization | complete after raw stage | bases NPZ, raw statistics | Randomized rank-64 approximation |
| Weight-vector intrinsic dimension | complete in baseline | intrinsic-dimension CSV | Raw and unit-normalized geometries |
| Full-matrix layer trajectory | complete | trajectory NPZ and local geometry | 80-point parameter path |
| Adjacent raw matrix change | complete after raw stage | adjacent raw changes | Fixed indices plus scale adjustment |
| Adjacent singular-subspace turnover | complete after subspace stage | adjacent subspace CSV | Randomized top subspaces |
| Within-block residual read/write circuits | complete after subspace stage | within-block subspace CSV | Linear weight-space alignment |
| Attention-head spectra and coupling | complete | baseline head CSV, distributions | All 40 heads by 80 blocks |
| Head permutation and reorganization | complete | transition and permutation CSV | Hungarian QKVO matching |
| FFN neuron norms, cosines, strength concentration | complete after raw stage | FFN NPZ and summary CSV | All 552,960 neuron-block pairs |
| Normalization vector distributions | complete | baseline and extended normalization CSV | All blocks and final norm |
| Normalization channel depth persistence | complete after raw stage | normalization NPZ and channel CSV | All 15,360 role-channel trajectories |
| ESMFold2 mixer concentration and distances | complete | mixer summary and pairwise CSV | Four related checkpoints |
| ESMFold2 projection pair geometry | complete after subspace stage | projection pairwise CSV | Raw and LN-scaled approximation |
| Projection-to-block/head alignment | complete | baseline and randomized alignment CSV | Shared projection per fold |
| Local and depth-adjusted anomaly tests | complete | anomaly CSVs | Multiple-testing scopes explicit |
| Mixer correlations and lag tests | complete | correlation CSVs | Exact circular-shift null |
| Cross-source omnibus scores | complete | omnibus CSVs | Both metric-weighted and source-balanced |
| Inputs, activations, hidden states, outputs | intentionally excluded | none | Outside weights-only scope |
| Functional ablation, P@L, folding accuracy | intentionally excluded | none | Requires model execution |
| SAE activation analysis | intentionally excluded | none | Requires activations |
