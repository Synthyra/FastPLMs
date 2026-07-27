# Metric definitions and data conventions

## Indexing and scope

- `block` is zero-based from 0 to 79.
- `produced_state = block + 1`. ESMC state 51 is produced by block 50 and consumed by block 51.
- State 0 is the embedding state and has no producing transformer block.
- No sequences, activations, hidden states, model outputs, or forward passes are used.
- Matrix families are Q, K, V, attention output O, SwiGLU gate, SwiGLU value, and FFN down projection.

## Linear spectra

- Singular energy is `s_i^2 / sum_j s_j^2`.
- Stable rank is `||W||_F^2 / ||W||_2^2`.
- Participation ratio is `(sum_i s_i^2)^2 / sum_i s_i^4`.
- Spectral effective rank is `exp(-sum_i p_i log p_i)` for energy probabilities `p_i`.
- Spectral flatness is the geometric mean divided by the arithmetic mean of positive singular values.
- Spectral and energy Gini coefficients quantify concentration on `[0, 1]`.
- HHI is the sum of squared normalized mass and effective count is the exponential entropy.
- Energy-at-rank and tail-energy fields are exact functions of saved complete spectra.
- Gap ratio at rank `k` is `s_k / s_(k+1)`.
- Power-law slopes fit log singular value against log rank; exponential slopes fit log singular value against rank. Head and bulk windows are recorded in the code.
- The log-spectrum knee is the maximum perpendicular deviation from the line connecting the first and final positive log singular values. It is a descriptive knee, not a model-selection criterion.
- Adjacent spectral Jensen-Shannon, total variation, rank-Wasserstein, correlation, and cosine compare normalized full spectral profiles.

## Raw parameter distributions

- Mean, variance, RMS, and exact zero fraction scan the complete tensor with FP64 NumPy accumulation.
- Quantiles, skewness, kurtosis, sign fraction, and relative near-zero rates use a deterministic evenly spaced sample of at most 1,000,000 parameters. Fields are suffixed `sampled`.
- Row and column norm distributions are complete, not sampled.
- Coherence uses 256 deterministic evenly spaced row or column vectors, unit normalizes them, and summarizes off-diagonal cosine magnitudes.

## Randomized singular vectors

- New singular-vector bases use deterministic Gaussian randomized SVD at target rank 64 by default, 16 oversamples, and two power iterations.
- Every tensor records captured Frobenius energy and left/right orthogonality error.
- Basis arrays are saved in `bases/*.npz` as float32. Subspace metrics are accumulated in float64.
- Singular-vector inverse participation ratio is `sum_i v_i^4`; its reciprocal is effective coordinate support.
- Chordal distance is `sqrt(r - sum_i cos(theta_i)^2)`.
- Normalized overlap is the mean squared canonical correlation between equal-rank subspaces.

## Layer trajectories

- Baseline trajectory Gram, cosine, and Frobenius-distance matrices use all parameters of each matrix family.
- Kernel-PCA coordinates come from the positive eigenspectrum of the centered 80 by 80 layer Gram matrix.
- Speed is adjacent Frobenius distance, acceleration is the norm of the second coordinate difference, and turning angle is the angle between consecutive trajectory steps.
- Nearest layer excludes self. Depth gap records whether geometric proximity is local or nonlocal in depth.

## Attention heads

- Q, K, and V use 64 by 2560 per-head matrices. O uses the matching 2560 by 64 block.
- Q-K and V-O overlaps are canonical subspace overlaps. Fold projection overlaps compare the head output space with the fold projection row space.
- Head distribution rows preserve mean, spread, quantiles, skewness, kurtosis, Gini, entropy, and top-head concentration for every baseline head metric.
- Hungarian permutations are decomposed into fixed points, cycles, displacements, inversions, and gain over fixed-index matching.

## FFN neurons and normalization channels

- FFN neuron arrays preserve gate norm, value norm, matching down-column norm, three pairwise cosines, and the descriptive triple-strength proxy `||g_i|| ||v_i|| ||d_i||` for all 6,912 neurons in every block.
- Triple strength is a weight-scale proxy only. It is not an activation, attribution, or functional importance score.
- Normalization vectors are preserved in `normalization_vectors.npz`; channel-depth rows summarize all 2,560 channels for each of six vector roles.
- Cross-role correlations compare matching channel indices. Adjacent changes report cosine, relative L2 movement, maximum channel movement, and thresholded change fractions.

## Compression

- Effective storage uses reported quantization storage when available and ideal nonzero-value storage for sparsity methods. Index and metadata overhead for sparse formats is not modeled.
- Pareto optimality means no earlier configuration at equal or lower effective storage has lower Frobenius error within the same block and family.
- All compression metrics are parameter reconstruction metrics, not functional performance measurements.

## Anomalies, correlations, and multiple testing

- Depth trends are cubic least-squares fits over normalized block depth. Residuals are standardized by median absolute deviation.
- Local robust z-scores compare a block against the five blocks on each side, excluding itself and truncating at endpoints.
- Empirical outlier p-values rank absolute residual deviations among 80 blocks. Normal-approximation p-values from robust z-scores are labeled as approximations.
- Benjamini-Hochberg q-values are reported within each complete 80-layer metric series for layer anomalies.
- Mixer associations include Pearson, Spearman, cubic depth-detrended Pearson, lags -2 through +2, and exact circular shifts over every unique nonzero shift.
- Exact shift p-values have minimum resolution 1/80 = 0.0125. Checkpoint-consistency q-values adjust across the four related ESMFold2 profiles for one metric and lag.
- The four folds are consistency checks, not independent statistical replicates.

## Null and non-finite semantics

- Null means structurally inapplicable, unavailable because a stage was not selected, or undefined because a denominator is zero. The field dictionary gives counts by column.
- Infinity is never accepted in shipped numerical tables. Validation checks enforce this.
- Tiny negative Gram eigenvalues are numerical roundoff only when they lie below the recorded tolerance; the baseline preserves clamped negative mass.
