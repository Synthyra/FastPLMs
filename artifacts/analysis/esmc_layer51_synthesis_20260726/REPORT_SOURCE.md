# ESMC-6B State 51 Weight Geometry

## Technical summary

The state-51 ESMFold2 mixing coefficient is real, large, and reproducible across all four supported folding checkpoints. State 51 receives 9.99%-10.65% of the total softmax mass and ranks fourth in every checkpoint, while states 77-80 together receive 60.44%-65.41%. The four 81-state profiles are extremely similar (pairwise Pearson 0.988-0.997), so the observation is not a single-checkpoint artifact.

The key indexing fact is that ESMC state 51 is the output of transformer block 50 and the input to block 51. The weight evidence is much stronger for block 50, the producer, than for block 51, the consumer. Block 50 combines three scale-invariant signatures:

1. Its FFN value subspace is unusually aligned with the shared ESMFold2 folding projection. The rank-16 overlap is about 0.160 against an isotropic expectation of 0.100, remains unusual after the folding LayerNorm scaling approximation, and survives family-wise false-discovery control.
2. Its attention heads undergo the strongest multi-head value-to-output coupling event in the network: 10 of 40 heads have V-O overlap above 0.5 and 6 exceed 0.7, both global maxima. The same block has unusually dispersed V-O coupling and elevated mean O stable rank.
3. Its SwiGLU gate/down and value/down neuron geometry reorganizes sharply. The mean gate-down cosine is the global minimum and a strong local outlier, while the important effect is the broadened distribution, not the tiny negative mean itself.

The best weights-only interpretation is therefore a **projection-aligned write/relay event**. Block 50 appears to reorganize several attention heads and the FFN so that the resulting residual state contains directions preferentially read by the folding projection. A contact-oriented P@L probe and an SAE reconstruction/perplexity criterion need not reward those same directions. Figure S26 in the paper independently shows a pronounced trough near this depth before the large layers-60-70 rise, which strengthens the conclusion that state 51 is a task-specific waypoint rather than a generic maximum of representation richness.

This is structural evidence, not a causal demonstration. No sequences, activations, attention maps, SAE activations, or folding outputs were generated. Weight-space alignment can show that a pathway is available, but not that proteins actually use it.

## The apparent contradiction is between objectives, not measurements of one quantity

The three observations are compatible:

- P@L asks whether a layer exposes linearly or attention-accessible residue-contact information.
- The Figure S26 SAE metrics ask how difficult the layer state is to reconstruct and how much masked-language-model perplexity changes after SAE reconstruction.
- ESMFold2 mixing asks which normalized, projected layer states help the folding objective after joint training.

These are not interchangeable definitions of "information." A direction can be weak for contact recovery, easy for an SAE to reconstruct, or irrelevant to masked-token prediction, yet still be valuable to a learned folding trunk. Conversely, a state can be rich in general sequence information but redundant after the folding projection.

The ESMFold2 implementation also sharpens this distinction. For state `s`, hidden vector `h_s in R^2560`, shared LayerNorm `N`, shared bias-free projection `W in R^(256 x 2560)`, and learned mixing logits `c_s`:

```text
alpha_s = exp(c_s) / sum_t exp(c_t)
z = sum_{s=0}^{80} alpha_s W N(h_s)
```

The code applies the same `LayerNorm -> Linear(2560, 256, bias=False)` to every state and combines the projected states afterward. There is one shared projection, not 81 projections. Because LayerNorm is nonlinear, this is not generally equal to applying LayerNorm after a raw-state weighted sum.

The coefficient `alpha_51 ~= 0.10` therefore means that state 51 receives about one tenth of the convex mixing mass. It does **not** mean that exactly one tenth of the final tensor's information, norm, variance, or causal effect comes from that state. Projected states can differ in direction, can cancel, and can be redundant.

## The layer-index mapping localizes the event to block 50

ESMC exposes 81 ordered states from 80 transformer blocks:

```text
state 0     token embedding state
state 1     output of block 0
...
state 51    output of block 50
state 80    output of block 79, followed contextually by final normalization
```

Thus state 51 is produced by block 50 and consumed by block 51. Treating "layer 51" as block 51 alone would shift the causal candidate by one block. The analysis prespecified both blocks 50 and 51, the wider 48-53 region, and blocks 76-79 as late-depth controls.

The evidence concentrates on the producer. Block 50 produces state 51, has 10 heads with V-O overlap above 0.5 and 6 above 0.7, and is unusual across several scale-invariant families. Block 51 produces state 52, has only 2 heads above 0.5 and none above 0.7, and is weak or inconsistent as an isolated weight anomaly. State 51 is therefore best treated as a readout waypoint, not evidence that block 51 itself is uniquely special.

## State 51 is consistently preferred by all four folding checkpoints

The exact state-51 and states-77-80 softmax masses are:

- ESMFold2: 10.654% and 65.412%.
- ESMFold2 experimental cutoff: 9.988% and 60.439%.
- ESMFold2 experimental fast: 10.253% and 62.210%.
- ESMFold2 fast: 10.134% and 63.480%.

State 51 ranks fourth in all four models.

The profiles have pairwise Pearson correlations of 0.988-0.997. This consistency is meaningful, but the checkpoints are correlated descendants with related architectures and training recipes. They are robustness checks, not four independent biological replicates.

The agreement is not explained by identical projection matrices. Pairwise raw projection Frobenius cosines are only 0.121-0.228, while rank-16 row-subspace overlaps are 0.345-0.451 and full rank-256 overlaps are 0.263-0.339. The four folding heads therefore read related but substantially non-identical 256-dimensional subspaces while converging on almost the same layer-mixing profile. This makes the state-51 preference more informative than literal checkpoint duplication, without making the checkpoints statistically independent.

The late-layer dominance and the state-51 bump coexist. A softmax mixer can allocate most mass to final states while retaining one earlier checkpoint because that earlier projected state supplies a complementary basis direction or an easier-to-read feature bundle.

## Projection alignment is the strongest direct explanation

For a rank-`r` weight subspace basis `U_r` and the 256-dimensional row space `P` of the folding projection, the normalized overlap is

```text
overlap_r(U, P) = ||U_r^T P||_F^2 / r
                = (1/r) sum_{i=1}^r cos^2(theta_i)
```

where `theta_i` are principal angles. The score is 1 for full containment and 0 for orthogonality. For an isotropically random rank-`r` subspace in 2560 dimensions compared with a rank-256 projection row space,

```text
E[overlap_r] = 256 / 2560 = 0.10.
```

The comparison used input-side right singular vectors for Q, K, V, gate, and value matrices, and residual-output left singular vectors for O and the FFN down projection. It was repeated at ranks 16, 32, 64, 128, and 256, both with the raw projection and with an approximation that incorporates the preceding LayerNorm channel scaling.

The clearest block-50 result is the FFN value subspace:

- Rank-16 raw overlap: about 0.15995, local robust z = 2.46, phase-randomized p = 0.0003, BH q = 0.0240.
- Rank-16 LayerNorm-scaled overlap: about 0.15872, local robust z = 2.32, phase-randomized p = 0.0005, BH q = 0.0400.
- Rank-32 raw overlap: about 0.15266, local robust z = 1.74, phase-randomized p = 0.0006, BH q = 0.0480.

Every folding checkpoint has a positive block-50 local delta. At rank 16, value-subspace deltas are +0.0288 to +0.0431 for the raw projection and +0.0280 to +0.0423 after LayerNorm scaling.

The gate subspace is even more locally peaked:

- Rank-16 raw overlap: about 0.16359, local robust z = 4.07.
- Four-checkpoint local deltas: +0.0340 to +0.0519.
- The phase-randomized p-value is 0.0033, but the broad family-wise BH correction is q = 0.132. This is strong supporting evidence, not a standalone discovery under the strict multiplicity rule.

Across all 80 blocks, ESMFold2 mixing weights track gate-projection alignment unusually well. For the four-checkpoint mean raw rank-16 gate overlap, depth-detrended Pearson correlations are 0.786-0.845 across the four mixer profiles. Every exact circular-shift p-value is 0.0125, the smallest possible with 79 nonzero circular shifts. The mean depth-detrended correlations remain strong across ranks:

| Weight subspace and rank | Mean depth-detrended Pearson r |
|---|---:|
| Gate, rank 16 | 0.823 |
| Gate, rank 32 | 0.814 |
| Gate, rank 64 | 0.797 |
| Value, rank 256 | 0.787 |
| Q, rank 256 | 0.765 |
| K, rank 256 | 0.720 |

Intuition: the mixer appears to prefer blocks whose learned operators expose a compact set of input directions that the folding projection can read. Block 50 is a local maximum of exactly that compatibility.

## Attention geometry indicates a coordinated multi-head event

Each Q, K, and V matrix was split into 40 heads of shape `64 x 2560`; O was split into 40 corresponding `2560 x 64` blocks. For each head, the analysis measured norms, ranks, Q-K principal angles, V-O coupling, projection overlap, within-layer redundancy, and adjacent-layer similarity. Adjacent heads were compared both by fixed index and by a Hungarian assignment over joint QKVO similarity.

Block 50 is exceptional at the layer level:

- 10 heads have V-O overlap above 0.5, the global maximum; local robust z = 3.60 and BH q = 0.0080.
- 6 heads exceed 0.7, also the global maximum; phase-based BH q = 0.0040.
- V-O overlap standard deviation is 0.25668; local robust z = 3.28 and BH q = 0.0080.
- Mean O stable rank is 31.7635; local robust z = 2.62 and BH q = 0.0080.

The most coupled heads are head 16 (V-O overlap 0.818; Q spectral norm 7.28), head 20 (0.814; Q spectral norm 14.34), head 22 (0.784), head 38 (0.779), and head 39 (0.765).

The event is not carried by a single head. Direct per-head overlap with the folding projection does not survive detrended significance testing; layer-mean correlations are only moderate (roughly 0.39-0.57, p about 0.06-0.10). The stronger signal appears after aggregating head behavior and considering the FFN/projection pathway.

The adjacent-layer matching analysis shows a local reorganization:

- Transition 49 -> 50 has the highest Hungarian-matched head similarity in the network, 0.09524.
- Transition 50 -> 51 is second, 0.09322.
- Transition 51 -> 52 is eighth, 0.08791.
- The corresponding matching-gain ranks are 2, 4, and 3.

Absolute similarities are low because full QKVO head operators are high-dimensional and change substantially with depth. The relevant feature is their relative ranking and the large benefit from allowing head permutation. This is consistent with head roles being reassigned or reorganized across the 49-52 boundary.

## The FFN shows redistribution, not simple amplification

For each of the 6912 SwiGLU neurons, gate and value rows were matched with the corresponding down-projection column. The analysis measured gate-value cosine, gate-down cosine, value-down cosine, norm ratios, subspace overlap, redundancy, and triple-strength concentration.

At block 50:

- Mean gate-down cosine is -0.008076, the global minimum and a very strong local outlier (local robust z = -4.91, BH q = 0.0080).
- Value-down cosine standard deviation is 0.31397, BH q = 0.0040.
- Gate-down cosine standard deviation is 0.30242, BH q = 0.0100.

The mean cosine is numerically tiny, so "anti-alignment" would be an overstatement. The robust finding is a broadened and reorganized neuron-wise coupling distribution: more neurons occupy unusually aligned and anti-aligned tails even though the average remains near zero. This fits a routing interpretation in which block 50 creates a heterogeneous collection of write directions, some of which are strongly accessible to the folding projection.

## Spectra point to compact dominant directions, but not a low-rank block

For each matrix `W` with singular values `s_1 >= ... >= s_m`, the analysis used the uncentered operator spectrum and centered row/column PCA. Core quantities were:

```text
p_i                       = s_i^2 / sum_j s_j^2
stable_rank(W)            = ||W||_F^2 / ||W||_2^2
participation_ratio(W)    = (sum_i s_i^2)^2 / sum_i s_i^4
effective_rank(W)         = exp(-sum_i p_i log p_i)
leading_energy_fraction   = s_1^2 / ||W||_F^2
rank_tau                  = min r such that sum_{i<=r} p_i >= tau
rank-r relative error     = sqrt(1 - sum_{i<=r} p_i)
rank-r storage fraction   = r(m+n) / (mn)
```

At block 50, the Q operator has:

- effective rank 718.51,
- participation ratio 175.08,
- stable rank 17.23,
- leading energy fraction 0.0580,
- spectral norm 25.95.

The combination of a high entropy-based effective rank and a low stable rank means that Q is not globally low-rank, yet its largest singular directions dominate operator norm. That is exactly the kind of spectrum in which a compact readout can select a meaningful leading subspace without the whole matrix being compressible to a tiny rank.

Across depth, mixing mass is inversely associated with Q spectral dimension after removing the smooth depth trend:

- Q participation ratio: mean depth-detrended r about -0.757.
- Q effective rank: mean r about -0.672.
- K effective rank: mean r about -0.675.
- Q stable rank: mean r about -0.561.

For Q effective rank, lagged correlations are -0.567, -0.652, -0.672, -0.554, and -0.489 for lags -2 through +2. The prespecified state-to-producing-block alignment is strongest. Raw correlations can have the opposite sign because both quantities trend with depth, so the detrended result is the relevant one.

Numerical checks support these measurements. Direct FP32 SVD on blocks 0, 50, 51, and 79 differs from the Gram-based leading-64 spectrum by at most 4.48e-4 relatively; captured-energy error is at most 6.29e-4; energy-rank thresholds differ by at most one. The negative Gram-eigenvalue mass for block-50 down projection is 7.17e-19, only 5.36e-20 of Frobenius-squared mass, which is numerical roundoff.

## Weight-vector intrinsic dimension does not provide a robust state-51 explanation

The weight-vector analyses treated normalized rows and columns as point clouds. These are dimensions of learned weight vectors, not dimensions of protein representations.

For nearest-neighbor distances `T_1 < T_2 < ...`, the TwoNN estimator uses `mu = T_2/T_1` and the relation

```text
-log(1 - F(mu)) = d log(mu),
```

while the local Levina-Bickel estimator is

```text
d_hat_i(k) = [ (1/(k-1)) sum_{j=1}^{k-1} log(T_k / T_j) ]^-1.
```

The raw Euclidean down-row analysis produces apparent dimension excursions near blocks 50-51, and one raw value-column TwoNN score at block 51 reaches a local z near 6.9. Those excursions disappear after unit normalization and cosine-equivalent analysis. They are therefore driven mainly by vector norms, not angular manifold complexity, and fail the prespecified normalization-sensitivity requirement.

This negative result is useful: the state-51 effect is not well described as a generic collapse or expansion of weight-vector intrinsic dimension.

## The 80-layer parameter trajectories remain high-dimensional

For each tensor family, the 80 vectorized matrices were treated as a parameter trajectory. Pairwise distances and similarities were

```text
d_F(i,j) = ||W_i - W_j||_F
cos(i,j) = <vec(W_i), vec(W_j)> / (||W_i||_F ||W_j||_F).
```

Centered kernel PCA on the `80 x 80` Gram matrix gives effective trajectory dimensions of 70.16 for down, 70.32 for gate, 73.24 for K, 76.85 for O, 72.47 for Q, 72.46 for V, and 66.32 for value. Every family has algebraic rank 79.

The layer paths are therefore not confined to a small global subspace. Several families increase step size entering block 50, but the turning angle and curvature are not uniquely extreme there. In other words, block 50 is a coordinated local change in several operator geometries, not a singular global kink in the entire 6B-parameter trajectory.

Change-point rankings support a broader transition zone. The 47 -> 48 derivative is the third-largest multi-family effective-rank change and the second-largest stable-rank change. Transitions 49 -> 50 and 50 -> 51 rank around 13-14 by effective-rank derivative, with 50 -> 51 eighth by stable-rank derivative. The evidence is better described as a blocks-48-53 regime with a particularly readable state emitted by block 50.

## Normalization, raw scale, and parameter reconstruction are not sufficient explanations

The analysis examined all attention/FFN LayerNorm weights and biases plus Q/K normalization vectors, including means, variance, coefficient of variation, tails, adjacent cosine, amplified/suppressed channel fractions, channel-depth outliers, and cross-role similarity.

No normalization statistic at blocks 50 or 51 survives 5% FDR. The closest is the block-50 attention-input-norm suppressed-channel fraction at q about 0.052. Likewise, ordinary matrix norms and scale statistics do not explain the effect after scale-invariant overlap and normalized-cosine checks.

Weight reconstruction also does not single out the region. The analysis evaluated truncated SVD, symmetric per-row INT8 and INT4, unstructured magnitude sparsity at 10%, 25%, 50%, and 75%, and deterministic 2:4 sparsity. For each reconstruction `W_hat`, it recorded

```text
relative Frobenius error = ||W - W_hat||_F / ||W||_F
flattened cosine         = <vec(W), vec(W_hat)> / (||W||_F ||W_hat||_F)
spectral distortion      = | ||W_hat||_2 - ||W||_2 | / ||W||_2.
```

No block-50 or block-51 quantization/sparsity reconstruction metric survives family-wise FDR. This argues against a simple claim that the state is preferred because its producer is uniquely easy to compress.

## Statistical treatment and why some striking scores were not promoted

For each metric family, the analysis fit a smooth depth trend, formed residuals, and calculated local and global robust anomaly scores. For a target value `x` and neighboring median `m` with median absolute deviation `MAD`,

```text
robust_z = 0.67448975 (x - m) / MAD.
```

Associations with the four mixing profiles used Pearson and Spearman correlation, depth-detrended correlation, lags -2 through +2, and circular-shift nulls. With 80 blocks, the exact nonzero-shift test has 79 null alignments, so its minimum attainable p-value is 1/80 = 0.0125. Broader 10,000-draw phase-randomized tests were used where specified. Benjamini-Hochberg correction controlled false discoveries within metric families.

Discrete metrics with zero local MAD can generate infinite or enormous robust z-scores even when their practical difference is one rank or one count. Those were retained in the raw archive but were not promoted unless a continuous companion metric, numerical sensitivity check, and normalization check agreed. The report likewise does not treat every exploratory p-value as independent, because thousands of related spectral and geometric metrics are highly correlated.

The exact Gram-spectrum results were checked against direct FP32 SVD on blocks 0, 50, 51, and 79. Across these validation cases, the maximum relative error among the leading 64 singular values was `4.48e-4`, the maximum energy discrepancy was `6.29e-4`, and the 90%, 95%, and 99% energy-rank thresholds differed by at most one. Negative Gram eigenvalue mass was negligible. By contrast, independent randomized rank-64 bases had a worst leading-spectrum difference of 10.2%, a median difference of 3.0%, and a minimum replicate subspace overlap of 0.526. Consequently, inferential claims in this report rely on exact Gram/direct-SVD quantities; randomized tail bases are supporting diagnostics only.

## Integrated interpretation

### Hypothesis 1: block 50 writes a folding-readable residual subspace

**Confidence: moderate, strongest weights-only hypothesis.** The FFN gate/value subspaces at block 50 align with the shared folding projection, the association tracks the mixer across depth and all four checkpoints, and the value result survives raw and LayerNorm-scaled sensitivity checks. This directly explains why the folding model can prefer state 51 even when a contact probe does not.

### Hypothesis 2: the write is distributed across heads and consolidated by the FFN

**Confidence: moderate.** Ten high-coupling V-O heads, six extremely high-coupling heads, elevated O stable rank, Hungarian head reorganization, and FFN coupling dispersion all coincide at block 50. The weak per-head projection correlation argues for an ensemble effect rather than a single specialized "folding head."

### Hypothesis 3: state 51 is a transient interface state

**Confidence: moderate-low.** State 51 may be a convenient intermediate basis that block 50 emits and block 51 immediately transforms. This explains why the producer is unusual, the consumer is much less so, and generic representational metrics can dip at the same depth. A downstream mixer can select a transient interface even if it is not the best standalone representation.

### Hypothesis 4: P@L and SAE metrics miss low-volume task-specific directions

**Confidence: plausible but not established by weights.** A low-dimensional direction can matter strongly to a 256-wide folding projection while contributing little to average reconstruction loss, masked-LM perplexity, or attention-derived contact precision. The spectral and overlap results make this geometrically plausible, but only activation interventions can demonstrate it.

### Alternative explanations that remain open

- The four mixers may agree because of shared initialization, data, or training lineage rather than independent convergence.
- Softmax coefficients can compensate for correlations and cancellations among projected states; a large coefficient is not a unique attribution.
- The folding projection may align with a weight subspace that is rarely occupied by real protein activations.
- A structure-specific signal could be encoded elsewhere in the 48-53 region, with state 51 acting only as the most convenient linear combination point.

## What the weights establish and what they do not

**Established:** state 51 is heavily weighted in every supported checkpoint; the relevant producer is block 50; and block 50 has unusual projection, head, and FFN geometry.

**Rejected or not supported:** the effect is only matrix scale; block 51 itself is a robust isolated anomaly; one attention head causes the state-51 mass; weight-vector intrinsic dimension explains the effect after unit normalization; or block 50 is uniquely more compressible.

**Unknown:** whether the aligned directions activate on real proteins, causally improve folding accuracy, or reflect biology rather than the particulars of training.

The overall validation rating is **ready to share with weights-only caveats**. All 64 saved data-quality checks pass, numerical SVD checks agree, the strongest projection result survives LayerNorm and scale sensitivity, and null findings are explicitly preserved. The causal claim remains intentionally withheld.

## Recommended activation follow-up

The next phase should be small and surgical rather than another broad survey:

1. Decompose the actual projected contribution `alpha_s W N(h_s)` by state on a fixed protein panel. Measure per-state norm, cancellation, covariance, and contribution to the final 256-vector.
2. At state 51, ablate only the rank-16 block-50 value subspace that overlaps the folding projection. Compare with an equal-rank orthogonal control and matched-norm random controls.
3. Intervene on the block-50 heads with the highest V-O coupling, individually and jointly. Test whether their effects are additive, redundant, or gated by the FFN.
4. Repeat with state swaps: replace state 51 by interpolations of states 50 and 52 after LayerNorm, holding the mixer and projection fixed.
5. Measure both folding metrics and the original P@L/SAE metrics. The hypothesis predicts a selective folding deficit larger than the change in generic contact or reconstruction scores.
6. Train or inspect SAEs specifically on the projection-aligned component and its orthogonal complement. A whole-state SAE can dilute a compact folding-relevant subspace.

The decisive experiment is an equal-rank, equal-norm, projection-aware ablation. If the aligned block-50 subspace is causal, removing it should hurt ESMFold2 more than removing an orthogonal block-50 subspace, even if both perturbations have similar effects on P@L or masked-LM perplexity.

## Scope and reproducibility

This pass analyzed the pinned Biohub ESMC-6B checkpoint at revision `45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a`, all 80 transformer blocks, and the four pinned ESMFold2 projection/mixer subsets. It streamed checkpoint tensors without instantiating or executing ESMC. The primary matrix families were Q, K, V, O, SwiGLU gate, SwiGLU value, and FFN down, plus all block normalization vectors.

The final server replication ran on an NVIDIA GH200 480GB system. Six ESMC shard SHA-256 hashes were verified before analysis. The evidence archive records every CSV field and NPZ array, completion/provenance manifests, the executed notebook, 300 dpi figures, and the 64/64 data-quality check result. No hidden states were collected and `model_execution` is false in provenance.

The published Figure S26 was visually inspected from the supplied [bioRxiv preprint](https://doi.org/10.64898/2026.06.03.729735). Its authors interpret layers 60-70 as carrying the highest SAE reconstruction/perplexity information, with gradual growth through layers 0-50 and a steep final decline. The present report uses that curve only as external context and does not digitize it or treat it as newly generated evidence.

## Further questions

- Does state 51 remain fourth-ranked if the mixer is refit with one state removed at a time, or are several states exchangeable?
- Are the projection-aligned directions enriched for geometry, secondary structure, chain-interface, or confidence features?
- Do monomers, multimers, disordered proteins, and long-repeat proteins use the state-51 component differently?
- Did the mixer bump emerge early in ESMFold2 training, or only after the folding trunk learned to exploit block-50 directions?
- Does an independently initialized folding head recover the same state-51 preference and subspace alignment?
- Is the Figure S26 trough preserved across SAE seeds, sparsity levels, normalization choices, and activation sampling panels?
