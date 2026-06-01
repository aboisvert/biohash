# BioHash: Paper Extraction and Implementation Notes

Implementation-oriented digest of **Bio-Inspired Hashing for Unsupervised Similarity Search** (Ryali, Hopfield, Grinberg, Krotov; ICML 2020).

Source: [biohash-ryali20a.pdf](./biohash-ryali20a.pdf)

---

## 1. Summary

**BioHash** is an unsupervised locality-sensitive hashing (LSH) method that:

1. **Expands** input vectors from dimension `d` to a high-dimensional latent space of size `m` (with `m > d`).
2. **Learns** synaptic weights `W` from data using a biologically plausible, local Hebbian/anti-Hebbian rule (from Krotov & Hopfield, 2019).
3. **Sparsifies** the latent representation via **k-Winner-Take-All (k-WTA)**, keeping only the top-`k` most activated hidden units.
4. **Encodes** each item as a sparse binary hash in `{-1, +1}^m` with exactly `k` active (`+1`) bits.

Compared to **FlyHash** (random projections + k-WTA), BioHash replaces random weights with **data-driven learned weights**, improving retrieval quality while remaining online, scalable, and biologically plausible.

The paper also proposes **BioConvHash**, a convolutional variant that learns normalized patch filters, applies cross-channel inhibition, max-pools, and then runs a BioHash layer on the resulting features.

---

## 2. Problem Formulation

### Similarity search

Given:

- Query `q ∈ R^d`
- Database `X` with `n` items in `R^d`
- Similarity measure `sim(q, x)`

Goal: retrieve the top-`R` database items most similar to `q`.

BioHash addresses this by mapping each item to a binary hash code and ranking candidates by **Hamming distance** in hash space.

### Notation

| Symbol | Meaning |
|--------|---------|
| `d` | Input dimension |
| `m` | Hash layer size (number of hidden units) |
| `k` | Hash length (number of active `+1` bits) |
| `W ∈ R^{m×d}` | Input-to-hash synaptic weight matrix |
| `W_μ` | Row `μ` of `W` (weights for hidden unit `μ`) |
| `x ∈ R^d` | Input data point |
| `y ∈ {-1, +1}^m` | Sparse binary hash code |
| `a = k/m` | Activity fraction (sparsity parameter) |
| `p` | Norm exponent for weight normalization / metric |
| `r` | Rank of the anti-winner unit receiving repulsive update |
| `Δ` | Anti-Hebbian strength for rank `r` |
| `τ` | Learning dynamics time constant |

**Hash length:** the paper defines hash length as `k` when exact Hamming distance computation is `O(k)`.

---

## 3. Relationship to Prior Work

### Classical LSH

- Maps `d`-dimensional input to **low**-dimensional codes (`m ≪ d`, typically `k = m`).
- Uses random projections (e.g., SimHash).
- Storage: `k` bits per database entry.

### FlyHash

- Maps to **high**-dimensional sparse codes (`m ≫ d`, only `k ≪ m` units active).
- Uses **random** weights `W`.
- Applies k-WTA to keep ~5% of units active.
- Baseline settings from the paper: `m = 10d`, PN→KC sampling rate `0.1`.

### BioHash

- Same sparse expansive architecture as FlyHash.
- **Learns** `W` from unlabeled training data.
- For `p = 2` and `Δ = 0`, reduces to an online, biologically plausible form of **spherical K-means**.

### NaiveBioHash (ablation baseline)

- Uses the same learning dynamics (Eq. 1).
- **No** sparse expansion: projects into `k` dense hidden units.
- Binarizes activations by sign → hash length `k` without k-WTA over `m > k` units.

---

## 4. Core Algorithm: BioHash

### 4.1 Weighted inner product

Each hidden unit `μ` uses a metric-weighted inner product:

```
⟨x, y⟩_μ = Σ_{i,j} η^μ_{i,j} x_i y_j
```

where:

```
η^μ_{i,j} = |W_{μi}|^{p-2} · δ_{ij}
```

For `p = 2`, this is the standard dot product: `⟨W_μ, x⟩ = W_μ · x`.

For `p = 1`, weights converge to the L1 unit sphere (`||W_μ||_1 = 1`), encouraging sparse synapses.

### 4.2 Learning rule (continuous time)

For each training example `x`, synapses update according to:

```
τ · dW_{μi}/dt = g[Rank(⟨W_μ, x⟩_μ)] · (x_i - ⟨W_μ, x⟩_μ · W_{μi})
```

**Rank operator:** sorts inner products from largest (`μ = 1`) to smallest (`μ = m`).

**Rank-dependent gain `g[μ]`:**

```
g[μ] =  1      if μ = 1        (winner: strongest response)
        -Δ     if μ = r        (anti-winner: repulsive update)
         0     otherwise
```

**Interpretation:**

- The **winner** unit (`μ = 1`, highest `⟨W_μ, x⟩`) moves toward the input (Hebbian).
- The **anti-winner** unit at rank `r` is repelled (anti-Hebbian).
- All other units are silent for this update step.

Updates depend only on pre- and post-synaptic activity → **local, biologically plausible** plasticity.

**Important:** this is **not** gradient descent on an explicit loss, though the dynamics decrease an associated energy function (Eq. 3 in the paper). Synapses converge to the **unit p-norm sphere**.

### 4.3 Special case: spherical K-means

When `p = 2` and `Δ = 0`:

- Energy reduces to online **spherical K-means** (Dhillon & Modha, 2001).
- Only the winning unit updates per example.
- Hyperparameters `p` and `Δ` generalize beyond spherical K-means while keeping biological plausibility.

The paper reports that **non-zero `Δ` improves performance empirically**.

### 4.3.1 Energy function and winner selection

The dynamics minimize (but not via gradient descent) an energy over training examples indexed by `A`:

```
E = - Σ_A  Σ_{μ=1}^{m}  g[Rank(⟨W_μ, x^A⟩_μ)] · ⟨W_μ, x^A⟩_μ / ||W_μ||_p
```

For each training example `x^A`, the **activated hidden unit** is:

```
μ̂ = argmax_μ  ⟨W_μ, x^A⟩_μ
```

The time derivative of `E` under the learning dynamics is non-positive (Eq. 4 in the paper; shown for `Δ = 0`, with a similar result for `Δ ≠ 0`).

### 4.4 Training procedure (implementation outline)

For each training example `x`:

1. **Compute scores** for all hidden units: `s_μ = ⟨W_μ, x⟩_μ`.
2. **Rank** units by `s_μ` descending.
3. **Update winner** (`rank 1`): apply learning rule with `g = 1`.
4. **Update anti-winner** (`rank r`): apply learning rule with `g = -Δ`.
5. **Optionally renormalize** each `W_μ` to unit p-norm (synapses converge to p-sphere under continuous dynamics).

Repeat over the training set for multiple passes (epoch count not specified in main text).

### 4.5 Hash encoding (inference)

After training, encode query/database item `x` as:

```
y_μ = +1   if ⟨W_μ, x⟩_μ is among the top k scores
      -1   otherwise
```

This is **k-WTA sparsification** followed by binarization, identical in structure to FlyHash but with learned weights.

**Retrieval:** rank database items by Hamming distance `d_H(y_query, y_db)`.

### 4.6 Hyperparameters

| Parameter | Role | Paper notes |
|-----------|------|-------------|
| `m` | Hash layer width | `m = k/a`; larger `m` → sparser codes at fixed `k` |
| `k` | Number of active bits | Evaluated in `{2, 4, 8, 16, 32, 64}` |
| `a = k/m` | Activity fraction | MNIST: `a = 0.05` (5%); CIFAR-10: `a = 0.005` (0.5%; validation optimum was 0.25% but gain over 0.5% was small) |
| `p` | Norm / metric exponent | `p = 2` → dot product; `p = 1` → sparse synapses |
| `r` | Anti-winner rank | Not explicitly fixed in main text |
| `Δ` | Anti-Hebbian strength | Non-zero values help empirically; exact value not in main text |
| `τ` | Learning time constant | Scales update magnitude; discrete step not specified |

**Example dimensions from paper:**

- MNIST, `k = 2`, `a = 0.05` → `m = 40`
- CIFAR-10, `k = 2`, `a = 0.005` → `m = 400`

---

## 5. BioConvHash (Convolutional Variant)

BioConvHash adds a convolutional front-end before the BioHash layer.

### 5.1 Architecture

```
Input image
  → Convolutional layer (learned filters, same plasticity rule as Eq. 1)
  → Cross-channel inhibition (k_CI-WTA per spatial location)
  → Max pooling
  → BioHash layer (Section 4)
  → Sparse binary hash
```

### 5.2 Key details

1. **Patch normalization:** image patches are normalized to **unit vectors** before computing inner products with filters. This acts like divisive normalization and improves robustness to local intensity variation ("shadows").

2. **Filter learning:** same local learning dynamics (Eq. 1) applied to image patches (following Grinberg et al., 2019).

3. **Cross-channel inhibition:** if there are `F` convolutional filters, keep only the top `k_CI` activations per spatial location; suppress the rest.

4. **Max pooling** after cross-channel inhibition.

5. **BioHash layer** on pooled features as in Section 4.

### 5.3 BioConvHash hyperparameters (from experiments)

| Dataset | Filters | Kernel sizes | Optimal `k_CI` (approx.) |
|---------|---------|--------------|--------------------------|
| MNIST | 500 | 3, 4 | 10 |
| CIFAR-10 | 400 | 3, 4, 10 | 1 |

Cross-channel inhibition is **critical** for good performance; dense channel activations degrade retrieval substantially (see paper Table 4).

---

## 6. Complexity and Storage

### Classical LSH

- Typically `k = m` and `m ≪ d`.
- Storage: **k bits** per database entry.
- Hamming distance: **O(k)**.

### BioHash / FlyHash

- Typically `m ≫ k` and `m > d`.
- Storage: **k · log₂(m) bits** per entry (store indices of active bits, not a dense `m`-bit vector).
- Hamming distance: **O(k)** if active indices are stored as sorted lists (intersection of two length-`k` lists).

### Metabolic / energy analogy

A sparse `m`-dimensional code with only `k` active units has similar "cost" to a dense `k`-dimensional code, but preserves more similarity information (per paper's empirical results).

---

## 7. Evaluation Protocol

### 7.1 Metric

**Mean Average Precision (mAP):**

```
AP(q)@R = (1 / Σ Rel(l)) · Σ_{l=1}^{R} Precision(l) · Rel(l)

mAP@R = (1 / |Q|) · Σ_{q∈Q} AP(q)@R
```

- `Rel(l) = 1` if retrieval `l` is relevant (same class label), else `0`.
- **mAP@All:** `R` = entire database (used for MNIST).
- **mAP@1000:** top 1000 retrievals (used for CIFAR-10).

Labels are used **only for evaluation**; training is fully unsupervised.

### 7.2 Datasets and splits

| Dataset | Size | Input | Classes | Query set | Train + database | Metric |
|---------|------|-------|---------|-----------|------------------|--------|
| MNIST | 70k | 28×28 greyscale | 10 digits | 100 per class (1k total) | remaining 69k | mAP@All |
| CIFAR-10 | 60k | 32×32×3 | 10 classes | 1000 per class (10k total) | remaining 50k | mAP@1000 |

Ground-truth relevance: **same class label**.

### 7.3 Hash lengths evaluated

`k ∈ {2, 4, 8, 16, 32, 64}` — the paper emphasizes strong performance at **small k**, where FlyHash also excels over classical LSH.

### 7.4 Baselines compared

- **Random:** LSH (SimHash), FlyHash
- **Data-driven classical:** PCAHash, Spectral Hashing (SH), ITQ
- **Deep hashing:** DeepBit, DH, USDH, UH-BNN, SAH, GreedyHash
- **Ablation:** NaiveBioHash

FlyHash baseline: `m = 10d`, sampling rate `0.1`.

### 7.5 Key empirical findings

1. **BioHash** substantially outperforms FlyHash and most baselines, especially at small `k`.
2. Performance saturates around `k = 16` on MNIST (diminishing returns beyond).
3. **BioConvHash** further improves results on raw images (MNIST, CIFAR-10).
4. **Functional smoothness:** BioHash preserves local input-space similarity in hash space much better than LSH, while keeping global similarities low (Table 3 in paper).
5. On **VGG16 fc7 features** of CIFAR-10, BioHash (`mAP@1000 = 63.47` at `k = 16`) beats recent deep unsupervised hashing methods.
6. Optimal sparsity varies by dataset (~5% activity MNIST, ~0.25–0.5% CIFAR-10).

---

## 8. Intuition

Hidden units behave like **particles** that:

- Are **attracted** to high-density regions of the data distribution (winner learns toward inputs in dense regions).
- **Repel** each other via the anti-winner update (prevent collapse onto a single point).

After learning, reference vectors `W_μ` are concentrated where data density is high but spread across the support of the distribution. For hashing:

- Nearby inputs activate **overlapping** top-`k` units → low Hamming distance.
- Distant inputs activate **disjoint** sets → high Hamming distance.

BioHash preferentially allocates resolution to **local** distances over global distances (verified via functional smoothness analysis).

---

## 9. Pseudocode

### BioHash training

```
initialize W ∈ R^{m×d}  // initialization not specified in main text
for epoch in 1..num_epochs:
  for x in training_set (shuffled):
    for μ in 1..m:
      s[μ] = ⟨W[μ], x⟩_μ
    ranks = argsort(s, descending=True)
    winner = ranks[0]
    anti_winner = ranks[r - 1]   // rank r (1-indexed)

    // When Δ = 0, only the winner receives a non-zero update (spherical K-means limit)
    for μ in {winner, anti_winner}:
      g = 1 if μ == winner else -Δ
      if g == 0: continue
      for i in 1..d:
        W[μ,i] += (dt/τ) * g * (x[i] - s[μ] * W[μ,i])
      normalize W[μ] to unit p-norm  // optional if enforcing explicitly
```

### BioHash encoding

```
function encode(x, W, k):
  for μ in 1..m:
    s[μ] = ⟨W[μ], x⟩_μ
  top_k = set of k indices with largest s[μ]
  for μ in 1..m:
    y[μ] = +1 if μ in top_k else -1
  return y
```

### Retrieval

```
function retrieve(query, database_hashes, R):
  scores = [(hamming_distance(query_hash, db_hash), index) for each db item]
  return top R items by lowest Hamming distance
```

For sparse codes, store only the `k` active indices and compute Hamming distance as:

```
d_H = 2k - 2 · |active(query) ∩ active(db)|
```

when both codes have exactly `k` active bits.

---

## 10. Open Implementation Questions

The main paper text leaves several details unspecified. These should be resolved from the supplementary material, reference code, or experimentation:

| Question | Status in main text |
|----------|---------------------|
| Weight initialization (`W` at t=0) | Not specified |
| Discrete learning rate `dt/τ` | Not specified |
| Number of training epochs | Not specified |
| Anti-winner rank `r` | Not specified |
| Anti-Hebbian strength `Δ` | Mentioned as helpful; value not given |
| Exact batch vs. pure online updates | Appears online (one sample at a time) |
| Whether inputs `x` are normalized | Not explicitly stated for flat BioHash (patches normalized in BioConvHash) |
| Random seed / data shuffling | Not specified |
| SOLHash comparison hyperparameters | Missing; direct reproduction not possible |

**Recommended first implementation scope:** flat **BioHash** on vector inputs (MNIST flattened pixels or pre-extracted features), then **BioConvHash** once the core learning and k-WTA encoding are validated.

---

## 11. References (Primary)

- Ryali et al., 2020. *Bio-Inspired Hashing for Unsupervised Similarity Search.* ICML.
- Krotov & Hopfield, 2019. *Unsupervised learning by competing hidden units.* PNAS. (Source of learning rule)
- Dasgupta et al., 2017. *A neural algorithm for a fundamental computing problem.* Science. (FlyHash)
- Grinberg et al., 2019. *Local Unsupervised Learning for Image Analysis.* (Convolutional filter learning for BioConvHash)
