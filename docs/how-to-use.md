# How to Use BioHash

Practical guide for training encoders, building a hash index, and retrieving top-k similar items with the current biohash library.

For algorithm notation and hyperparameters, see [biohash-ryali20a.md](./biohash-ryali20a.md). For incremental text-index updates (append, improve encoder, consolidate), see [incremental-index-updates.md](./incremental-index-updates.md).

---

## Overview

Similarity search is a **two-step pipeline**:

1. **Encode** — map feature vectors to sparse binary codes (`SparseHash`) with a learned or random encoder.
2. **Retrieve** — rank corpus items by **Hamming distance** between query and database hashes.

There is no single `BioHash.retrieveTopK(...)` method on the encoder. Encoding and retrieval are separate modules: `HashEncoder` (e.g. `BioHash`) and `Retrieval`.

---

## Important naming distinction

| Parameter | Meaning |
|-----------|---------|
| **`BioHashConfig.k`** | Hash length — number of active (+1) bits in each code (k-WTA during encoding). |
| **Retrieval `r`** | How many nearest neighbors to return (your “top-k similar items”). |

These are unrelated. A config with `k = 8` produces 8-bit sparse codes; `Retrieval.retrieveTopR(..., r = 100)` returns the 100 closest corpus items.

---

## Basic workflow

### 1. Configure and train an encoder

```scala
import io.github.aboisvert.biohash._

val config = BioHashConfig.paper(
  inputDim = 784,
  m = 40,   // hash layer width (number of hidden units)
  k = 2     // hash length (active bits per code)
)
val encoder = BioHash(config)
encoder.train(trainingVectors)
```

Any type implementing `HashEncoder` works the same way for encoding and retrieval:

- `BioHash` — learned weights (Hebbian/anti-Hebbian)
- `FlyHash` — random projection baseline
- `NaiveBioHash` — dense k-unit ablation

Restore a trained encoder from persisted weights:

```scala
val encoder = BioHash.fromWeights(config, weights, trainingSteps = steps)
```

### 2. Build the corpus index

Encode every corpus vector once and keep the resulting hashes:

```scala
val corpusHashes: IndexedSeq[SparseHash] = encoder.encodeAll(corpusVectors)
```

`encodeAll` parallelizes over large corpora. Each `SparseHash` stores sorted indices of the `k` active bits.

### 3. Query for top-R similar items

```scala
val queryHash = encoder.encode(queryVector)
val r = 10  // top-10 nearest neighbors

val results: IndexedSeq[RetrievalResult] =
  Retrieval.retrieveTopR(queryHash, corpusHashes, r)

// results(i).index    — corpus index (position in corpusHashes)
// results(i).distance — Hamming distance (lower = more similar)
```

Results are sorted by ascending Hamming distance. Ties break on lower corpus index.

### 4. Map indices back to your ids

The library stores hashes in order; you maintain the mapping from index to external id (document id, label, URL, etc.):

```scala
val corpusIds: IndexedSeq[String] = ...  // parallel to corpusVectors / corpusHashes
val topIds = results.map(r => corpusIds(r.index))
```

---

## How similarity is defined

**Encoding.** The encoder scores the input against all `m` weight rows, then selects the top `k` units (k-WTA):

```scala
def encode(x: Array[Double]): SparseHash
```

**Distance.** Retrieval uses Hamming distance between two k-sparse codes, computed in O(k) via sorted intersection:

```
distance = 2k - 2 × |intersection(active bits)|
```

So “similar” means **fewest differing bits** in hash space, not cosine or dot product in the original feature space.

---

## Batch queries

Encode all queries, then retrieve in parallel:

```scala
val queryHashes = encoder.encodeAll(queryVectors)
val allResults: IndexedSeq[IndexedSeq[RetrievalResult]] =
  Retrieval.batchRetrieveTopR(queryHashes, corpusHashes, r)
```

---

## Other retrieval APIs

| API | Purpose |
|-----|---------|
| `Retrieval.retrieveAll(query, db)` | Rank the entire corpus by Hamming distance |
| `Retrieval.retrieveTopR(..., excludeIndices = Set(i))` | Skip items (e.g. exclude the query vector if it is in the database) |
| `Retrieval.batchRetrieveTopR(queries, db, r)` | Top-R for many queries in parallel |
| `Metrics.evaluateRetrieval(...)` | Mean average precision (mAP) at cutoff R |
| `Metrics.recallAtR(...)` | Recall@R against ground-truth nearest neighbors |

Example — exclude self-match when the query is also in the corpus:

```scala
Retrieval.retrieveTopR(queryHash, corpusHashes, r, excludeIndices = Set(queryIndex))
```

Example — evaluate retrieval quality with labels:

```scala
val mAP = Metrics.evaluateRetrieval(
  queryHashes,
  corpusHashes,
  queryLabels,
  corpusLabels,
  r = 100
)
```

---

## Text benchmark and segmented indexes

For persisted text artifacts (BEIR-style benchmarks), train and query via CLI:

```sh
scala-cli run . --main-class io.github.aboisvert.biohash.trainTextBenchmark -- \
  --dataset scifact --method biohash --k 32

scala-cli run . --main-class io.github.aboisvert.biohash.queryTextBenchmark -- \
  --dataset scifact --dense-baseline true
```

When the index has **multiple segments** (after append / improve-encoder operations), use `SegmentedRetrieval`:

```scala
import io.github.aboisvert.biohash.eval.SegmentedRetrieval

val topDocIds: IndexedSeq[String] =
  SegmentedRetrieval.retrieveTopR(queryVector, artifact.segments, r = 100)
```

Within a single segment, retrieval is **exact** (same encoder for query and corpus). Across segments, merge is **approximate** because hashes from different encoder snapshots are not strictly comparable. Consolidate segments when you need one globally consistent hash space — see [incremental-index-updates.md](./incremental-index-updates.md).

---

## End-to-end example

```scala
import io.github.aboisvert.biohash._

// Train
val config = BioHashConfig.mnistDefault(k = 2)
val encoder = BioHash(config)
encoder.train(trainVectors)

// Index
val dbHashes = encoder.encodeAll(databaseVectors)

// Query
val queryHash = encoder.encode(queryVector)
val neighbors = Retrieval.retrieveTopR(queryHash, dbHashes, r = 10)

neighbors.foreach { result =>
  println(s"index=${result.index}  hamming=${result.distance}")
}
```

The evaluation harness follows the same pattern in `EvalRunner` and `TextBenchmarkRunner`.

---

## Limitations

1. **No built-in vector database** — you hold `IndexedSeq[SparseHash]` (and your own id mapping) in memory or persist them yourself via text benchmark artifacts.
2. **Exact linear scan** — every corpus item is compared; there is no LSH bucketing or graph-based ANN index in the core library.
3. **Same encoder required** — query and corpus hashes must be produced by the same encoder weights for exact Hamming ranking. After encoder updates, re-encode or use segmented retrieval / consolidation.

For large-scale ANN benchmarks (SIFT, GIST), the library evaluates recall@R with this linear-scan retrieval; production systems would typically add an approximate index on top of the same hash codes.
