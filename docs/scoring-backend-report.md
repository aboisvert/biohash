# Scoring Backend Comparison Report

Comparison of the three numeric scoring backends in BioHash on the primary hot path: **matrix-vector scoring** (`scoresGemv`) at text-realistic dimensions.

## Summary

On an Apple M4 Pro (aarch64) with JDK 17 and a **3200 × 384** weight matrix (SciFact/text scale):

| Backend | Latency | Speedup vs scalar |
|---------|---------|-------------------|
| **vector** (JDK Vector API) | **216.5 µs/op** | **2.05×** |
| scalar (reference loops) | 442.9 µs/op | 1.00× |
| blas (netlib-java JNIBLAS) | 545.5 µs/op | 0.81× |

**Recommendation:** Keep **vector** as the runtime default (already configured in `ScoringBackend.default`). Use **scalar** for portability testing or when the incubator vector module is unavailable. Avoid **blas** in its current form — the per-row `ddot` loop adds JNI call overhead that outweighs native SIMD on this workload.

End-to-end encode/query throughput also depends on TopK selection, retrieval, and parallelism; this report covers only the scoring kernel.

## Methodology

### Benchmark

- **Tool:** JMH 1.37 via `scala-cli --jmh`
- **Class:** `io.github.aboisvert.biohash.bench.ScoringBackendJmh`
- **Method:** `scoresGemv` — compute all `m` row scores against one input vector with `p = 2.0`
- **Matrix size:** `m = 3200` rows, `d = 384` columns (~1.23M multiply-adds per call)
- **Parameter sweep:** `backendName ∈ {scalar, vector, blas}`
- **Mode:** Average time, microseconds per operation
- **Warmup:** 5 × 500 ms
- **Measurement:** 5 × 500 ms
- **Forks:** 1

### Command

```bash
just jmh "-i 5 -wi 5 -f1 ScoringBackendJmh"
```

Equivalent:

```bash
_JAVA_OPTIONS='--add-modules=jdk.incubator.vector' \
  scala-cli --jmh . --java-opt --add-modules=jdk.incubator.vector -- \
  -i 5 -wi 5 -f1 ScoringBackendJmh
```

### Environment

| Item | Value |
|------|-------|
| Date | 2026-06-01 |
| CPU | Apple M4 Pro (aarch64) |
| JMH JVM | Azul Zulu 17.0.18+8-LTS (aarch64) |
| Project JVM target | 17 (`project.scala`) |
| Vector module | `--add-modules=jdk.incubator.vector` |
| BLAS binding | `dev.ludovic.netlib.blas.JNIBLAS` (native JNI) |
| netlib-java version | 3.2.0 |

### Correctness gate

All three backends are verified numerically identical by `ScoringBackendSuite`:

- `dot` product agreement (tolerance 1e-9)
- `scoresGemv` output agreement on a 3 × 3 test matrix

## Results

### JMH output (text scale: d=384, m=3200)

```
Benchmark                     (backendName)  Mode  Cnt    Score    Error  Units
ScoringBackendJmh.scoresGemv         scalar  avgt    5  442.856 ±  7.858  us/op
ScoringBackendJmh.scoresGemv         vector  avgt    5  216.505 ±  8.910  us/op
ScoringBackendJmh.scoresGemv           blas  avgt    5  545.521 ± 13.348  us/op
```

### Derived metrics

| Backend | Score (µs/op) | 99.9% CI | Relative to scalar | Approx. throughput |
|---------|---------------|----------|--------------------|--------------------|
| vector | 216.5 ± 8.9 | [207.6, 225.4] | **2.05× faster** | ~5.7 GFLOPS |
| scalar | 442.9 ± 7.9 | [435.0, 450.7] | baseline | ~2.8 GFLOPS |
| blas | 545.5 ± 13.3 | [532.2, 558.9] | **1.23× slower** | ~2.3 GFLOPS |

Throughput estimated as `(m × d) / latency` with `m=3200`, `d=384`.

## Implementation notes

All backends operate on a flat row-major [`WeightMatrix`](../src/main/scala/io/github/aboisvert/biohash/WeightMatrix.scala) and are selected via [`ScoringBackend`](../src/main/scala/io/github/aboisvert/biohash/ScoringBackend.scala):

```bash
-Dbiohash.backend=scalar   # reference loops
-Dbiohash.backend=vector    # JDK Vector API (default when available)
-Dbiohash.backend=blas      # netlib-java BLAS
```

### scalar

Hand-written `while` loops over `Array[Double]`. For `p = 2.0`, each output row is a dot product of length `d`. No external dependencies. Serves as the correctness baseline and fallback when vector SIMD is unavailable.

### vector

Uses the JDK Vector API (`jdk.incubator.vector.DoubleVector`) in [`VectorApiBackendImpl`](../src/main/scala/io/github/aboisvert/biohash/VectorApiBackendImpl.scala). Processes `d` elements in SIMD lanes (`SPECIES_PREFERRED`), with a scalar tail for the remainder. Requires `--add-modules=jdk.incubator.vector` at runtime.

This backend wins on the benchmarked workload because it vectorizes the inner dimension loop in-process with no per-row call overhead.

### blas

Uses netlib-java's native `JNIBLAS` binding. The current implementation calls `ddot` once per row (`m = 3200` JNI calls per `scoresGemv`), not a single `dgemv`. Each call crosses the JNI boundary, which dominates at this matrix size on Apple Silicon.

A future improvement would be a single `dgemv` over the contiguous flat buffer; that may change the ranking.

## Trade-offs

| Backend | Pros | Cons |
|---------|------|------|
| **scalar** | Zero deps; always available; simplest to debug | Slowest vectorized path; no SIMD |
| **vector** | ~2× faster at text scale; pure JVM; no native artifacts | Incubator module required; lane width varies by CPU |
| **blas** | Native BLAS when properly batched; standard API | Current per-row `ddot` is slower than both alternatives; adds netlib-java dependency; platform-specific JNI |

## Recommendations

1. **Production default:** `vector` — best measured performance with no native library distribution.
2. **CI / portability:** `scalar` — use `-Dbiohash.backend=scalar` when testing on JVMs without the vector incubator module.
3. **BLAS:** Do not enable by default. Revisit only after replacing the per-row loop with a single `dgemv` call on the flat `WeightMatrix` buffer.
4. **End-to-end benchmarks:** Re-run `BioHashTextJmh` and text benchmark evals (`just train-text-scifact` / `just query-text-scifact`) to quantify how much scoring improvements translate to total encode time, given TopK and I/O overhead.

## Reproducing

```bash
# Run the comparison benchmark
just jmh "-i 5 -wi 5 -f1 ScoringBackendJmh"

# Run correctness tests
scala-cli test . --test-only '*ScoringBackendSuite*'
```
