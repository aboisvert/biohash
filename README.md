# biohash

Scala implementation of **BioHash** — bio-inspired locality-sensitive hashing for unsupervised similarity search ([Ryali et al., ICML 2020](docs/biohash-ryali20a.md)).

## Features

- **BioHash**: online Hebbian/anti-Hebbian weight learning + k-WTA sparse binary encoding
- **FlyHash** baseline: random sparse projections + k-WTA
- **NaiveBioHash** ablation: dense k-unit learning + sign binarization
- Sparse active-index hash storage with O(k) Hamming distance
- mAP evaluation (MNIST, CIFAR-10, Fashion-MNIST) and recall@R (SIFT/GIST ANN benchmarks)
- Real-world text retrieval benchmark (BEIR-style corpus with train/query split)
- MUnit test suite and JMH microbenchmarks

## Build & Test

```sh
scala-cli test .
```

## Run

Scala CLI generates one main class per `@main` method. Pass `--main-class biohash.<command>`:

```sh
# List subcommands
scala-cli run . --main-class biohash.run

# Runtime microbenchmarks (non-JMH)
scala-cli run . --main-class biohash.microbench

# Evaluate on MNIST (requires data — see data/README.md)
scala-cli run . --main-class biohash.evalMnist -- --method biohash --k 2 --epochs 5

# Sweep hash lengths
scala-cli run . --main-class biohash.sweepMnist -- --method biohash --ks 2,4,8,16

# FlyHash baseline
scala-cli run . --main-class biohash.evalMnist -- --method flyhash --k 2

# Fashion-MNIST / CIFAR-10
scala-cli run . --main-class biohash.evalFashion -- --k 2
scala-cli run . --main-class biohash.evalCifar -- --k 2

# ANN benchmarks (SIFT/GIST — see data/README.md)
scala-cli run . --main-class biohash.evalAnn -- --dataset sift10k --k 8

# Synthetic text retrieval (in-memory clustered embeddings)
scala-cli run . --main-class biohash.evalSyntheticText -- --method biohash --k 8

# Real-world text benchmark (BEIR-style — see data/README.md)
python scripts/prepare_beir_embeddings.py --dataset scifact
scala-cli run . --main-class biohash.trainTextBenchmark -- --dataset scifact --method biohash --k 32
scala-cli run . --main-class biohash.queryTextBenchmark -- --dataset scifact --dense-baseline true
```

## JMH Benchmarks

```sh
scala-cli --jmh . -- -i 3 -wi 3 -f1 BioHashJmh
```

On Apple M4 Pro with JDK 25, the Vector API scoring backend runs `scoresGemv` at about **222.2 µs/op** — roughly **2.10×** faster 
than the scalar backend (466.0 µs/op). BLAS is slower in the current per-row `ddot` implementation (557.1 µs/op). 
See [docs/scoring-backend-report.md](docs/scoring-backend-report.md) for methodology, environment, and full numbers.

## Project Layout

```
src/main/scala/io/github/aboisvert/biohash/
  BioHash.scala       # core learning + encoding
  FlyHash.scala       # random projection baseline
  NaiveBioHash.scala  # ablation baseline
  VectorOps.scala     # dot product, p-norm, scoring
  TopK.scala          # k-WTA index selection
  SparseHash.scala    # sparse codes + Hamming distance
  Retrieval.scala     # top-R retrieval
  Metrics.scala       # mAP, recall@R
  data/               # dataset loaders (MNIST, CIFAR, ANN, text)
  eval/               # evaluation harness
scripts/
  prepare_beir_embeddings.py  # optional BEIR embedding preparation
bench/
  BioHashJmh.scala    # JMH microbenchmarks
docs/
  biohash-ryali20a.md # algorithm reference
```

## Algorithm Reference

See [docs/biohash-ryali20a.md](docs/biohash-ryali20a.md) for notation, learning rule, hyperparameters, and evaluation protocol.

## License

Copyright (c) Alex Boisvert, 2026.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.
