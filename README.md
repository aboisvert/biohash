# biohash

Scala implementation of **BioHash** — bio-inspired locality-sensitive hashing for unsupervised similarity search ([Ryali et al., ICML 2020](docs/biohash-ryali20a.md)).

## Features

- **BioHash**: online Hebbian/anti-Hebbian weight learning + k-WTA sparse binary encoding
- **FlyHash** baseline: random sparse projections + k-WTA
- **NaiveBioHash** ablation: dense k-unit learning + sign binarization
- Sparse active-index hash storage with O(k) Hamming distance
- mAP evaluation (MNIST, CIFAR-10, Fashion-MNIST) and recall@R (SIFT/GIST ANN benchmarks)
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
```

## JMH Benchmarks

```sh
scala-cli --jmh . -- -i 3 -wi 3 -f1 BioHashJmh
```

## Project Layout

```
src/main/scala/biohash/
  BioHash.scala       # core learning + encoding
  FlyHash.scala       # random projection baseline
  NaiveBioHash.scala  # ablation baseline
  VectorOps.scala     # dot product, p-norm, scoring
  TopK.scala          # k-WTA index selection
  SparseHash.scala    # sparse codes + Hamming distance
  Retrieval.scala     # top-R retrieval
  Metrics.scala       # mAP, recall@R
  data/               # dataset loaders (MNIST, CIFAR, ANN)
  eval/               # evaluation harness
bench/
  BioHashJmh.scala    # JMH microbenchmarks
docs/
  biohash-ryali20a.md # algorithm reference
```

## Algorithm Reference

See [docs/biohash-ryali20a.md](docs/biohash-ryali20a.md) for notation, learning rule, hyperparameters, and evaluation protocol.
