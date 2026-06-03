# Datasets

Place downloaded datasets in this directory. Evaluation commands check for files and exit with instructions if missing.

## MNIST

Download from http://yann.lecun.com/exdb/mnist/ into `data/mnist/`:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

## Fashion-MNIST

Download from https://github.com/zalandoresearch/fashion-mnist into `data/fashion-mnist/` (same IDX filenames as MNIST).

## CIFAR-10

Download the binary version from https://www.cs.toronto.edu/~kriz/cifar.html into `data/cifar-10-batches-bin/`:

- `data_batch_1.bin` … `data_batch_5.bin`
- `test_batch.bin`

## ANN Benchmarks (SIFT / GIST)

Download from http://corpus-texmex.irisa.fr/ into `data/ann/`:

**SIFT10K (TEXMEX siftsmall)** — run `just download-sift10k`; files live in `data/ann/siftsmall/`:
- `siftsmall_base.fvecs`
- `siftsmall_query.fvecs`
- `siftsmall_groundtruth.ivecs`

Use `--dataset sift10k` for this corpus (~10k database vectors).

**SIFT1M** — run `just download-sift`; files live directly in `data/ann/`:
- `sift_base.bvecs`
- `sift_query.bvecs`
- `sift_groundtruth.ivecs`

Use `--dataset sift1m` for the full 1M-vector set.

**GIST1M:**
- `gist_base.fvecs`
- `gist_query.fvecs`
- `gist_groundtruth.ivecs`

## VGG16 fc7 Features (optional)

For paper-comparable CIFAR-10 results on deep features, place pre-extracted features at `data/cifar10-vgg-fc7.csv` (label in last column) or a custom binary format loadable via `Cifar10.loadVggFc7Features`.

## Text Retrieval Benchmark (BEIR-style)

Real-world text evaluation uses precomputed dense embeddings plus BEIR qrels. BioHash is trained on the corpus once, persisted as a binary hash index, and queried separately.

Prepare a dataset (SciFact is the quick smoke benchmark):

```sh
python scripts/prepare_beir_embeddings.py --dataset scifact
```

Expected layout under `data/text/<dataset>/`:

- `corpus.jsonl`, `queries.jsonl` — raw BEIR text (optional for validation)
- `qrels/test.tsv` — query-id, corpus-id, score
- `corpus.fvecs`, `query.fvecs` — float32 TEXMEX-style vectors
- `corpus.ids`, `query.ids` — one external id per vector row
- `manifest.properties` — embedding model metadata

Train and query:

```sh
just prepare-text-scifact
just train-text-scifact
just query-text-scifact
```

Interactive search over the trained index (install Python deps once with `just install-python-deps`):

```sh
just text-search-repl
# or: just text-search-repl dataset=scifact
```

Defaults: `k=32`, `activity=0.01`, `epochs=3`, `normalizeInputs=true`, `retrievalLimit=100`.

Query output reports `nDCG@10`, `MAP@100`, `Recall@10`, `Recall@100`, query latency, artifact size, and an optional dense cosine baseline on smaller corpora.

For a larger benchmark after SciFact smoke tests, rerun preparation with another BEIR dataset such as `fiqa` or `quora`:

```sh
python scripts/prepare_beir_embeddings.py --dataset fiqa
scala-cli run . --main-class io.github.aboisvert.biohash.trainTextBenchmark -- --dataset fiqa
scala-cli run . --main-class io.github.aboisvert.biohash.queryTextBenchmark -- --dataset fiqa --dense-baseline true
```

## Literary corpora

For book-like interactive search (chunked public-domain novels and narrative QA), use the Gutenberg or NarrativeQA prep scripts. Both write the same layout under `data/text/<dataset>/`.

**Project Gutenberg classics** (`gutenberg`) — ~10 novels chunked into passage-sized segments with bundled literary questions:

```sh
just prepare-text-gutenberg
just train-text-gutenberg
just text-search-repl-gutenberg
```

Or directly:

```sh
python scripts/prepare_gutenberg_embeddings.py
scala-cli run . --main-class io.github.aboisvert.biohash.trainTextBenchmark -- --dataset gutenberg
just text-search-repl dataset=gutenberg
```

Smoke test with smaller limits:

```sh
python scripts/prepare_gutenberg_embeddings.py --max-corpus 100 --max-queries 5
```

**NarrativeQA books** (`narrativeqa` / `narrativeqa-literary`) — streams book documents from Hugging Face, chunks them, and exports narrative questions. Use `--literary-only` to exclude plays, scripts, and poetry:

```sh
just prepare-text-narrative
just train-text-narrative
just text-search-repl-narrative
```

Or directly:

```sh
python scripts/prepare_narrative_embeddings.py --literary-only --max-books 10 --max-queries 50
scala-cli run . --main-class io.github.aboisvert.biohash.trainTextBenchmark -- --dataset narrativeqa-literary
just text-search-repl dataset=narrativeqa-literary
```

Requires `datasets` in addition to `sentence-transformers` (`pip install -r scripts/requirements.txt`).
