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
