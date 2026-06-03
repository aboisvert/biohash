# BioHash task runner — see README.md and data/README.md

set shell := ["sh", "-uc"]

# List available recipes
default:
    @just --list

# ── Build & Test ─────────────────────────────────────────────────────────────

# Format all Scala source files
fmt:
    scala-cli fmt .

# Run the MUnit test suite
test:
    scala-cli test .

# List available Scala CLI subcommands
run:
    scala-cli run . --main-class io.github.aboisvert.biohash.run

# Runtime microbenchmarks (non-JMH)
microbench:
    scala-cli run . --main-class io.github.aboisvert.biohash.microbench

# JMH microbenchmarks
jmh *args="-i 3 -wi 3 -f1 BioHashJmh":
    #!/usr/bin/env sh
    set -eu
    export _JAVA_OPTIONS="--add-modules=jdk.incubator.vector"
    exec scala-cli --jmh . --java-opt --add-modules=jdk.incubator.vector -- {{args}}

# ── Evaluation ───────────────────────────────────────────────────────────────

# Evaluate on MNIST (requires data/mnist — run `just download-mnist`)
eval-mnist $method="biohash" $k="2" $epochs="5" *$extra="":
    #!/usr/bin/env sh
    set -eu
    method="${method#method=}"
    k="${k#k=}"
    epochs="${epochs#epochs=}"
    exec scala-cli run . --main-class io.github.aboisvert.biohash.evalMnist -- --method "$method" --k "$k" --epochs "$epochs" ${extra:-}

# Sweep hash lengths on MNIST
sweep-mnist $method="biohash" $ks="2,4,8,16" $epochs="5" *$extra="":
    #!/usr/bin/env sh
    set -eu
    method="${method#method=}"
    ks="${ks#ks=}"
    epochs="${epochs#epochs=}"
    exec scala-cli run . --main-class io.github.aboisvert.biohash.sweepMnist -- --method "$method" --ks "$ks" --epochs "$epochs" ${extra:-}

# FlyHash baseline on MNIST
eval-flyhash k="2" *extra="":
    just eval-mnist flyhash {{k}} 5 {{extra}}

# Evaluate on Fashion-MNIST (requires data/fashion-mnist)
eval-fashion $method="biohash" $k="2" $epochs="5" *$extra="":
    #!/usr/bin/env sh
    set -eu
    method="${method#method=}"
    k="${k#k=}"
    epochs="${epochs#epochs=}"
    exec scala-cli run . --main-class io.github.aboisvert.biohash.evalFashion -- --method "$method" --k "$k" --epochs "$epochs" ${extra:-}

# Evaluate on CIFAR-10 (requires data/cifar-10-batches-bin)
eval-cifar $method="biohash" $k="2" $epochs="5" *$extra="":
    #!/usr/bin/env sh
    set -eu
    method="${method#method=}"
    k="${k#k=}"
    epochs="${epochs#epochs=}"
    exec scala-cli run . --main-class io.github.aboisvert.biohash.evalCifar -- --method "$method" --k "$k" --epochs "$epochs" ${extra:-}

# ANN benchmarks on SIFT/GIST (requires data/ann — run `just download-sift10k` or `just download-sift`)
eval-ann $dataset="sift10k" $k="8" $epochs="3" *$extra="":
    #!/usr/bin/env sh
    set -eu
    dataset="${dataset#dataset=}"
    k="${k#k=}"
    epochs="${epochs#epochs=}"
    exec scala-cli run . --main-class io.github.aboisvert.biohash.evalAnn -- --dataset "$dataset" --k "$k" --epochs "$epochs" ${extra:-}

# Synthetic text retrieval benchmark (in-memory)
eval-synthetic-text $method="biohash" $k="8" $epochs="3" *$extra="":
    #!/usr/bin/env sh
    set -eu
    method="${method#method=}"
    k="${k#k=}"
    epochs="${epochs#epochs=}"
    exec scala-cli run . --main-class io.github.aboisvert.biohash.evalSyntheticText -- --method "$method" --k "$k" --epochs "$epochs" ${extra:-}

# Create .venv and install sentence-transformers for text embed / REPL
install-python-deps:
    #!/usr/bin/env sh
    set -eu
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r scripts/requirements.txt
    echo "Installed into .venv — text-search-repl will use .venv/bin/python automatically."

# Prepare text benchmark embeddings (BEIR datasets; requires Python + sentence-transformers)
prepare-text $dataset="scifact" *$extra="":
    #!/usr/bin/env sh
    set -eu
    dataset="${dataset#dataset=}"
    if [ -d "data/text/$dataset" ] && [ -f "data/text/$dataset/corpus.fvecs" ]; then
      echo "Text data already exists in data/text/$dataset"
      exit 0
    fi
    if [ -x .venv/bin/python ]; then
      python=.venv/bin/python
    else
      python=python3
    fi
    case "$dataset" in
      gutenberg)
        "$python" scripts/prepare_gutenberg_embeddings.py ${extra:-}
        ;;
      narrativeqa|narrativeqa-literary)
        literary=""
        if [ "$dataset" = "narrativeqa-literary" ]; then
          literary="--literary-only --dataset-name narrativeqa-literary"
        fi
        "$python" scripts/prepare_narrative_embeddings.py $literary ${extra:-}
        ;;
      *)
        "$python" scripts/prepare_beir_embeddings.py --dataset "$dataset" ${extra:-}
        ;;
    esac

# Train BioHash on a text corpus
train-text $dataset="scifact" $method="biohash" $hash_k="32" $epochs="3" *$extra="":
    #!/usr/bin/env sh
    set -eu
    dataset="${dataset#dataset=}"
    method="${method#method=}"
    hash_k="${hash_k#hash_k=}"
    [ -n "$hash_k" ] || hash_k=32
    epochs="${epochs#epochs=}"
    exec scala-cli run . --main-class io.github.aboisvert.biohash.trainTextBenchmark -- --dataset "$dataset" --method "$method" --k "$hash_k" --epochs "$epochs" ${extra:-}

# Query text benchmark against trained artifact
query-text $dataset="scifact" $method="biohash" $hash_k="32" $epochs="3" *$extra="":
    #!/usr/bin/env sh
    set -eu
    dataset="${dataset#dataset=}"
    method="${method#method=}"
    hash_k="${hash_k#hash_k=}"
    [ -n "$hash_k" ] || hash_k=32
    epochs="${epochs#epochs=}"
    exec scala-cli run . --main-class io.github.aboisvert.biohash.queryTextBenchmark -- --dataset "$dataset" --method "$method" --k "$hash_k" --epochs "$epochs" --dense-baseline true ${extra:-}

# Interactive text search REPL (run `just install-python-deps` once for query embedding)
text-search-repl $dataset="scifact" $hash_k="32" *$extra="":
    #!/usr/bin/env sh
    set -eu
    dataset="${dataset#dataset=}"
    hash_k="${hash_k#hash_k=}"
    [ -n "$hash_k" ] || hash_k=32
    if [ -x .venv/bin/python ]; then
      python=.venv/bin/python
    else
      python=python3
    fi
    exec scala-cli run . --main-class io.github.aboisvert.biohash.textSearchRepl -- \
      --dataset "$dataset" --k "$hash_k" --python "$python" ${extra:-}

# Literary corpus shortcuts
prepare-text-gutenberg *extra="": (prepare-text "gutenberg" extra)
prepare-text-narrative *extra="": (prepare-text "narrativeqa-literary" extra)
train-text-gutenberg *extra="": (train-text "gutenberg" extra)
train-text-narrative *extra="": (train-text "narrativeqa-literary" extra)
text-search-repl-gutenberg *extra="": (text-search-repl "gutenberg" extra)
text-search-repl-narrative *extra="": (text-search-repl "narrativeqa-literary" extra)

# Backward-compatible aliases
text-repl $dataset="scifact" $hash_k="32" *$extra="": (text-search-repl dataset hash_k extra)
text-repl-gutenberg *extra="": (text-search-repl "gutenberg" extra)
text-repl-narrative *extra="": (text-search-repl "narrativeqa-literary" extra)
prepare-text-scifact: (prepare-text "scifact")
train-text-scifact: (train-text "scifact")
query-text-scifact: (query-text "scifact")
scifact-repl: (text-search-repl "scifact")

# ── Dataset downloads ────────────────────────────────────────────────────────

# Download MNIST IDX gzip files into data/mnist/
download-mnist:
    #!/usr/bin/env sh
    set -eu
    dir="data/mnist"
    base="https://storage.googleapis.com/cvdf-datasets/mnist"
    mkdir -p "$dir"
    for file in \
      train-images-idx3-ubyte.gz \
      train-labels-idx1-ubyte.gz \
      t10k-images-idx3-ubyte.gz \
      t10k-labels-idx1-ubyte.gz
    do
      dest="$dir/$file"
      if [ ! -f "$dest" ]; then
        echo "Downloading $dest ..."
        curl -sfL -o "$dest" "$base/$file"
      else
        echo "Already have $dest"
      fi
    done

# Download Fashion-MNIST IDX gzip files into data/fashion-mnist/
download-fashion-mnist:
    #!/usr/bin/env sh
    set -eu
    dir="data/fashion-mnist"
    base="https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"
    mkdir -p "$dir"
    for file in \
      train-images-idx3-ubyte.gz \
      train-labels-idx1-ubyte.gz \
      t10k-images-idx3-ubyte.gz \
      t10k-labels-idx1-ubyte.gz
    do
      dest="$dir/$file"
      if [ ! -f "$dest" ]; then
        echo "Downloading $dest ..."
        curl -sfL -o "$dest" "$base/$file"
      else
        echo "Already have $dest"
      fi
    done

# Download CIFAR-10 binary batches into data/cifar-10-batches-bin/
download-cifar10:
    #!/usr/bin/env sh
    set -eu
    root="data"
    dest="$root/cifar-10-batches-bin"
    if [ -f "$dest/data_batch_1.bin" ] && [ -f "$dest/test_batch.bin" ]; then
      echo "Already have CIFAR-10 in $dest"
      exit 0
    fi
    archive="$root/cifar-10-binary.tar.gz"
    url="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    mkdir -p "$root"
    if [ ! -f "$archive" ]; then
      echo "Downloading $archive ..."
      curl -sfL -o "$archive" "$url"
    fi
    echo "Extracting CIFAR-10 into $root ..."
    tar -xzf "$archive" -C "$root"
    rm -f "$archive"

# Download SIFT10K (TEXMEX siftsmall) into data/ann/siftsmall/
download-sift10k:
    #!/usr/bin/env sh
    set -eu
    dir="data/ann"
    url="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
    mkdir -p "$dir"
    if [ -f "$dir/siftsmall/siftsmall_base.fvecs" ] && \
       [ -f "$dir/siftsmall/siftsmall_query.fvecs" ] && \
       [ -f "$dir/siftsmall/siftsmall_groundtruth.ivecs" ]; then
      echo "Already have SIFT10K (siftsmall) in $dir/siftsmall"
      exit 0
    fi
    archive="$dir/siftsmall.tar.gz"
    if [ ! -f "$archive" ]; then
      echo "Downloading $archive ..."
      curl -sfL -o "$archive" "$url"
    fi
    echo "Extracting SIFT10K into $dir ..."
    tar -xzf "$archive" -C "$dir"
    rm -f "$archive"

# Download full SIFT1M (TEXMEX) into data/ann/
download-sift:
    #!/usr/bin/env sh
    set -eu
    dir="data/ann"
    url="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    mkdir -p "$dir"
    if [ -f "$dir/sift_base.bvecs" ] && [ -f "$dir/sift_query.bvecs" ] && [ -f "$dir/sift_groundtruth.ivecs" ]; then
      echo "Already have SIFT1M files in $dir"
      exit 0
    fi
    archive="$dir/sift.tar.gz"
    if [ ! -f "$archive" ]; then
      echo "Downloading $archive (~161MB) ..."
      curl -sfL -o "$archive" "$url"
    fi
    echo "Extracting SIFT1M into $dir ..."
    tar -xzf "$archive" -C "$dir"
    rm -f "$archive"

# Download GIST1M (TEXMEX) into data/ann/ (~2.6GB)
download-gist:
    #!/usr/bin/env sh
    set -eu
    dir="data/ann"
    url="ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
    mkdir -p "$dir"
    if [ -f "$dir/gist_base.fvecs" ] && [ -f "$dir/gist_query.fvecs" ] && [ -f "$dir/gist_groundtruth.ivecs" ]; then
      echo "Already have GIST1M files in $dir"
      exit 0
    fi
    archive="$dir/gist.tar.gz"
    if [ ! -f "$archive" ]; then
      echo "Downloading $archive (~2.6GB) ..."
      curl -sfL -o "$archive" "$url"
    fi
    echo "Extracting GIST1M into $dir ..."
    tar -xzf "$archive" -C "$dir"
    rm -f "$archive"

# Print instructions for optional VGG16 fc7 CIFAR-10 features
vgg-fc7-note:
    @echo "Optional VGG16 fc7 features for paper-comparable CIFAR-10 results:"
    @echo "  Place pre-extracted features at data/cifar10-vgg-fc7.csv (label in last column)"
    @echo "  or a custom binary format loadable via Cifar10.loadVggFc7Features."
    @echo "No canonical public download URL is documented in data/README.md."

# Download MNIST, Fashion-MNIST, and CIFAR-10
download-core-data: download-mnist download-fashion-mnist download-cifar10

# Download ANN benchmark data for default eval (SIFT10K)
download-ann-data: download-sift10k

# Download core datasets plus SIFT10K ANN data
download-data: download-core-data download-ann-data
