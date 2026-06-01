#!/usr/bin/env python3
"""Prepare BEIR text benchmark files for BioHash.

Downloads a BEIR dataset, embeds corpus and queries with sentence-transformers,
and writes the local layout expected by the Scala text benchmark loader:

  data/text/<dataset>/
    corpus.jsonl
    queries.jsonl
    qrels/test.tsv
    corpus.fvecs
    query.fvecs
    corpus.ids
    query.ids
    manifest.properties
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ROOT = Path("data/text")
BEIR_ZIP_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"


def download_beir(dataset: str, dest: Path) -> Path:
    dataset_dir = dest / dataset
    if (dataset_dir / "corpus.jsonl").exists() and (dataset_dir / "queries.jsonl").exists():
        return dataset_dir

    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{dataset}.zip"
    if not archive.exists():
        url = BEIR_ZIP_URL.format(dataset=dataset)
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, archive)
    print(f"Extracting {archive} ...")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest)
    return dataset_dir


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            query_id, corpus_id, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                qrels.setdefault(query_id, {})[corpus_id] = score
    return qrels


def write_fvecs(path: Path, vectors) -> None:
    with path.open("wb") as handle:
        for vector in vectors:
            dim = len(vector)
            handle.write(struct.pack("<i", dim))
            handle.write(struct.pack(f"<{dim}f", *vector))


def write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def embed_texts(model, texts: list[str], batch_size: int):
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name (default: scifact)")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Output root directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-transformers model name")
    parser.add_argument("--split", default="test", help="Qrels split to export (default: test)")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--max-corpus", type=int, default=0, help="Optional corpus size cap for smoke tests")
    parser.add_argument("--max-queries", type=int, default=0, help="Optional query size cap for smoke tests")
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "Missing dependency: sentence-transformers\n"
            "Install with: pip install sentence-transformers",
            file=sys.stderr,
        )
        return 1

    dataset_dir = download_beir(args.dataset, args.root)
    corpus_rows = read_jsonl(dataset_dir / "corpus.jsonl")
    query_rows = read_jsonl(dataset_dir / "queries.jsonl")
    qrels_path = dataset_dir / "qrels" / f"{args.split}.tsv"
    if not qrels_path.exists():
        print(f"Qrels split not found: {qrels_path}", file=sys.stderr)
        return 1
    qrels = read_qrels(qrels_path)

    if args.max_corpus > 0:
        corpus_rows = corpus_rows[: args.max_corpus]
    if args.max_queries > 0:
        query_rows = query_rows[: args.max_queries]

    corpus_ids = [row["_id"] for row in corpus_rows]
    query_ids = [row["_id"] for row in query_rows]

    def passage_text(row: dict) -> str:
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        return f"{title}\n{text}".strip()

    print(f"Loading embedding model {args.model} ...")
    model = SentenceTransformer(args.model)

    print(f"Embedding {len(corpus_rows)} corpus passages ...")
    corpus_vectors = embed_texts(model, [passage_text(row) for row in corpus_rows], args.batch_size)
    print(f"Embedding {len(query_rows)} queries ...")
    query_vectors = embed_texts(model, [row["text"] for row in query_rows], args.batch_size)

    out_dir = args.root / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    write_fvecs(out_dir / "corpus.fvecs", corpus_vectors)
    write_fvecs(out_dir / "query.fvecs", query_vectors)
    write_ids(out_dir / "corpus.ids", corpus_ids)
    write_ids(out_dir / "query.ids", query_ids)

    qrels_out = out_dir / "qrels"
    qrels_out.mkdir(parents=True, exist_ok=True)
    with (qrels_out / "test.tsv").open("w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for query_id, docs in sorted(qrels.items()):
            if query_id not in query_ids:
                continue
            for corpus_id, score in sorted(docs.items()):
                if corpus_id in corpus_ids:
                    handle.write(f"{query_id}\t{corpus_id}\t{score}\n")

    manifest = out_dir / "manifest.properties"
    manifest.write_text(
        "\n".join(
            [
                f"dataset={args.dataset}",
                f"embeddingModel={args.model}",
                f"corpusSize={len(corpus_ids)}",
                f"querySize={len(query_ids)}",
                f"vectorDim={len(corpus_vectors[0]) if len(corpus_vectors) else 0}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote benchmark files to {out_dir}")
    print("Next steps:")
    print(
        "  scala-cli run . --main-class io.github.aboisvert.biohash.trainTextBenchmark -- "
        f"--dataset {args.dataset}"
    )
    print(
        "  scala-cli run . --main-class io.github.aboisvert.biohash.queryTextBenchmark -- "
        f"--dataset {args.dataset} --dense-baseline true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
