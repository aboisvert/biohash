"""Shared helpers for BioHash text benchmark preparation scripts."""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ROOT = Path("data/text")
MIN_CHUNK_WORDS = 250
MAX_CHUNK_WORDS = 350


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def passage_text(row: dict) -> str:
    title = row.get("title", "") or ""
    text = row.get("text", "") or ""
    return f"{title}\n{text}".strip()


def word_count(text: str) -> int:
    return len(text.split())


def split_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_paragraphs(paragraphs: list[str], min_words: int = MIN_CHUNK_WORDS, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Merge paragraphs into passage-sized chunks."""
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        words = word_count(paragraph)
        if current and current_words + words > max_words:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_words = words
        else:
            current.append(paragraph)
            current_words += words
            if current_words >= min_words:
                chunks.append("\n\n".join(current))
                current = []
                current_words = 0

    if current:
        if chunks and current_words < min_words // 2:
            chunks[-1] = chunks[-1] + "\n\n" + "\n\n".join(current)
        else:
            chunks.append("\n\n".join(current))

    return chunks


def strip_gutenberg_boilerplate(text: str) -> str:
    start_marker = "*** START OF"
    end_marker = "*** END OF"
    start_idx = text.find(start_marker)
    if start_idx >= 0:
        line_end = text.find("\n", start_idx)
        text = text[line_end + 1 :] if line_end >= 0 else text[start_idx:]
    end_idx = text.find(end_marker)
    if end_idx >= 0:
        text = text[:end_idx]
    return text.strip()


def write_qrels(path: Path, query_to_corpus: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for query_id in sorted(query_to_corpus):
            for corpus_id in sorted(query_to_corpus[query_id]):
                handle.write(f"{query_id}\t{corpus_id}\t1\n")


def write_manifest(path: Path, dataset: str, model: str, corpus_size: int, query_size: int, vector_dim: int) -> None:
    path.write_text(
        "\n".join(
            [
                f"dataset={dataset}",
                f"embeddingModel={model}",
                f"corpusSize={corpus_size}",
                f"querySize={query_size}",
                f"vectorDim={vector_dim}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def print_next_steps(dataset: str) -> None:
    print(f"Wrote benchmark files to data/text/{dataset}")
    print("Next steps:")
    print(
        "  scala-cli run . --main-class io.github.aboisvert.biohash.trainTextBenchmark -- "
        f"--dataset {dataset}"
    )
    print(
        "  just text-search-repl dataset="
        f"{dataset}"
    )
