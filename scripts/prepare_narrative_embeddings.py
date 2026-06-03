#!/usr/bin/env python3
"""Prepare NarrativeQA literary corpus for BioHash text search.

Streams the Hugging Face NarrativeQA dataset, keeps book documents (Gutenberg),
chunks them into passages, embeds corpus and questions, and writes the standard
text benchmark layout under data/text/<dataset-name>/.

Use --literary-only for a stricter book filter (excludes plays, scripts, poetry).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from text_prep_common import (
    DEFAULT_MODEL,
    DEFAULT_ROOT,
    chunk_paragraphs,
    embed_texts,
    passage_text,
    print_next_steps,
    split_paragraphs,
    strip_gutenberg_boilerplate,
    write_fvecs,
    write_ids,
    write_jsonl,
    write_manifest,
    write_qrels,
)

LITERARY_TITLE_BLOCKLIST = (
    "play",
    "screenplay",
    "script",
    "poems",
    "poetry",
    "sonnets",
    "dialogues",
    "theatre",
    "theater",
)


def is_gutenberg_book(doc: dict) -> bool:
    return doc.get("kind") == "gutenberg"


def is_literary_book(doc: dict) -> bool:
    if not is_gutenberg_book(doc):
        return False
    summary = doc.get("summary") or {}
    title = (summary.get("title") or "").lower()
    return not any(term in title for term in LITERARY_TITLE_BLOCKLIST)


def document_title(doc: dict) -> str:
    summary = doc.get("summary") or {}
    return summary.get("title") or doc.get("id", "unknown")


def document_text(doc: dict) -> str:
    text = doc.get("text") or ""
    if doc.get("kind") == "gutenberg":
        text = strip_gutenberg_boilerplate(text)
    return text.strip()


def split_html_paragraphs(text: str) -> list[str]:
    plain = re.sub(r"<[^>]+>", " ", text)
    plain = re.sub(r"[ \t]+", " ", plain)
    return split_paragraphs(re.sub(r"\n{3,}", "\n\n", plain))


def load_narrativeqa_rows(split: str):
    from datasets import load_dataset

    return load_dataset("deepmind/narrativeqa", split=split, streaming=True)


def collect_documents_and_queries(
    split: str,
    literary_only: bool,
    max_books: int,
    max_queries: int,
) -> tuple[dict[str, dict], list[dict]]:
    doc_filter = is_literary_book if literary_only else is_gutenberg_book
    documents: dict[str, dict] = {}
    queries: list[dict] = []
    seen_query_keys: set[tuple[str, str]] = set()

    for row in load_narrativeqa_rows(split):
        doc = row["document"]
        if not doc_filter(doc):
            continue
        doc_id = doc["id"]
        if doc_id not in documents:
            if max_books > 0 and len(documents) >= max_books:
                continue
            text = document_text(doc)
            if not text:
                continue
            documents[doc_id] = doc

        if doc_id not in documents:
            continue

        question = row["question"]["text"].strip()
        if not question:
            continue
        query_key = (doc_id, question)
        if query_key in seen_query_keys:
            continue
        seen_query_keys.add(query_key)
        if max_queries > 0 and len(queries) >= max_queries:
            continue
        query_id = f"nqa-{len(queries):05d}"
        queries.append({"_id": query_id, "text": question, "doc_id": doc_id})

    return documents, queries


def build_corpus_rows(documents: dict[str, dict], max_corpus: int) -> list[dict]:
    rows: list[dict] = []
    for doc_id, doc in documents.items():
        title = document_title(doc)
        text = document_text(doc)
        paragraphs = split_html_paragraphs(text) if "<" in text[:200] else split_paragraphs(text)
        chunks = chunk_paragraphs(paragraphs)
        print(f"  {title}: {len(chunks)} passages")
        for idx, chunk in enumerate(chunks):
            rows.append(
                {
                    "_id": f"{doc_id}-{idx:04d}",
                    "title": title,
                    "text": chunk,
                    "doc_id": doc_id,
                }
            )
            if max_corpus > 0 and len(rows) >= max_corpus:
                return rows
    return rows


def build_qrels(queries: list[dict], corpus_rows: list[dict]) -> dict[str, list[str]]:
    chunks_by_doc: dict[str, list[str]] = {}
    for row in corpus_rows:
        chunks_by_doc.setdefault(row["doc_id"], []).append(row["_id"])

    qrels: dict[str, list[str]] = {}
    for query in queries:
        chunk_ids = chunks_by_doc.get(query["doc_id"], [])
        if chunk_ids:
            qrels[query["_id"]] = chunk_ids
    return qrels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Output root directory")
    parser.add_argument("--dataset-name", default="", help="Output dataset directory name")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-transformers model name")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"), help="NarrativeQA split")
    parser.add_argument("--literary-only", action="store_true", help="Keep curated literary books only")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--max-books", type=int, default=20, help="Maximum number of books to include (0 = all)")
    parser.add_argument("--max-corpus", type=int, default=0, help="Optional corpus passage cap for smoke tests")
    parser.add_argument("--max-queries", type=int, default=200, help="Maximum queries to export (0 = all)")
    args = parser.parse_args()

    dataset_name = args.dataset_name or ("narrativeqa-literary" if args.literary_only else "narrativeqa")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "Missing dependency: sentence-transformers\n"
            "Install with: pip install -r scripts/requirements.txt",
            file=sys.stderr,
        )
        return 1

    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        print(
            "Missing dependency: datasets\n"
            "Install with: pip install -r scripts/requirements.txt",
            file=sys.stderr,
        )
        return 1

    print(f"Collecting NarrativeQA split={args.split} literary_only={args.literary_only} ...")
    documents, query_rows = collect_documents_and_queries(
        args.split,
        args.literary_only,
        args.max_books,
        args.max_queries,
    )
    if not documents:
        print("No matching book documents found.", file=sys.stderr)
        return 1

    print(f"Building corpus from {len(documents)} books ...")
    corpus_rows = build_corpus_rows(documents, args.max_corpus)
    included_doc_ids = {row["doc_id"] for row in corpus_rows}
    query_rows = [row for row in query_rows if row["doc_id"] in included_doc_ids]

    corpus_ids = [row["_id"] for row in corpus_rows]
    query_ids = [row["_id"] for row in query_rows]
    qrels = build_qrels(query_rows, corpus_rows)

    print(f"Loading embedding model {args.model} ...")
    model = SentenceTransformer(args.model)

    print(f"Embedding {len(corpus_rows)} corpus passages ...")
    corpus_vectors = embed_texts(model, [passage_text(row) for row in corpus_rows], args.batch_size)
    print(f"Embedding {len(query_rows)} queries ...")
    query_vectors = embed_texts(model, [row["text"] for row in query_rows], args.batch_size)

    out_dir = args.root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(
        out_dir / "corpus.jsonl",
        [{"_id": row["_id"], "title": row["title"], "text": row["text"]} for row in corpus_rows],
    )
    write_jsonl(out_dir / "queries.jsonl", [{"_id": row["_id"], "text": row["text"]} for row in query_rows])
    write_fvecs(out_dir / "corpus.fvecs", corpus_vectors)
    write_fvecs(out_dir / "query.fvecs", query_vectors)
    write_ids(out_dir / "corpus.ids", corpus_ids)
    write_ids(out_dir / "query.ids", query_ids)
    write_qrels(out_dir / "qrels" / "test.tsv", qrels)
    write_manifest(
        out_dir / "manifest.properties",
        dataset_name,
        args.model,
        len(corpus_ids),
        len(query_ids),
        len(corpus_vectors[0]) if len(corpus_vectors) else 0,
    )

    print_next_steps(dataset_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
