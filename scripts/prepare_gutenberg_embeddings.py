#!/usr/bin/env python3
"""Prepare Project Gutenberg literary corpus for BioHash text search.

Downloads public-domain books, chunks them into passage-sized segments,
embeds corpus and bundled queries, and writes the standard text benchmark layout:

  data/text/gutenberg/
    corpus.jsonl, queries.jsonl, qrels/test.tsv
    corpus.fvecs, query.fvecs, corpus.ids, query.ids
    manifest.properties
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from text_prep_common import (
    DEFAULT_MODEL,
    DEFAULT_ROOT,
    chunk_paragraphs,
    embed_texts,
    passage_text,
    print_next_steps,
    read_jsonl,
    split_paragraphs,
    strip_gutenberg_boilerplate,
    write_fvecs,
    write_ids,
    write_jsonl,
    write_manifest,
    write_qrels,
)

DEFAULT_BOOKS: dict[int, tuple[str, str]] = {
    1342: ("pride-prejudice-1342", "Pride and Prejudice"),
    2701: ("moby-dick-2701", "Moby-Dick"),
    84: ("frankenstein-84", "Frankenstein"),
    11: ("alice-11", "Alice's Adventures in Wonderland"),
    1661: ("sherlock-1661", "The Adventures of Sherlock Holmes"),
    345: ("dracula-345", "Dracula"),
    74: ("tom-sawyer-74", "The Adventures of Tom Sawyer"),
    1260: ("jane-eyre-1260", "Jane Eyre"),
    5200: ("metamorphosis-5200", "Metamorphosis"),
    98: ("tale-two-cities-98", "A Tale of Two Cities"),
}

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
QUERIES_FILE = Path(__file__).parent / "data" / "gutenberg_queries.jsonl"
DATASET_NAME = "gutenberg"


def download_book(book_id: int) -> str:
    url = GUTENBERG_URL.format(book_id=book_id)
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=120) as response:
        raw = response.read()
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return strip_gutenberg_boilerplate(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    return strip_gutenberg_boilerplate(raw.decode("utf-8", errors="replace"))


def build_corpus_rows(book_ids: list[int]) -> list[dict]:
    rows: list[dict] = []
    for book_id in book_ids:
        slug, title = DEFAULT_BOOKS[book_id]
        text = download_book(book_id)
        chunks = chunk_paragraphs(split_paragraphs(text))
        print(f"  {title}: {len(chunks)} passages")
        for idx, chunk in enumerate(chunks):
            rows.append(
                {
                    "_id": f"{slug}-{idx:04d}",
                    "title": title,
                    "text": chunk,
                    "book_slug": slug,
                }
            )
    return rows


def load_queries(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Bundled queries not found: {path}")
    return read_jsonl(path)


def build_qrels(queries: list[dict], corpus_rows: list[dict]) -> dict[str, list[str]]:
    chunks_by_book: dict[str, list[str]] = {}
    for row in corpus_rows:
        book_slug = row.get("book_slug", "")
        chunks_by_book.setdefault(book_slug, []).append(row["_id"])

    qrels: dict[str, list[str]] = {}
    for query in queries:
        book_slug = query.get("book_slug", "")
        chunk_ids = chunks_by_book.get(book_slug, [])
        if chunk_ids:
            qrels[query["_id"]] = chunk_ids
    return qrels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Output root directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-transformers model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument(
        "--book-ids",
        default=",".join(str(book_id) for book_id in DEFAULT_BOOKS),
        help="Comma-separated Project Gutenberg book ids",
    )
    parser.add_argument("--queries-file", type=Path, default=QUERIES_FILE, help="Bundled queries jsonl")
    parser.add_argument("--max-corpus", type=int, default=0, help="Optional corpus size cap for smoke tests")
    parser.add_argument("--max-queries", type=int, default=0, help="Optional query size cap for smoke tests")
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "Missing dependency: sentence-transformers\n"
            "Install with: pip install -r scripts/requirements.txt",
            file=sys.stderr,
        )
        return 1

    book_ids = [int(part.strip()) for part in args.book_ids.split(",") if part.strip()]
    unknown = [book_id for book_id in book_ids if book_id not in DEFAULT_BOOKS]
    if unknown:
        print(f"Unknown book ids (add to DEFAULT_BOOKS): {unknown}", file=sys.stderr)
        return 1

    print(f"Building corpus from {len(book_ids)} books ...")
    corpus_rows = build_corpus_rows(book_ids)
    query_rows = load_queries(args.queries_file)
    query_rows = [row for row in query_rows if row.get("book_slug") in {DEFAULT_BOOKS[b][0] for b in book_ids}]

    if args.max_corpus > 0:
        corpus_rows = corpus_rows[: args.max_corpus]
    if args.max_queries > 0:
        query_rows = query_rows[: args.max_queries]

    corpus_ids = [row["_id"] for row in corpus_rows]
    query_ids = [row["_id"] for row in query_rows]
    qrels = build_qrels(query_rows, corpus_rows)

    print(f"Loading embedding model {args.model} ...")
    model = SentenceTransformer(args.model)

    print(f"Embedding {len(corpus_rows)} corpus passages ...")
    corpus_vectors = embed_texts(model, [passage_text(row) for row in corpus_rows], args.batch_size)
    print(f"Embedding {len(query_rows)} queries ...")
    query_vectors = embed_texts(model, [row["text"] for row in query_rows], args.batch_size)

    out_dir = args.root / DATASET_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(out_dir / "corpus.jsonl", [{k: v for k, v in row.items() if k != "book_slug"} for row in corpus_rows])
    write_jsonl(out_dir / "queries.jsonl", [{k: v for k, v in row.items() if k in ("_id", "text")} for row in query_rows])
    write_fvecs(out_dir / "corpus.fvecs", corpus_vectors)
    write_fvecs(out_dir / "query.fvecs", query_vectors)
    write_ids(out_dir / "corpus.ids", corpus_ids)
    write_ids(out_dir / "query.ids", query_ids)
    write_qrels(out_dir / "qrels" / "test.tsv", qrels)
    write_manifest(
        out_dir / "manifest.properties",
        DATASET_NAME,
        args.model,
        len(corpus_ids),
        len(query_ids),
        len(corpus_vectors[0]) if len(corpus_vectors) else 0,
    )

    print_next_steps(DATASET_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
