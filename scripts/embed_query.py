#!/usr/bin/env python3

"""Embed query strings for BioHash text search.

One-shot mode (stdout: space-separated floats):
  python embed_query.py --text "your query" [--manifest path/to/manifest.properties]

Server mode (length-prefixed stdin/stdout framing for long-lived REPL use):
  python embed_query.py --server [--manifest path/to/manifest.properties]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QUIT_PAYLOAD = b"__QUIT__"


def resolve_model_name(manifest: Path | None, model: str) -> str:
    model_name = model
    if manifest is not None and manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.startswith("embeddingModel="):
                model_name = line.split("=", 1)[1].strip()
                break
    return model_name


def import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "Missing dependency: sentence-transformers\n"
            "Install with: pip install -r scripts/requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    return SentenceTransformer


def load_model(model_name: str) -> SentenceTransformer:
    SentenceTransformer = import_sentence_transformers()
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return SentenceTransformer(model_name)


def encode_text(model: SentenceTransformer, text: str) -> str:
    vector = model.encode(
        [text],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0]
    return " ".join(f"{v:.8g}" for v in vector)


def write_frame(payload: bytes) -> None:
    sys.stdout.buffer.write(f"{len(payload)}\n".encode())
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def read_frame() -> bytes | None:
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    n = int(line.strip())
    payload = sys.stdin.buffer.read(n)
    if len(payload) != n:
        return None
    return payload


def run_server(model_name: str) -> int:
    model = load_model(model_name)
    write_frame(b"READY")
    while True:
        payload = read_frame()
        if payload is None:
            break
        if payload == QUIT_PAYLOAD:
            break
        text = payload.decode("utf-8")
        if not text:
            write_frame(b"ERR\tempty query")
            continue
        try:
            write_frame(encode_text(model, text).encode("utf-8"))
        except Exception as exc:  # noqa: BLE001 — return error to client
            write_frame(f"ERR\t{exc}".encode("utf-8"))
    return 0


def run_once(model_name: str, text: str) -> int:
    model = load_model(model_name)
    print(encode_text(model, text))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Query text to embed (one-shot mode)")
    parser.add_argument("--server", action="store_true", help="Run as long-lived embed server")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-transformers model name")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest.properties; overrides --model with embeddingModel=",
    )
    args = parser.parse_args()

    if args.server and args.text:
        print("Cannot use --server with --text", file=sys.stderr)
        return 1
    if not args.server and not args.text:
        print("Either --text or --server is required", file=sys.stderr)
        return 1

    model_name = resolve_model_name(args.manifest, args.model)
    if args.server:
        return run_server(model_name)
    return run_once(model_name, args.text)


if __name__ == "__main__":
    raise SystemExit(main())
