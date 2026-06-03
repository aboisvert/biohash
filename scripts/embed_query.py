#!/usr/bin/env python3

"""Embed a single query string for BioHash text search (stdout: space-separated floats)."""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Query text to embed")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-transformers model name")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest.properties; overrides --model with embeddingModel=",
    )
    args = parser.parse_args()

    model_name = args.model
    if args.manifest is not None and args.manifest.exists():
        for line in args.manifest.read_text(encoding="utf-8").splitlines():
            if line.startswith("embeddingModel="):
                model_name = line.split("=", 1)[1].strip()
                break

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "Missing dependency: sentence-transformers\n"
            "Install with: pip install -r scripts/requirements.txt",
            file=sys.stderr,
        )
        return 1

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model = SentenceTransformer(model_name)
    vector = model.encode(
        [args.text],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0]
    print(" ".join(f"{v:.8g}" for v in vector))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
