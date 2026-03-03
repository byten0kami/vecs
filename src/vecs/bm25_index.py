from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """BM25 keyword search index with persistence."""

    def __init__(self, path: Path):
        self.path = path
        self.bm25: BM25Okapi | None = None
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []

    def build(self, docs: list[dict]) -> None:
        """Build index from list of {"id": ..., "text": ...} dicts."""
        self.doc_ids = [d["id"] for d in docs]
        self.doc_texts = [d["text"] for d in docs]
        tokenized = [_tokenize(d["text"]) for d in docs]
        if tokenized:
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def search(self, query: str, n: int = 5) -> list[dict]:
        """Search the index. Returns list of {"id", "text", "score"}."""
        if self.bm25 is None or not self.doc_ids:
            return []

        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:n]

        return [
            {
                "id": self.doc_ids[i],
                "text": self.doc_texts[i],
                "score": float(score),
            }
            for i, score in ranked
            if score > 0
        ]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(
                {
                    "doc_ids": self.doc_ids,
                    "doc_texts": self.doc_texts,
                },
                f,
            )

    def load(self) -> bool:
        """Load from disk. Returns True if loaded, False if file missing."""
        if not self.path.exists():
            return False
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        self.doc_ids = data["doc_ids"]
        self.doc_texts = data["doc_texts"]
        # Rebuild BM25 from tokenized texts (BM25Okapi doesn't pickle well)
        tokenized = [_tokenize(text) for text in self.doc_texts]
        if tokenized:
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None
        return True
