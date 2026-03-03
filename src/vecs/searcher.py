from __future__ import annotations

import chromadb
import voyageai

from vecs.config import (
    CHROMADB_DIR,
    CODE_COLLECTION,
    CODE_MODEL,
    SESSIONS_COLLECTION,
    SESSIONS_MODEL,
)


def format_results(raw: dict) -> list[dict]:
    """Format ChromaDB query results into a clean list."""
    if not raw["ids"] or not raw["ids"][0]:
        return []

    results = []
    ids = raw["ids"][0]
    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    distances = raw.get("distances", [[None] * len(ids)])[0]

    for id_, doc, meta, dist in zip(ids, docs, metas, distances):
        results.append(
            {
                "id": id_,
                "text": doc,
                "metadata": meta,
                "distance": dist,
            }
        )
    return results


def search(
    query: str,
    collection_name: str | None = None,
    n_results: int = 5,
) -> list[dict]:
    """Search across one or both collections."""
    vo = voyageai.Client()
    db = chromadb.PersistentClient(path=str(CHROMADB_DIR))

    targets = []
    if collection_name is None or collection_name == "code":
        targets.append((CODE_COLLECTION, CODE_MODEL))
    if collection_name is None or collection_name == "sessions":
        targets.append((SESSIONS_COLLECTION, SESSIONS_MODEL))

    all_results = []
    for col_name, model in targets:
        try:
            collection = db.get_collection(col_name)
        except Exception:
            continue

        embedding = vo.embed([query], model=model, input_type="query").embeddings[0]

        raw = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        results = format_results(raw)
        for r in results:
            r["collection"] = col_name
        all_results.extend(results)

    # Sort by distance (lower = more similar)
    all_results.sort(key=lambda r: r.get("distance") or float("inf"))
    return all_results[:n_results]
