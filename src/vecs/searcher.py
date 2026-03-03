from __future__ import annotations

from cachetools import TTLCache

from vecs.bm25_index import BM25Index
from vecs.clients import get_voyage_client, get_chromadb_client
from vecs.config import (
    CODE_MODEL,
    SESSIONS_MODEL,
    VECS_DIR,
    load_config,
)

# Cache embeddings by (query, model) for 5 minutes
_embedding_cache: TTLCache = TTLCache(maxsize=256, ttl=300)


def _clear_caches() -> None:
    """Clear all caches."""
    _embedding_cache.clear()


def _cached_embed(vo, query: str, model: str) -> list[float]:
    """Embed a query, using cache when available."""
    key = (query, model)
    if key in _embedding_cache:
        return _embedding_cache[key]
    embedding = vo.embed([query], model=model, input_type="query").embeddings[0]
    _embedding_cache[key] = embedding
    return embedding


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


def deduplicate_results(results: list[dict], threshold: float = 0.7) -> list[dict]:
    """Remove results with high line overlap.

    For each pair, if Jaccard similarity of their line sets exceeds threshold,
    the lower-ranked (higher distance) result is dropped.
    """
    if len(results) <= 1:
        return results

    keep = []
    for r in results:
        r_lines = set(r["text"].split("\n"))
        is_dup = False
        for kept in keep:
            k_lines = set(kept["text"].split("\n"))
            intersection = len(r_lines & k_lines)
            union = len(r_lines | k_lines)
            if union > 0 and intersection / union > threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(r)
    return keep


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Merge vector and BM25 results using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) across both result lists.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, r in enumerate(vector_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0) + 1 / (k + rank + 1)
        if rid not in doc_map:
            doc_map[rid] = r

    for rank, r in enumerate(bm25_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0) + 1 / (k + rank + 1)
        if rid not in doc_map:
            doc_map[rid] = {
                "id": rid,
                "text": r["text"],
                "metadata": r.get("metadata", {}),
                "distance": None,
            }

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[rid] for rid, _ in ranked]


def search(
    query: str,
    collection_name: str | None = None,
    n_results: int = 5,
    path_filter: str | None = None,
    project: str | None = None,
) -> list[dict]:
    """Search across one or both collections, optionally filtered by path.

    Args:
        query: Search query string.
        collection_name: "code" or "sessions" or None (both).
        n_results: Number of results to return.
        path_filter: Filter results to paths containing this substring.
        project: Search a specific project (default: all).
    """
    vo = get_voyage_client()
    db = get_chromadb_client()
    config = load_config()

    projects = (
        {project: config.projects[project]}
        if project and project in config.projects
        else config.projects
    )

    targets = []
    for proj_name, proj in projects.items():
        if collection_name is None or collection_name == "code":
            targets.append((proj.code_collection, CODE_MODEL, proj_name))
        if collection_name is None or collection_name == "sessions":
            targets.append((proj.sessions_collection, SESSIONS_MODEL, proj_name))

    all_results = []
    for col_name, model, proj_name in targets:
        try:
            collection = db.get_collection(col_name)
        except Exception:
            continue

        embedding = _cached_embed(vo, query, model)

        where = None
        if path_filter:
            where = {"file_path": {"$contains": path_filter}}

        # Request extra results to account for dedup filtering
        fetch_n = n_results * 2

        try:
            raw = collection.query(
                query_embeddings=[embedding],
                n_results=fetch_n,
                include=["documents", "metadatas", "distances"],
                where=where,
            )
        except Exception:
            # where clause may fail on sessions collection (no file_path)
            if path_filter:
                continue
            raise

        results = format_results(raw)
        for r in results:
            r["collection"] = col_name
            r["project"] = proj_name
        all_results.extend(results)

    # BM25 keyword search
    bm25_results = []
    bm25_dir = VECS_DIR / "bm25"
    for col_name, model, proj_name in targets:
        suffix = "code" if ":code" in col_name else "sessions"
        bm25_path = bm25_dir / f"{proj_name}_{suffix}.pkl"
        bm25 = BM25Index(bm25_path)
        if bm25.load():
            hits = bm25.search(query, n=fetch_n)
            for h in hits:
                h["collection"] = col_name
                h["project"] = proj_name
            bm25_results.extend(hits)

    if bm25_results:
        all_results = reciprocal_rank_fusion(all_results, bm25_results)
    else:
        all_results.sort(key=lambda r: r.get("distance") or float("inf"))

    all_results = deduplicate_results(all_results)
    return all_results[:n_results]
