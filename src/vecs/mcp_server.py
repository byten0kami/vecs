from mcp.server.fastmcp import FastMCP

from vecs.searcher import search

mcp = FastMCP("vecs")


@mcp.tool()
def semantic_search(
    query: str,
    collection: str | None = None,
    n_results: int = 5,
) -> str:
    """Search Bloomly code and Claude Code session transcripts semantically.

    Args:
        query: Natural language search query.
        collection: Optional filter — "code" or "sessions". Searches both if omitted.
        n_results: Number of results to return (default 5).
    """
    results = search(query, collection_name=collection, n_results=n_results)
    if not results:
        return "No results found."

    parts = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        source = meta.get("file_path") or f"session:{meta.get('session_id', '?')}"
        dist = r.get("distance")
        dist_str = f" (distance: {dist:.4f})" if dist is not None else ""
        header = f"--- Result {i} [{r.get('collection', '?')}] {source}{dist_str} ---"
        text = r["text"]
        if len(text) > 2000:
            text = text[:2000] + "\n... [truncated]"
        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)
