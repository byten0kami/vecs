from mcp.server.fastmcp import FastMCP

from vecs.indexer import run_index, get_status
from vecs.searcher import search

mcp = FastMCP("vecs")


@mcp.tool()
def semantic_search(
    query: str,
    collection: str | None = None,
    n_results: int = 5,
    path_filter: str | None = None,
    project: str | None = None,
) -> str:
    """Search code and session transcripts semantically.

    Args:
        query: Natural language search query.
        collection: Optional filter — "code" or "sessions". Searches both if omitted.
        n_results: Number of results to return (default 5).
        path_filter: Filter results to file paths containing this substring (e.g. "Services/Analytics/").
        project: Search a specific project (default: all).
    """
    results = search(
        query,
        collection_name=collection,
        n_results=n_results,
        path_filter=path_filter,
        project=project,
    )
    if not results:
        return "No results found."

    parts = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        source = meta.get("file_path") or f"session:{meta.get('session_id', '?')}"
        dist = r.get("distance")
        dist_str = f" (distance: {dist:.4f})" if dist is not None else ""
        proj = r.get("project", "?")
        header = f"--- Result {i} [{proj}:{r.get('collection', '?')}] {source}{dist_str} ---"
        text = r["text"]
        if len(text) > 2000:
            text = text[:2000] + "\n... [truncated]"
        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


@mcp.tool()
def reindex(project: str | None = None) -> str:
    """Trigger incremental reindexing of code and session files.

    Args:
        project: Reindex a specific project (default: all configured projects).
    """
    try:
        run_index(project_name=project)
        status = get_status(project_name=project)
        return f"Reindex complete. {status['total_code_chunks']} code chunks, {status['total_session_chunks']} session chunks."
    except Exception as e:
        return f"Reindex failed: {e}"


@mcp.tool()
def index_status(project: str | None = None) -> str:
    """Check the current index status — chunk counts and tracked files.

    Args:
        project: Status for a specific project (default: all).
    """
    status = get_status(project_name=project)
    lines = []
    for name, info in status.get("projects", {}).items():
        lines.append(f"[{name}] code: {info['code_chunks']} chunks, sessions: {info['session_chunks']} chunks")
    lines.append(f"Total: {status['total_code_chunks']} code + {status['total_session_chunks']} sessions")
    lines.append(f"Tracked files: {status.get('manifest_entries', 0)}")
    return "\n".join(lines)
