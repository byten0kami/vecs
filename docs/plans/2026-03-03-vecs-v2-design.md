# vecs v2 Design

## Context

Feedback on vecs v1 identified 11 issues across high, medium, and low impact. This design addresses all of them.

## Issues Addressed

### High Impact

#### 1. AST-aware chunking (C# + TypeScript)

Current line-based chunking (200 lines, 50 overlap) splits code at arbitrary boundaries. A 250-line class gets cut mid-method, producing poor embeddings.

**Solution:** Use tree-sitter to parse C# and TypeScript files and chunk at class/method/function boundaries.

- New `ast_chunker.py` module with extension-to-parser registry
- Registry: `{".cs": "c_sharp", ".ts": "typescript", ".tsx": "typescript"}`
- Chunk at top-level declarations: classes, structs, enums, interfaces, methods, functions
- Max chunk size cap of 500 lines — if a single declaration exceeds it, split within using line-based fallback
- Small adjacent declarations (under 20 lines) are merged into a single chunk
- Unsupported file types fall back to existing line-based chunking
- More grammars can be added later by installing the grammar and adding the mapping

#### 2. MCP-triggered reindex (no daemon)

Currently requires switching to terminal to run `vecs index`.

**Solution:** Expose `reindex` and `index_status` as MCP tools.

- `reindex(project?: string)` — triggers incremental reindex for one or all projects
- `index_status(project?: string)` — returns chunk counts and manifest info
- No file watcher daemon for now; can be added later

#### 3. Multi-project config

Hardcoded `BLOOMLY_CODE_DIR` in config.py prevents reuse.

**Solution:** Central YAML config at `~/.vecs/config.yaml` with CLI management commands.

Config structure:
```yaml
projects:
  bloomly:
    code_dir: ~/Repositories/Bloomly/Assets/
    extensions: [.cs, .shader, .asmdef, .json]
    sessions_dir: ~/.claude/projects/-Users-darynavoloshyna-Repositories-Bloomly/
  supabase-functions:
    code_dir: ~/Repositories/Bloomly/supabase/functions/
    extensions: [.ts, .tsx]
```

CLI commands:
- `vecs project add <name> --code-dir <path> --ext .cs,.ts [--sessions-dir <path>]`
- `vecs project remove <name>`
- `vecs project list`

Collections become `{project}:code` and `{project}:sessions`.

#### 4. MCP tools for index management

Only `semantic_search` is exposed via MCP.

**Solution:** Add two more MCP tools (see #2 above).

### Medium Impact

#### 5. Path/namespace filtering in search

Can't scope search to a subdirectory.

**Solution:** Add `path_filter` parameter to search and MCP tool.

- Uses ChromaDB `where` clause: `{"file_path": {"$contains": path_filter}}`
- Example: `semantic_search(query="animation", path_filter="Services/Analytics/")`

#### 6. Deduplicate overlapping results

50-line overlap causes near-duplicate results in top-N.

**Solution:** Post-processing deduplication in `searcher.py`.

- After retrieving results, compute Jaccard similarity on line sets between consecutive results
- If two results share >70% of lines, keep only the higher-ranked one
- Applied after rank fusion (if hybrid search is active)

#### 7. Multi-file-type support

Only indexes `.cs` files.

**Solution:** Configurable per project via `extensions` list in config.

- Default: `[".cs"]` for backward compat
- Supports any text file: `.shader`, `.asmdef`, `.json`, `.ts`, `.tsx`, etc.
- Files with tree-sitter grammar get AST chunking; others get line-based

#### 8. Hybrid search (BM25 + vector + RRF)

Pure vector search misses exact keyword matches.

**Solution:** Add BM25 keyword matching alongside vector search, merged via Reciprocal Rank Fusion.

- Build BM25 index during indexing, stored as `~/.vecs/bm25/{project}_{collection}.pkl`
- At search time: run both BM25 and vector queries
- Merge using RRF: `score = sum(1 / (k + rank))` for k=60
- BM25 index rebuilt during reindex

### Low Impact

#### 9. Singleton Voyage client

`searcher.py` creates a new `voyageai.Client()` per call.

**Solution:** Module-level lazy singleton.

```python
_vo_client = None

def get_voyage_client():
    global _vo_client
    if _vo_client is None:
        _vo_client = voyageai.Client()
    return _vo_client
```

#### 10. Session chunk overlap

Session chunks have hard 10-message boundaries with no overlap.

**Solution:** Add configurable overlap (default: 2 messages).

- Same sliding window pattern as code chunks
- 10-message chunks with 2-message overlap preserves context at boundaries

#### 11. Query caching

No caching of embeddings or results.

**Solution:** In-memory TTL cache using `cachetools`.

- Cache embedding results by `(query_string, model)`, TTL = 5 minutes
- Cache search results by `(query, collection, n_results, path_filter)`, TTL = 5 minutes
- Lightweight, in-memory only, no persistence

## New Dependencies

| Package | Purpose |
|---------|---------|
| tree-sitter | AST parsing framework |
| tree-sitter-c-sharp | C# grammar |
| tree-sitter-typescript | TypeScript/TSX grammar |
| rank-bm25 | BM25 keyword search |
| cachetools | TTL caching |
| pyyaml | YAML config parsing |

## Migration

- Existing `~/.vecs/manifest.json` and ChromaDB data will need to be rebuilt (one-time reindex)
- Old config constants in `config.py` replaced by YAML-driven config
- `config.py` retains defaults (models, chunk sizes, batch size) but project-specific paths move to YAML
