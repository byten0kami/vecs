# vecs — Semantic Search for Bloomly

## Overview

A Python CLI (`vecs`) + MCP server that provides semantic search over Bloomly's codebase and Claude Code session transcripts using Voyage AI embeddings and ChromaDB.

## Scope

### Two collections
- **`code`** — .cs files from `Bloomly/Assets/` (330 files, ~1.9 MB)
- **`sessions`** — Claude Code session transcripts from `~/.claude/projects/-Users-darynavoloshyna-Repositories-Bloomly/` (198 sessions, ~345 MB raw, ~60 MB after preprocessing)

### Backup
Session transcripts backed up to `~/.claude/projects/vecs/` before any processing.

## CLI Interface

```bash
vecs index                                          # index everything (incremental)
vecs search "animation state machine"               # search across both collections
vecs search "animation" --collection code            # code only
vecs search "that bug with sprites" --collection sessions  # sessions only
```

## MCP Interface

Tool: `semantic_search(query, collection?)`

Configured globally in `~/.claude/settings.json` so Claude Code can access it from any project directory.

## Preprocessing

### Code files
- Split by class/method boundaries when possible
- Fallback: line-based chunks (~200 lines, 50 line overlap)

### Session transcripts
- Strip repeated system prompts
- Strip base64/binary content
- Strip tool call metadata, keep results (summarized)
- Keep: user messages, assistant responses, meaningful tool results
- Chunk by conversation turns (~10 messages per chunk)

## Incremental Indexing

- Tracks file hashes + last-modified timestamps in a local manifest (`~/.vecs/manifest.json`)
- `vecs index` only processes new/changed files
- Skips already-indexed content

## Stack

- Python 3.14 + `uv`
- `voyageai` SDK — `voyage-code-3` for code, `voyage-3` for sessions
- ChromaDB local — stored in `~/.vecs/chromadb/`
- MCP server via `stdio` transport

## File Structure

```
~/Repositories/vecs/
  pyproject.toml
  src/
    vecs/
      __init__.py
      cli.py          # CLI entry point (click)
      indexer.py       # preprocessing + chunking + embedding
      searcher.py      # query logic
      mcp_server.py    # MCP wrapper
      config.py        # paths, collections, Voyage settings
```

## Cost Estimate

### Initial index
- Code: ~500k tokens → ~$0.03
- Sessions: ~15M tokens (after preprocessing) → ~$0.90
- **Total: ~$1**

### Ongoing
- New sessions: pennies each
- Code re-index on changes: negligible

## Future Extensibility

- `vecs add-project <path>` to index additional codebases
- Additional collections as needed
