# vecs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CLI + MCP server that provides semantic search over Bloomly's code and session transcripts using Voyage AI embeddings and ChromaDB.

**Architecture:** Python package using `uv`, with `click` for CLI, `voyageai` SDK for embeddings, `chromadb` for vector storage, and `mcp[cli]` FastMCP for the MCP server. Data stored in `~/.vecs/`. Two collections: `code` (Bloomly .cs files) and `sessions` (Claude Code transcripts).

**Tech Stack:** Python 3.14, uv, voyageai, chromadb, click, mcp[cli]

---

### Task 1: Project Scaffolding

**Files:**
- Create: `~/Repositories/vecs/pyproject.toml`
- Create: `~/Repositories/vecs/src/vecs/__init__.py`
- Create: `~/Repositories/vecs/src/vecs/config.py`

**Step 1: Initialize uv project**

Run:
```bash
cd ~/Repositories/vecs
uv init --lib --name vecs
```

**Step 2: Replace pyproject.toml with correct config**

```toml
[project]
name = "vecs"
version = "0.1.0"
description = "Semantic search for Bloomly code and Claude Code sessions"
requires-python = ">=3.12"
dependencies = [
    "voyageai>=0.3",
    "chromadb>=1.0",
    "click>=8.0",
    "mcp[cli]>=1.2",
]

[project.scripts]
vecs = "vecs.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Step 3: Install dependencies**

Run:
```bash
cd ~/Repositories/vecs
uv sync
```
Expected: Dependencies install successfully.

**Step 4: Create config module**

`src/vecs/config.py`:
```python
from pathlib import Path

VECS_DIR = Path.home() / ".vecs"
CHROMADB_DIR = VECS_DIR / "chromadb"
MANIFEST_PATH = VECS_DIR / "manifest.json"

BLOOMLY_CODE_DIR = Path.home() / "Repositories" / "Bloomly" / "Assets"
BLOOMLY_SESSIONS_DIR = (
    Path.home()
    / ".claude"
    / "projects"
    / "-Users-darynavoloshyna-Repositories-Bloomly"
)

CODE_COLLECTION = "code"
SESSIONS_COLLECTION = "sessions"

CODE_EXTENSIONS = {".cs"}
CODE_MODEL = "voyage-code-3"
SESSIONS_MODEL = "voyage-3"

CODE_CHUNK_LINES = 200
CODE_CHUNK_OVERLAP = 50
SESSION_CHUNK_MESSAGES = 10

VOYAGE_BATCH_SIZE = 128
VOYAGE_MAX_TOKENS_PER_BATCH = 120_000
```

`src/vecs/__init__.py`:
```python
"""vecs — Semantic search for Bloomly."""
```

**Step 5: Verify import works**

Run:
```bash
cd ~/Repositories/vecs
uv run python -c "from vecs.config import VECS_DIR; print(VECS_DIR)"
```
Expected: `/Users/darynavoloshyna/.vecs`

**Step 6: Initialize git and commit**

Run:
```bash
cd ~/Repositories/vecs
git init
echo "__pycache__/\n*.egg-info/\n.venv/\ndist/" > .gitignore
git add pyproject.toml src/ .gitignore docs/
git commit -m "feat: project scaffolding with config"
```

---

### Task 2: Code Chunker

**Files:**
- Create: `~/Repositories/vecs/src/vecs/chunkers.py`
- Create: `~/Repositories/vecs/tests/test_chunkers.py`

**Step 1: Write failing test for line-based chunking**

`tests/test_chunkers.py`:
```python
from vecs.chunkers import chunk_code_file


def test_chunk_code_small_file():
    """A file smaller than chunk size returns one chunk."""
    content = "line1\nline2\nline3"
    chunks = chunk_code_file(content, file_path="Test.cs", chunk_lines=5, overlap=2)
    assert len(chunks) == 1
    assert chunks[0]["text"] == content
    assert chunks[0]["metadata"]["file_path"] == "Test.cs"
    assert chunks[0]["metadata"]["chunk_index"] == 0


def test_chunk_code_large_file():
    """A file larger than chunk size returns overlapping chunks."""
    lines = [f"line{i}" for i in range(10)]
    content = "\n".join(lines)
    chunks = chunk_code_file(content, file_path="Big.cs", chunk_lines=4, overlap=2)
    assert len(chunks) >= 3
    # Verify overlap: last 2 lines of chunk 0 == first 2 lines of chunk 1
    chunk0_lines = chunks[0]["text"].split("\n")
    chunk1_lines = chunks[1]["text"].split("\n")
    assert chunk0_lines[-2:] == chunk1_lines[:2]


def test_chunk_code_empty_file():
    """Empty file returns no chunks."""
    chunks = chunk_code_file("", file_path="Empty.cs", chunk_lines=5, overlap=2)
    assert len(chunks) == 0
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd ~/Repositories/vecs
uv run pytest tests/test_chunkers.py -v
```
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

**Step 3: Implement code chunker**

`src/vecs/chunkers.py`:
```python
from __future__ import annotations


def chunk_code_file(
    content: str,
    file_path: str,
    chunk_lines: int = 200,
    overlap: int = 50,
) -> list[dict]:
    """Split a code file into overlapping line-based chunks."""
    if not content.strip():
        return []

    lines = content.split("\n")
    chunks = []
    start = 0

    while start < len(lines):
        end = min(start + chunk_lines, len(lines))
        chunk_text = "\n".join(lines[start:end])
        chunks.append(
            {
                "text": chunk_text,
                "metadata": {
                    "file_path": file_path,
                    "chunk_index": len(chunks),
                    "start_line": start + 1,
                    "end_line": end,
                },
            }
        )
        if end >= len(lines):
            break
        start += chunk_lines - overlap

    return chunks
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd ~/Repositories/vecs
uv run pytest tests/test_chunkers.py -v
```
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
cd ~/Repositories/vecs
git add src/vecs/chunkers.py tests/
git commit -m "feat: code file chunker with overlap"
```

---

### Task 3: Session Preprocessor

**Files:**
- Modify: `~/Repositories/vecs/src/vecs/chunkers.py` (add session functions)
- Create: `~/Repositories/vecs/tests/test_session_chunker.py`

**Step 1: Write failing test for session preprocessing**

`tests/test_session_chunker.py`:
```python
import json

from vecs.chunkers import preprocess_session, chunk_session


def _make_message(role: str, content: str, msg_type: str = "user") -> str:
    """Create a minimal JSONL line matching Claude Code format."""
    obj = {
        "type": msg_type,
        "message": {"role": role, "content": content},
        "sessionId": "test-session",
        "timestamp": "2026-01-01T00:00:00Z",
        "uuid": "fake-uuid",
    }
    return json.dumps(obj)


def test_preprocess_strips_system_prompts():
    """System reminder content is stripped."""
    lines = [
        _make_message("user", "<system-reminder>blah</system-reminder>"),
        _make_message("user", "real question"),
        _make_message("assistant", "real answer", msg_type="assistant"),
    ]
    messages = preprocess_session("\n".join(lines))
    texts = [m["text"] for m in messages]
    assert not any("<system-reminder>" in t for t in texts)
    assert any("real question" in t for t in texts)


def test_preprocess_strips_base64():
    """Base64 image data is replaced with placeholder."""
    lines = [
        _make_message("user", "here is image data:iVBORw0KGgoAAAAN" + "A" * 200),
    ]
    messages = preprocess_session("\n".join(lines))
    assert not any("iVBORw0KGgo" in m["text"] for m in messages)


def test_preprocess_skips_metadata_lines():
    """Lines with type file-history-snapshot or progress are skipped."""
    lines = [
        json.dumps({"type": "file-history-snapshot", "snapshot": {}}),
        _make_message("user", "real message"),
    ]
    messages = preprocess_session("\n".join(lines))
    assert len(messages) == 1


def test_chunk_session_groups_messages():
    """Messages are grouped into chunks of N."""
    lines = [_make_message("user", f"msg {i}") for i in range(25)]
    messages = preprocess_session("\n".join(lines))
    chunks = chunk_session(messages, session_id="test", chunk_size=10)
    assert len(chunks) >= 2
    assert chunks[0]["metadata"]["session_id"] == "test"
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd ~/Repositories/vecs
uv run pytest tests/test_session_chunker.py -v
```
Expected: FAIL with `ImportError`

**Step 3: Implement session preprocessor**

Add to `src/vecs/chunkers.py`:
```python
import json
import re

SKIP_TYPES = {"file-history-snapshot", "progress"}
SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
BASE64_RE = re.compile(r"[A-Za-z0-9+/]{100,}={0,2}")


def preprocess_session(raw_jsonl: str) -> list[dict]:
    """Parse a session JSONL file and extract clean messages."""
    messages = []
    for line in raw_jsonl.strip().split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type", "")
        if msg_type in SKIP_TYPES:
            continue

        message = obj.get("message")
        if not message:
            continue

        role = message.get("role", "")
        content = message.get("content", "")

        # Handle content that's a list (multimodal messages)
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part["text"])
            content = "\n".join(text_parts)

        if not isinstance(content, str) or not content.strip():
            continue

        # Strip system reminders
        content = SYSTEM_REMINDER_RE.sub("", content)
        # Strip base64
        content = BASE64_RE.sub("[binary data removed]", content)
        # Strip thinking blocks content (keep the fact that thinking happened)
        content = content.strip()

        if not content:
            continue

        messages.append(
            {
                "role": role,
                "text": content,
                "timestamp": obj.get("timestamp", ""),
            }
        )

    return messages


def chunk_session(
    messages: list[dict],
    session_id: str,
    chunk_size: int = 10,
) -> list[dict]:
    """Group preprocessed messages into chunks."""
    chunks = []
    for i in range(0, len(messages), chunk_size):
        group = messages[i : i + chunk_size]
        combined = "\n\n".join(
            f"[{m['role']}]: {m['text']}" for m in group
        )
        first_ts = group[0].get("timestamp", "")
        last_ts = group[-1].get("timestamp", "")
        chunks.append(
            {
                "text": combined,
                "metadata": {
                    "session_id": session_id,
                    "chunk_index": len(chunks),
                    "start_timestamp": first_ts,
                    "end_timestamp": last_ts,
                    "message_count": len(group),
                },
            }
        )
    return chunks
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd ~/Repositories/vecs
uv run pytest tests/test_session_chunker.py -v
```
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
cd ~/Repositories/vecs
git add src/vecs/chunkers.py tests/test_session_chunker.py
git commit -m "feat: session preprocessor and chunker"
```

---

### Task 4: Indexer (Voyage + ChromaDB)

**Files:**
- Create: `~/Repositories/vecs/src/vecs/indexer.py`
- Create: `~/Repositories/vecs/tests/test_indexer.py`

**Step 1: Write failing test for manifest tracking**

`tests/test_indexer.py`:
```python
import json
from pathlib import Path

from vecs.indexer import Manifest


def test_manifest_new_file(tmp_path):
    """A new file is detected as needing indexing."""
    manifest_path = tmp_path / "manifest.json"
    m = Manifest(manifest_path)
    test_file = tmp_path / "test.cs"
    test_file.write_text("hello")
    assert m.needs_indexing(test_file) is True


def test_manifest_already_indexed(tmp_path):
    """A file that hasn't changed is skipped."""
    manifest_path = tmp_path / "manifest.json"
    m = Manifest(manifest_path)
    test_file = tmp_path / "test.cs"
    test_file.write_text("hello")
    m.mark_indexed(test_file)
    m.save()
    # Reload
    m2 = Manifest(manifest_path)
    assert m2.needs_indexing(test_file) is False


def test_manifest_changed_file(tmp_path):
    """A file with new content is detected as needing re-indexing."""
    manifest_path = tmp_path / "manifest.json"
    m = Manifest(manifest_path)
    test_file = tmp_path / "test.cs"
    test_file.write_text("hello")
    m.mark_indexed(test_file)
    m.save()
    test_file.write_text("changed")
    m2 = Manifest(manifest_path)
    assert m2.needs_indexing(test_file) is True
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd ~/Repositories/vecs
uv run pytest tests/test_indexer.py -v
```
Expected: FAIL with `ImportError`

**Step 3: Implement Manifest class**

`src/vecs/indexer.py`:
```python
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import chromadb
import voyageai

from vecs.chunkers import chunk_code_file, preprocess_session, chunk_session
from vecs.config import (
    BLOOMLY_CODE_DIR,
    BLOOMLY_SESSIONS_DIR,
    CHROMADB_DIR,
    CODE_CHUNK_LINES,
    CODE_CHUNK_OVERLAP,
    CODE_COLLECTION,
    CODE_EXTENSIONS,
    CODE_MODEL,
    MANIFEST_PATH,
    SESSION_CHUNK_MESSAGES,
    SESSIONS_COLLECTION,
    SESSIONS_MODEL,
    VECS_DIR,
    VOYAGE_BATCH_SIZE,
)


class Manifest:
    """Tracks which files have been indexed and their content hashes."""

    def __init__(self, path: Path = MANIFEST_PATH):
        self.path = path
        self.data: dict[str, str] = {}
        if path.exists():
            self.data = json.loads(path.read_text())

    def _file_hash(self, file_path: Path) -> str:
        content = file_path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    def needs_indexing(self, file_path: Path) -> bool:
        key = str(file_path)
        if key not in self.data:
            return True
        return self.data[key] != self._file_hash(file_path)

    def mark_indexed(self, file_path: Path) -> None:
        self.data[str(file_path)] = self._file_hash(file_path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _embed_and_store(
    chunks: list[dict],
    collection: chromadb.Collection,
    model: str,
    vo: voyageai.Client,
    id_prefix: str,
) -> int:
    """Embed chunks in batches and store in ChromaDB. Returns count stored."""
    if not chunks:
        return 0

    stored = 0
    for i in range(0, len(chunks), VOYAGE_BATCH_SIZE):
        batch = chunks[i : i + VOYAGE_BATCH_SIZE]
        texts = [c["text"] for c in batch]
        result = vo.embed(texts, model=model, input_type="document")

        ids = [f"{id_prefix}-{i + j}" for j in range(len(batch))]
        metadatas = [c["metadata"] for c in batch]

        collection.upsert(
            ids=ids,
            embeddings=result.embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        stored += len(batch)
        _log(f"  Indexed {stored}/{len(chunks)} chunks")

    return stored


def index_code(vo: voyageai.Client, db: chromadb.ClientAPI) -> int:
    """Index Bloomly .cs files from Assets/. Returns count of new chunks."""
    manifest = Manifest()
    collection = db.get_or_create_collection(CODE_COLLECTION)

    files = [
        f
        for f in BLOOMLY_CODE_DIR.rglob("*")
        if f.suffix in CODE_EXTENSIONS and f.is_file()
    ]

    to_index = [f for f in files if manifest.needs_indexing(f)]
    if not to_index:
        _log("Code: nothing new to index.")
        return 0

    _log(f"Code: {len(to_index)} files to index ({len(files)} total)")

    all_chunks = []
    for f in to_index:
        content = f.read_text(errors="replace")
        rel_path = str(f.relative_to(BLOOMLY_CODE_DIR))
        chunks = chunk_code_file(
            content, rel_path, CODE_CHUNK_LINES, CODE_CHUNK_OVERLAP
        )
        all_chunks.extend(chunks)

    stored = _embed_and_store(all_chunks, collection, CODE_MODEL, vo, "code")

    for f in to_index:
        manifest.mark_indexed(f)
    manifest.save()

    return stored


def index_sessions(vo: voyageai.Client, db: chromadb.ClientAPI) -> int:
    """Index Claude Code session transcripts. Returns count of new chunks."""
    manifest = Manifest()
    collection = db.get_or_create_collection(SESSIONS_COLLECTION)

    files = sorted(BLOOMLY_SESSIONS_DIR.glob("*.jsonl"))
    to_index = [f for f in files if manifest.needs_indexing(f)]

    if not to_index:
        _log("Sessions: nothing new to index.")
        return 0

    _log(f"Sessions: {len(to_index)} files to index ({len(files)} total)")

    all_chunks = []
    for f in to_index:
        raw = f.read_text(errors="replace")
        session_id = f.stem
        messages = preprocess_session(raw)
        chunks = chunk_session(messages, session_id, SESSION_CHUNK_MESSAGES)
        all_chunks.extend(chunks)

    stored = _embed_and_store(
        all_chunks, collection, SESSIONS_MODEL, vo, "session"
    )

    for f in to_index:
        manifest.mark_indexed(f)
    manifest.save()

    return stored


def run_index() -> None:
    """Run full incremental index."""
    VECS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMADB_DIR.mkdir(parents=True, exist_ok=True)

    vo = voyageai.Client()
    db = chromadb.PersistentClient(path=str(CHROMADB_DIR))

    _log("Starting index...")
    code_count = index_code(vo, db)
    session_count = index_sessions(vo, db)
    _log(f"Done. Indexed {code_count} code chunks, {session_count} session chunks.")
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd ~/Repositories/vecs
uv run pytest tests/test_indexer.py -v
```
Expected: All 3 manifest tests PASS

**Step 5: Commit**

```bash
cd ~/Repositories/vecs
git add src/vecs/indexer.py tests/test_indexer.py
git commit -m "feat: indexer with manifest tracking and Voyage+ChromaDB integration"
```

---

### Task 5: Searcher

**Files:**
- Create: `~/Repositories/vecs/src/vecs/searcher.py`
- Create: `~/Repositories/vecs/tests/test_searcher.py`

**Step 1: Write failing test**

`tests/test_searcher.py`:
```python
from vecs.searcher import format_results


def test_format_results_empty():
    results = format_results({"ids": [[]], "documents": [[]], "metadatas": [[]]})
    assert results == []


def test_format_results_with_data():
    results = format_results(
        {
            "ids": [["id1", "id2"]],
            "documents": [["doc one", "doc two"]],
            "metadatas": [[{"file_path": "a.cs"}, {"session_id": "abc"}]],
            "distances": [[0.1, 0.5]],
        }
    )
    assert len(results) == 2
    assert results[0]["text"] == "doc one"
    assert results[0]["metadata"]["file_path"] == "a.cs"
    assert results[0]["distance"] == 0.1
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd ~/Repositories/vecs
uv run pytest tests/test_searcher.py -v
```
Expected: FAIL with `ImportError`

**Step 3: Implement searcher**

`src/vecs/searcher.py`:
```python
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
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd ~/Repositories/vecs
uv run pytest tests/test_searcher.py -v
```
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
cd ~/Repositories/vecs
git add src/vecs/searcher.py tests/test_searcher.py
git commit -m "feat: searcher with multi-collection query support"
```

---

### Task 6: CLI

**Files:**
- Create: `~/Repositories/vecs/src/vecs/cli.py`

**Step 1: Implement CLI**

`src/vecs/cli.py`:
```python
import click

from vecs.indexer import run_index
from vecs.searcher import search


@click.group()
def main():
    """vecs — Semantic search for Bloomly."""
    pass


@main.command()
def index():
    """Index code and session transcripts (incremental)."""
    run_index()


@main.command()
@click.argument("query")
@click.option(
    "--collection", "-c",
    type=click.Choice(["code", "sessions"], case_sensitive=False),
    default=None,
    help="Search a specific collection (default: both).",
)
@click.option("--limit", "-n", default=5, help="Number of results.")
def search_cmd(query: str, collection: str | None, limit: int):
    """Search code and sessions semantically."""
    results = search(query, collection_name=collection, n_results=limit)
    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        source = meta.get("file_path") or f"session:{meta.get('session_id', '?')}"
        dist = r.get("distance")
        dist_str = f" (distance: {dist:.4f})" if dist is not None else ""
        click.echo(f"\n--- Result {i} [{r.get('collection', '?')}] {source}{dist_str} ---")
        # Truncate long results for display
        text = r["text"]
        if len(text) > 1000:
            text = text[:1000] + "\n... [truncated]"
        click.echo(text)

# Alias so `vecs search` works
main.add_command(search_cmd, "search")
```

**Step 2: Verify CLI wires up**

Run:
```bash
cd ~/Repositories/vecs
uv run vecs --help
```
Expected: Shows `index` and `search` commands.

**Step 3: Commit**

```bash
cd ~/Repositories/vecs
git add src/vecs/cli.py
git commit -m "feat: CLI with index and search commands"
```

---

### Task 7: MCP Server

**Files:**
- Create: `~/Repositories/vecs/src/vecs/mcp_server.py`

**Step 1: Implement MCP server**

`src/vecs/mcp_server.py`:
```python
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
```

**Step 2: Test MCP server starts**

Run:
```bash
cd ~/Repositories/vecs
uv run mcp dev src/vecs/mcp_server.py
```
Expected: MCP Inspector launches, shows `semantic_search` tool.

Kill the inspector after verifying (Ctrl+C).

**Step 3: Commit**

```bash
cd ~/Repositories/vecs
git add src/vecs/mcp_server.py
git commit -m "feat: MCP server exposing semantic_search tool"
```

---

### Task 8: Global Install & MCP Config

**Step 1: Install globally with uv**

Run:
```bash
cd ~/Repositories/vecs
uv tool install --editable .
```

Verify:
```bash
vecs --help
```
Expected: Shows help from anywhere.

**Step 2: Set VOYAGE_API_KEY**

Ask the user for their Voyage API key, then:

Run:
```bash
# Add to shell profile
echo 'export VOYAGE_API_KEY="<their-key>"' >> ~/.zshrc
source ~/.zshrc
```

**Step 3: Configure MCP in Claude Code global settings**

Read `~/.claude/settings.json` first, then add the MCP server config:

Add to `mcpServers` in `~/.claude/settings.json`:
```json
{
  "mcpServers": {
    "vecs": {
      "command": "uv",
      "args": ["run", "--directory", "/Users/darynavoloshyna/Repositories/vecs", "mcp", "run", "src/vecs/mcp_server.py"]
    }
  }
}
```

**Step 4: Test end-to-end**

Run:
```bash
vecs index
```
Expected: Indexes code and sessions, shows progress.

Then:
```bash
vecs search "animation state machine"
```
Expected: Returns relevant code and/or session chunks.

**Step 5: Commit**

```bash
cd ~/Repositories/vecs
git add -A
git commit -m "docs: update plans with implementation details"
```

---

## Task Summary

| Task | Description | Depends On |
|------|------------|------------|
| 1 | Project scaffolding + config | — |
| 2 | Code chunker | 1 |
| 3 | Session preprocessor | 1 |
| 4 | Indexer (Voyage + ChromaDB) | 2, 3 |
| 5 | Searcher | 4 |
| 6 | CLI | 4, 5 |
| 7 | MCP server | 5 |
| 8 | Global install + MCP config | 6, 7 |

Tasks 2 and 3 can run in parallel. Tasks 6 and 7 can run in parallel.
