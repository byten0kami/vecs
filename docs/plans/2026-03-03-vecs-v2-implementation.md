# vecs v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address all 11 feedback items — AST-aware chunking, multi-project config, hybrid search, MCP index tools, path filtering, dedup, multi-file-types, session overlap, singleton client, query caching.

**Architecture:** Replace hardcoded Bloomly config with YAML-driven multi-project system. Add tree-sitter AST chunking for C#/TypeScript with line-based fallback. Layer BM25 keyword search alongside vector search with Reciprocal Rank Fusion. Expose reindex/status via MCP.

**Tech Stack:** tree-sitter + tree-sitter-c-sharp + tree-sitter-typescript, rank-bm25, cachetools, pyyaml, chromadb, voyageai, click, mcp[cli]

---

### Task 1: Add new dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add dependencies to pyproject.toml**

Add to `dependencies` list:
```
"tree-sitter>=0.24",
"tree-sitter-c-sharp>=0.23",
"tree-sitter-typescript>=0.23",
"rank-bm25>=0.2",
"cachetools>=5.0",
"pyyaml>=6.0",
```

**Step 2: Install dependencies**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv sync`
Expected: All packages install successfully

**Step 3: Verify imports work**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run python -c "import tree_sitter; import tree_sitter_c_sharp; import tree_sitter_typescript; import rank_bm25; import cachetools; import yaml; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add tree-sitter, rank-bm25, cachetools, pyyaml dependencies"
```

---

### Task 2: YAML-based multi-project config

Replaces hardcoded Bloomly paths with a YAML config file at `~/.vecs/config.yaml`. Each project defines its code directory, file extensions, and optional sessions directory.

**Files:**
- Modify: `src/vecs/config.py`
- Create: `tests/test_config.py`

**Step 1: Write failing tests for config loading**

Create `tests/test_config.py`:
```python
import yaml
from pathlib import Path
from vecs.config import load_config, ProjectConfig, DEFAULT_CONFIG_PATH


def test_load_config_from_yaml(tmp_path):
    """Config loads projects from YAML file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "myproject": {
                "code_dir": "/tmp/code",
                "extensions": [".cs", ".ts"],
            }
        }
    }))
    config = load_config(config_file)
    assert "myproject" in config.projects
    p = config.projects["myproject"]
    assert p.code_dir == Path("/tmp/code")
    assert p.extensions == {".cs", ".ts"}
    assert p.sessions_dir is None


def test_load_config_with_sessions(tmp_path):
    """Config supports optional sessions_dir."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "proj": {
                "code_dir": "/tmp/code",
                "extensions": [".cs"],
                "sessions_dir": "/tmp/sessions",
            }
        }
    }))
    config = load_config(config_file)
    assert config.projects["proj"].sessions_dir == Path("/tmp/sessions")


def test_load_config_missing_file(tmp_path):
    """Missing config file returns empty config."""
    config = load_config(tmp_path / "nonexistent.yaml")
    assert config.projects == {}


def test_project_config_collection_names():
    """Project collection names are prefixed with project name."""
    p = ProjectConfig(
        name="bloomly",
        code_dir=Path("/tmp"),
        extensions={".cs"},
    )
    assert p.code_collection == "bloomly:code"
    assert p.sessions_collection == "bloomly:sessions"


def test_save_config(tmp_path):
    """Config can be saved and re-loaded."""
    config_file = tmp_path / "config.yaml"
    config = load_config(config_file)
    config.add_project("test", code_dir=Path("/tmp/code"), extensions={".cs"})
    config.save()
    reloaded = load_config(config_file)
    assert "test" in reloaded.projects


def test_remove_project(tmp_path):
    """Projects can be removed."""
    config_file = tmp_path / "config.yaml"
    config = load_config(config_file)
    config.add_project("test", code_dir=Path("/tmp/code"), extensions={".cs"})
    config.save()
    config.remove_project("test")
    config.save()
    reloaded = load_config(config_file)
    assert "test" not in reloaded.projects
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest tests/test_config.py -v`
Expected: FAIL — `load_config`, `ProjectConfig` not defined

**Step 3: Rewrite config.py**

Replace `src/vecs/config.py` with:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

VECS_DIR = Path.home() / ".vecs"
CHROMADB_DIR = VECS_DIR / "chromadb"
MANIFEST_PATH = VECS_DIR / "manifest.json"
DEFAULT_CONFIG_PATH = VECS_DIR / "config.yaml"

# Embedding models
CODE_MODEL = "voyage-code-3"
SESSIONS_MODEL = "voyage-3"

# Chunking defaults
CODE_CHUNK_LINES = 200
CODE_CHUNK_OVERLAP = 50
SESSION_CHUNK_MESSAGES = 10
SESSION_CHUNK_OVERLAP = 2

# API
VOYAGE_BATCH_SIZE = 128


@dataclass
class ProjectConfig:
    """Configuration for a single project."""

    name: str
    code_dir: Path
    extensions: set[str] = field(default_factory=lambda: {".cs"})
    sessions_dir: Path | None = None

    @property
    def code_collection(self) -> str:
        return f"{self.name}:code"

    @property
    def sessions_collection(self) -> str:
        return f"{self.name}:sessions"


@dataclass
class VecsConfig:
    """Top-level config holding all projects."""

    path: Path
    projects: dict[str, ProjectConfig] = field(default_factory=dict)

    def add_project(
        self,
        name: str,
        code_dir: Path,
        extensions: set[str] | None = None,
        sessions_dir: Path | None = None,
    ) -> None:
        self.projects[name] = ProjectConfig(
            name=name,
            code_dir=code_dir,
            extensions=extensions or {".cs"},
            sessions_dir=sessions_dir,
        )

    def remove_project(self, name: str) -> None:
        self.projects.pop(name, None)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {"projects": {}}
        for name, p in self.projects.items():
            proj: dict = {
                "code_dir": str(p.code_dir),
                "extensions": sorted(p.extensions),
            }
            if p.sessions_dir:
                proj["sessions_dir"] = str(p.sessions_dir)
            data["projects"][name] = proj
        self.path.write_text(yaml.dump(data, default_flow_style=False))


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> VecsConfig:
    """Load config from YAML. Returns empty config if file missing."""
    config = VecsConfig(path=path)
    if not path.exists():
        return config
    raw = yaml.safe_load(path.read_text()) or {}
    for name, proj in raw.get("projects", {}).items():
        config.projects[name] = ProjectConfig(
            name=name,
            code_dir=Path(proj["code_dir"]),
            extensions=set(proj.get("extensions", [".cs"])),
            sessions_dir=Path(proj["sessions_dir"]) if proj.get("sessions_dir") else None,
        )
    return config
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest tests/test_config.py -v`
Expected: All 6 tests PASS

**Step 5: Run all existing tests to check for regressions**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest -v`
Expected: Some existing tests may fail if they import old constants from config — note which ones. The old constants `BLOOMLY_CODE_DIR`, `BLOOMLY_SESSIONS_DIR`, `CODE_COLLECTION`, `SESSIONS_COLLECTION`, `CODE_EXTENSIONS` are removed but still imported in `indexer.py`. This is expected — we'll fix those in a later task.

**Step 6: Commit**

```bash
git add src/vecs/config.py tests/test_config.py
git commit -m "feat: YAML-based multi-project config system"
```

---

### Task 3: CLI project management commands

Add `vecs project add/remove/list` commands.

**Files:**
- Modify: `src/vecs/cli.py`

**Step 1: Add project subgroup to cli.py**

Add a `project` command group with three subcommands:

```python
@main.group()
def project():
    """Manage indexed projects."""
    pass


@project.command("add")
@click.argument("name")
@click.option("--code-dir", required=True, type=click.Path(exists=True), help="Root directory of source code.")
@click.option("--ext", required=True, help="Comma-separated file extensions (e.g. .cs,.ts)")
@click.option("--sessions-dir", default=None, type=click.Path(exists=True), help="Claude Code sessions directory.")
def project_add(name: str, code_dir: str, ext: str, sessions_dir: str | None):
    """Register a project for indexing."""
    from vecs.config import load_config
    config = load_config()
    extensions = {e.strip() if e.strip().startswith(".") else f".{e.strip()}" for e in ext.split(",")}
    config.add_project(
        name=name,
        code_dir=Path(code_dir).resolve(),
        extensions=extensions,
        sessions_dir=Path(sessions_dir).resolve() if sessions_dir else None,
    )
    config.save()
    click.echo(f"Added project '{name}'")


@project.command("remove")
@click.argument("name")
def project_remove(name: str):
    """Unregister a project."""
    from vecs.config import load_config
    config = load_config()
    if name not in config.projects:
        click.echo(f"Project '{name}' not found.")
        return
    config.remove_project(name)
    config.save()
    click.echo(f"Removed project '{name}'")


@project.command("list")
def project_list():
    """List registered projects."""
    from vecs.config import load_config
    config = load_config()
    if not config.projects:
        click.echo("No projects configured. Use 'vecs project add' to register one.")
        return
    for name, p in config.projects.items():
        exts = ", ".join(sorted(p.extensions))
        click.echo(f"  {name}: {p.code_dir} [{exts}]")
        if p.sessions_dir:
            click.echo(f"    sessions: {p.sessions_dir}")
```

Add `from pathlib import Path` to cli.py imports.

**Step 2: Test manually**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run vecs project list`
Expected: "No projects configured..."

**Step 3: Commit**

```bash
git add src/vecs/cli.py
git commit -m "feat: CLI project management commands (add/remove/list)"
```

---

### Task 4: AST-aware chunking with tree-sitter

New module that parses C# and TypeScript files using tree-sitter, extracting class/method/function boundaries for smarter chunking. Falls back to line-based for unsupported languages.

**Files:**
- Create: `src/vecs/ast_chunker.py`
- Create: `tests/test_ast_chunker.py`

**Step 1: Write failing tests**

Create `tests/test_ast_chunker.py`:
```python
from vecs.ast_chunker import chunk_code_file_ast


CS_CODE = """using System;

namespace MyApp
{
    public class Player
    {
        public int Health { get; set; }

        public void TakeDamage(int amount)
        {
            Health -= amount;
            if (Health < 0) Health = 0;
        }

        public void Heal(int amount)
        {
            Health += amount;
        }
    }

    public class Enemy
    {
        public void Attack(Player target)
        {
            target.TakeDamage(10);
        }
    }
}
"""

TS_CODE = """
export function greet(name: string): string {
    return `Hello, ${name}!`;
}

export class UserService {
    private users: Map<string, User> = new Map();

    addUser(user: User): void {
        this.users.set(user.id, user);
    }

    getUser(id: string): User | undefined {
        return this.users.get(id);
    }
}

interface User {
    id: string;
    name: string;
}
"""


def test_cs_chunks_at_class_boundaries():
    """C# file is chunked at class boundaries."""
    chunks = chunk_code_file_ast(CS_CODE, "Player.cs")
    # Should have chunks for Player and Enemy classes
    assert len(chunks) >= 2
    texts = [c["text"] for c in chunks]
    assert any("Player" in t and "TakeDamage" in t for t in texts)
    assert any("Enemy" in t and "Attack" in t for t in texts)


def test_ts_chunks_at_boundaries():
    """TypeScript file is chunked at function/class boundaries."""
    chunks = chunk_code_file_ast(TS_CODE, "user.ts")
    assert len(chunks) >= 2
    texts = [c["text"] for c in chunks]
    assert any("greet" in t for t in texts)
    assert any("UserService" in t for t in texts)


def test_metadata_has_file_path():
    """Chunks carry file_path and line numbers in metadata."""
    chunks = chunk_code_file_ast(CS_CODE, "Player.cs")
    for c in chunks:
        assert c["metadata"]["file_path"] == "Player.cs"
        assert "start_line" in c["metadata"]
        assert "end_line" in c["metadata"]
        assert "chunk_index" in c["metadata"]


def test_unknown_extension_falls_back():
    """Unsupported file extensions fall back to line-based chunking."""
    content = "\n".join(f"line {i}" for i in range(300))
    chunks = chunk_code_file_ast(content, "data.shader", chunk_lines=200, overlap=50)
    assert len(chunks) >= 2  # 300 lines / 200 = at least 2 chunks


def test_empty_file_returns_empty():
    """Empty file returns no chunks."""
    chunks = chunk_code_file_ast("", "Empty.cs")
    assert chunks == []


def test_large_class_is_split():
    """A class exceeding max_chunk_lines is split into sub-chunks."""
    # Generate a class with many methods (>500 lines)
    methods = []
    for i in range(60):
        methods.append(f"""
        public void Method{i}()
        {{
            var x = {i};
            var y = x + 1;
            var z = y * 2;
            Console.WriteLine(z);
            // padding line
            // more padding
        }}""")
    content = f"public class BigClass\n{{\n{''.join(methods)}\n}}"
    chunks = chunk_code_file_ast(content, "Big.cs", max_chunk_lines=500)
    assert len(chunks) >= 2
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest tests/test_ast_chunker.py -v`
Expected: FAIL — `ast_chunker` module not found

**Step 3: Implement ast_chunker.py**

Create `src/vecs/ast_chunker.py`:
```python
from __future__ import annotations

import tree_sitter_c_sharp as ts_cs
import tree_sitter_typescript as ts_ts
from tree_sitter import Language, Parser

from vecs.chunkers import chunk_code_file

CS_LANGUAGE = Language(ts_cs.language())
TS_LANGUAGE = Language(ts_ts.language_typescript())
TSX_LANGUAGE = Language(ts_ts.language_tsx())

# Map file extensions to tree-sitter languages
LANGUAGE_MAP: dict[str, Language] = {
    ".cs": CS_LANGUAGE,
    ".ts": TS_LANGUAGE,
    ".tsx": TSX_LANGUAGE,
}

# Node types that represent top-level declarations worth chunking
CHUNK_NODE_TYPES: dict[str, set[str]] = {
    ".cs": {
        "class_declaration",
        "struct_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
        "namespace_declaration",
    },
    ".ts": {
        "class_declaration",
        "function_declaration",
        "interface_declaration",
        "enum_declaration",
        "type_alias_declaration",
        "export_statement",
    },
    ".tsx": {
        "class_declaration",
        "function_declaration",
        "interface_declaration",
        "enum_declaration",
        "type_alias_declaration",
        "export_statement",
    },
}

# Minimum lines for a chunk to stand alone (otherwise merge with adjacent)
MIN_CHUNK_LINES = 5


def _get_extension(file_path: str) -> str:
    """Extract file extension from path."""
    dot = file_path.rfind(".")
    return file_path[dot:] if dot >= 0 else ""


def _extract_declarations(root, node_types: set[str]) -> list[tuple[int, int]]:
    """Walk the AST and return (start_line, end_line) for top-level declarations."""
    declarations = []

    def walk(node, depth=0):
        if node.type in node_types and depth <= 2:
            declarations.append((node.start_point[0], node.end_point[0]))
            return  # Don't recurse into matched nodes
        for child in node.children:
            walk(child, depth + 1)

    walk(root)
    return declarations


def chunk_code_file_ast(
    content: str,
    file_path: str,
    max_chunk_lines: int = 500,
    chunk_lines: int = 200,
    overlap: int = 50,
) -> list[dict]:
    """Chunk a code file using AST boundaries when possible.

    Falls back to line-based chunking for unsupported languages or
    when AST parsing fails.

    Args:
        content: File content as string.
        file_path: Relative file path (used to detect language and for metadata).
        max_chunk_lines: Maximum lines per chunk. Declarations exceeding this are sub-split.
        chunk_lines: Line-based chunk size for fallback.
        overlap: Line overlap for fallback chunking.
    """
    if not content.strip():
        return []

    ext = _get_extension(file_path)
    language = LANGUAGE_MAP.get(ext)

    if language is None:
        return chunk_code_file(content, file_path, chunk_lines, overlap)

    parser = Parser(language)
    tree = parser.parse(content.encode())

    node_types = CHUNK_NODE_TYPES.get(ext, set())
    declarations = _extract_declarations(tree.root_node, node_types)

    if not declarations:
        return chunk_code_file(content, file_path, chunk_lines, overlap)

    lines = content.split("\n")
    total_lines = len(lines)

    # Sort declarations by start line
    declarations.sort()

    # Build chunks from declarations, merging small ones
    raw_chunks: list[tuple[int, int]] = []
    for start, end in declarations:
        if raw_chunks and (start - raw_chunks[-1][1] <= 1) and (end - raw_chunks[-1][0] + 1 < MIN_CHUNK_LINES * 3):
            # Merge with previous if they're adjacent and combined is small
            raw_chunks[-1] = (raw_chunks[-1][0], end)
        else:
            raw_chunks.append((start, end))

    # Include any preamble (imports, using statements) before first declaration
    if raw_chunks and raw_chunks[0][0] > 0:
        preamble_end = raw_chunks[0][0] - 1
        # Attach preamble to first chunk
        raw_chunks[0] = (0, raw_chunks[0][1])

    # Include any trailing code after last declaration
    if raw_chunks and raw_chunks[-1][1] < total_lines - 1:
        raw_chunks[-1] = (raw_chunks[-1][0], total_lines - 1)

    # Fill gaps between declarations
    filled: list[tuple[int, int]] = []
    for i, (start, end) in enumerate(raw_chunks):
        if i > 0:
            prev_end = filled[-1][1]
            if start > prev_end + 1:
                # Gap — attach to previous chunk or next chunk
                filled[-1] = (filled[-1][0], start - 1)
        filled.append((start, end))

    # Build final chunks, splitting any that exceed max_chunk_lines
    chunks = []
    for start, end in filled:
        chunk_text = "\n".join(lines[start : end + 1])
        line_count = end - start + 1

        if line_count > max_chunk_lines:
            # Sub-split using line-based chunking
            sub_chunks = chunk_code_file(chunk_text, file_path, chunk_lines, overlap)
            for sc in sub_chunks:
                sc["metadata"]["start_line"] = start + sc["metadata"]["start_line"]
                sc["metadata"]["end_line"] = start + sc["metadata"]["end_line"]
                sc["metadata"]["chunk_index"] = len(chunks)
                chunks.append(sc)
        else:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "file_path": file_path,
                    "chunk_index": len(chunks),
                    "start_line": start + 1,
                    "end_line": end + 1,
                },
            })

    return chunks
```

**Step 4: Run tests**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest tests/test_ast_chunker.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/vecs/ast_chunker.py tests/test_ast_chunker.py
git commit -m "feat: AST-aware chunking for C# and TypeScript via tree-sitter"
```

---

### Task 5: Session chunk overlap

Add overlap parameter to `chunk_session()` so context at chunk boundaries is preserved.

**Files:**
- Modify: `src/vecs/chunkers.py`
- Modify: `tests/test_session_chunker.py`

**Step 1: Write failing test**

Add to `tests/test_session_chunker.py`:
```python
def test_chunk_session_with_overlap():
    """Session chunks overlap by N messages."""
    lines = [_make_message("user", f"msg {i}") for i in range(15)]
    messages = preprocess_session("\n".join(lines))
    chunks = chunk_session(messages, session_id="test", chunk_size=10, overlap=2)
    assert len(chunks) == 2
    # Second chunk should start 2 messages before where first chunk ended
    # First chunk: msgs 0-9, second chunk: msgs 8-14
    assert "msg 8" in chunks[1]["text"]
    assert "msg 9" in chunks[1]["text"]
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest tests/test_session_chunker.py::test_chunk_session_with_overlap -v`
Expected: FAIL — `chunk_session() got an unexpected keyword argument 'overlap'`

**Step 3: Add overlap parameter to chunk_session**

Modify `chunk_session` in `src/vecs/chunkers.py` to accept `overlap` parameter and use sliding window:

```python
def chunk_session(
    messages: list[dict],
    session_id: str,
    chunk_size: int = 10,
    overlap: int = 0,
) -> list[dict]:
    """Group preprocessed messages into overlapping chunks."""
    chunks = []
    step = max(1, chunk_size - overlap)
    start = 0

    while start < len(messages):
        group = messages[start : start + chunk_size]
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
        if start + chunk_size >= len(messages):
            break
        start += step

    return chunks
```

**Step 4: Run all session chunker tests**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest tests/test_session_chunker.py -v`
Expected: All tests PASS (including existing `test_chunk_session_groups_messages`)

**Step 5: Commit**

```bash
git add src/vecs/chunkers.py tests/test_session_chunker.py
git commit -m "feat: add overlap parameter to session chunking"
```

---

### Task 6: Singleton Voyage client

Centralize client creation so `searcher.py` doesn't create a new client per call.

**Files:**
- Create: `src/vecs/clients.py`
- Modify: `src/vecs/indexer.py` (import change only — full refactor in Task 7)
- Modify: `src/vecs/searcher.py` (import change only — full refactor in Task 8)

**Step 1: Create clients.py**

Create `src/vecs/clients.py`:
```python
from __future__ import annotations

import chromadb
import voyageai

from vecs.config import CHROMADB_DIR

_vo_client: voyageai.Client | None = None
_db_client: chromadb.ClientAPI | None = None


def get_voyage_client() -> voyageai.Client:
    """Return a singleton Voyage client."""
    global _vo_client
    if _vo_client is None:
        _vo_client = voyageai.Client()
    return _vo_client


def get_chromadb_client() -> chromadb.ClientAPI:
    """Return a singleton ChromaDB persistent client."""
    global _db_client
    if _db_client is None:
        CHROMADB_DIR.mkdir(parents=True, exist_ok=True)
        _db_client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
    return _db_client
```

**Step 2: Commit**

```bash
git add src/vecs/clients.py
git commit -m "feat: singleton Voyage and ChromaDB clients"
```

---

### Task 7: Refactor indexer for multi-project support

The indexer currently has hardcoded Bloomly paths. Refactor to accept a `ProjectConfig` and iterate over all configured projects.

**Files:**
- Modify: `src/vecs/indexer.py`
- Modify: `tests/test_indexer.py`

**Step 1: Update test_indexer.py**

The existing manifest tests don't import from config, so they should keep passing. Just verify:

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest tests/test_indexer.py -v`

**Step 2: Rewrite indexer.py**

Replace `src/vecs/indexer.py`. Key changes:
- Remove all `BLOOMLY_*` imports
- `index_code` takes `ProjectConfig` instead of using hardcoded paths
- `index_sessions` takes `ProjectConfig`
- `run_index` loads config and iterates over all projects
- Use `ast_chunker.chunk_code_file_ast` instead of `chunk_code_file`
- Use `clients.py` singletons
- Pass `SESSION_CHUNK_OVERLAP` to `chunk_session`

```python
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import chromadb
import voyageai

from vecs.ast_chunker import chunk_code_file_ast
from vecs.chunkers import preprocess_session, chunk_session
from vecs.clients import get_voyage_client, get_chromadb_client
from vecs.config import (
    CHROMADB_DIR,
    CODE_CHUNK_LINES,
    CODE_CHUNK_OVERLAP,
    CODE_MODEL,
    MANIFEST_PATH,
    SESSION_CHUNK_MESSAGES,
    SESSION_CHUNK_OVERLAP,
    SESSIONS_MODEL,
    VECS_DIR,
    VOYAGE_BATCH_SIZE,
    ProjectConfig,
    load_config,
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


def _make_chunk_id(source_key: str, chunk_index: int) -> str:
    return f"{source_key}:{chunk_index}"


def _delete_stale_chunks(
    collection: chromadb.Collection,
    metadata_key: str,
    metadata_value: str,
) -> None:
    try:
        existing = collection.get(where={metadata_key: metadata_value})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass


def _embed_and_store(
    chunks: list[dict],
    collection: chromadb.Collection,
    model: str,
    vo: voyageai.Client,
) -> int:
    if not chunks:
        return 0

    stored = 0
    for i in range(0, len(chunks), VOYAGE_BATCH_SIZE):
        batch = chunks[i : i + VOYAGE_BATCH_SIZE]
        texts = [c["text"] for c in batch]

        for attempt in range(5):
            try:
                result = vo.embed(texts, model=model, input_type="document")
                break
            except Exception as e:
                if "RateLimitError" in type(e).__name__ or "rate" in str(e).lower():
                    wait = 20 * (attempt + 1)
                    _log(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        else:
            _log(f"  Failed after 5 retries, skipping batch at {i}")
            continue

        ids = [c["id"] for c in batch]
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


def index_code(project: ProjectConfig, vo: voyageai.Client, db: chromadb.ClientAPI) -> int:
    """Index code files for a project. Returns count of new chunks."""
    manifest = Manifest()
    collection = db.get_or_create_collection(project.code_collection)

    if not project.code_dir.exists():
        _log(f"Code dir not found: {project.code_dir}")
        return 0

    files = [
        f
        for f in project.code_dir.rglob("*")
        if f.suffix in project.extensions and f.is_file()
    ]

    to_index = [f for f in files if manifest.needs_indexing(f)]
    if not to_index:
        _log(f"[{project.name}] Code: nothing new to index.")
        return 0

    _log(f"[{project.name}] Code: {len(to_index)} files to index ({len(files)} total)")

    all_chunks = []
    for f in to_index:
        content = f.read_text(errors="replace")
        rel_path = str(f.relative_to(project.code_dir))
        _delete_stale_chunks(collection, "file_path", rel_path)
        chunks = chunk_code_file_ast(
            content, rel_path, chunk_lines=CODE_CHUNK_LINES, overlap=CODE_CHUNK_OVERLAP
        )
        for c in chunks:
            c["id"] = _make_chunk_id(f"code:{rel_path}", c["metadata"]["chunk_index"])
        all_chunks.extend(chunks)

    stored = _embed_and_store(all_chunks, collection, CODE_MODEL, vo)

    for f in to_index:
        manifest.mark_indexed(f)
    manifest.save()

    return stored


def index_sessions(project: ProjectConfig, vo: voyageai.Client, db: chromadb.ClientAPI) -> int:
    """Index Claude Code session transcripts for a project. Returns count of new chunks."""
    if not project.sessions_dir or not project.sessions_dir.exists():
        return 0

    manifest = Manifest()
    collection = db.get_or_create_collection(project.sessions_collection)

    files = sorted(project.sessions_dir.glob("*.jsonl"))
    to_index = [f for f in files if manifest.needs_indexing(f)]

    if not to_index:
        _log(f"[{project.name}] Sessions: nothing new to index.")
        return 0

    _log(f"[{project.name}] Sessions: {len(to_index)} files to index ({len(files)} total)")

    all_chunks = []
    for f in to_index:
        raw = f.read_text(errors="replace")
        session_id = f.stem
        _delete_stale_chunks(collection, "session_id", session_id)
        messages = preprocess_session(raw)
        chunks = chunk_session(
            messages, session_id, SESSION_CHUNK_MESSAGES, overlap=SESSION_CHUNK_OVERLAP
        )
        for c in chunks:
            c["id"] = _make_chunk_id(f"session:{session_id}", c["metadata"]["chunk_index"])
        all_chunks.extend(chunks)

    stored = _embed_and_store(all_chunks, collection, SESSIONS_MODEL, vo)

    for f in to_index:
        manifest.mark_indexed(f)
    manifest.save()

    return stored


def get_status(project_name: str | None = None) -> dict:
    """Get index status info, optionally filtered to one project."""
    config = load_config()
    db = get_chromadb_client()
    projects = (
        {project_name: config.projects[project_name]}
        if project_name and project_name in config.projects
        else config.projects
    )

    status: dict = {"projects": {}, "total_code_chunks": 0, "total_session_chunks": 0}

    for name, p in projects.items():
        code_count = 0
        session_count = 0
        try:
            col = db.get_collection(p.code_collection)
            code_count = col.count()
        except Exception:
            pass
        try:
            col = db.get_collection(p.sessions_collection)
            session_count = col.count()
        except Exception:
            pass
        status["projects"][name] = {
            "code_chunks": code_count,
            "session_chunks": session_count,
        }
        status["total_code_chunks"] += code_count
        status["total_session_chunks"] += session_count

    if MANIFEST_PATH.exists():
        data = json.loads(MANIFEST_PATH.read_text())
        status["manifest_entries"] = len(data)
    else:
        status["manifest_entries"] = 0

    return status


def run_index(project_name: str | None = None) -> None:
    """Run incremental index for one or all projects."""
    VECS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMADB_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    if not config.projects:
        _log("No projects configured. Use 'vecs project add' to register one.")
        return

    vo = get_voyage_client()
    db = get_chromadb_client()

    projects = (
        {project_name: config.projects[project_name]}
        if project_name and project_name in config.projects
        else config.projects
    )

    _log("Starting index...")
    total_code = 0
    total_sessions = 0
    for name, project in projects.items():
        _log(f"\nProject: {name}")
        total_code += index_code(project, vo, db)
        total_sessions += index_sessions(project, vo, db)

    _log(f"\nDone. Indexed {total_code} code chunks, {total_sessions} session chunks.")
```

**Step 3: Run manifest tests**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest tests/test_indexer.py -v`
Expected: All 3 manifest tests PASS (they don't depend on config imports)

**Step 4: Update cli.py to use new indexer API**

Update the `index` command in `cli.py` to accept optional `--project` flag:

```python
@main.command()
@click.option("--project", "-p", default=None, help="Index a specific project (default: all).")
def index(project: str | None):
    """Index code and session transcripts (incremental)."""
    run_index(project_name=project)
```

Update the `status` command:
```python
@main.command()
@click.option("--project", "-p", default=None, help="Status for a specific project.")
def status(project: str | None):
    """Show index status."""
    s = get_status(project_name=project)
    for name, info in s.get("projects", {}).items():
        click.echo(f"\n  [{name}]")
        click.echo(f"    Code chunks:    {info['code_chunks']}")
        click.echo(f"    Session chunks: {info['session_chunks']}")
    click.echo(f"\nTotal code chunks:    {s['total_code_chunks']}")
    click.echo(f"Total session chunks: {s['total_session_chunks']}")
    click.echo(f"Tracked files:        {s.get('manifest_entries', 0)}")
```

**Step 5: Run all tests**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/vecs/indexer.py src/vecs/cli.py
git commit -m "feat: refactor indexer for multi-project support with AST chunking"
```

---

### Task 8: Path filtering and result deduplication in search

Add `path_filter` parameter to search and deduplicate overlapping results.

**Files:**
- Modify: `src/vecs/searcher.py`
- Modify: `tests/test_searcher.py`

**Step 1: Write failing tests**

Add to `tests/test_searcher.py`:
```python
from vecs.searcher import format_results, deduplicate_results


def test_deduplicate_removes_overlapping():
    """Results with >70% line overlap are deduplicated."""
    shared = "\n".join(f"line {i}" for i in range(80))
    results = [
        {"id": "a", "text": shared + "\nextra_a1\nextra_a2", "metadata": {}, "distance": 0.1},
        {"id": "b", "text": shared + "\nextra_b1", "metadata": {}, "distance": 0.2},
        {"id": "c", "text": "totally different content\nnothing in common", "metadata": {}, "distance": 0.3},
    ]
    deduped = deduplicate_results(results)
    assert len(deduped) == 2
    assert deduped[0]["id"] == "a"
    assert deduped[1]["id"] == "c"


def test_deduplicate_keeps_unique():
    """Non-overlapping results are all kept."""
    results = [
        {"id": "a", "text": "alpha\nbeta\ngamma", "metadata": {}, "distance": 0.1},
        {"id": "b", "text": "delta\nepsilon\nzeta", "metadata": {}, "distance": 0.2},
    ]
    deduped = deduplicate_results(results)
    assert len(deduped) == 2
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest tests/test_searcher.py -v`
Expected: FAIL — `deduplicate_results` not defined

**Step 3: Rewrite searcher.py**

Replace `src/vecs/searcher.py`:
```python
from __future__ import annotations

from vecs.clients import get_voyage_client, get_chromadb_client
from vecs.config import (
    CODE_MODEL,
    SESSIONS_MODEL,
    load_config,
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

        embedding = vo.embed([query], model=model, input_type="query").embeddings[0]

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

    all_results.sort(key=lambda r: r.get("distance") or float("inf"))
    all_results = deduplicate_results(all_results)
    return all_results[:n_results]
```

**Step 4: Run tests**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest tests/test_searcher.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/vecs/searcher.py tests/test_searcher.py
git commit -m "feat: path filtering and result deduplication in search"
```

---

### Task 9: Hybrid search with BM25 and Reciprocal Rank Fusion

Add BM25 keyword index alongside vector search, merge results using RRF.

**Files:**
- Create: `src/vecs/bm25_index.py`
- Create: `tests/test_bm25.py`
- Modify: `src/vecs/searcher.py`
- Modify: `src/vecs/indexer.py`

**Step 1: Write failing tests for BM25 index**

Create `tests/test_bm25.py`:
```python
from pathlib import Path
from vecs.bm25_index import BM25Index


def test_bm25_build_and_search(tmp_path):
    """BM25 index returns results ranked by keyword relevance."""
    idx = BM25Index(tmp_path / "test.pkl")
    docs = [
        {"id": "a", "text": "the player takes damage and health decreases"},
        {"id": "b", "text": "the enemy attacks the player with a sword"},
        {"id": "c", "text": "the menu shows options for settings and audio"},
    ]
    idx.build(docs)
    results = idx.search("player damage", n=2)
    assert len(results) == 2
    assert results[0]["id"] == "a"  # most relevant


def test_bm25_save_and_load(tmp_path):
    """BM25 index can be saved and loaded."""
    idx = BM25Index(tmp_path / "test.pkl")
    docs = [
        {"id": "a", "text": "hello world"},
        {"id": "b", "text": "goodbye world"},
    ]
    idx.build(docs)
    idx.save()

    idx2 = BM25Index(tmp_path / "test.pkl")
    idx2.load()
    results = idx2.search("hello", n=1)
    assert results[0]["id"] == "a"


def test_bm25_empty_index(tmp_path):
    """Empty index returns no results."""
    idx = BM25Index(tmp_path / "test.pkl")
    idx.build([])
    results = idx.search("anything", n=5)
    assert results == []
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest tests/test_bm25.py -v`
Expected: FAIL — `bm25_index` module not found

**Step 3: Implement bm25_index.py**

Create `src/vecs/bm25_index.py`:
```python
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
                    "bm25": self.bm25,
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
        self.bm25 = data["bm25"]
        self.doc_ids = data["doc_ids"]
        self.doc_texts = data["doc_texts"]
        return True
```

**Step 4: Run BM25 tests**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest tests/test_bm25.py -v`
Expected: All 3 tests PASS

**Step 5: Write failing test for RRF in searcher**

Add to `tests/test_searcher.py`:
```python
from vecs.searcher import reciprocal_rank_fusion


def test_rrf_merges_rankings():
    """RRF combines two rankings into a merged ranking."""
    vector_results = [
        {"id": "a", "text": "alpha", "metadata": {}, "distance": 0.1},
        {"id": "b", "text": "beta", "metadata": {}, "distance": 0.3},
    ]
    bm25_results = [
        {"id": "b", "text": "beta", "score": 5.0},
        {"id": "c", "text": "gamma", "score": 3.0},
    ]
    merged = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    ids = [r["id"] for r in merged]
    # "b" appears in both, should rank high
    assert "b" in ids
    assert "a" in ids
    assert "c" in ids
```

**Step 6: Implement RRF in searcher.py**

Add to `src/vecs/searcher.py`:
```python
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
```

**Step 7: Integrate BM25 into indexer**

In `src/vecs/indexer.py`, after `_embed_and_store` in `index_code` and `index_sessions`, build and save the BM25 index:

Add imports at top of `indexer.py`:
```python
from vecs.bm25_index import BM25Index
```

At the end of `index_code`, after `manifest.save()`, add:
```python
    # Build BM25 index for this project
    bm25_dir = VECS_DIR / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    try:
        all_docs = collection.get(include=["documents"])
        bm25_docs = [
            {"id": id_, "text": text}
            for id_, text in zip(all_docs["ids"], all_docs["documents"])
        ]
        bm25 = BM25Index(bm25_dir / f"{project.name}_code.pkl")
        bm25.build(bm25_docs)
        bm25.save()
    except Exception:
        pass  # BM25 is best-effort
```

Same pattern at end of `index_sessions`.

**Step 8: Integrate BM25 into search**

In `src/vecs/searcher.py`, update the `search` function to also query BM25 and merge:

After getting `all_results` from vector search, add:
```python
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
```

Add these imports to searcher.py:
```python
from vecs.bm25_index import BM25Index
from vecs.config import VECS_DIR
```

**Step 9: Run all tests**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest -v`
Expected: All tests PASS

**Step 10: Commit**

```bash
git add src/vecs/bm25_index.py tests/test_bm25.py src/vecs/searcher.py src/vecs/indexer.py
git commit -m "feat: hybrid search with BM25 keyword matching and reciprocal rank fusion"
```

---

### Task 10: Query caching

Cache embedding and search results in memory with TTL.

**Files:**
- Modify: `src/vecs/searcher.py`
- Create: `tests/test_cache.py`

**Step 1: Write failing test**

Create `tests/test_cache.py`:
```python
from vecs.searcher import _embedding_cache, _clear_caches


def test_cache_exists():
    """Embedding cache is a TTLCache."""
    from cachetools import TTLCache
    assert isinstance(_embedding_cache, TTLCache)


def test_clear_caches():
    """Caches can be cleared."""
    _embedding_cache["test_key"] = [0.1, 0.2]
    _clear_caches()
    assert len(_embedding_cache) == 0
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest tests/test_cache.py -v`
Expected: FAIL

**Step 3: Add caching to searcher.py**

Add at the top of `src/vecs/searcher.py`:
```python
from cachetools import TTLCache

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
```

Then replace the `vo.embed(...)` call in the `search` function with `_cached_embed(vo, query, model)`.

**Step 4: Run tests**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest tests/test_cache.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/vecs/searcher.py tests/test_cache.py
git commit -m "feat: TTL caching for query embeddings"
```

---

### Task 11: MCP tools for index management

Add `reindex` and `index_status` MCP tools.

**Files:**
- Modify: `src/vecs/mcp_server.py`

**Step 1: Update mcp_server.py**

Replace `src/vecs/mcp_server.py`:
```python
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
```

**Step 2: Commit**

```bash
git add src/vecs/mcp_server.py
git commit -m "feat: MCP tools for reindex and index_status"
```

---

### Task 12: Update CLI search and description

Update the CLI search command to pass through new parameters and update the tool description to be project-generic.

**Files:**
- Modify: `src/vecs/cli.py`

**Step 1: Update cli.py**

Update the `main` group description and `search_cmd` to accept `--path-filter` and `--project`:

```python
@main.group()
def main():
    """vecs — Semantic search for your codebase."""
    pass
```

```python
@main.command()
@click.argument("query")
@click.option("--collection", "-c", type=click.Choice(["code", "sessions"], case_sensitive=False), default=None)
@click.option("--limit", "-n", default=5, help="Number of results.")
@click.option("--path-filter", "-f", default=None, help="Filter to paths containing this substring.")
@click.option("--project", "-p", default=None, help="Search a specific project.")
def search_cmd(query: str, collection: str | None, limit: int, path_filter: str | None, project: str | None):
    """Search code and sessions semantically."""
    results = search(query, collection_name=collection, n_results=limit, path_filter=path_filter, project=project)
    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        source = meta.get("file_path") or f"session:{meta.get('session_id', '?')}"
        dist = r.get("distance")
        dist_str = f" (distance: {dist:.4f})" if dist is not None else ""
        proj = r.get("project", "?")
        click.echo(f"\n--- Result {i} [{proj}:{r.get('collection', '?')}] {source}{dist_str} ---")
        text = r["text"]
        if len(text) > 1000:
            text = text[:1000] + "\n... [truncated]"
        click.echo(text)
```

**Step 2: Run all tests**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/vecs/cli.py
git commit -m "feat: CLI search with path filtering and project scoping"
```

---

### Task 13: Final integration test and cleanup

Run the full test suite, fix any remaining issues, verify the tool works end-to-end.

**Step 1: Run full test suite**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run pytest -v`
Expected: All tests PASS

**Step 2: Verify CLI help**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run vecs --help`
Expected: Shows all commands including `project`, `index`, `search`, `status`

**Step 3: Verify project commands**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv run vecs project --help`
Expected: Shows add, remove, list subcommands

**Step 4: Verify MCP server starts**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && timeout 3 uv run mcp run src/vecs/mcp_server.py 2>&1 || true`
Expected: Server starts without import errors

**Step 5: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: integration cleanup for vecs v2"
```
