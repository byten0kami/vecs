from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import chromadb
import voyageai

from vecs.ast_chunker import chunk_code_file_ast
from vecs.bm25_index import BM25Index
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

    return stored


def index_sessions(project: ProjectConfig, vo: voyageai.Client, db: chromadb.ClientAPI) -> int:
    """Index Claude Code session transcripts for a project."""
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

    # Build BM25 index for this project
    bm25_dir = VECS_DIR / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    try:
        all_docs = collection.get(include=["documents"])
        bm25_docs = [
            {"id": id_, "text": text}
            for id_, text in zip(all_docs["ids"], all_docs["documents"])
        ]
        bm25 = BM25Index(bm25_dir / f"{project.name}_sessions.pkl")
        bm25.build(bm25_docs)
        bm25.save()
    except Exception:
        pass  # BM25 is best-effort

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
