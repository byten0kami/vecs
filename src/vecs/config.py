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
