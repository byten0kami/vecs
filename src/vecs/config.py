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
