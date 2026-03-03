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
