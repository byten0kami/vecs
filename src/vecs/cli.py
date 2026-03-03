from pathlib import Path

import click


@click.group()
def main():
    """vecs — Semantic search for your codebase."""
    pass


@main.command()
@click.option("--project", "-p", default=None, help="Index a specific project (default: all).")
def index(project: str | None):
    """Index code and session transcripts (incremental)."""
    from vecs.indexer import run_index
    run_index(project_name=project)


@main.command()
@click.argument("query")
@click.option(
    "--collection", "-c",
    type=click.Choice(["code", "sessions"], case_sensitive=False),
    default=None,
    help="Search a specific collection (default: both).",
)
@click.option("--limit", "-n", default=5, help="Number of results.")
@click.option("--path-filter", "-f", default=None, help="Filter to paths containing this substring.")
@click.option("--project", "-p", default=None, help="Search a specific project.")
def search_cmd(query: str, collection: str | None, limit: int, path_filter: str | None, project: str | None):
    """Search code and sessions semantically."""
    from vecs.searcher import search
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


@main.command()
@click.option("--project", "-p", default=None, help="Status for a specific project.")
def status(project: str | None):
    """Show index status."""
    from vecs.indexer import get_status
    s = get_status(project_name=project)
    for name, info in s.get("projects", {}).items():
        click.echo(f"\n  [{name}]")
        click.echo(f"    Code chunks:    {info['code_chunks']}")
        click.echo(f"    Session chunks: {info['session_chunks']}")
    click.echo(f"\nTotal code chunks:    {s['total_code_chunks']}")
    click.echo(f"Total session chunks: {s['total_session_chunks']}")
    click.echo(f"Tracked files:        {s.get('manifest_entries', 0)}")


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


# Alias so `vecs search` works
main.add_command(search_cmd, "search")
