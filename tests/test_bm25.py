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
        {"id": "a", "text": "hello world greeting"},
        {"id": "b", "text": "goodbye world farewell"},
        {"id": "c", "text": "something else entirely"},
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
