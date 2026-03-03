from vecs.searcher import format_results, deduplicate_results, reciprocal_rank_fusion


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
