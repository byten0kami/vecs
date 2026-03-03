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
