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
