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
