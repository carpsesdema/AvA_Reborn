# File: tests/test_context_cache.py

import pytest
from core.workflow_engine import ContextCache


@pytest.fixture
def cache():
    # Create a ContextCache with a small max size for eviction testing
    return ContextCache(max_cache_size=3)


def test_store_and_retrieve_simple(cache):
    """
    Verify that storing contexts and retrieving them by key/type works.
    """
    cache.store_context("key1", "ctx1", "typeA", relevance_score=0.2)
    cache.store_context("key2", "ctx2", "typeA", relevance_score=0.8)

    # Retrieval should return exactly what was stored
    assert cache.get_context("key1", "typeA") == "ctx1"
    assert cache.get_context("key2", "typeA") == "ctx2"


def test_store_with_different_types(cache):
    """
    Storing contexts under different types should not collide.
    """
    cache.store_context("key", "ctxA", "typeA", relevance_score=0.5)
    cache.store_context("key", "ctxB", "typeB", relevance_score=0.5)

    # Same key but different type—each should return its own context
    assert cache.get_context("key", "typeA") == "ctxA"
    assert cache.get_context("key", "typeB") == "ctxB"


def test_eviction_lru_behavior(cache):
    """
    Fill the cache beyond its max size and ensure the least-relevant entry is evicted.
    """
    # Insert three items with explicit relevance scores
    cache.store_context("k1", "c1", "t", relevance_score=0.1)
    cache.store_context("k2", "c2", "t", relevance_score=0.2)
    cache.store_context("k3", "c3", "t", relevance_score=0.3)

    # At this point, cache size == 3
    stats_before = cache.get_stats()
    assert stats_before["cache_size"] == 3

    # Insert a fourth item; since max_cache_size=3, one must be pruned
    cache.store_context("k4", "c4", "t", relevance_score=0.4)
    stats_after = cache.get_stats()
    assert stats_after["cache_size"] == 3

    # The item with lowest relevance_score ("k1") should have been removed
    assert cache.get_context("k1", "t") is None
    # The others should still be available
    assert cache.get_context("k2", "t") == "c2"
    assert cache.get_context("k3", "t") == "c3"
    assert cache.get_context("k4", "t") == "c4"


def test_prune_preserves_top_n_relevance(cache):
    """
    If multiple entries tie for lowest relevance, the cache should still prune down to max size.
    """
    cache.store_context("k1", "c1", "t", relevance_score=0.5)
    cache.store_context("k2", "c2", "t", relevance_score=0.5)
    cache.store_context("k3", "c3", "t", relevance_score=0.5)

    # All three have identical relevance; next insert will cause one to be evicted arbitrarily among the lowest.
    cache.store_context("k4", "c4", "t", relevance_score=0.5)
    stats = cache.get_stats()
    assert stats["cache_size"] == 3

    # Exactly one of ["k1", "k2", "k3"] should be missing, but we don't know which.
    missing = [key for key in ["k1", "k2", "k3"] if cache.get_context(key, "t") is None]
    assert len(missing) == 1
