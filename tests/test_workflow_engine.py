# File: tests/test_workflow_engine.py

import pytest
import asyncio

from core.workflow_engine import WorkflowEngine
from core.llm_client import LLMClient


@pytest.fixture
def dummy_llm_client(monkeypatch):
    """
    Create a dummy LLMClient that has minimal required attributes.
    We monkeypatch its models dict so role assignment always has something to pick.
    """
    client = LLMClient()

    # Ensure there is at least one “mock-model” with all roles allowed
    class DummyModel:
        def __init__(self):
            # Simulate a model that claims it can do everything
            self.suitable_roles = [
                client.LLMRole.CHAT,
                client.LLMRole.CODER,
                client.LLMRole.PLANNER,
                client.LLMRole.ASSEMBLER,
                client.LLMRole.REVIEWER,
            ]

    # Overwrite the models dictionary
    client.models = {"mock-model": DummyModel()}

    # Force‐assign roles immediately so WorkflowEngine can rely on them
    client._assign_roles()
    return client


@pytest.fixture
def workflow_engine(dummy_llm_client):
    """
    Create a WorkflowEngine with a dummy LLMClient.
    Pass None for terminal_window and code_viewer since we won't open any real UI.
    """
    return WorkflowEngine(
        llm_client=dummy_llm_client,
        terminal_window=None,
        code_viewer=None
    )


@pytest.mark.asyncio
async def test_is_build_request_simple(workflow_engine):
    """
    Ensure that _is_build_request returns True when the text looks like a 'build' instruction.
    """
    we = workflow_engine

    # Too short / not containing 'build' -> False
    assert not we._is_build_request("Hi there!")
    # Fewer than 3 tokens, even with 'build' -> False
    assert not we._is_build_request("build")
    assert not we._is_build_request("please build")

    # Contains the word "build" and at least 3 tokens -> True
    assert we._is_build_request("build a REST API")
    assert we._is_build_request("could you build me something? build a CLI tool")


@pytest.mark.asyncio
async def test_get_workflow_stats_initial_empty(workflow_engine):
    """
    After initialization, the workflow cache should be empty.
    """
    stats = workflow_engine.get_workflow_stats()
    # Check that cache_stats exist and are zeroed
    assert "cache_stats" in stats
    cache_stats = stats["cache_stats"]
    assert cache_stats["cache_size"] == 0
    assert pytest.approx(cache_stats["hit_rate"], rel=1e-3) == 0.0


@pytest.mark.asyncio
async def test_context_integration_increment_hit_rate(workflow_engine):
    """
    Store a context via the workflow engine and retrieve it, verifying hit/miss counts.
    """
    we = workflow_engine

    # At first, retrieving will be a miss
    before_stats = we.get_workflow_stats()["cache_stats"]
    assert before_stats["miss_count"] == 0
    assert before_stats["hit_count"] == 0

    # Store a context
    we.context_cache.store_context("q1", "ctx1", "typeX", relevance_score=0.7)

    # Retrieve it once (should count as a hit)
    retrieved = we.context_cache.get_context("q1", "typeX")
    assert retrieved == "ctx1"

    mid_stats = we.get_workflow_stats()["cache_stats"]
    assert mid_stats["hit_count"] == 1
    assert mid_stats["miss_count"] == 0

    # Looking up a non-existent context is a miss
    _ = we.context_cache.get_context("nope", "typeX")
    after_stats = we.get_workflow_stats()["cache_stats"]
    assert after_stats["hit_count"] == 1
    assert after_stats["miss_count"] == 1
