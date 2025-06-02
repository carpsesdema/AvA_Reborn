# File: tests/test_llm_client.py

import pytest

from core.llm_client import LLMClient


@pytest.fixture
def minimal_llm_client(monkeypatch):
    """
    Build an LLMClient with exactly one dummy model so that role assignment always succeeds.
    """
    client = LLMClient()

    # Create a dummy “ModelInfo”-like object with all roles allowed
    class DummyModelInfo:
        def __init__(self):
            self.suitable_roles = [
                client.LLMRole.CHAT,
                client.LLMRole.CODER,
                client.LLMRole.PLANNER,
                client.LLMRole.ASSEMBLER,
                client.LLMRole.REVIEWER,
            ]

    # Monkeypatch the client.models dictionary
    client.models = {"mock-model": DummyModelInfo()}

    # Immediately assign roles
    client._assign_roles()
    return client


def test_every_role_assigned(minimal_llm_client):
    """
    Ensure that after _assign_roles(), each logical role has some model assigned.
    """
    client = minimal_llm_client
    assignments = client.get_role_assignments()

    # We expect each role name to be a key in the assignment dict
    for role_name in ["planner", "coder", "assembler", "reviewer", "chat"]:
        assert role_name in assignments
        # Since only one model exists, it must be assigned to everything
        assert assignments[role_name] == "mock-model"


def test_override_models_and_reassign(monkeypatch):
    """
    If we add a second dummy model with a restricted set of roles, the client should pick
    the “best fit” for each role.
    """
    client = LLMClient()

    class ModelA:
        def __init__(self):
            # Only suitable for planning and chatting
            self.suitable_roles = [client.LLMRole.PLANNER, client.LLMRole.CHAT]

    class ModelB:
        def __init__(self):
            # Only suitable for coding and assembling
            self.suitable_roles = [client.LLMRole.CODER, client.LLMRole.ASSEMBLER]

    # Two models, each specialized
    client.models = {
        "modelA": ModelA(),
        "modelB": ModelB(),
    }

    client._assign_roles()
    assignments = client.get_role_assignments()

    # Check that planner/chat went to modelA
    assert assignments["planner"] == "modelA"
    assert assignments["chat"] == "modelA"

    # coder/assembler should go to modelB
    assert assignments["coder"] == "modelB"
    assert assignments["assembler"] == "modelB"

    # reviewer wasn’t explicitly covered, so it should fall back to whichever model has “most capacity.”
    # In this simple test, both have the same number of roles (2), so either is OK; just check that
    # reviewer is set to something valid.
    assert assignments["reviewer"] in {"modelA", "modelB"}
