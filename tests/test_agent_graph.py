"""Agent wiring tests with a fake LLM -- no network, no API key."""
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agent import agent_answer
from src.config import load_config


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, q, k=None):
        self.queries.append(q)
        return [{"id": f"x{len(self.queries)}", "text": f"excerpt for {q}",
                 "doc": "a.pdf", "page": 1, "score": 0.5}]


def _cfg(**agent):
    cfg = load_config()
    cfg["agent"].update(agent)
    return cfg


def test_critic_satisfied_goes_straight_to_answer():
    llm = FakeListChatModel(responses=['["sub one"]', '{"sufficient": true}', "final answer [1]"])
    r = FakeRetriever()
    out = agent_answer("q?", _cfg(enabled=True, decompose=True, critic=True), r, llm)
    assert out["answer"] == "final answer [1]"
    assert r.queries == ["sub one"]


def test_insufficient_triggers_one_requery_then_stops():
    llm = FakeListChatModel(responses=[
        '["sub one"]',
        '{"sufficient": false, "followup_query": "better query"}',
        '{"sufficient": false, "followup_query": "even better"}',
        "final answer",
    ])
    r = FakeRetriever()
    out = agent_answer("q?", _cfg(enabled=True, max_requeries=1), r, llm)
    assert "better query" in r.queries
    assert out["answer"] == "final answer"


def test_malformed_json_falls_back_to_original_question():
    llm = FakeListChatModel(responses=["not json at all", '{"sufficient": true}', "ans"])
    r = FakeRetriever()
    agent_answer("original?", _cfg(enabled=True), r, llm)
    assert r.queries == ["original?"]


def test_flags_off_skip_stages():
    llm = FakeListChatModel(responses=["ans"])
    r = FakeRetriever()
    out = agent_answer("q?", _cfg(enabled=True, decompose=False, critic=False), r, llm)
    assert r.queries == ["q?"] and out["answer"] == "ans"
