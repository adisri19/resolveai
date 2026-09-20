"""Chunking, retrieval, grounding and failure handling of the AI pipeline."""
import pytest
from pydantic import ValidationError

from src import decision as decision_mod
from src import retrieval
from src.decision import _ground, build_prompt, decide
from src.retrieval import Chunk, load_chunks, retrieve
from src.schemas import ACTIONS, Decision, TicketIn


def test_chunks_are_one_rule_each_and_carry_their_source():
    chunks = load_chunks()
    assert len(chunks) >= 25
    assert {c.source for c in chunks} >= {"damaged_goods.md", "returns.md", "shipping.md"}
    assert all(c.text and "rule" in c.text for c in chunks)
    # A chunk must never merge two numbered rules.
    assert all(c.text.count("(rule ") == 1 for c in chunks)


def test_ingest_is_idempotent(app_env):
    assert retrieval.ingest() == 0  # fixture already ingested; nothing new to embed


def test_retrieval_surfaces_the_relevant_policy(app_env):
    top = retrieve("My package arrived damaged and the box is crushed.", top_k=5)
    assert "damaged_goods.md" in {c.source for c in top}
    assert top[0].score >= top[-1].score  # ordered best-first

    shipping = retrieve("My parcel was dispatched 9 days ago and still has not arrived.", top_k=5)
    assert "shipping.md" in {c.source for c in shipping}


def test_prompt_contains_the_retrieved_rules_and_the_ticket_facts(app_env):
    ticket = TicketIn(message="Damaged order.", order_value_inr=3500, days_since_delivery=1)
    chunks = retrieve(ticket.message, top_k=3)
    prompt = build_prompt(ticket, chunks)
    for chunk in chunks:
        assert chunk.text in prompt
    assert "3500" in prompt
    assert "Days since dispatch: unknown" in prompt  # absent facts are shown as unknown, not dropped


def test_decision_schema_rejects_invented_actions():
    with pytest.raises(ValidationError):
        Decision(action="GIVE_FREE_STUFF", confidence=0.9, reason="x", sources=[])
    with pytest.raises(ValidationError):
        Decision(action=ACTIONS[0], confidence=1.4, reason="x", sources=[])


def test_grounding_drops_citations_that_were_never_retrieved():
    chunks = [Chunk("damaged_goods.md", "text", 0.9)]
    dirty = Decision(action="REQUEST_PHOTOS", confidence=0.9, reason="x",
                     sources=["damaged_goods.md", "invented_policy.md"])
    assert _ground(dirty, chunks).sources == ["damaged_goods.md"]


def test_grounding_falls_back_to_top_hit_when_nothing_real_is_cited():
    chunks = [Chunk("returns.md", "text", 0.9)]
    dirty = Decision(action="APPROVE_RETURN", confidence=0.8, reason="x", sources=["made_up.md"])
    assert _ground(dirty, chunks).sources == ["returns.md"]


def test_llm_failure_raises_instead_of_inventing_a_decision(app_env, stub_llm):
    """A dead model must not look like a NEEDS_MORE_INFORMATION decision.

    That action means "the facts are too thin"; an unreachable API means "no answer".
    Collapsing the two would persist a fabricated decision and make an outage read as
    poor accuracy in the evaluation runner.
    """
    stub_llm["decision"] = RuntimeError("Gemini unavailable")
    with pytest.raises(decision_mod.DecisionUnavailable) as caught:
        decide(TicketIn(message="My order arrived damaged."))
    assert "Gemini unavailable" in str(caught.value)
    assert caught.value.chunks  # retrieval still happened and is reported for debugging


def test_retry_happens_once_before_giving_up(app_env, monkeypatch):
    calls = []

    def flaky(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return Decision(action="APPROVE_RETURN", confidence=0.7, reason="ok", sources=["returns.md"])

    monkeypatch.setattr(decision_mod, "_generate", flaky)
    result, _ = decide(TicketIn(message="I changed my mind, unopened non-food item."))
    assert len(calls) == 2
    assert result.action == "APPROVE_RETURN"


def test_embedding_failure_is_also_a_decision_unavailable(app_env, stub_llm, monkeypatch):
    """Retrieval is an API call too. A dead embedding endpoint must surface the same
    way as a dead generation endpoint, not as an unhandled 500."""
    def dead(texts, task_type):
        raise ConnectionError("embedding endpoint returned 403")

    monkeypatch.setattr(retrieval, "_embed", dead)
    with pytest.raises(decision_mod.DecisionUnavailable) as caught:
        decide(TicketIn(message="My order arrived damaged."))
    assert "403" in str(caught.value)
    assert caught.value.chunks == []


def test_misconfiguration_is_not_disguised_as_a_transient_failure(app_env):
    """An empty knowledge base needs a human to run the ingest, so it keeps its own
    actionable message instead of becoming a generic 'try again'."""
    from src import database

    with database.connect() as conn:
        conn.execute("DELETE FROM kb_chunks")
    with pytest.raises(RuntimeError, match="Knowledge base is empty"):
        decide(TicketIn(message="My order arrived damaged."))
