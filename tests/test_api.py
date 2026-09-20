"""Ticket submission, persistence and retrieval through the HTTP API."""
import json

from src import database
from src.schemas import Decision

from .conftest import register_and_login

TICKET = {
    "message": "My order arrived damaged yesterday.",
    "order_value_inr": 3500,
    "days_since_delivery": 1,
    "product_type": "non_food",
    "opened_status": "opened",
    "order_status": "delivered",
}


def test_ticket_returns_structured_decision(client):
    headers = register_and_login(client, "a@example.com")
    body = client.post("/tickets", json=TICKET, headers=headers).json()
    decision = body["decision"]
    assert decision["action"] == "REQUEST_PHOTOS"
    assert 0.0 <= decision["confidence"] <= 1.0
    assert decision["sources"] == ["damaged_goods.md"]
    assert decision["reason"]


def test_ticket_and_decision_are_persisted(client):
    headers = register_and_login(client, "a@example.com")
    ticket_id = client.post("/tickets", json=TICKET, headers=headers).json()["id"]
    with database.connect() as conn:
        ticket = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        decision = conn.execute("SELECT * FROM decisions WHERE ticket_id = ?", (ticket_id,)).fetchone()
    assert ticket["message"] == TICKET["message"]
    assert ticket["order_value_inr"] == 3500
    assert decision["action"] == "REQUEST_PHOTOS"
    assert json.loads(decision["sources"]) == ["damaged_goods.md"]


def test_history_lists_only_own_tickets_newest_first(client):
    headers = register_and_login(client, "a@example.com")
    first = client.post("/tickets", json=TICKET, headers=headers).json()["id"]
    second = client.post("/tickets", json={**TICKET, "message": "Second issue."}, headers=headers).json()["id"]
    ids = [t["id"] for t in client.get("/tickets", headers=headers).json()]
    assert ids == [second, first]


def test_missing_ticket_is_404(client):
    headers = register_and_login(client, "a@example.com")
    assert client.get("/tickets/9999", headers=headers).status_code == 404


def test_invalid_payloads_are_rejected(client):
    headers = register_and_login(client, "a@example.com")
    assert client.post("/tickets", json={"message": ""}, headers=headers).status_code == 422
    assert client.post("/tickets", json={"message": "hi", "order_value_inr": -5}, headers=headers).status_code == 422
    assert client.post("/tickets", json={"message": "hi", "product_type": "liquid"}, headers=headers).status_code == 422
    assert client.post("/register", json={"email": "not-an-email", "password": "longenough"}).status_code == 422
    assert client.post("/register", json={"email": "a@b.com", "password": "short"}).status_code == 422


def test_model_outage_returns_503_and_persists_nothing(client, stub_llm):
    """A ticket row with no decision is a worse artefact than no ticket at all,
    and a guessed decision is worse than both."""
    headers = register_and_login(client, "a@example.com")
    stub_llm["decision"] = RuntimeError("Gemini unavailable")

    resp = client.post("/tickets", json=TICKET, headers=headers)
    assert resp.status_code == 503

    with database.connect() as conn:
        assert conn.execute("SELECT COUNT(*) AS c FROM tickets").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) AS c FROM decisions").fetchone()["c"] == 0
    assert client.get("/tickets", headers=headers).json() == []
