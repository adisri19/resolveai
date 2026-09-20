"""Authentication and — the part that actually matters — authorization."""
import json

from src import database
from src.auth import hash_password, verify_password

from .conftest import register_and_login

TICKET = {"message": "My order arrived damaged.", "order_value_inr": 3500, "days_since_delivery": 1}


def test_password_is_hashed_not_stored(client, app_env):
    client.post("/register", json={"email": "a@example.com", "password": "correct-horse"})
    with database.connect() as conn:
        row = conn.execute("SELECT password_hash FROM users").fetchone()
    assert "correct-horse" not in row["password_hash"]
    assert row["password_hash"].startswith("$2b$")
    assert verify_password("correct-horse", row["password_hash"])


def test_hash_is_salted():
    assert hash_password("same") != hash_password("same")


def test_register_login_me(client):
    headers = register_and_login(client, "a@example.com")
    body = client.get("/me", headers=headers).json()
    assert body["email"] == "a@example.com"


def test_duplicate_email_rejected(client):
    client.post("/register", json={"email": "a@example.com", "password": "correct-horse"})
    resp = client.post("/register", json={"email": "A@Example.com", "password": "other-pass1"})
    assert resp.status_code == 409


def test_bad_password_is_401(client):
    client.post("/register", json={"email": "a@example.com", "password": "correct-horse"})
    assert client.post("/login", json={"email": "a@example.com", "password": "wrong-password"}).status_code == 401


def test_unknown_email_is_401_not_404(client):
    # Identical response to a wrong password: no user-enumeration oracle.
    assert client.post("/login", json={"email": "nobody@example.com", "password": "whatever1"}).status_code == 401


def test_protected_endpoints_reject_missing_and_bogus_tokens(client):
    for endpoint in ("/me", "/tickets"):
        assert client.get(endpoint).status_code == 401
        assert client.get(endpoint, headers={"Authorization": "Bearer not.a.jwt"}).status_code == 401


def test_token_signed_with_another_secret_is_rejected(client, monkeypatch):
    headers = register_and_login(client, "a@example.com")
    import jwt

    forged = jwt.encode({"sub": "1"}, "attacker-secret", algorithm="HS256")
    assert client.get("/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    assert client.get("/me", headers=headers).status_code == 200


def test_alice_cannot_read_bobs_ticket(client):
    """The authorization requirement, stated explicitly."""
    alice = register_and_login(client, "alice@example.com")
    bob = register_and_login(client, "bob@example.com")

    bobs_ticket = client.post("/tickets", json=TICKET, headers=bob).json()["id"]

    assert client.get(f"/tickets/{bobs_ticket}", headers=bob).status_code == 200
    assert client.get(f"/tickets/{bobs_ticket}", headers=alice).status_code == 404
    assert client.get("/tickets", headers=alice).json() == []
