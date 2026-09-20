"""Test fixtures.

The suite runs entirely offline: the Gemini embedding call is swapped for a
deterministic bag-of-words vectoriser (so retrieval is still genuinely exercised,
just without the network) and the Gemini generation call is stubbed per test.
"""
import re
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import auth, database, decision, retrieval
from src.schemas import Decision

DIM = 256


def fake_embed(texts, task_type):
    """Hashed bag of words. Crude, but similarity still tracks word overlap, which
    is all the retrieval tests need to assert."""
    matrix = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in re.findall(r"[a-z0-9₹,]+", text.lower()):
            matrix[i, zlib.crc32(word.encode()) % DIM] += 1.0
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-a-real-one")
    monkeypatch.setattr(retrieval, "_embed", fake_embed)
    database.init_db()
    retrieval.ingest()
    return tmp_path


@pytest.fixture
def stub_llm(monkeypatch):
    """Canned model output; individual tests override `holder["decision"]`."""
    holder = {
        "decision": Decision(
            action="REQUEST_PHOTOS",
            confidence=0.9,
            reason="Order is above the photo threshold.",
            sources=["damaged_goods.md"],
        )
    }

    def _generate(prompt):
        result = holder["decision"]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(decision, "_generate", _generate)
    return holder


@pytest.fixture
def client(app_env, stub_llm):
    from src.api import app

    with TestClient(app) as c:
        yield c


def register_and_login(client, email, password="correct-horse"):
    client.post("/register", json={"email": email, "password": password})
    token = client.post("/login", json={"email": email, "password": password}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
