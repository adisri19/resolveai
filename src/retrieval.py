"""RAG over the policy knowledge base.

Pipeline: load .md policies -> split into one chunk per numbered rule -> embed with
Gemini -> cache vectors in SQLite -> embed the ticket -> cosine KNN -> return chunks.

The knowledge base is six short documents, so the vector store is a numpy matrix
built from a SQLite table. FAISS/Chroma would be a dependency buying nothing at
this size -- exact search over ~30 vectors is microseconds.
"""
import hashlib
import os
import re
from pathlib import Path
from typing import List, NamedTuple

import numpy as np

from .database import connect

KB_DIR = Path(os.getenv("KB_DIR", "knowledge_base"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768          # gemini-embedding-001 supports truncated output dims
TOP_K = int(os.getenv("TOP_K", "8"))


class Chunk(NamedTuple):
    source: str          # filename, e.g. "damaged_goods.md" -- this is what gets cited
    text: str
    score: float = 0.0


def _client():
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return genai.Client(api_key=api_key)


def load_chunks() -> List[Chunk]:
    """One chunk per numbered policy rule.

    These documents are already written as atomic numbered rules, so the document's
    own structure is a better split than any fixed token window -- no rule gets cut
    in half and no chunk mixes two unrelated rules. Each chunk carries its policy
    title so the embedding has topical context and not just a bare sentence.
    """
    chunks = []
    for path in sorted(KB_DIR.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title = next((l.lstrip("# ").strip() for l in lines if l.startswith("#")), path.stem)
        for line in lines:
            rule = re.match(r"^\s*(\d+)\.\s+(.*\S)", line)
            if rule:
                chunks.append(Chunk(path.name, f"{title} (rule {rule.group(1)}): {rule.group(2)}"))
    if not chunks:
        raise RuntimeError(f"No policy rules found in {KB_DIR.resolve()}")
    return chunks


def _embed(texts: List[str], task_type: str) -> np.ndarray:
    from google.genai import types

    client = _client()
    vectors = []
    for text in texts:
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM),
        )
        vectors.append(resp.embeddings[0].values)
    matrix = np.asarray(vectors, dtype=np.float32)
    # Truncated gemini-embedding output is not unit-length; normalise so a dot
    # product *is* cosine similarity.
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)


def ingest(verbose: bool = False) -> int:
    """Embed any policy rule we have not embedded before and drop stale ones.

    Content-hash keyed, so this is idempotent: editing one policy file re-embeds
    that file's rules only. Returns the number of newly embedded chunks.
    """
    chunks = load_chunks()
    hashes = {c: hashlib.sha256((c.source + c.text).encode()).hexdigest() for c in chunks}

    with connect() as conn:
        known = {r["hash"] for r in conn.execute("SELECT hash FROM kb_chunks")}
        conn.execute(
            f"DELETE FROM kb_chunks WHERE hash NOT IN ({','.join('?' * len(hashes))})",
            tuple(hashes.values()),
        )

    missing = [c for c in chunks if hashes[c] not in known]
    if not missing:
        if verbose:
            print(f"Knowledge base already ingested ({len(chunks)} chunks).")
        return 0

    if verbose:
        print(f"Embedding {len(missing)} new chunk(s) with {EMBED_MODEL}...")
    vectors = _embed([c.text for c in missing], "RETRIEVAL_DOCUMENT")
    with connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO kb_chunks (source, text, hash, embedding) VALUES (?, ?, ?, ?)",
            [(c.source, c.text, hashes[c], v.tobytes()) for c, v in zip(missing, vectors)],
        )
    if verbose:
        print(f"Ingested {len(missing)} chunk(s); {len(chunks)} total.")
    return len(missing)


def retrieve(query: str, top_k: int = TOP_K) -> List[Chunk]:
    """Embed the ticket and return the most similar policy rules, best first."""
    with connect() as conn:
        rows = conn.execute("SELECT source, text, embedding FROM kb_chunks").fetchall()
    if not rows:
        raise RuntimeError("Knowledge base is empty. Run: python -m src.retrieval")

    matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    scores = matrix @ _embed([query], "RETRIEVAL_QUERY")[0]
    best = np.argsort(-scores)[:top_k]
    return [Chunk(rows[i]["source"], rows[i]["text"], float(scores[i])) for i in best]


if __name__ == "__main__":  # python -m src.retrieval  -- build/refresh the vector store
    from .database import init_db

    init_db()
    ingest(verbose=True)
