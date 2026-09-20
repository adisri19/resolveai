# ResolveAI — Support Ticket Decision Assistant

[![Vercel Deployment](https://img.shields.io/badge/Vercel-Live%20Demo-brightgreen?logo=vercel)](https://resolveai-ai.vercel.app)
[![API Docs](https://img.shields.io/badge/FastAPI-Swagger%20Docs-009688?logo=fastapi)](https://resolveai-ai.vercel.app/docs)

**Live Web App**: [https://resolveai-ai.vercel.app](https://resolveai-ai.vercel.app)  
**API Documentation**: [https://resolveai-ai.vercel.app/docs](https://resolveai-ai.vercel.app/docs)

An end-to-end AI decision API: a user registers, logs in with a JWT, submits a support
ticket, and gets back a structured, evidence-backed recommendation grounded in the
company's policy knowledge base. Tickets and decisions are persisted in SQLite.

```
Streamlit  ──HTTP──▶  FastAPI  ──▶  retrieval (Gemini embeddings + cosine KNN over SQLite)
                         │                │
                         │                ▼
                         │         top-k policy rules
                         │                │
                         ▼                ▼
                     SQLite  ◀──  Gemini (structured JSON, validated)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

On macOS/Linux activate with `source .venv/bin/activate`. After copying `.env.example`,
fill in `GEMINI_API_KEY` and `JWT_SECRET`.

Get a Gemini key at <https://aistudio.google.com/u/0/api-keys>. Generate a JWT secret with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## Running

Two processes. Backend first:

```bash
uvicorn src.api:app --reload
```

Then the frontend:

```bash
streamlit run streamlit_app.py
```

Open <http://localhost:8501>, register, log in, and submit a ticket. Interactive API
docs are at <http://127.0.0.1:8000/docs>.

The backend ingests the knowledge base on startup. To build or refresh the vector
store by hand:

```bash
python -m src.retrieval
```

## Evaluation

```bash
python -m src.evaluate
```

That runs the five supplied sample cases. To run the historical tickets instead:

```bash
python -m src.evaluate --dataset data/tickets.csv --limit 50
```

Current result on the five supplied cases:

```
PASS    S01  expected=REQUEST_PHOTOS                got=REQUEST_PHOTOS                conf=1.00
PASS    S02  expected=APPROVE_RETURN                got=APPROVE_RETURN                conf=1.00
PASS    S03  expected=OPEN_SHIPPING_INVESTIGATION   got=OPEN_SHIPPING_INVESTIGATION   conf=1.00
PASS    S04  expected=REPLACE_CORRECT_ITEM          got=REPLACE_CORRECT_ITEM          conf=1.00
PASS    S05  expected=NEEDS_MORE_INFORMATION        got=NEEDS_MORE_INFORMATION        conf=0.00

5 test cases
Correct: 5
Incorrect: 0
Accuracy: 100%
```

Failures print the model's reason, the sources it cited and the documents retrieved,
so a wrong answer can be traced to either bad retrieval or bad reasoning. Cases where
the API call itself failed are reported as `ERROR` and excluded from the accuracy
figure — they are not wrong answers, and counting them as such would hide an outage
behind a plausible-looking score.

Verified on both `gemini-3.6-flash` (the default) and `gemini-3.5-flash`.

The free Gemini tier allows 20 `generateContent` requests per **project per model**
per day, so a long `--limit` run against `data/tickets.csv` will exhaust it — and a
fresh API key from the same AI Studio project shares the same bucket. Switching model
gets a fresh one:

```bash
GEMINI_MODEL=gemini-3.5-flash python -m src.evaluate
```

The runner aborts after three consecutive failures rather than burning the rest of the
dataset on calls that cannot succeed.

## Tests

```bash
pytest -q
```

The suite runs fully offline. The Gemini embedding call is replaced with a
deterministic bag-of-words vectoriser — retrieval is still exercised end to end,
just without the network — and generation is stubbed per test. No API key needed.

## API

| Method | Endpoint | Auth | Purpose |
| --- | --- | --- | --- |
| POST | `/register` | — | Create a user account |
| POST | `/login` | — | Verify credentials, return a JWT |
| GET | `/me` | Bearer | Return the authenticated user |
| POST | `/tickets` | Bearer | Submit a ticket, generate and store a decision |
| GET | `/tickets` | Bearer | List the caller's tickets |
| GET | `/tickets/{id}` | Bearer | One ticket and its decision |

Protected endpoints expect `Authorization: Bearer <JWT>`.

A ticket response looks like this:

```json
{
  "id": 1,
  "message": "My order arrived damaged yesterday.",
  "created_at": "2026-09-20 10:15:00",
  "decision": {
    "action": "REQUEST_PHOTOS",
    "confidence": 0.95,
    "reason": "The order is valued at ₹3,500, above the ₹2,000 threshold, and was delivered 1 day ago, so photographs of the product and packaging must be requested first.",
    "sources": ["damaged_goods.md"]
  }
}
```

## Database

SQLite, created on first start.

| Table | Columns |
| --- | --- |
| `users` | `id` PK, `email` UNIQUE, `password_hash`, `created_at` |
| `tickets` | `id` PK, `user_id` → `users.id`, `message`, `order_value_inr`, `days_since_delivery`, `days_since_dispatch`, `product_type`, `opened_status`, `order_status`, `created_at` |
| `decisions` | `id` PK, `ticket_id` → `tickets.id`, `action`, `reason`, `confidence`, `sources` (JSON), `created_at` |
| `kb_chunks` | `id` PK, `source`, `text`, `hash` UNIQUE, `embedding` (BLOB) |

`tickets` carries the structured facts the policies actually need (order value,
delivery/dispatch age, product type, opened status) alongside the free-text message.
Those facts are what let a decision be *withheld* rather than guessed: when one is
missing and it changes the outcome, the answer is `NEEDS_MORE_INFORMATION`.

## How the RAG pipeline works

1. **Load** the six markdown policies from `knowledge_base/`.
2. **Chunk** one chunk per numbered policy rule. These documents are already written
   as atomic numbered rules, so the document's own structure beats any fixed token
   window: no rule gets split in half and no chunk mixes two unrelated rules. Each
   chunk is prefixed with its policy title so the embedding has topical context.
3. **Embed** with `gemini-embedding-001` (768 dims, `RETRIEVAL_DOCUMENT` task type),
   normalised to unit length.
4. **Store** the vectors as BLOBs in the `kb_chunks` SQLite table, keyed by content
   hash — re-ingesting is free and editing one policy re-embeds only that file.
5. **Retrieve** by embedding the incoming ticket (`RETRIEVAL_QUERY`), then cosine
   KNN as a single numpy matrix-vector product. Top 8 rules by default (`TOP_K`).
6. **Ground** the model on those rules only, and drop any citation pointing at a
   document that was not retrieved.

## Design decisions

**Why no FAISS/Chroma/LangChain.** The knowledge base is ~30 short rules. Exact
cosine search over a 30×768 numpy array takes microseconds; an ANN index and a
framework would be pure dependency weight. `numpy` plus a SQLite BLOB column is the
whole vector store.

**Why one chunk per rule, not a token window.** The policies *are* a list of atomic
rules. Splitting on that boundary means a retrieved chunk is always a complete,
independently-true statement — which is exactly what you want to hand a model that
must not interpolate.

**Why RAG and not CAG here.** The full knowledge base is small enough to fit in a
prompt, so caching it would work. Retrieval is kept because it is what makes
`sources` meaningful: a decision cites the rules that were actually selected for it,
which is what makes a wrong answer debuggable — bad retrieval versus bad reasoning,
and the evaluation runner prints both. At ten times the policy count, this is also
the version that still works.

**Why sqlite3 and not SQLAlchemy.** Three tables and about a dozen queries. The
stdlib driver is less code than the ORM configuration would be.

**Why the action glossary in the prompt describes what actions do, not when they
apply.** The eligibility conditions come from the retrieved rules. Putting thresholds
and time windows in the system prompt would make retrieval decorative and the
citations dishonest.

**Failure handling, and why "no answer" is not an action.** `temperature=0` and
Gemini's `response_schema` constrain generation to the `Decision` model; the result is
re-validated with Pydantic, so an action outside the fifteen-item vocabulary cannot
reach the database.

`NEEDS_MORE_INFORMATION` means the *model* decided the facts are too thin. An
unreachable API, an exhausted quota or unparseable output means the model did not
decide at all, and that raises `DecisionUnavailable` instead. Collapsing the two would
be the easy shortcut and is the wrong one: it persists a decision nobody made, and it
turns an outage into a fake accuracy number in the evaluation runner. So `/tickets`
returns 503 and writes nothing — a ticket row with no decision is a worse artefact
than no ticket, and a guessed decision is worse than both — and `src/evaluate.py`
counts errored cases separately from wrong ones, aborting after three in a row.

**Security.** Passwords are bcrypt-hashed with a per-password salt and capped at 72
bytes (bcrypt silently truncates past that, which would let two different long
passwords open the same account). Login returns the same 401 for an unknown email
and a wrong password, so the endpoint is not a user-enumeration oracle. Authorization
is enforced in the SQL predicate: `/tickets/{id}` filters on `user_id`, so another
user's ticket is indistinguishable from one that does not exist — 404, not 403.
Covered by `test_alice_cannot_read_bobs_ticket`.

## Known limits

- The vector store is loaded from SQLite on every query. Fine at 30 chunks; cache the
  matrix in memory if the knowledge base grows past a few thousand.
- Tokens are not revocable before expiry (no refresh tokens, no denylist) — out of
  scope for this exercise.
- No rate limiting on `/tickets`, which is the endpoint that costs money. The free
  Gemini tier's own 20-requests-per-day cap is currently doing that job by accident.
- `TOP_K` is 8 by judgement rather than a sweep; tuning it needs more daily quota than
  the free tier allows.

## Layout

```
├── README.md
├── DEVELOPMENT.md          AI coding-agent usage
├── requirements.txt
├── .env.example
├── data/
│   ├── tickets.csv         historical tickets (extended evaluation set)
│   └── DATA_NOTES.md
├── knowledge_base/         the six policy documents
├── sample_test_cases.json
├── src/
│   ├── api.py              FastAPI app and endpoints
│   ├── auth.py             bcrypt hashing, JWT issue/verify, auth dependency
│   ├── database.py         schema and connection helper
│   ├── decision.py         prompt construction, Gemini call, grounding
│   ├── retrieval.py        chunk, embed, store, KNN  (also: python -m src.retrieval)
│   ├── evaluate.py         accuracy runner
│   └── schemas.py          Pydantic models and the action vocabulary
├── streamlit_app.py
└── tests/
    ├── conftest.py         offline fixtures
    ├── test_auth.py        authentication and authorization
    ├── test_api.py         endpoints and persistence
    └── test_rag.py         chunking, retrieval, grounding, failure handling
```
