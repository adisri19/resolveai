# Development notes: AI coding-agent usage

This project was built with Claude Code (Opus 5) as a pair, not as an autopilot. This
file records how it was used, the decisions that needed a human call, what was
rejected, and how the generated code was verified.

## How it was used

The brief, the six policy documents, `sample_test_cases.json` and `tickets.csv` were
read in full before any code was written. The action vocabulary was not invented: it
is the fifteen distinct values of `resolved_action` in `tickets.csv`, so the system
can only emit actions the business already uses.

The modules were then written in dependency order — `database` → `schemas` → `auth`
→ `retrieval` → `decision` → `api` → `evaluate` → `streamlit_app` → tests — each one
read before the next was started. The working instruction to the agent throughout was
to stop at the simplest thing that holds: no abstraction without a second caller, no
dependency for what a few lines of stdlib does.

## Decisions that needed a human call

These are the points where the obvious generated answer and the right answer differ.

**Embeddings live in the database, not beside it.** The natural shape is a
`vectors.npy` file next to `app.db`. Two stores that can disagree about what has been
ingested is a bug waiting to happen, so vectors go in a `kb_chunks` table in the same
SQLite file, keyed by content hash. Ingestion is idempotent and a deleted policy rule
is removed from the store rather than retrieved forever. `test_ingest_is_idempotent`
pins that.

**The prompt describes what actions do, not when they apply.** Listing eligibility
conditions in the system prompt ("REQUEST_PHOTOS — damaged, above ₹2,000, within 7
days") would score well on the sample cases and be a lie: the policy would live in the
prompt, retrieval would be decorative, and `sources` would cite documents the model
never used. The glossary in `decision.py` is deliberately condition-free; every
threshold and time window has to come from a retrieved rule.

**Chunk on rule boundaries, not a token window.** The policies are already numbered
atomic rules, so a fixed-size window would only ever split one in half or merge two.
One chunk per rule means a retrieved chunk is always a complete, independently-true
statement. `test_chunks_are_one_rule_each_and_carry_their_source` asserts no chunk
merges two rules.

**Embeddings are explicitly normalised.** `gemini-embedding-001` output is unit-length
only at its native 3072 dimensions. This code requests 768 to keep the store small, so
vectors are normalised before storage — without that the dot product is not cosine
similarity and the ranking is quietly wrong rather than loudly broken.

**Passwords are capped at 72 bytes.** bcrypt truncates silently past that, which means
two different long passwords can open the same account. The cap is in
`Credentials.password` so the truncation cannot happen.

**Another user's ticket is a 404, not a 403.** A 403 confirms the ticket exists. The
ownership check is a predicate in the SQL rather than a branch after the fetch, so
there is no code path that loads someone else's row at all.

**A failed model call is not a `NEEDS_MORE_INFORMATION` decision.** This one was
caught by running the evaluation, not by reading the code. The first version caught
every generation failure and returned `NEEDS_MORE_INFORMATION` with confidence `0.0`
— which looks like graceful degradation and reads fine in review. Then the free-tier
daily quota ran out mid-run and the runner reported 20% accuracy: every quota error
had been silently scored as a wrong answer. The failure mode is worse inside the API,
where it would have persisted a decision that no model ever made.

Generation failures now raise `DecisionUnavailable`. `/tickets` turns that into a 503
and writes nothing; the evaluation runner counts those cases separately and aborts
after three in a row. `test_llm_failure_raises_instead_of_inventing_a_decision` and
`test_model_outage_returns_503_and_persists_nothing` pin both halves.

## Rejected

- **LangChain / LlamaIndex / FAISS / Chroma.** Thirty chunks. `numpy` and a SQLite
  BLOB column are the entire vector store; an ANN index over 30 vectors costs more in
  import time than the exact search costs to run.
- **SQLAlchemy.** Three tables, about a dozen queries. The model definitions would be
  longer than the SQL.
- **A `config.py` settings class.** Six environment variables, read with `os.getenv`
  where they are used.
- **Streaming the LLM response.** The output is a four-field JSON object.
- **A retry/backoff library.** One retry in a `for` loop, then raise.

## Verifying the agent's output

Generated code was not trusted on inspection alone.

- `pytest -q` — 24 tests, fully offline. Embeddings are swapped for a deterministic
  bag-of-words vectoriser so retrieval is genuinely exercised without the network;
  generation is stubbed per test.
- The tests that matter most are the negative ones: a forged JWT signed with the wrong
  secret is rejected, Alice gets a 404 on Bob's ticket, an action outside the
  vocabulary fails Pydantic validation before it can be stored, and a dead model
  produces a 503 with nothing written rather than an invented decision.
- `python -m src.evaluate` scores the pipeline against the supplied labelled cases and
  prints, for every failure, the model's reason, its citations and the documents
  retrieved — which separates a retrieval problem from a reasoning problem.
- A module-level assertion in `decision.py` fails at import if the prompt's action
  glossary ever drifts from `schemas.ACTIONS`.

## Measured result

`python -m src.evaluate` scores 5/5 on the supplied sample cases, reproduced on two
models: `gemini-3.6-flash` (the default) and `gemini-3.5-flash`. Retrieval picks the
correct policy document on all five — confirmed independently, because during the
quota failure above the runner still printed which documents had been retrieved for
every case.

The model default is `gemini-3.6-flash` rather than `gemini-2.5-flash`: the latter is
listed by `models.list()` but returns 404 on `generateContent` for new keys.

## Still open

- `TOP_K` is 8 by judgement, not by a sweep. The evaluation runner is the instrument
  for tuning it against `data/tickets.csv`.
- Accuracy on the full 214-row historical set has not been measured. The free Gemini
  tier allows 20 `generateContent` calls per project per model per day — and a new API
  key in the same AI Studio project shares that bucket — which is the binding
  constraint, not the code.
