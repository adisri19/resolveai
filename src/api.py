"""FastAPI backend. Streamlit talks to this over HTTP; nothing but this process
touches the database."""
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse

from .auth import authenticate, create_token, create_user, current_user
from .database import connect, init_db
from .decision import DecisionUnavailable, decide
from .schemas import Credentials, Decision, TicketIn, TicketOut, Token, User

load_dotenv()

TICKET_FIELDS = (
    "message", "order_value_inr", "days_since_delivery", "days_since_dispatch",
    "product_type", "opened_status", "order_status",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        from .retrieval import ingest

        ingest(verbose=True)
    except Exception as exc:  # noqa: BLE001
        # Auth and history still work without the vector store; only /tickets needs it.
        print(f"[warn] knowledge-base ingest skipped: {exc}")
    yield


app = FastAPI(title="Support Ticket Decision API", version="1.0", lifespan=lifespan)
STATIC_INDEX = Path(__file__).resolve().parent.parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/api/index.py", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    if STATIC_INDEX.exists():
        return FileResponse(STATIC_INDEX)
    return HTMLResponse("<h1>ResolveAI API</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


_TICKET_QUERY = """
SELECT t.id, t.message, t.created_at, d.action, d.reason, d.confidence, d.sources
FROM tickets t LEFT JOIN decisions d ON d.ticket_id = t.id
"""


def _row_to_ticket(row) -> TicketOut:
    decision = None
    if row["action"] is not None:
        decision = Decision(
            action=row["action"],
            reason=row["reason"],
            confidence=row["confidence"],
            sources=json.loads(row["sources"]),
        )
    return TicketOut(id=row["id"], message=row["message"], created_at=row["created_at"], decision=decision)


@app.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
def register(creds: Credentials):
    user_id = create_user(creds.email, creds.password)
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return User(id=row["id"], email=row["email"], created_at=row["created_at"])


@app.post("/login", response_model=Token)
def login(creds: Credentials):
    user = authenticate(creds.email, creds.password)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return Token(access_token=create_token(user["id"]))


@app.get("/me", response_model=User)
def me(user=Depends(current_user)):
    return User(id=user["id"], email=user["email"], created_at=user["created_at"])


@app.post("/tickets", response_model=TicketOut, status_code=status.HTTP_201_CREATED)
def submit_ticket(ticket: TicketIn, user=Depends(current_user)):
    try:
        decision, _ = decide(ticket)
    except DecisionUnavailable as exc:
        # Nothing is written: a ticket with no decision is a worse artefact than no
        # ticket, and a guessed decision is worse than both. The caller retries.
        print(f"[error] decision unavailable for user {user['id']}: {exc}")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The decision service is temporarily unavailable. Please try again.",
        )
    except RuntimeError as exc:  # misconfiguration, e.g. empty knowledge base
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))

    payload = ticket.model_dump()
    with connect() as conn:
        cur = conn.execute(
            f"INSERT INTO tickets (user_id, {', '.join(TICKET_FIELDS)}) "
            f"VALUES (?, {', '.join('?' * len(TICKET_FIELDS))})",
            (user["id"], *(payload[f] for f in TICKET_FIELDS)),
        )
        ticket_id = cur.lastrowid
        conn.execute(
            "INSERT INTO decisions (ticket_id, action, reason, confidence, sources) VALUES (?, ?, ?, ?, ?)",
            (ticket_id, decision.action, decision.reason, decision.confidence, json.dumps(decision.sources)),
        )
        row = conn.execute(_TICKET_QUERY + " WHERE t.id = ?", (ticket_id,)).fetchone()
    return _row_to_ticket(row)


@app.get("/tickets", response_model=List[TicketOut])
def list_tickets(user=Depends(current_user)):
    with connect() as conn:
        rows = conn.execute(
            _TICKET_QUERY + " WHERE t.user_id = ? ORDER BY t.id DESC", (user["id"],)
        ).fetchall()
    return [_row_to_ticket(r) for r in rows]


@app.get("/tickets/{ticket_id}", response_model=TicketOut)
def get_ticket(ticket_id: int, user=Depends(current_user)):
    with connect() as conn:
        # The user_id predicate is the authorization check: another user's ticket is
        # indistinguishable from one that does not exist.
        row = conn.execute(
            _TICKET_QUERY + " WHERE t.id = ? AND t.user_id = ?", (ticket_id, user["id"])
        ).fetchone()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found")
    return _row_to_ticket(row)
