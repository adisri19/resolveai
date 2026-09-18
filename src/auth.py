"""Password hashing + JWT issue/verify, and the FastAPI dependency that turns a
bearer token back into a user row."""
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .database import connect

ALGORITHM = "HS256"
TOKEN_TTL_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

bearer = HTTPBearer(auto_error=False)


def _secret() -> str:
    # Read at call time so tests (and a late-loading .env) can set it.
    secret = os.getenv("JWT_SECRET")
    if not secret:
        raise RuntimeError("JWT_SECRET is not set. Copy .env.example to .env and fill it in.")
    return secret


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False


def create_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "iat": now, "exp": now + timedelta(minutes=TOKEN_TTL_MINUTES)}
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM)


def create_user(email: str, password: str) -> int:
    with connect() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                (email.lower(), hash_password(password)),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
        return cur.lastrowid


def authenticate(email: str, password: str):
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
    # Same 401 whether the email is unknown or the password is wrong -- don't
    # hand out a user-enumeration oracle.
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return row


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)):
    """Depend on this to make an endpoint require `Authorization: Bearer <JWT>`."""
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "Invalid or missing credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if creds is None:
        raise unauthorized
    try:
        payload = jwt.decode(creds.credentials, _secret(), algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise unauthorized

    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise unauthorized
    return row
