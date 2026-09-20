import os
import sys
from pathlib import Path

# Ensure root directory is on the python search path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

os.environ["VERCEL"] = "1"

from src.api import app as fastapi_app


class VercelPathFixMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            path = scope.get("path", "")
            if path.startswith("/api/index.py"):
                sub = path[len("/api/index.py"):]
                if not sub or sub == "/":
                    scope["path"] = "/"
                else:
                    scope["path"] = sub if sub.startswith("/") else f"/{sub}"
        await self.app(scope, receive, send)


app = VercelPathFixMiddleware(fastapi_app)
