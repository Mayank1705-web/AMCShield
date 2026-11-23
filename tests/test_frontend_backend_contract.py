"""Frontend <-> backend contract tests for AMCShield.

Run from the repository root:

    python -m unittest tests.test_frontend_backend_contract -v

The test scans every API call made by the UI in
``frontend/AMCShield_All_In_One`` and checks that the FastAPI app really
exposes it (same path, same method).  It also exercises the live app with
``TestClient`` to verify status codes, CSRF enforcement and admin gating.

Only stdlib ``unittest`` and FastAPI's ``TestClient`` are used.  A random
``SESSION_SECRET`` and ``ADMIN_PASSWORD`` are set in ``os.environ`` for this
process only; no ``.env`` file is read or written, and any database or report
file created by the tests is removed again.
"""

from __future__ import annotations

import os
import re
import secrets
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "AMCShield_All_In_One"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "amcshield_auth.db"
GENERATED_DIR = ROOT / "reports" / "generated"

# Browser redirects that are not JSON API calls.
ALLOWED_UNMATCHED = {("GET", "/api/auth/github/login")}

# GET routes that cannot be called blindly: a redirect and a route that
# needs a one-time invitation token.  Their existence is still verified.
GET_SKIP = {"/api/auth/github/login", "/api/account/invite/validate"}

# Query strings the UI always appends for these routes.
GET_QUERY = {"/api/dataset/sample": "?class_name=BPSK&snr=10"}

QUOTED_API = re.compile(r"""(["'`])(/api/[^"'`\s]*)\1""")
RELATIVE_CALL = re.compile(
    r"""\b(api|fetchAPI|command)\(\s*(["'`])(/[^"'`\s]*)\2"""
)
ARRAY_ENTRY = re.compile(r"""\[\s*(["'`])(/[a-z][^"'`\s]*)\1\s*,""")
BASE_DEF = re.compile(
    r"""\bAPI_BASE(?:_URL)?\s*=\s*["'](/[^"']*)["']"""
)
METHOD = re.compile(r"""method\s*:\s*["'`](\w+)["'`]""", re.I)
STATIC_ASSETS = (".html", ".js", ".css", ".png", ".jpg", ".svg", ".json")


def _call_window(text: str, pos: int, limit: int = 1500) -> str:
    """Text from ``pos`` up to the end of the enclosing call/statement."""
    depth = 0
    i = pos
    stop = min(len(text), pos + limit)
    while i < stop:
        ch = text[i]
        if ch in "\"'`":
            quote = ch
            i += 1
            while i < stop and text[i] != quote:
                if text[i] == "\\":
                    i += 1
                i += 1
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                return text[pos:i]
            depth -= 1
        elif ch == ";" and depth == 0:
            return text[pos:i]
        i += 1
    return text[pos:stop]


def _infer_method(text: str, end: int, default: str = "GET") -> str:
    found = METHOD.search(_call_window(text, end))
    return found.group(1).upper() if found else default


def _normalise(path: str) -> str:
    path = re.sub(r"\$\{[^}]*\}", "{}", path)
    path = re.sub(r"\{[^}]*\}", "{}", path)
    return path.split("?", 1)[0].split("#", 1)[0]


def scan_frontend_calls() -> dict[tuple[str, str], set[str]]:
    """Return {(METHOD, "/api/path"): {source files}} for every UI API call."""
    calls: dict[tuple[str, str], set[str]] = {}

    def add(method: str, path: str, source: str) -> None:
        calls.setdefault((method, _normalise(path)), set()).add(source)

    files = sorted(FRONTEND.glob("*.js")) + sorted(FRONTEND.glob("*.html"))
    for file in files:
        text = file.read_text(encoding="utf-8", errors="replace")
        name = file.name
        base_match = BASE_DEF.search(text)
        base = base_match.group(1).rstrip("/") if base_match else None

        # 1. Literal "/api/..." strings.
        for match in QUOTED_API.finditer(text):
            path, end = match.group(2), match.end()
            const_name = None
            prefix = text[max(0, match.start() - 80):match.start()]
            declared = re.search(r"(?:const|let|var)\s+(\w+)\s*=\s*$", prefix)
            if declared:
                const_name = declared.group(1)
            if const_name:
                usages = [
                    m for m in re.finditer(rf"\b{re.escape(const_name)}\b", text)
                    if not re.search(r"(?:const|let|var)\s+$", text[max(0, m.start() - 8):m.start()])
                ]
                if usages:
                    for usage in usages:
                        add(_infer_method(text, usage.end()), path, name)
                else:
                    add("GET", path, name)
            else:
                add(_infer_method(text, end), path, name)

        if base is None:
            continue

        # 2. api("/x") / fetchAPI("/x") / command("/x") relative to the base.
        for match in RELATIVE_CALL.finditer(text):
            func, path, end = match.group(1), match.group(3), match.end()
            if path.startswith("/api/"):
                continue  # already handled above
            default = "POST" if func == "command" else "GET"
            add(_infer_method(text, end, default), base + path, name)

        # 3. [["/x", renderer], ...] endpoint tables.
        for match in ARRAY_ENTRY.finditer(text):
            path = match.group(2)
            if path.startswith("/api/") or path.endswith(STATIC_ASSETS):
                continue
            add("GET", base + path, name)

    return calls


class FrontendBackendContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_env = {
            key: os.environ.get(key) for key in ("SESSION_SECRET", "ADMIN_PASSWORD")
        }
        cls.bootstrap_pw = secrets.token_urlsafe(24)
        os.environ["SESSION_SECRET"] = secrets.token_urlsafe(48)
        os.environ["ADMIN_PASSWORD"] = cls.bootstrap_pw

        cls._data_dir_existed = DATA_DIR.exists()
        cls._db_existed = DB_PATH.exists()
        cls._generated_before = (
            {p.name for p in GENERATED_DIR.glob("*")} if GENERATED_DIR.is_dir() else set()
        )
        cls._generated_dir_existed = GENERATED_DIR.exists()
        cls._test_user_ids: list[int] = []

        from fastapi.testclient import TestClient

        from backend.main import app

        cls.app = app
        cls.TestClient = TestClient
        cls.client = TestClient(app)
        cls.openapi_paths = {
            _normalise(path): {m.upper() for m in methods}
            for path, methods in app.openapi()["paths"].items()
        }
        cls.calls = scan_frontend_calls()

    @classmethod
    def tearDownClass(cls):
        try:
            from backend.database.db import SessionLocal, engine
            from backend.database.models import User

            if cls._test_user_ids:
                with SessionLocal() as db:
                    for user_id in cls._test_user_ids:
                        user = db.get(User, user_id)
                        if user is not None:
                            db.delete(user)
                    db.commit()
            engine.dispose()
        finally:
            if not cls._db_existed:
                for suffix in ("", "-journal", "-wal", "-shm"):
                    path = Path(str(DB_PATH) + suffix)
                    if path.exists():
                        path.unlink()
            if not cls._data_dir_existed and DATA_DIR.is_dir() and not any(DATA_DIR.iterdir()):
                DATA_DIR.rmdir()

            if GENERATED_DIR.is_dir():
                for path in GENERATED_DIR.glob("*"):
                    if path.name not in cls._generated_before:
                        path.unlink()
                if not cls._generated_dir_existed and not any(GENERATED_DIR.iterdir()):
                    GENERATED_DIR.rmdir()

            for key, value in cls._old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    # ------------------------------------------------------------ helpers

    def _login_admin(self, client):
        response = client.post(
            "/api/auth/login",
            json={"identity": "admin", "password": self.bootstrap_pw},
        )
        self.assertEqual(response.status_code, 200, response.text)

    def _csrf(self, client) -> dict[str, str]:
        response = client.get("/api/auth/csrf")
        self.assertEqual(response.status_code, 200, response.text)
        return {"X-CSRF-Token": response.json()["csrf_token"]}

    def _admin_client(self):
        client = self.TestClient(self.app)
        self._login_admin(client)
        return client

    # -------------------------------------------------------------- tests

    def test_01_frontend_calls_exist_in_backend(self):
        self.assertGreater(len(self.calls), 40, "Scanner found too few API calls.")
        unmatched = []
        for (method, path), sources in sorted(self.calls.items()):
            if (method, path) in ALLOWED_UNMATCHED:
                continue
            if path.endswith("/"):  # dynamic suffix, e.g. "/visualization/" + view
                if not any(p.startswith(path) for p in self.openapi_paths):
                    unmatched.append(f"{method} {path}* ({', '.join(sorted(sources))})")
                continue
            methods = self.openapi_paths.get(path)
            if methods is None or method not in methods:
                unmatched.append(f"{method} {path} ({', '.join(sorted(sources))})")
        self.assertEqual(
            unmatched, [],
            "Frontend calls with no matching backend route:\n  " + "\n  ".join(unmatched),
        )

    def test_02_every_frontend_get_is_healthy(self):
        client = self._admin_client()
        failures = []
        checked = 0
        for (method, path), _sources in sorted(self.calls.items()):
            if method != "GET" or "{}" in path or path.endswith("/") or path in GET_SKIP:
                continue
            checked += 1
            response = client.get(path + GET_QUERY.get(path, ""))
            ok = response.status_code == 200 or (
                response.status_code == 503 and path.startswith("/api/dataset/")
            )
            if not ok:
                failures.append(f"GET {path} -> {response.status_code} {response.text[:120]}")
        self.assertGreater(checked, 30)
        self.assertEqual(failures, [], "\n  ".join(failures))

    def test_03_logout_requires_csrf_and_clears_session(self):
        client = self._admin_client()
        denied = client.post("/api/auth/logout")
        self.assertEqual(denied.status_code, 403)
        self.assertEqual(client.get("/api/auth/me").status_code, 200)

        allowed = client.post("/api/auth/logout", headers=self._csrf(client))
        self.assertEqual(allowed.status_code, 200, allowed.text)
        self.assertEqual(client.get("/api/dashboard/summary").status_code, 401)

    def test_04_live_attack_execution_is_disabled(self):
        client = self._admin_client()
        headers = self._csrf(client)
        for path in ("/api/attacks/run", "/api/attacks/stop"):
            self.assertEqual(client.post(path).status_code, 403, path)
            response = client.post(path, headers=headers)
            self.assertEqual(response.status_code, 501, path)
            self.assertIn("precomputed offline", response.json()["detail"])

    def test_05_report_generation_is_validated_and_csrf_protected(self):
        client = self._admin_client()
        headers = self._csrf(client)
        body = {"report_type": "executive", "scope": "all", "format": "json", "version": "latest"}
        self.assertEqual(client.post("/api/reports/generate", json=body).status_code, 403)
        self.assertEqual(client.post("/api/reports/preview", json=body).status_code, 403)

        bad = dict(body, report_type="../../etc/passwd")
        self.assertEqual(
            client.post("/api/reports/generate", json=bad, headers=headers).status_code, 422
        )
        good = client.post("/api/reports/generate", json=body, headers=headers)
        self.assertEqual(good.status_code, 200, good.text)
        self.assertEqual(good.json()["data"]["status"], "completed")
        preview = client.post("/api/reports/preview", json=body, headers=headers)
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertNotIn("url", preview.json()["data"])

        self.assertEqual(client.get("/api/reports/export?format=json").status_code, 200)
        self.assertEqual(client.get("/api/reports/export?format=csv").status_code, 400)

    def test_06_non_admin_can_view_but_not_control_training(self):
        from backend.database.db import SessionLocal, init_db
        from backend.database.models import User
        from backend.services.security_service import hash_password

        init_db()
        email = f"contract-test-{secrets.token_hex(4)}@example.invalid"
        password = secrets.token_urlsafe(24)
        with SessionLocal() as db:
            user = User(
                name="Contract Test",
                email=email,
                password_hash=hash_password(password),
                role="user",
                status="active",
                email_verified=True,
            )
            db.add(user)
            db.commit()
            self.__class__._test_user_ids.append(user.id)

        client = self.TestClient(self.app)
        login = client.post("/api/auth/login", json={"identity": email, "password": password})
        if login.status_code != 200:
            self.skipTest(f"Could not create a non-admin session cheaply ({login.status_code}).")

        self.assertEqual(client.get("/api/training/overview").status_code, 200)
        headers = self._csrf(client)
        for path in ("/api/training/start", "/api/training/resume", "/api/training/stop"):
            self.assertEqual(client.post(path, headers=headers).status_code, 403, path)
        self.assertEqual(client.post("/api/attacks/run", headers=headers).status_code, 403)

    def test_07_csp_header_is_strict(self):
        csp = self.client.get("/api/health").headers.get("content-security-policy", "")
        self.assertIn("script-src 'self'", csp)
        self.assertIn("style-src 'self' https://fonts.googleapis.com", csp)
        self.assertNotIn("unsafe-inline", csp)

    def test_08_no_inline_styles_or_handlers_in_frontend(self):
        offenders = []
        for file in sorted(FRONTEND.glob("*.js")) + sorted(FRONTEND.glob("*.html")):
            for number, line in enumerate(
                file.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if 'style="' in line or "style='" in line:
                    offenders.append(f"{file.name}:{number}: inline style attribute")
                if re.search(r"\son[a-z]+\s*=", line):
                    offenders.append(f"{file.name}:{number}: inline event handler")
                if "<style" in line.lower() or "javascript:" in line.lower():
                    offenders.append(f"{file.name}:{number}: inline <style>/javascript: URL")
        self.assertEqual(offenders, [], "CSP violations:\n  " + "\n  ".join(offenders))


if __name__ == "__main__":
    unittest.main()
