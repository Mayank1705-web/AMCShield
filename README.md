# AMCShield — FastAPI + Frontend + GitHub Authentication

This package keeps the existing AMCShield frontend and project layout, and adds a FastAPI authentication layer.

## Project layout

```text
AMCShield/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── routers/
│   │   └── auth.py
│   └── services/
├── frontend/
│   └── AMCShield_All_In_One/
│       ├── index.html
│       ├── login.css
│       ├── login.js
│       ├── auth-guard.js
│       └── ...
├── src/
├── results/
├── checkpoints/
└── data/
```

## Install

From the AMCShield project root:

```powershell
pip install -r backend\requirements.txt
```

## Configure local GitHub OAuth

1. On GitHub, create an OAuth App for AMCShield.
2. Set the Authorization callback URL exactly to:

```text
http://127.0.0.1:8000/api/auth/github/callback
```

3. Copy the client ID and client secret into the project's `.env`:

```text
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...
GITHUB_REDIRECT_URI=http://127.0.0.1:8000/api/auth/github/callback
SESSION_SECRET=<long-random-secret>
```

Do **not** put the GitHub client secret in JavaScript or HTML.

## Run

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

## Authentication endpoints

```text
GET  /api/auth/github/login
GET  /api/auth/github/callback
POST /api/auth/login
GET  /api/auth/me
POST /api/auth/logout
```

### GitHub flow

```text
AMCShield Login
      ↓
Continue with GitHub
      ↓
/api/auth/github/login
      ↓
GitHub authorization
      ↓
/api/auth/github/callback
      ↓
AMCShield session cookie
      ↓
/dashboard.html
```

### Local development login

Set `ADMIN_PASSWORD` in `.env`. The existing Email/Username + Password form will then authenticate against the configured local admin identity. Credentials are never hardcoded into frontend JavaScript.

## Security notes

- The GitHub client secret stays server-side.
- OAuth `state` is generated and verified to prevent CSRF.
- The browser receives a signed session cookie, not the GitHub access token.
- For production, use HTTPS and set `https_only=True` for the session cookie.
- Replace the development `SESSION_SECRET` with a strong random secret.
- A production multi-user system should use a database-backed user/session model rather than the local `.env` admin credential.
