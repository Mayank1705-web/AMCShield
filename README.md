# AMCShield — FastAPI + Frontend + GitHub Authentication

This package keeps the existing AMCShield frontend and project layout, and adds a FastAPI authentication layer.

## Project layout

```text
AMCShield/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── routers/            # auth, dashboard, dataset, training, attacks, ...
│   └── services/
├── frontend/
│   └── AMCShield_All_In_One/   # the only UI copy, served by the backend
│       ├── index.html
│       ├── login.js
│       ├── security.js         # CSRF-aware fetch helper
│       └── ...
├── tests/                  # frontend/backend contract tests
├── scripts/
├── src/                    # research pipeline (not touched by the web layer)
├── results/                # precomputed evaluation artifacts
├── checkpoints/
├── data/                   # local datasets and the SQLite auth DB (git-ignored)
└── .env.example
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

## Environment file

Never commit `.env`. Create it from the template and fill in the secrets:

```powershell
copy .env.example .env
python -c "import secrets;print(secrets.token_urlsafe(48))"   # use as SESSION_SECRET
```

`SESSION_SECRET` (at least 32 characters) is required; the app refuses to start without it.

## Required local data files

The dataset pages need files that are git-ignored because of their size:

```text
data/raw/GOLD_XYZ_OSC.0001_1024.hdf5
data/raw/classes-fixed.json
data/processed/*_indices.npy
```

Without them `/api/dataset/metadata`, `/class-distribution`, `/snr-distribution`,
`/splits` and `/sample` return **HTTP 503** and the Dataset page shows
"unavailable" placeholders. This is expected, not a bug.

## API routes

"Signed-in" means a valid session is required; "Admin" additionally requires
the administrator role; "CSRF" means the `X-CSRF-Token` header (from
`GET /api/auth/csrf`) is required. Responses use `{"success": true, "data": ...}`.

| Method | Path | Auth | Used by page |
|---|---|---|---|
| `POST` | `/api/account/invite/complete` | Public + CSRF | invite |
| `GET` | `/api/account/invite/validate` | Public | invite |
| `GET` | `/api/admin/messages` | Admin | admin |
| `GET` | `/api/admin/messages/{message_id}` | Admin | — |
| `POST` | `/api/admin/messages/{message_id}/approve` | Admin + CSRF | admin |
| `POST` | `/api/admin/messages/{message_id}/close` | Admin + CSRF | admin |
| `POST` | `/api/admin/messages/{message_id}/resend` | Admin + CSRF | admin |
| `POST` | `/api/admin/users/invite` | Admin + CSRF | — |
| `GET` | `/api/attacks/config` | Signed-in | adversarial-attacks |
| `GET` | `/api/attacks/constraints` | Signed-in | adversarial-attacks |
| `GET` | `/api/attacks/logs` | Signed-in | adversarial-attacks |
| `GET` | `/api/attacks/methods` | Signed-in | adversarial-attacks |
| `GET` | `/api/attacks/overview` | Signed-in | adversarial-attacks |
| `GET` | `/api/attacks/results` | Signed-in | adversarial-attacks |
| `POST` | `/api/attacks/run` | Admin + CSRF | adversarial-attacks |
| `GET` | `/api/attacks/snr-profile` | Signed-in | adversarial-attacks |
| `POST` | `/api/attacks/stop` | Admin + CSRF | adversarial-attacks |
| `POST` | `/api/auth/contact-admin` | Public + CSRF | login |
| `GET` | `/api/auth/csrf` | Public | admin, invite, security |
| `GET` | `/api/auth/github/callback` | Public | — |
| `GET` | `/api/auth/github/login` | Public | login |
| `POST` | `/api/auth/login` | Public | login |
| `POST` | `/api/auth/logout` | Public + CSRF | about, admin, adversarial-attacks, dashboard, dataset, evaluation-results, metrics-reference, model-training, reports, visualization |
| `GET` | `/api/auth/me` | Public | admin, auth-guard, dashboard, model-training, nav-admin |
| `GET` | `/api/dashboard/attacks` | Signed-in | dashboard |
| `GET` | `/api/dashboard/generalization-gap` | Signed-in | dashboard |
| `GET` | `/api/dashboard/performance` | Signed-in | dashboard |
| `GET` | `/api/dashboard/recent-updates` | Signed-in | dashboard |
| `GET` | `/api/dashboard/snr` | Signed-in | dashboard |
| `GET` | `/api/dashboard/summary` | Signed-in | dashboard |
| `GET` | `/api/dataset/class-distribution` | Signed-in | dataset |
| `GET` | `/api/dataset/metadata` | Signed-in | dataset |
| `GET` | `/api/dataset/overview` | Signed-in | — |
| `GET` | `/api/dataset/quality` | Signed-in | dataset |
| `GET` | `/api/dataset/sample` | Signed-in | dataset |
| `GET` | `/api/dataset/snr-distribution` | Signed-in | dataset |
| `GET` | `/api/dataset/splits` | Signed-in | dataset |
| `GET` | `/api/evaluation/accuracy-by-snr` | Signed-in | evaluation-results |
| `GET` | `/api/evaluation/artifacts` | Signed-in | — |
| `GET` | `/api/evaluation/attack-summary` | Signed-in | dashboard, evaluation-results |
| `GET` | `/api/evaluation/comparison` | Signed-in | evaluation-results |
| `GET` | `/api/evaluation/filters` | Signed-in | evaluation-results |
| `GET` | `/api/evaluation/generalization-gap` | Signed-in | evaluation-results |
| `GET` | `/api/evaluation/insights` | Signed-in | evaluation-results |
| `GET` | `/api/evaluation/results` | Signed-in | evaluation-results |
| `GET` | `/api/evaluation/summary` | Signed-in | evaluation-results |
| `GET` | `/api/health` | Public | metrics-reference |
| `GET` | `/api/project/attacks` | Signed-in | about |
| `GET` | `/api/project/context` | Signed-in | about |
| `GET` | `/api/project/facts` | Signed-in | about |
| `GET` | `/api/project/mission` | Signed-in | about |
| `GET` | `/api/project/models` | Signed-in | about |
| `GET` | `/api/project/overview` | Signed-in | about |
| `GET` | `/api/project/resources` | Signed-in | about |
| `GET` | `/api/reports/artifacts` | Signed-in | reports |
| `GET` | `/api/reports/checks` | Signed-in | reports |
| `GET` | `/api/reports/executive-summary` | Signed-in | reports |
| `GET` | `/api/reports/export` | Signed-in | reports |
| `GET` | `/api/reports/findings` | Signed-in | reports |
| `POST` | `/api/reports/generate` | Admin + CSRF | reports |
| `GET` | `/api/reports/methodology` | Signed-in | reports |
| `GET` | `/api/reports/metrics` | Signed-in | reports |
| `GET` | `/api/reports/options` | Signed-in | reports |
| `POST` | `/api/reports/preview` | Admin + CSRF | reports |
| `GET` | `/api/reports/summary` | Signed-in | reports |
| `GET` | `/api/training/checkpoints` | Signed-in | model-training |
| `GET` | `/api/training/config` | Signed-in | model-training |
| `GET` | `/api/training/history` | Signed-in | model-training |
| `GET` | `/api/training/logs` | Signed-in | model-training |
| `GET` | `/api/training/models` | Signed-in | model-training |
| `GET` | `/api/training/overview` | Signed-in | model-training |
| `GET` | `/api/training/progress` | Signed-in | model-training |
| `POST` | `/api/training/resume` | Admin + CSRF | model-training |
| `POST` | `/api/training/start` | Admin + CSRF | model-training |
| `GET` | `/api/training/status` | Signed-in | — |
| `POST` | `/api/training/stop` | Admin + CSRF | model-training |
| `GET` | `/api/visualization/accuracy-by-snr` | Signed-in | visualization |
| `GET` | `/api/visualization/artifacts` | Signed-in | — |
| `GET` | `/api/visualization/asr-by-snr` | Signed-in | visualization |
| `GET` | `/api/visualization/class-accuracy` | Signed-in | visualization |
| `GET` | `/api/visualization/filters` | Signed-in | visualization |
| `GET` | `/api/visualization/generalization-gap` | Signed-in | visualization |
| `GET` | `/api/visualization/insights` | Signed-in | visualization |
| `GET` | `/api/visualization/model-comparison` | Signed-in | visualization |
| `GET` | `/api/visualization/summary` | Signed-in | visualization |

`POST /api/attacks/run` and `/stop` intentionally return HTTP 501: attacks are
precomputed offline and never started from the dashboard.

## Tests

```powershell
python -m unittest tests.test_frontend_backend_contract -v
```

The test scans every API call in the UI, checks it against the FastAPI route
table, and verifies CSRF, admin gating and CSP-safe markup. It sets a random
`SESSION_SECRET` for its own process only and does not read or write `.env`.

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
