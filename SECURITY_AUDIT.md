# AMCShield Security Audit

## 1. Executive summary

This audit hardened the supplied AMCShield backend and frontend archives across the requested Phase 2–Phase 9 security areas. Changes were targeted to the existing FastAPI + SQLAlchemy + vanilla JavaScript architecture.

The most significant original finding was that GitHub OAuth authenticated a GitHub profile and immediately created an AMCShield session without checking whether the verified GitHub identity belonged to an existing active AMCShield account. That flow was changed so GitHub authentication is followed by verified-email lookup, existing-account lookup, active-status validation, and database-derived role assignment.

The research/application split was preserved. No model architecture, attack implementation, dataset, checkpoint, or experimental result was changed. The uploaded archives did not contain `src/`, `checkpoints/`, `results/`, or research `data/` trees, so those artifacts could not be modified or regenerated.

## 2. Architecture reviewed

### Backend
- FastAPI application in `backend/main.py`
- Starlette signed `SessionMiddleware`
- SQLAlchemy database models and SQLite auth database
- Authentication and CSRF helpers in `backend/services/security_service.py`
- Authentication in `backend/routers/auth.py`
- Administrative routes in `backend/routers/admin.py` and `admin_users.py`
- Invitation/password setup in `backend/routers/invitations.py`
- Dataset/training/attack/evaluation/visualization/report/project read APIs

### Frontend
- Active served tree: `frontend/AMCShield_All_In_One/`
- Duplicate outer frontend tree was compared with the served tree and kept synchronized
- Vanilla HTML/CSS/JavaScript
- Central CSRF helper in `security.js`
- Client-side authentication guards retained as UX controls only

## 3. Phase 2 — GitHub OAuth

### Findings and changes
- OAuth state already used `secrets.token_urlsafe()` and `secrets.compare_digest()`.
- State is now single-use and has a bounded lifetime using `GITHUB_STATE_MAX_AGE`.
- OAuth state and its timestamp are removed from the session before callback processing.
- GitHub email is obtained from the GitHub email endpoint rather than trusting an arbitrary client value or only the public profile email.
- A verified email is required.
- GitHub users must map to an existing AMCShield database user.
- Pending/disabled/unknown accounts are rejected.
- AMCShield account status and role come from the database.
- GitHub access tokens are not stored in the browser.
- Successful OAuth replaces the pre-authentication session state.

### Verification
- Missing OAuth state: HTTP 302 to a safe error page.
- Invalid OAuth state: HTTP 302 to a safe error page.
- LIVE GitHub OAuth: **NOT TESTED** — no live GitHub credentials were used during local verification.

## 4. Phase 3 — Sessions

- Session secret is environment-driven and must be at least 32 characters.
- Session cookie uses `HttpOnly` by Starlette, `SameSite=Lax`, configurable `Secure`, and absolute `max_age`.
- Production configuration requires `SESSION_HTTPS_ONLY=true`.
- Production configuration requires HTTPS application and OAuth URLs.
- Login replaces session contents using `request.session.clear()`.
- Logout requires CSRF and clears the session.
- Protected requests reject stale/disabled database sessions.
- HTML and API responses receive `no-store` cache controls.

### Verification
- Login: PASS
- Authenticated request after login: PASS
- Logout without CSRF: rejected (403)
- Logout with invalid CSRF: rejected (403)
- Logout with valid CSRF: PASS
- Authenticated request after logout: rejected (401)

## 5. Phase 4 — RBAC / authorization

All existing research/data read APIs are now protected by a centralized authenticated-session dependency.

### Public
- `GET /health`
- `GET /api/health`
- `GET /api/auth/github/login`
- `GET /api/auth/github/callback`
- `GET /api/auth/csrf`
- `POST /api/auth/login`
- `POST /api/auth/contact-admin` with CSRF
- `GET /api/account/invite/validate`
- `POST /api/account/invite/complete` with CSRF

### Authenticated
- `GET /api/auth/me`
- `/api/dashboard/*`
- `/api/dataset/*`
- `/api/training/*`
- `/api/attacks/*`
- `/api/evaluation/*`
- `/api/visualization/*`
- `/api/reports/*`
- `/api/project/*`

### Admin
- `/api/admin/*`
- `/api/admin/users/*`
- `/admin.html`
- `/admin`
- `/admin.js`
- `/admin.css`

### Verification
- Anonymous research API: HTTP 401
- Anonymous admin API: HTTP 401
- Normal user research API: allowed
- Normal user admin API: HTTP 403
- Normal user admin page: redirected
- Bootstrap administrator admin API: allowed
- Database administrator admin API: allowed

Roles are not accepted from request bodies. Protected admin authorization uses trusted server-side database state.

## 6. Phase 5 — Security headers / CSP

Implemented:
- `Content-Security-Policy`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- restrictive `Permissions-Policy`
- conditional `Strict-Transport-Security`
- `Cache-Control: no-store` for API and HTML responses

CSP was tightened from inline-style allowance to a policy that permits only self-hosted scripts/styles plus the existing Google Fonts resources. The admin inline `<style>` block was moved into `admin.css`.

Production also disables FastAPI Swagger/OpenAPI/ReDoc endpoints.

## 7. Phase 6 — Authentication / invitation / abuse

### Passwords
- Argon2id password hashing retained.
- 12–128 character password policy.
- At least three character classes.
- No plaintext password logging or password URLs.

### Login abuse
- Independent identity and source-IP rate-limit buckets.
- Successful authentication clears relevant failure buckets.
- Login enumeration was reduced by returning the same generic authentication error for unknown, pending, and disabled accounts.

### Invitations
- Cryptographically random tokens.
- SHA-256 token hashes stored in the database.
- Expiration.
- Atomic one-time claiming.
- Previous active tokens invalidated where applicable.
- Frontend invitation token is held in memory and removed from the URL immediately after parsing.
- CSRF required for completion.

### Verification
- Six failed-login sequence activated HTTP 429 protection.
- Invitation without CSRF: HTTP 403.
- First invitation completion: PASS.
- Replay of used invitation: HTTP 409.
- Contact-admin without CSRF: HTTP 403.
- Contact-admin with valid CSRF: PASS.

## 8. Phase 7 — Dependency / application security

### Static application audit
Searched for dangerous patterns including:
- `eval`
- `exec`
- `pickle.loads`
- `os.system`
- `shell=True`
- unsafe `yaml.load`

No occurrences requiring remediation were found in the supplied backend.

### Filesystem security
- Existing results path helper already enforced resolved-path containment.
- Research/data/training metadata endpoints no longer disclose absolute server filesystem paths.
- Frontend static serving is restricted to the configured frontend directory.
- Sensitive path requests such as `.env`, backend source, database path, and checkpoints path returned HTTP 404.

### Frontend XSS
- Existing escaped rendering was retained.
- Dynamic report/visualization/profile rendering was hardened with HTML escaping.
- External resource URLs in the About page are protocol-validated before insertion.
- Same-origin API URLs replaced hard-coded `http://localhost:8000/api` values.

### Dependency scanner
`pip-audit` was **NOT TESTED / NOT AVAILABLE** in the execution environment (`No module named pip_audit`). Dependencies were reviewed statically; no blind upgrades were performed.

## 9. Phase 8 — Production hardening

- Explicit CORS origins retained; wildcard CORS is rejected by configuration validation.
- Credentials remain enabled only with explicit origins.
- Production requires secure cookies and HTTPS application/OAuth URLs.
- Secrets are environment-driven.
- `.env.example` added with placeholders only.
- `.gitignore` added for `.env`, Python caches, virtual environments, logs, and the local authentication DB.
- Sensitive exception details are not returned to OAuth clients.
- GitHub OAuth logging now records only the exception type.
- Production API documentation is disabled.
- Static frontend access does not expose backend/data/checkpoint directories.

## 10. Phase 9 — Verification

| Test | Result | Evidence |
| --- | --- | --- |
| Backend syntax | PASS | `python -m compileall -q backend` |
| Frontend JavaScript syntax | PASS | `node --check` on all frontend JS files |
| Uvicorn startup | PASS | `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8017`; `/health` returned 200 |
| Anonymous protected API | PASS | HTTP 401 |
| Normal user login | PASS | HTTP 200 |
| Normal user research API | PASS | HTTP 200 |
| Normal user → admin API | PASS | HTTP 403 |
| Admin login/API | PASS | HTTP 200 |
| Admin asset protection | PASS | anonymous admin page redirected; admin JS returned 404; admin session received 200 |
| CSRF missing/invalid | PASS | HTTP 403 |
| CSRF valid | PASS | logout/contact/invitation operations accepted |
| Logout invalidation | PASS | post-logout `/api/auth/me` and protected API returned 401 |
| Invitation first use | PASS | HTTP 200 |
| Invitation replay | PASS | HTTP 409 |
| Login rate limiting | PASS | sequence `[401, 401, 401, 401, 429, 429]` |
| OAuth state checks | PASS | missing/invalid state redirected with safe error |
| LIVE GitHub OAuth | NOT TESTED | external credentials/live provider flow not used |
| SMTP-dependent approval/resend | NOT TESTED | no SMTP credentials; authorization gating was tested |
| Security headers | PASS | expected headers present |
| Production HSTS | PASS | HSTS emitted with HTTPS-only configuration |
| Production docs disabled | PASS | OpenAPI/Swagger/ReDoc URLs are `None` |
| Secret scan | PASS | no credential-value patterns found in final source tree |
| Dependency vulnerability scan | NOT TESTED | `pip-audit` unavailable |
| Research retraining | NOT PERFORMED | no training command executed |
| Research results modification | NOT PERFORMED | no research artifact trees were present in supplied archives |

## 11. Remaining limitations

1. **Live GitHub OAuth is not tested.** The local implementation was statically audited and the state-validation paths were exercised without contacting GitHub.
2. **SMTP approval/resend is not tested end-to-end** because no SMTP credentials were supplied.
3. **`pip-audit` was unavailable** in the execution environment, so dependency CVE verification could not be claimed.
4. The supplied backend exposes a smaller API surface than several frontend pages currently call. For example, some frontend pages reference endpoints such as dashboard performance/filter/detail routes that are not present in the uploaded backend. This pre-existing API-contract mismatch was not filled with invented data or behavior because doing so would change application functionality rather than harden it.
5. The two supplied archives contained backend/frontend trees but did not contain the separate research `src/`, `checkpoints/`, `results/`, or dataset trees described by the project prompt. Those missing artifacts were not fabricated.

## 12. Research integrity

No model was retrained. No model architecture, attack implementation, dataset, checkpoint, accuracy value, attack success rate, or experimental result was changed.

## 13. Final packaging

The final archive contains the complete contents of the two supplied archives, with security documentation/configuration additions and without generated `__pycache__`, local authentication DB files, or real credentials.

## Admin Inbox Access-Control Hardening Addendum

A targeted second-pass audit hardened the Admin Inbox page and its directly loaded assets. The server now registers explicit `/admin`, `/admin.html`, `/admin.js`, and `/admin.css` routes before the frontend `StaticFiles` mount. The page is served only to an authenticated administrator; anonymous requests redirect to `/`, authenticated non-admin requests redirect to `/dashboard.html`, and admin assets return `404` to non-admin callers. Admin responses use `Cache-Control: no-store, max-age=0` and `Pragma: no-cache`.

Authorization is centralized in `backend/dependencies.py`. The check normalizes `role_key` with a legacy display-role fallback, treats the `.env` bootstrap admin as a server-side special case, rejects GitHub sessions from DB-ID authorization, and verifies database-backed administrators are both active and still assigned the admin role on every request.

The session now carries a normalized `role_key`, including `admin`/`user` for local sessions and `user` for GitHub sessions. The Admin Inbox navigation item is created dynamically only after `/api/auth/me` confirms an administrator; it is not present as static HTML for non-admin pages.

### Targeted Acceptance Verification

An isolated SQLite TestClient run executed the requested Admin Inbox matrix. **52/52 checks passed**, including anonymous access, normal-user access, GitHub numeric-ID collision, bootstrap-admin access, approve/close/resend behavior, database-admin demotion/disablement without logout, navigation inclusion/absence checks, dashboard authentication guarding, and no-store cache headers.

The live GitHub provider itself was not contacted; the GitHub acceptance case used a signed test session with a GitHub provider and an ID deliberately equal to the DB administrator's numeric ID, verifying the required server-side collision defense.
