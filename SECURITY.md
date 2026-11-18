# AMCShield Security

## Authentication

AMCShield uses FastAPI/Starlette signed sessions. Local accounts are backed by the application database and passwords are hashed with Argon2id via `argon2-cffi`.

GitHub OAuth is an authentication mechanism, not an authorization mechanism. The callback requires a cryptographically random, single-use OAuth `state`, obtains a verified GitHub email, maps that email to an existing AMCShield account, and requires that account to be active. The role is read from the AMCShield database.

## Authorization

Protected research APIs require an authenticated active account. Administrative APIs require an active database administrator account or the explicitly configured bootstrap administrator session. Frontend guards are only UX controls; backend dependencies enforce authorization.

## Sessions and cookies

Session cookies use:
- `HttpOnly` (Starlette default)
- `SameSite=Lax`
- configurable `Secure` behavior through `SESSION_HTTPS_ONLY`
- a configurable absolute `SESSION_MAX_AGE`

Production deployments must enable HTTPS-only cookies.

## CSRF

State-changing routes use a per-session CSRF token in the `X-CSRF-Token` header. Login and OAuth callback are deliberately exempt because they establish authentication state; OAuth is protected by its state parameter.

## Invitations

Invitation and password-setup tokens are generated with cryptographic randomness, stored only as SHA-256 hashes, expire, are atomically claimed, and are invalidated after use. Raw tokens are sent only in the one-time activation URL and are kept in frontend memory.

## Passwords

Passwords are validated at 12–128 characters with at least three character classes and are hashed using Argon2id. Passwords are not logged or placed in URLs.

## Abuse controls

Local login uses independent normalized identity and source-IP failure buckets with configurable lockout settings. Successful authentication clears the relevant buckets.

## Security headers

The application emits CSP, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, and conditional HSTS. API and HTML responses are marked `no-store`.

## Production deployment

Set:
- `ENVIRONMENT=production`
- `SESSION_HTTPS_ONLY=true`
- an HTTPS `APP_BASE_URL`
- an HTTPS `GITHUB_REDIRECT_URI`
- explicit `ALLOWED_ORIGINS`
- strong random `SESSION_SECRET`
- strong administrator credentials
- SMTP credentials through environment variables

Do not expose the backend project directory, database, datasets, checkpoints, results, or `.env` through a reverse proxy/static-file mount.

## Reporting

Do not place secrets in bug reports. Report security issues privately to the AMCShield project maintainer and include reproducible steps without credential material.
