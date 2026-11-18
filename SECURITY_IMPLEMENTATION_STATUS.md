# AMCShield Security Hardening

Implemented in this bundle:

- Login rate limiting with separate HMAC identity/IP buckets.
- Session cookie hardening configuration.
- CSRF token endpoint and API middleware protection.
- Security response headers and conditional HSTS.
- Frontend CSRF support for invitation activation and Admin Inbox actions.
- Secure `.env.example` template.
- Password email service changed to send a one-time setup link instead of a temporary password.

Still requires the current `backend/routers/admin.py` before the password-setup email flow can be completed safely, because that router currently creates the temporary password and calls `send_credentials_email()` with the old signature.

Also verify any other frontend JavaScript that performs POST/PUT/PATCH/DELETE requests and add the same CSRF header mechanism.
