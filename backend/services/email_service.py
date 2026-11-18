import html
import smtplib
import ssl

from email.message import EmailMessage

from ..config import settings


def _send_email(
    to_email: str,
    subject: str,
    text_body: str,
    html_body: str,
):
    if not settings.smtp_host:
        raise RuntimeError(
            "SMTP_HOST is not configured."
        )

    if not settings.smtp_username:
        raise RuntimeError(
            "SMTP_USERNAME is not configured."
        )

    if not settings.smtp_password:
        raise RuntimeError(
            "SMTP_PASSWORD is not configured."
        )

    message = EmailMessage()

    from_address = (
        settings.smtp_from
        or settings.smtp_username
    )

    if settings.smtp_from_name:
        message["From"] = (
            f"{settings.smtp_from_name} "
            f"<{from_address}>"
        )
    else:
        message["From"] = from_address

    message["To"] = to_email
    message["Subject"] = subject

    message.set_content(text_body)

    message.add_alternative(
        html_body,
        subtype="html",
    )

    context = ssl.create_default_context()

    if settings.smtp_port == 465:
        with smtplib.SMTP_SSL(
            settings.smtp_host,
            settings.smtp_port,
            timeout=20,
            context=context,
        ) as server:
            server.login(
                settings.smtp_username,
                settings.smtp_password,
            )
            server.send_message(message)

        return

    with smtplib.SMTP(
        settings.smtp_host,
        settings.smtp_port,
        timeout=20,
    ) as server:
        server.starttls(
            context=context
        )

        server.login(
            settings.smtp_username,
            settings.smtp_password,
        )

        server.send_message(message)


# ============================================================
# EXISTING INVITATION FLOW
# Used by backend/routers/admin_users.py
# ============================================================

def send_invitation_email(
    name: str,
    email: str,
    invitation_url: str,
):
    safe_name = html.escape(name)
    safe_email = html.escape(email)
    safe_url = html.escape(
        invitation_url,
        quote=True,
    )

    subject = (
        "Welcome to AMCShield — "
        "Activate your account"
    )

    text = f"""
Hello {name},

An AMCShield administrator created an account for you.

Activate your account using this one-time link:

{invitation_url}

This link expires in
{settings.invitation_minutes} minutes.

You will choose your own AMCShield password during activation.

If you did not expect this invitation,
contact the AMCShield administrator.

AMCShield
Defending the Airwaves with AI
"""

    html_body = f"""
<!DOCTYPE html>
<html>
<body style="
    margin:0;
    padding:30px;
    background:#050807;
    color:#f5f5f5;
    font-family:Arial,sans-serif;
">

<div style="
    max-width:620px;
    margin:auto;
    padding:35px;
    background:#07100d;
    border:1px solid #43552c;
    border-radius:14px;
">

<h2 style="color:#f3c63f;">
    Welcome to AMCShield
</h2>

<p>
Hello {safe_name},
</p>

<p>
An AMCShield administrator created an account for you.
</p>

<p>
Click the button below to activate your account.
</p>

<p>
<a
    href="{safe_url}"
    style="
        display:inline-block;
        padding:14px 24px;
        background:#f3c63f;
        color:#111;
        text-decoration:none;
        border-radius:8px;
        font-weight:bold;
    "
>
ACTIVATE ACCOUNT
</a>
</p>

<p>
This one-time link expires in
<strong>
{settings.invitation_minutes} minutes
</strong>.
</p>

<p>
You will choose your own password during activation.
</p>

<p>
If you did not expect this invitation,
you can ignore this email.
</p>

<hr>

<p>
<strong>AMCShield</strong><br>
Defending the Airwaves with AI
</p>

</div>

</body>
</html>
"""

    _send_email(
        email,
        subject,
        text,
        html_body,
    )


# ============================================================
# CONTACT ADMIN → APPROVE FLOW
# Used by backend/routers/admin.py
# ============================================================

def send_credentials_email(
    name: str,
    email: str,
    login_url: str,
    change_password_url: str,
    expires_hours: int,
):
    """Send account approval with a one-time password setup link.

    No password is generated or transmitted by email. The recipient creates
    the password through the short-lived, single-use setup token.
    """
    safe_name = html.escape(name)
    safe_email = html.escape(email)
    safe_login_url = html.escape(login_url, quote=True)
    safe_change_url = html.escape(change_password_url, quote=True)

    subject = "Your AMCShield access has been approved"

    text = f"""
Hello {name},

Your AMCShield access request was approved.

Login email:
{email}

Set your AMCShield password using this one-time link:
{change_password_url}

The password setup link expires in {expires_hours} hours and can only be used once.
No password is included in this email.

After creating your password, sign in here:
{login_url}

If you did not request access, you can ignore this email.

AMCShield
Defending the Airwaves with AI
"""

    html_body = f"""
<!DOCTYPE html>
<html>
<body style="margin:0;padding:30px;background:#050807;color:#f5f5f5;font-family:Arial,sans-serif;">
<div style="max-width:620px;margin:auto;padding:35px;background:#07100d;border:1px solid #43552c;border-radius:14px;">
<h2 style="color:#f3c63f;">AMCShield Access Approved</h2>
<p>Hello {safe_name},</p>
<p>Your AMCShield access request was approved.</p>
<p><strong>Login email:</strong><br>{safe_email}</p>
<p>Before signing in, create your own password using the secure one-time link below.</p>
<p>
<a href="{safe_change_url}" style="display:inline-block;padding:14px 24px;background:#00d9ff;color:#061011;text-decoration:none;border-radius:8px;font-weight:bold;">
SET YOUR PASSWORD
</a>
</p>
<p>The password setup link expires in <strong>{expires_hours} hours</strong> and can only be used once.</p>
<p>No password is included in this email.</p>
<p>After setting your password:</p>
<p>
<a href="{safe_login_url}" style="display:inline-block;padding:14px 24px;background:#f3c63f;color:#111;text-decoration:none;border-radius:8px;font-weight:bold;">
LOG IN TO AMCShield
</a>
</p>
<p>If you did not request access, you can ignore this email.</p>
<hr>
<p><strong>AMCShield</strong><br>Defending the Airwaves with AI</p>
</div>
</body>
</html>
"""

    _send_email(
        email,
        subject,
        text,
        html_body,
    )
