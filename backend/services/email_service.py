import smtplib

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

    message["From"] = (
        f"{settings.smtp_from_name} "
        f"<{settings.smtp_from}>"
    )

    message["To"] = to_email
    message["Subject"] = subject

    message.set_content(
        text_body
    )

    message.add_alternative(
        html_body,
        subtype="html",
    )

    with smtplib.SMTP(
        settings.smtp_host,
        settings.smtp_port,
        timeout=20,
    ) as server:

        server.starttls()

        server.login(
            settings.smtp_username,
            settings.smtp_password,
        )

        server.send_message(
            message
        )


def send_invitation_email(
    name: str,
    email: str,
    invitation_url: str,
):

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

    html = f"""
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

<p>Hello {name},</p>

<p>
An AMCShield administrator created an account for you.
</p>

<p>
Click the button below to activate your account.
</p>

<p>
<a href="{invitation_url}"
style="
display:inline-block;
padding:14px 24px;
background:#f3c63f;
color:#111;
text-decoration:none;
border-radius:8px;
font-weight:bold;
">
ACTIVATE ACCOUNT
</a>
</p>

<p>
This one-time link expires in
<strong>{settings.invitation_minutes} minutes</strong>.
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
        html,
    )