const loginForm = document.getElementById("loginForm");
const identityInput = document.getElementById("identity");
const passwordInput = document.getElementById("password");
const rememberInput = document.getElementById("remember");
const loginButton = document.getElementById("loginButton");
const passwordToggle = document.querySelector(".password-toggle");
const identityError = document.getElementById("identityError");
const passwordError = document.getElementById("passwordError");
const formMessage = document.getElementById("formMessage");
const forgotButton = document.getElementById("forgotButton");
const githubButton = document.getElementById("githubButton");
const contactButton = document.getElementById("contactButton");

// Later, replace the simulated login below with the real AMCShield
// authentication API. No credentials are hardcoded here.
const AUTH_ENDPOINT = "/api/auth/login";
const REMEMBER_KEY = "amcshield_login_identity";

function clearErrors() {
  identityError.textContent = "";
  passwordError.textContent = "";
  formMessage.textContent = "";
  formMessage.className = "form-message";
}

function setMessage(message, type = "info") {
  formMessage.textContent = message;
  formMessage.className = `form-message ${type}`;
}

function validateForm() {
  clearErrors();

  let valid = true;
  const identity = identityInput.value.trim();
  const password = passwordInput.value;

  if (!identity) {
    identityError.textContent = "Please enter your email or username.";
    valid = false;
  }

  if (!password) {
    passwordError.textContent = "Please enter your password.";
    valid = false;
  } else if (password.length < 8) {
    passwordError.textContent = "Password must contain at least 8 characters.";
    valid = false;
  }

  return valid;
}

passwordToggle.addEventListener("click", () => {
  const visible = passwordInput.type === "text";
  passwordInput.type = visible ? "password" : "text";
  passwordToggle.classList.toggle("is-visible", !visible);
  passwordToggle.setAttribute(
    "aria-label",
    visible ? "Show password" : "Hide password"
  );
});

const savedIdentity = localStorage.getItem(REMEMBER_KEY);
if (savedIdentity) {
  identityInput.value = savedIdentity;
  rememberInput.checked = true;
}

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!validateForm()) {
    return;
  }

  const identity = identityInput.value.trim();

  if (rememberInput.checked) {
    localStorage.setItem(REMEMBER_KEY, identity);
  } else {
    localStorage.removeItem(REMEMBER_KEY);
  }

  loginButton.disabled = true;
  loginButton.classList.add("loading");
  setMessage("Authenticating...", "info");

  try {
    const response = await fetch(AUTH_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      credentials: "include",
      body: JSON.stringify({
        identity,
        password: passwordInput.value
      })
    });

    const body = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(body.detail || "Authentication failed.");
    }

    setMessage("Login successful. Opening AMCShield Console...", "success");
    window.setTimeout(() => {
      window.location.href = "/dashboard.html";
    }, 250);
  } catch (error) {
    setMessage(error.message || "Unable to authenticate.", "error");
  } finally {
    loginButton.disabled = false;
    loginButton.classList.remove("loading");
  }
});

forgotButton.addEventListener("click", () => {
  setMessage("Password recovery will be connected to the AMCShield backend.", "info");
});

githubButton.addEventListener("click", () => {
  githubButton.disabled = true;
  githubButton.classList.add("loading");
  setMessage("Redirecting to GitHub...", "info");
  window.location.href = "/api/auth/github/login";
});

contactButton.addEventListener("click", () => {
  setMessage("Contact Admin action will be connected to the project backend.", "info");
});

[identityInput, passwordInput].forEach((input) => {
  input.addEventListener("input", () => {
    if (input === identityInput) identityError.textContent = "";
    if (input === passwordInput) passwordError.textContent = "";
    formMessage.textContent = "";
    formMessage.className = "form-message";
  });
});


const authParams = new URLSearchParams(window.location.search);
const authError = authParams.get("auth_error");
if (authError) {
  const messages = {
    github_not_configured: "GitHub login is not configured on the AMCShield server.",
    invalid_oauth_state: "GitHub sign-in could not be verified. Please try again.",
    github_auth_failed: "GitHub authentication failed. Please try again."
  };
  setMessage(messages[authError] || "Authentication failed. Please try again.", "error");
  window.history.replaceState({}, document.title, window.location.pathname);
}

document.addEventListener("DOMContentLoaded", () => {

    const contactLink =
        document.getElementById("contactAdminLink");

    const modal =
        document.getElementById("contactAdminModal");

    const closeButton =
        document.getElementById("closeContactAdmin");

    const backdrop =
        document.querySelector(".contact-modal-backdrop");

    const form =
        document.getElementById("contactAdminForm");

    const status =
        document.getElementById("contactAdminStatus");

    const submitButton =
        document.getElementById("contactSubmit");


    function openContactModal() {
        if (!modal) return;

        modal.classList.remove("hidden");
        document.body.style.overflow = "hidden";

        const nameField =
            document.getElementById("contactName");

        if (nameField) {
            nameField.focus();
        }
    }


    function closeContactModal() {
        if (!modal) return;

        modal.classList.add("hidden");
        document.body.style.overflow = "";

        if (status) {
            status.textContent = "";
            status.className = "contact-admin-status";
        }
    }


    contactLink?.addEventListener("click", (event) => {
        event.preventDefault();
        openContactModal();
    });


    closeButton?.addEventListener(
        "click",
        closeContactModal
    );


    backdrop?.addEventListener(
        "click",
        closeContactModal
    );


    document.addEventListener("keydown", (event) => {
        if (
            event.key === "Escape" &&
            modal &&
            !modal.classList.contains("hidden")
        ) {
            closeContactModal();
        }
    });


    form?.addEventListener("submit", async (event) => {

        event.preventDefault();

        status.textContent = "";
        status.className = "contact-admin-status";

        submitButton.disabled = true;
        submitButton.innerHTML = "SUBMITTING...";


        const formData = new FormData(form);

        const payload = {
            name: formData.get("name")?.trim(),
            email: formData.get("email")?.trim(),
            subject: formData.get("subject")?.trim(),
            message: formData.get("message")?.trim()
        };


        try {

            const response = await fetch(
                "/api/auth/contact-admin",
                {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify(payload)
                }
            );


            const data = await response.json();


            if (!response.ok) {
                throw new Error(
                    data.detail ||
                    "Unable to submit your message."
                );
            }


            status.textContent =
                "Message submitted successfully. " +
                "The AMCShield administrator will review it.";

            status.className =
                "contact-admin-status success";

            form.reset();


            setTimeout(() => {
                closeContactModal();
            }, 2200);


        } catch (error) {

            console.error(
                "Contact admin error:",
                error
            );

            status.textContent =
                error.message ||
                "Something went wrong.";

            status.className =
                "contact-admin-status error";

        } finally {

            submitButton.disabled = false;

            submitButton.innerHTML =
                'SEND MESSAGE <span>→</span>';
        }

    });

});