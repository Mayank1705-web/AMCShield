(() => {
  "use strict";

  /*
   * AMCShield invite activation flow.
   *
   * Expected backend contract:
   *   GET  /api/invitations/validate?token=<token>
   *        -> { valid: true, user: { name, email } }
   *
   *   POST /api/invitations/activate
   *        body: { token, password }
   *        -> { success: true, message?: string }
   *
   * The invitation token is intentionally kept only in memory.
   * It is never written to localStorage/sessionStorage.
   */

  const API_BASE = "/api";
  const VALIDATE_URL = `${API_BASE}/invitations/validate`;
  const ACTIVATE_URL = `${API_BASE}/invitations/activate`;

  const els = {
    loading: document.getElementById("loadingState"),
    form: document.getElementById("activationForm"),
    invalid: document.getElementById("invalidState"),
    invalidTitle: document.getElementById("invalidTitle"),
    invalidText: document.getElementById("invalidText"),
    success: document.getElementById("successState"),
    successText: document.getElementById("successText"),
    message: document.getElementById("stateMessage"),
    userName: document.getElementById("userName"),
    userEmail: document.getElementById("userEmail"),
    password: document.getElementById("password"),
    confirm: document.getElementById("confirmPassword"),
    passwordError: document.getElementById("passwordError"),
    confirmError: document.getElementById("confirmError"),
    policy: document.getElementById("acceptPolicy"),
    policyError: document.getElementById("policyError"),
    activateBtn: document.getElementById("activateBtn"),
    activateBtnText: document.getElementById("activateBtnText"),
    togglePassword: document.getElementById("togglePassword"),
    toggleConfirm: document.getElementById("toggleConfirm"),
    strengthLabel: document.getElementById("strengthLabel"),
    strengthBar: document.getElementById("strengthBar"),
    ruleLength: document.getElementById("ruleLength"),
    ruleUpper: document.getElementById("ruleUpper"),
    ruleNumber: document.getElementById("ruleNumber"),
    ruleSpecial: document.getElementById("ruleSpecial")
  };

  let invitationToken = null;

  function getTokenFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");
    return token && token.trim() ? token.trim() : null;
  }

  function clearTokenFromAddressBar() {
    // Do not remove it until activation is complete: validation/activation may
    // need the token. On success, replace the URL so the token is no longer visible.
    const cleanUrl = `${window.location.origin}${window.location.pathname}`;
    window.history.replaceState({}, document.title, cleanUrl);
  }

  function showMessage(text, type = "error") {
    els.message.textContent = text;
    els.message.className = `state-message ${type}`;
  }

  function hideMessage() {
    els.message.className = "state-message hidden";
    els.message.textContent = "";
  }

  function showInvalid(title, text) {
    els.loading.classList.add("hidden");
    els.form.classList.add("hidden");
    els.success.classList.add("hidden");
    els.invalidTitle.textContent = title;
    els.invalidText.textContent = text;
    els.invalid.classList.remove("hidden");
  }

  function showForm(user) {
    els.loading.classList.add("hidden");
    els.invalid.classList.add("hidden");
    els.success.classList.add("hidden");
    els.userName.textContent = user?.name || "AMCShield user";
    els.userEmail.textContent = user?.email || "Invited account";
    els.form.classList.remove("hidden");
  }

  function showSuccess(message) {
    els.loading.classList.add("hidden");
    els.invalid.classList.add("hidden");
    els.form.classList.add("hidden");
    els.successText.textContent =
      message || "Your password has been created successfully. You can now sign in to AMCShield.";
    els.success.classList.remove("hidden");
    clearTokenFromAddressBar();
  }

  function normalizeError(payload, fallback) {
    if (!payload) return fallback;
    if (typeof payload.detail === "string") return payload.detail;
    if (typeof payload.message === "string") return payload.message;
    if (typeof payload.error === "string") return payload.error;
    if (Array.isArray(payload.detail)) {
      return payload.detail.map(item => item?.msg || "Invalid request").join(". ");
    }
    return fallback;
  }

  async function readJson(response) {
    const text = await response.text();
    if (!text) return {};
    try { return JSON.parse(text); }
    catch { return { message: text }; }
  }

  async function validateInvitation() {
    invitationToken = getTokenFromUrl();

    if (!invitationToken) {
      showInvalid(
        "Invitation link incomplete",
        "This activation link does not contain an invitation token. Ask the administrator to send a new invitation."
      );
      return;
    }

    try {
      const response = await fetch(
        `${VALIDATE_URL}?token=${encodeURIComponent(invitationToken)}`,
        {
          method: "GET",
          credentials: "same-origin",
          headers: { "Accept": "application/json" }
        }
      );

      const data = await readJson(response);

      if (!response.ok || data.valid === false) {
        showInvalid(
          "Invitation unavailable",
          normalizeError(
            data,
            "This invitation is invalid, expired, or has already been used."
          )
        );
        return;
      }

      const user = data.user || data.account || {
        name: data.name,
        email: data.email
      };

      showForm(user);
    } catch (error) {
      console.error("Invitation validation failed:", error);
      showInvalid(
        "Unable to verify invitation",
        "The AMCShield authentication service could not be reached. Please try again later."
      );
    }
  }

  function toggleVisibility(input, button) {
    const visible = input.type === "text";
    input.type = visible ? "password" : "text";
    button.textContent = visible ? "SHOW" : "HIDE";
    button.setAttribute("aria-label", visible ? "Show password" : "Hide password");
  }

  function passwordRules(password) {
    return {
      length: password.length >= 12,
      upper: /[A-Z]/.test(password) && /[a-z]/.test(password),
      number: /\d/.test(password),
      special: /[^A-Za-z0-9]/.test(password)
    };
  }

  function updateStrength() {
    const value = els.password.value;
    const rules = passwordRules(value);
    const score = Object.values(rules).filter(Boolean).length;

    els.ruleLength.classList.toggle("ok", rules.length);
    els.ruleUpper.classList.toggle("ok", rules.upper);
    els.ruleNumber.classList.toggle("ok", rules.number);
    els.ruleSpecial.classList.toggle("ok", rules.special);

    const labels = ["—", "WEAK", "WEAK", "GOOD", "STRONG"];
    els.strengthLabel.textContent = labels[score];
    els.strengthBar.style.width = `${score * 25}%`;

    if (score <= 1) els.strengthBar.style.background = "#ff7b7b";
    else if (score <= 2) els.strengthBar.style.background = "#e9b949";
    else els.strengthBar.style.background = "#52d273";
  }

  function validateForm() {
    let valid = true;
    const password = els.password.value;
    const confirm = els.confirm.value;

    els.passwordError.textContent = "";
    els.confirmError.textContent = "";
    els.policyError.textContent = "";
    hideMessage();

    const rules = passwordRules(password);
    if (!password) {
      els.passwordError.textContent = "Please create a password.";
      valid = false;
    } else if (!(rules.length && rules.upper && rules.number && rules.special)) {
      els.passwordError.textContent =
        "Use at least 12 characters with upper/lowercase letters, a number, and a special character.";
      valid = false;
    }

    if (!confirm) {
      els.confirmError.textContent = "Please confirm your password.";
      valid = false;
    } else if (password !== confirm) {
      els.confirmError.textContent = "Passwords do not match.";
      valid = false;
    }

    if (!els.policy.checked) {
      els.policyError.textContent = "Please confirm that you understand the password requirement.";
      valid = false;
    }

    return valid;
  }

  async function activateAccount(event) {
    event.preventDefault();

    if (!validateForm() || !invitationToken) return;

    els.activateBtn.disabled = true;
    els.activateBtnText.textContent = "ACTIVATING…";
    hideMessage();

    try {
      const response = await fetch(ACTIVATE_URL, {
        method: "POST",
        credentials: "same-origin",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({
          token: invitationToken,
          password: els.password.value
        })
      });

      const data = await readJson(response);

      if (!response.ok || data.success === false) {
        showMessage(
          normalizeError(
            data,
            response.status === 409
              ? "This invitation has already been used."
              : "The password could not be set. Please request a new invitation."
          )
        );
        return;
      }

      showSuccess(data.message);
    } catch (error) {
      console.error("Account activation failed:", error);
      showMessage(
        "The AMCShield authentication service could not be reached. Please try again."
      );
    } finally {
      els.activateBtn.disabled = false;
      els.activateBtnText.textContent = "ACTIVATE ACCOUNT";
    }
  }

  els.togglePassword.addEventListener("click", () =>
    toggleVisibility(els.password, els.togglePassword)
  );
  els.toggleConfirm.addEventListener("click", () =>
    toggleVisibility(els.confirm, els.toggleConfirm)
  );
  els.password.addEventListener("input", updateStrength);
  els.form.addEventListener("submit", activateAccount);

  validateInvitation();
})();
