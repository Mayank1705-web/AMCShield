/*
 * AMCShield frontend security helper.
 *
 * All authenticated state-changing requests should use amcFetch().
 * Login remains intentionally exempt because the backend login route
 * does not require a CSRF token.
 */
(function () {
  let csrfToken = null;
  let csrfPromise = null;

  async function fetchCsrfToken(forceRefresh = false) {
    if (csrfToken && !forceRefresh) {
      return csrfToken;
    }

    if (csrfPromise && !forceRefresh) {
      return csrfPromise;
    }

    csrfPromise = fetch("/api/auth/csrf", {
      method: "GET",
      credentials: "include",
      headers: {
        "Accept": "application/json"
      },
      cache: "no-store"
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`CSRF token request failed (${response.status})`);
        }

        const body = await response.json();

        if (!body || typeof body.csrf_token !== "string" || !body.csrf_token) {
          throw new Error("CSRF token was not returned by the server.");
        }

        csrfToken = body.csrf_token;
        return csrfToken;
      })
      .finally(() => {
        csrfPromise = null;
      });

    return csrfPromise;
  }

  async function amcFetch(url, options = {}) {
    const requestOptions = { ...options };
    const method = String(requestOptions.method || "GET").toUpperCase();

    requestOptions.credentials = requestOptions.credentials || "include";
    requestOptions.headers = new Headers(requestOptions.headers || {});

    requestOptions.headers.set("Accept", "application/json");

    const stateChanging =
      !["GET", "HEAD", "OPTIONS"].includes(method);

    if (stateChanging) {
      let token = await fetchCsrfToken();

      requestOptions.headers.set("X-CSRF-Token", token);

      let response = await fetch(url, requestOptions);

      // If the session/token changed, refresh once and retry.
      if (response.status === 403) {
        try {
          token = await fetchCsrfToken(true);
          requestOptions.headers.set("X-CSRF-Token", token);
          response = await fetch(url, requestOptions);
        } catch (_) {
          // Preserve the original 403 when refresh itself fails.
        }
      }

      return response;
    }

    return fetch(url, requestOptions);
  }

  window.AMCShieldSecurity = Object.freeze({
    getCsrfToken: fetchCsrfToken,
    fetch: amcFetch
  });
})();
