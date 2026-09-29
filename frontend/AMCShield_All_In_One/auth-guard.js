(async function () {
  try {
    const response = await fetch("/api/auth/me", {
      headers: { "Accept": "application/json" },
      credentials: "include"
    });
    if (!response.ok) {
      window.location.replace("/");
      return;
    }
    const body = await response.json();
    window.AMCShieldUser = body.user || null;
  } catch (error) {
    console.error("AMCShield auth check failed:", error);
    window.location.replace("/");
  }
})();
