(() => {
  async function addAdminLink() {
    const sideNav = document.querySelector(".side-nav");
    if (!sideNav || sideNav.querySelector('[data-page="admin"]')) return;

    try {
      const response = await fetch("/api/auth/me", {
        credentials: "include"
      });

      if (!response.ok) return;

      const body = await response.json();
      const user = body.user || {};
      const isAdmin =
        user.role_key === "admin" ||
        String(user.role || "").toLowerCase() === "administrator";

      if (!isAdmin) return;

      const link = document.createElement("a");
      link.className = "nav-item";
      link.href = "/admin.html";
      link.dataset.page = "admin";
      link.innerHTML =
        '<span class="nav-icon">⚙</span><span>Admin Inbox</span>';

      sideNav.appendChild(link);
    } catch (_) {
      // Do not create a privileged navigation item when auth cannot be verified.
    }
  }

  addAdminLink();
})();
