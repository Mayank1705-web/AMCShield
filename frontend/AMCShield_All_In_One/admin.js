(() => {
    "use strict";

    const state = {
        messages: [],
        selectedId: null,
        filter: "all",
        search: ""
    };

    const $ = (id) =>
        document.getElementById(id);


    // =========================================================
    // Security / formatting helpers
    // =========================================================

    const esc = (v) =>
        String(v ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");


    const initials = (name) =>
        String(name || "A")
            .trim()
            .split(/\s+/)
            .slice(0, 2)
            .map((x) => x[0])
            .join("")
            .toUpperCase() || "A";


    function formatDate(value) {
        if (!value) {
            return "—";
        }

        const d = new Date(value);

        if (Number.isNaN(d.getTime())) {
            return value;
        }

        return new Intl.DateTimeFormat(
            undefined,
            {
                day: "2-digit",
                month: "short",
                year: "numeric",
                hour: "2-digit",
                minute: "2-digit"
            }
        ).format(d);
    }


    function statusClass(status) {
        return `status-${String(
            status || "new"
        ).toLowerCase()}`;
    }


    function statusLabel(status) {
        return String(
            status || "new"
        )
            .replaceAll("_", " ")
            .toUpperCase();
    }


    function toast(message) {
        const element = $("toast");

        if (!element) {
            return;
        }

        element.textContent = message;
        element.classList.add("show");

        clearTimeout(toast.timer);

        toast.timer = setTimeout(() => {
            element.classList.remove("show");
        }, 2800);
    }


    // =========================================================
    // CSRF
    // =========================================================

    let csrfToken = null;

    async function getCsrfToken() {
        if (csrfToken) return csrfToken;

        const response = await fetch("/api/auth/csrf", {
            method: "GET",
            credentials: "same-origin",
            headers: { "Accept": "application/json" }
        });

        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data.csrf_token) {
            throw new Error("Unable to establish a secure session.");
        }

        csrfToken = data.csrf_token;
        return csrfToken;
    }

    // =========================================================
    // API
    // =========================================================

    async function api(
        url,
        options = {}
    ) {
        const method = String(options.method || "GET").toUpperCase();
        const headers = {
            ...(options.body
                ? { "Content-Type": "application/json" }
                : {}),
            ...(options.headers || {})
        };

        if (["POST", "PUT", "PATCH", "DELETE"].includes(method)) {
            headers["X-CSRF-Token"] = await getCsrfToken();
        }

        const response = await fetch(
            url,
            {
                credentials: "same-origin",
                ...options,
                headers
            }
        );

        if (response.status === 403 && method !== "GET") {
            // A rotated session may have invalidated the cached token.
            csrfToken = null;
        }


        if (response.status === 401) {
            location.href = "/";

            throw new Error(
                "Authentication required."
            );
        }


        if (response.status === 403) {

            document.body.innerHTML = `
                <div
                    style="
                        min-height:100vh;
                        display:grid;
                        place-items:center;
                        background:#030706;
                        color:#fff;
                        font-family:Inter,sans-serif;
                        text-align:center;
                    "
                >
                    <div>

                        <h1
                            style="
                                font-family:Rajdhani;
                                color:#ffd54f;
                            "
                        >
                            ADMIN ACCESS REQUIRED
                        </h1>

                        <p style="color:#999">
                            Your account does not have
                            administrator privileges.
                        </p>

                        <a
                            href="/"
                            style="color:#00d9ff"
                        >
                            Return to login
                        </a>

                    </div>
                </div>
            `;

            throw new Error(
                "Administrator access required."
            );
        }


        let data = {};

        try {
            data = await response.json();
        } catch {
            // Non-JSON response.
        }


        if (!response.ok) {
            throw new Error(
                data.detail ||
                `Request failed (${response.status})`
            );
        }


        return data;
    }


    // =========================================================
    // Counts
    // =========================================================

    function counts(messages) {
        return {
            total: messages.length,

            new: messages.filter(
                (x) => x.status === "new"
            ).length,

            read: messages.filter(
                (x) => x.status === "read"
            ).length,

            approved: messages.filter(
                (x) => x.status === "approved"
            ).length,

            closed: messages.filter(
                (x) => x.status === "closed"
            ).length
        };
    }


    function setCounts(data) {

        if ($("countTotal")) {
            $("countTotal").textContent =
                data.total || 0;
        }

        if ($("countNew")) {
            $("countNew").textContent =
                data.new || 0;
        }

        if ($("countRead")) {
            $("countRead").textContent =
                data.read || 0;
        }

        if ($("countApproved")) {
            $("countApproved").textContent =
                data.approved || 0;
        }

        if ($("countClosed")) {
            $("countClosed").textContent =
                data.closed || 0;
        }

        if ($("sidebarNewCount")) {
            $("sidebarNewCount").textContent =
                data.new || 0;
        }
    }


    // =========================================================
    // Filtering
    // =========================================================

    function filteredMessages() {

        const query =
            state.search
                .trim()
                .toLowerCase();


        return state.messages.filter(
            (message) => {

                const matchesFilter =
                    state.filter === "all" ||
                    message.status ===
                        state.filter;


                const searchable = [
                    message.name,
                    message.email,
                    message.subject,
                    message.message
                ]
                    .join(" ")
                    .toLowerCase();


                const matchesSearch =
                    !query ||
                    searchable.includes(query);


                return (
                    matchesFilter &&
                    matchesSearch
                );
            }
        );
    }


    // =========================================================
    // Render inbox
    // =========================================================

    function render() {

        const list =
            $("messageList");

        if (!list) {
            return;
        }


        const messages =
            filteredMessages();


        if ($("resultCount")) {
            $("resultCount").textContent =
                `${messages.length} request${
                    messages.length === 1
                        ? ""
                        : "s"
                }`;
        }


        if (!messages.length) {

            list.innerHTML = `
                <div class="empty">

                    <div>◌</div>

                    <h3>
                        No requests found
                    </h3>

                    <p>
                        Try another filter
                        or search.
                    </p>

                </div>
            `;

            return;
        }


        list.innerHTML =
            messages
                .map(
                    (message) => `
                        <article
                            class="
                                message-item
                                ${
                                    Number(
                                        message.id
                                    ) ===
                                    state.selectedId
                                        ? "selected"
                                        : ""
                                }
                            "
                            data-id="${Number(
                                message.id
                            )}"
                        >

                            <div class="message-top">

                                <span
                                    class="sender-name"
                                >
                                    ${esc(
                                        message.name ||
                                        "Unknown sender"
                                    )}
                                </span>

                                <span
                                    class="
                                        status-pill
                                        ${statusClass(
                                            message.status
                                        )}
                                    "
                                >
                                    ${esc(
                                        statusLabel(
                                            message.status
                                        )
                                    )}
                                </span>

                            </div>


                            <div class="message-subject">
                                ${esc(
                                    message.subject ||
                                    "No subject"
                                )}
                            </div>


                            <p class="message-preview">
                                ${esc(
                                    message.message ||
                                    ""
                                )}
                            </p>


                            <div class="message-bottom">

                                <span class="message-date">
                                    ${esc(
                                        formatDate(
                                            message.created_at
                                        )
                                    )}
                                </span>

                                <span class="message-date">
                                    #${esc(
                                        message.id
                                    )}
                                </span>

                            </div>

                        </article>
                    `
                )
                .join("");


        list
            .querySelectorAll(
                ".message-item"
            )
            .forEach((element) => {

                element.onclick = () => {

                    select(
                        Number(
                            element.dataset.id
                        )
                    );

                };

            });
    }


    // =========================================================
    // Error display
    // =========================================================

    function renderInboxError(error) {

        const list =
            $("messageList");

        if (!list) {
            return;
        }


        list.innerHTML = `
            <div
                class="empty"
                role="alert"
            >

                <div>⚠</div>

                <h3>
                    Unable to load Admin Inbox
                </h3>

                <p>
                    ${esc(
                        error?.message ||
                        "Unable to load requests."
                    )}
                </p>

                <button
                    type="button"
                    id="retryInbox"
                    class="admin-action-btn resend-btn"
                >
                    Retry
                </button>

            </div>
        `;


        $("retryInbox")?.addEventListener(
            "click",
            async () => {

                try {

                    await load(
                        state.selectedId
                    );

                    toast(
                        "Inbox loaded successfully."
                    );

                } catch (retryError) {

                    renderInboxError(
                        retryError
                    );

                }

            }
        );
    }


    // =========================================================
    // Select request
    // =========================================================

    function select(id) {

        const message =
            state.messages.find(
                (item) =>
                    Number(item.id) ===
                    Number(id)
            );


        if (!message) {
            return;
        }


        state.selectedId =
            Number(id);


        $("detailEmpty")
            ?.classList.add("hidden");

        $("detailContent")
            ?.classList.remove("hidden");


        if ($("detailSubject")) {
            $("detailSubject").textContent =
                message.subject ||
                "No subject";
        }


        if ($("detailStatus")) {

            $("detailStatus")
                .textContent =
                    statusLabel(
                        message.status
                    );

            $("detailStatus")
                .className =
                    `status-pill ${statusClass(
                        message.status
                    )}`;
        }


        if ($("detailAvatar")) {
            $("detailAvatar").textContent =
                initials(
                    message.name
                );
        }


        if ($("detailName")) {
            $("detailName").textContent =
                message.name ||
                "Unknown sender";
        }


        if ($("detailEmail")) {

            $("detailEmail")
                .textContent =
                    message.email ||
                    "No email";

            $("detailEmail").href =
                message.email
                    ? `mailto:${encodeURIComponent(
                          message.email
                      )}`
                    : "#";
        }


        if ($("detailId")) {
            $("detailId").textContent =
                `#${message.id}`;
        }


        if ($("detailDate")) {
            $("detailDate").textContent =
                formatDate(
                    message.created_at
                );
        }


        if ($("detailMessage")) {
            $("detailMessage").textContent =
                message.message ||
                "";
        }


        if ($("detailNotice")) {
            $("detailNotice").textContent =
                "";

            $("detailNotice")
                .className = "";
        }


        if ($("closeBox")) {
            $("closeBox")
                .classList.remove("show");
        }


        updateActionButtons(
            message.status
        );


        render();
    }


    // =========================================================
    // Button state
    // =========================================================

    function updateActionButtons(
        status
    ) {

        const approve =
            $("approveBtn");

        const close =
            $("closeBtn");

        const resend =
            $("resendBtn");


        if (!approve ||
            !close ||
            !resend) {
            return;
        }


        /*
         * NEW / READ
         * Can be approved or closed.
         */
        if (
            status === "new" ||
            status === "read"
        ) {

            approve.disabled = false;
            close.disabled = false;
            resend.disabled = true;

            approve.textContent =
                "✓ APPROVE & EMAIL";

            return;
        }


        /*
         * APPROVED
         * Cannot approve or close again.
         * Credentials may be resent.
         */
        if (status === "approved") {

            approve.disabled = true;
            close.disabled = true;
            resend.disabled = false;

            resend.textContent =
                "↻ RESEND CREDENTIALS";

            return;
        }


        /*
         * CLOSED
         */
        if (status === "closed") {

            approve.disabled = true;
            close.disabled = true;
            resend.disabled = true;

            return;
        }


        approve.disabled = false;
        close.disabled = false;
        resend.disabled = true;
    }


    // =========================================================
    // Load messages
    // =========================================================

    async function load(
        preferredId =
            state.selectedId
    ) {

        const list =
            $("messageList");


        if (list) {

            list.innerHTML = `
                <div class="empty">

                    <div>◌</div>

                    <h3>
                        Loading inbox...
                    </h3>

                    <p>
                        Retrieving administrator
                        messages.
                    </p>

                </div>
            `;
        }


        const data =
            await api(
                "/api/admin/messages"
            );


        state.messages =
            Array.isArray(
                data.messages
            )
                ? data.messages
                : [];


        setCounts(
            data.counts ||
            counts(
                state.messages
            )
        );


        render();


        if (
            preferredId !== null &&
            preferredId !== undefined &&
            state.messages.some(
                (item) =>
                    Number(item.id) ===
                    Number(preferredId)
            )
        ) {

            select(
                Number(preferredId)
            );

        } else {

            state.selectedId =
                null;

            $("detailEmpty")
                ?.classList.remove(
                    "hidden"
                );

            $("detailContent")
                ?.classList.add(
                    "hidden"
                );
        }
    }


    // =========================================================
    // Approve
    // =========================================================

    async function approveRequest() {

        if (
            state.selectedId === null
        ) {
            return;
        }


        const button =
            $("approveBtn");

        const notice =
            $("detailNotice");


        button.disabled = true;
        button.textContent =
            "APPROVING...";


        notice.textContent =
            "Creating account and sending credentials...";

        notice.className =
            "info";


        try {

            const data =
                await api(
                    `/api/admin/messages/${state.selectedId}/approve`,
                    {
                        method: "POST"
                    }
                );


            notice.textContent =
                data.message ||
                "Request approved and credentials emailed.";

            notice.className =
                "success";


            toast(
                "Access approved successfully."
            );


            await load(
                state.selectedId
            );


            /*
             * Re-select the request after reload
             * so the administrator can see its
             * APPROVED status.
             */
            if (
                state.messages.some(
                    (item) =>
                        Number(item.id) ===
                        Number(
                            state.selectedId
                        )
                )
            ) {
                select(
                    Number(
                        state.selectedId
                    )
                );
            }


        } catch (error) {

            console.error(
                "Approve failed:",
                error
            );

            notice.textContent =
                error.message ||
                "Unable to approve request.";

            notice.className =
                "error";

            button.disabled = false;
            button.textContent =
                "✓ APPROVE & EMAIL";
        }
    }


    // =========================================================
    // Close UI
    // =========================================================

    function showCloseBox() {

        $("closeBox")
            ?.classList.add("show");

        $("adminNote")?.focus();
    }


    function hideCloseBox() {

        $("closeBox")
            ?.classList.remove("show");

        if ($("adminNote")) {
            $("adminNote").value = "";
        }
    }


    // =========================================================
    // Close request
    // =========================================================

    async function closeRequest() {

        if (
            state.selectedId === null
        ) {
            return;
        }


        const button =
            $("confirmCloseBtn");

        const notice =
            $("detailNotice");

        const adminNote =
            $("adminNote")
                ?.value
                ?.trim() || "";


        button.disabled = true;
        button.textContent =
            "CLOSING...";


        try {

            const data =
                await api(
                    `/api/admin/messages/${state.selectedId}/close`,
                    {
                        method: "POST",

                        body: JSON.stringify({
                            admin_note:
                                adminNote ||
                                null
                        })
                    }
                );


            hideCloseBox();


            notice.textContent =
                data.message ||
                "Access request closed.";

            notice.className =
                "success";


            toast(
                "Access request closed."
            );


            await load(
                state.selectedId
            );


            if (
                state.messages.some(
                    (item) =>
                        Number(item.id) ===
                        Number(
                            state.selectedId
                        )
                )
            ) {
                select(
                    Number(
                        state.selectedId
                    )
                );
            }

        } catch (error) {

            console.error(
                "Close failed:",
                error
            );

            notice.textContent =
                error.message ||
                "Unable to close request.";

            notice.className =
                "error";

        } finally {

            button.disabled = false;
            button.textContent =
                "Confirm Close";
        }
    }


    // =========================================================
    // Resend credentials
    // =========================================================

    async function resendCredentials() {

        if (
            state.selectedId === null
        ) {
            return;
        }


        const button =
            $("resendBtn");

        const notice =
            $("detailNotice");


        button.disabled = true;
        button.textContent =
            "SENDING...";


        notice.textContent =
            "Generating new credentials and sending email...";

        notice.className =
            "info";


        try {

            const data =
                await api(
                    `/api/admin/messages/${state.selectedId}/resend`,
                    {
                        method: "POST"
                    }
                );


            notice.textContent =
                data.message ||
                "Credentials re-sent successfully.";

            notice.className =
                "success";


            toast(
                "Credentials re-sent successfully."
            );


            await load(
                state.selectedId
            );


            if (
                state.messages.some(
                    (item) =>
                        Number(item.id) ===
                        Number(
                            state.selectedId
                        )
                )
            ) {
                select(
                    Number(
                        state.selectedId
                    )
                );
            }


        } catch (error) {

            console.error(
                "Resend failed:",
                error
            );

            notice.textContent =
                error.message ||
                "Unable to resend credentials.";

            notice.className =
                "error";

        } finally {

            button.disabled = false;

            /*
             * Restore appropriate button text.
             */
            button.textContent =
                "↻ RESEND CREDENTIALS";
        }
    }


    // =========================================================
    // Initialisation
    // =========================================================

    async function init() {

        try {

            const me =
                await api(
                    "/api/auth/me"
                );


            const user =
                me.user || {};


            if (
                String(
                    user.role || ""
                ).toLowerCase() !==
                "administrator"
            ) {

                throw new Error(
                    "Administrator access required."
                );
            }


            if ($("adminName")) {
                $("adminName")
                    .textContent =
                        user.name ||
                        user.login ||
                        "Administrator";
            }


            if ($("adminRole")) {
                $("adminRole")
                    .textContent =
                        user.role ||
                        "Administrator";
            }


            if ($("userAvatar")) {
                $("userAvatar")
                    .textContent =
                        initials(
                            user.name ||
                            user.login
                        );
            }


            await load();

        } catch (error) {

            console.error(
                "Admin inbox initialization failed:",
                error
            );

            renderInboxError(
                error
            );
        }
    }


    // =========================================================
    // Event listeners
    // =========================================================

    document
        .querySelectorAll(".filter")
        .forEach((button) => {

            button.onclick = () => {

                document
                    .querySelectorAll(
                        ".filter"
                    )
                    .forEach(
                        (item) =>
                            item.classList
                                .remove(
                                    "active"
                                )
                    );


                button.classList.add(
                    "active"
                );


                state.filter =
                    button.dataset.filter ||
                    "all";


                render();
            };

        });


    if ($("searchInput")) {

        $("searchInput").oninput =
            (event) => {

                state.search =
                    event.target.value ||
                    "";

                render();
            };
    }


    if ($("approveBtn")) {

        $("approveBtn").onclick =
            approveRequest;
    }


    if ($("closeBtn")) {

        $("closeBtn").onclick =
            showCloseBox;
    }


    if ($("confirmCloseBtn")) {

        $("confirmCloseBtn").onclick =
            closeRequest;
    }


    if ($("cancelCloseBtn")) {

        $("cancelCloseBtn").onclick =
            hideCloseBox;
    }


    if ($("resendBtn")) {

        $("resendBtn").onclick =
            resendCredentials;
    }


    if ($("refreshBtn")) {

        $("refreshBtn").onclick =
            async () => {

                try {

                    await load(
                        state.selectedId
                    );

                    toast(
                        "Inbox refreshed."
                    );

                } catch (error) {

                    renderInboxError(
                        error
                    );
                }
            };
    }


    if ($("logoutBtn")) {

        $("logoutBtn").onclick =
            async () => {

                try {

                    await api(
                        "/api/auth/logout",
                        {
                            method: "POST"
                        }
                    );

                } finally {

                    location.href = "/";
                }
            };
    }


    document.addEventListener(
        "DOMContentLoaded",
        init
    );

})();