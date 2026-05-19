# IMPLEMENTATION_PLAN.md

## 0. Document Purpose

This is an execution manual derived from the **INIDS Frontend Reconstruction & Audit Report** (Phases 0–6, including the File Inventory, Architecture Mapping, Comprehensive Frontend Issue Audit, Root Cause Synthesis, Prioritized Fix Roadmap, and Surgical Implementation Plan). It is not a re-audit. Every task in this document is traceable to a specific finding (ISSUE-001 through ISSUE-033) or to an explicitly enumerated Wave 2–7 roadmap item from the source report. This document is intended to be followed line-by-line by an AI or engineer to modify the actual INIDS repository with zero ambiguity.

---

## 1. Source Traceability Matrix

Every Issue ID from the source audit appears in at least one task below.

| Issue ID | Issue Title | Severity | Phase | Task IDs |
|----------|-------------|----------|-------|----------|
| ISS-001 | Detection/Health/Realtime templates have dual `{% block content %}` and orphaned CSS | Critical | 2 | T-2.1, T-2.2, T-2.3 |
| ISS-002 | Missing CSS variables (`--border`, `--mono`, `--bg-card`, etc.) | Critical | 4 | T-4.1 |
| ISS-003 | `/dashboard/main` route renders nonexistent template; recursive 500 | Critical | 1 | T-1.1 |
| ISS-004 | `alerts.page.js` uses non-existent `GlobalState.state` API | Critical | 1 | T-1.2 |
| ISS-005 | `monitor.page.js` accesses `Socket.socket.on(...)` while socket is null | Critical | 1 | T-1.3 |
| ISS-006 | `AlertCard` block-action response check (`response.ok`) is always false | Critical | 3 | T-3.1 |
| ISS-007 | `dashboard.page.js` calls `ModuleCard(dataObject)` instead of `ModuleCard(id, config)` | Critical | 3 | T-3.2 |
| ISS-008 | `AlertCard` reads `alert.src_ip`/`alert.dst_ip` instead of normalized `source_ip` | Critical | 3 | T-3.3 |
| ISS-009 | Alert audio files missing (`/static/sfx/alert_*.mp3` return 404) | Critical | 9 | T-9.1 |
| ISS-010 | `home.html` tile links to `/threat_intel` (underscore) — route is `/threat-intel` | High | 5 | T-5.1 |
| ISS-011 | Undefined CSS classes (`.page-header`, `.panel-header`, `.reconcile-strip`, etc.) | High | 4 | T-4.2 |
| ISS-012 | Duplicate templates: `threat_intel.html` (old) and `threat-intel.html` (stub) | High | 7 | T-7.1 |
| ISS-013 | Sidebar exposes only 5 of 15+ routes | High | 5 | T-5.2 |
| ISS-014 | `detection.html` loads legacy `detection.js`, not modern `detection.page.js` | High | 2 | T-2.1 |
| ISS-015 | Tailwind CDN play mode in production (no SRI, JIT in browser) | High | 8 | T-8.1 |
| ISS-016 | `socket-core.js` imports `./ui_core.js` (wrong path) and uses old `set(object)` API | High | 7 | T-7.2 |
| ISS-017 | `base-module-controller.js` `openSettings()` uses native `alert()` | High | 6 | T-6.1 |
| ISS-018 | Three competing CSS frameworks (Bootstrap + Tailwind CDN + custom inline) | High | 8 | T-8.2 |
| ISS-019 | `<script>` placed inside `{% block content %}` rather than `{% block scripts %}` | Medium | 6 | T-6.2 |
| ISS-020 | `AppToast.success('msg', 'string')` — second arg should be numeric duration | Medium | 6 | T-6.3 |
| ISS-021 | `home.html` uses `onclick="location.href=..."` instead of `<a href>` | Medium | 5 | T-5.3 |
| ISS-022 | No favicon — every page 404s on `/favicon.ico` | Medium | 5 | T-5.4 |
| ISS-023 | `base-module-controller.js` uses global `getElementById` instead of element-scoped queries | Medium | 6 | T-6.4 |
| ISS-024 | Orphaned legacy `state.js` and `socket.js` still present in static dir | Medium | 7 | T-7.3 |
| ISS-025 | `generate_alerts.py` exposed under `/static/sfx/` | Medium | 7 | T-7.4 |
| ISS-026 | `openSettings()` is a stub (duplicates ISS-017 root) | Medium | 6 | T-6.1 |
| ISS-027 | `loading-spinner.js` imported in `monitor.page.js` but never used | Medium | 7 | T-7.5 |
| ISS-028 | No FOUC prevention while Tailwind CDN JIT runs | Medium | 8 | T-8.1 |
| ISS-029 | `bootstrap.bundle.min.js` (~80KB) loaded on every page, mostly unused | Medium | 8 | T-8.3 |
| ISS-030 | `console.log` statements left in production JS | Low | 9 | T-9.2 |
| ISS-031 | `GlobalState.modules` initialized as `{}` but consumer checks `Array.isArray` | Low | 6 | T-6.5 |
| ISS-032 | `AppModal.setContent(html)` uses `innerHTML` with no sanitization | Low | 9 | T-9.3 |
| ISS-033 | `UICard.divider()` uses arbitrary Tailwind value `before:content-['']` requiring JIT | Low | 8 | T-8.1 |

---

## 2. Architectural Snapshot (from report, not invented)

Pulled verbatim or near-verbatim from Phase 1 of the audit report:

- **Framework / runtime:** Flask (Python) backend with Jinja2 SSR templates. Frontend runs in browser only — no Node runtime.
- **Build / bundler:** None. No bundler, no minification, no tree-shaking. All static files served raw from Flask's static server.
- **Module system:** Dual-mode. (A) Modern ES modules loaded via `<script type="module">` with explicit imports (`core/global-state.js`, `core/socket-manager.js`, `core/http-client.js`, all `pages/*.page.js`). (B) Legacy IIFE scripts (`state.js`, `socket.js`, `socket-core.js`, `detection.js`, `dashboard.js`, etc.). Several legacy files are orphaned (never loaded); `detection.js` is still actively loaded by `detection.html`.
- **Template system:** Jinja2 with `{% extends "base.html" %}` and `{% block content %}` / `{% block extra_css %}` / `{% block scripts %}` inheritance. `base.html` is the single root layout (sidebar, topbar, design tokens inline).
- **CSS methodology and token layer:** Four-layer conflict. (1) Bootstrap 5.2.3 vendored (~238KB CSS). (2) Tailwind CDN play (JIT in browser). (3) Inline `<style>` block in `base.html` (~700 lines, the canonical dark design system). (4) Per-template `{% block extra_css %}` inline CSS. Token layer is incomplete — variables such as `--border`, `--mono`, `--bg-card`, `--text-muted`, `--text-primary`, `--text-secondary`, `--red`, `--green`, `--amber`, `--red-lt` are referenced but never defined.
- **Component model:** JS factory functions returning DOM elements (e.g., `AlertCard(data)`, `UICard.create(opts)`, `ModuleCard(id, config)`). No virtual DOM, no web-components spec, no React.
- **State management:** `core/global-state.js` — slice-based observer pattern, exposes `data` (flat map of slices), `get(key)`, `set(key, value)`, `push(key, item)`, `subscribe(slice, cb)`. Also exposed as `window.GlobalState` for legacy access. Legacy `state.js` is orphaned with an incompatible API.
- **Routing / navigation:** Server-side Flask routing only. No client-side router. URL changes cause full page reloads. Navigation is rendered in `base.html` sidebar.
- **Frontend ↔ backend integration shape:** All API calls go through `HttpClient_Instance` (ES module singleton, also on `window.HttpClient`). REST endpoints under `/api/*`. Real-time channel is a Socket.IO WebSocket at `/events` managed by `SocketManager`. Key consumed endpoints: `/api/auth/login`, `/api/auth/refresh`, `/api/dashboard/metrics`, `/api/alerts?limit=200`, `/api/perception/pulse`, `/api/actions`, `/api/alerts/dismiss`.
- **Migration state (from → to):** From an older design system (legacy `ds-card` classes, `state.js`/`socket.js` IIFE globals, `detection.js`/`dashboard.js` page controllers, `var(--border)`/`var(--mono)` token names) **to** a Tailwind-styled UI on a new ES-module foundation (`core/global-state.js`, `core/socket-manager.js`, `pages/*.page.js` controllers, raw color values, modern component factories). Migration is incomplete; both systems coexist in several files (notably `detection.html`, `health.html`, `realtime.html`).

---

## 3. Dependency Graph

### Adjacency listing (issue-level blocks/independent)

- **ISS-002** → blocks → [ISS-011 partially: visual rendering of templates fixed by ISS-011 still requires tokens from ISS-002]
- **ISS-001** → blocks → [ISS-014]: detection.html cannot load the modern controller until the dual-block structure is resolved.
- **ISS-014** → depends on → [ISS-001]
- **ISS-004** → independent (isolated to `alerts.page.js`)
- **ISS-005** → independent (isolated to `monitor.page.js`)
- **ISS-008** → blocks → [ISS-006]: the block action sends `target_ip` derived from `alert.src_ip` (undefined). Field-name fix must land or be coupled with the response-check fix.
- **ISS-006** → depends on → [ISS-008] (coupled)
- **ISS-007** → blocks → [ISS-031]: the module rendering pathway must accept the right shape before the slice initialization mismatch is meaningful.
- **ISS-031** → depends on → [ISS-007]
- **ISS-003** → independent (route handler self-contained)
- **ISS-009** → independent
- **ISS-010** → independent (single string fix)
- **ISS-013** → independent (sidebar markup)
- **ISS-021** → coupled with → [ISS-010]: both touch `home.html` navigation; do together.
- **ISS-022** → independent
- **ISS-011** → depends on → [ISS-002]: many of the undefined classes use the undefined tokens internally.
- **ISS-012** → independent (template housekeeping)
- **ISS-015** ⟂ **ISS-018** ⟂ **ISS-028** ⟂ **ISS-033**: all four are the same root cause (Tailwind CDN + competing frameworks). Resolve as a coordinated Phase 8 effort.
- **ISS-016** → independent (orphaned file)
- **ISS-017** → independent
- **ISS-019** → independent (template hygiene)
- **ISS-020** → independent (one-line fix in `dashboard.page.js`)
- **ISS-023** → independent
- **ISS-024** → depends on → [Phase-5 completion]: do not delete legacy `state.js`/`socket.js` until replacements are confirmed in production behavior.
- **ISS-025** → independent (file move)
- **ISS-027** → independent (dead import)
- **ISS-029** → independent (bundle reduction)
- **ISS-030** → independent (logging cleanup)
- **ISS-032** → independent (sanitizer addition)
- **ISS-026** is the same defect as ISS-017 (audit lists it twice); single task covers both.

### Topologically sorted execution order (issue level)

1. ISS-003 (route crash)
2. ISS-004 (runtime TypeError)
3. ISS-005 (runtime TypeError)
4. ISS-001 (template render correctness)
5. ISS-014 (controller wiring) — depends on ISS-001
6. ISS-008 (schema)
7. ISS-006 (response handling) — coupled to ISS-008
8. ISS-007 (component contract)
9. ISS-031 (state slice shape) — depends on ISS-007
10. ISS-002 (token layer)
11. ISS-011 (class definitions) — depends on ISS-002
12. ISS-010 (route string)
13. ISS-021 (anchor tags) — coupled to ISS-010
14. ISS-013 (sidebar expansion)
15. ISS-022 (favicon)
16. ISS-019 (script block hygiene)
17. ISS-020 (toast duration)
18. ISS-017 / ISS-026 (alert → modal)
19. ISS-023 (element-scoped DOM lookups)
20. ISS-012 (duplicate template)
21. ISS-016 (orphaned `socket-core.js`)
22. ISS-024 (legacy file removal)
23. ISS-025 (generator script move)
24. ISS-027 (dead import)
25. ISS-015 / ISS-018 / ISS-028 / ISS-033 (Tailwind/Bootstrap rationalization)
26. ISS-029 (Bootstrap JS removal)
27. ISS-009 (audio assets)
28. ISS-030 (production logging)
29. ISS-032 (modal sanitization)

---

## 4. Global Execution Rules

- **Branching strategy per phase:** Create one branch per phase (e.g., `phase-1/critical-runtime`, `phase-2/template-repair`). Merge to integration branch only after the phase Exit Criteria are met.
- **Commit granularity rule:** One task = one commit, unless a task contains an explicit sub-step list where each sub-step says "commit separately." Commit message format: `T-<PHASE>.<N>: <short title> (ISS-<id>)`.
- **Required checks before moving to the next phase:**
  1. The phase's defined **Frontend Stability Checkpoint** passes.
  2. The phase's defined **UI Consistency Checkpoint** passes (where applicable).
  3. The phase's **Production-Readiness Checkpoint** passes (Phases 8–10 only).
  4. All tasks in the phase have their Validation/Testing Instructions executed and recorded.
- **Definition of "task complete":** Implementation Actions are executed exactly as written; the task's Expected Frontend Behavior is observed; all Validation/Testing Instructions pass; no regressions to any earlier completed task; the commit is pushed.
- **Definition of "phase complete":** All tasks in the phase are complete; the phase's Exit Criteria are observed live; the phase's checkpoint sections all pass; the phase branch is merged.

---

## 5. Phased Execution Plan

---

### Phase 1 — Critical Runtime Stabilization

**Phase Objective:** Eliminate the three runtime defects the audit identifies as preventing the app from booting or rendering specific pages without crash/500.

**Entry Preconditions:**
- Source audit report is available at the project root.
- Local dev environment can run the Flask app and load `/`, `/alerts`, `/monitor`, `/dashboard/main` in a browser.

**Exit Criteria:**
- `/dashboard/main` no longer returns 500.
- Loading `/alerts` and interacting with the severity filter, search input, bulk dismiss, and `alert.dismissed` socket events produces no `TypeError`.
- Loading `/monitor` produces no `TypeError: Cannot read properties of null (reading 'on')` at module init.

**Stability Checkpoint (Phase 1):**
- Routes `/`, `/alerts`, `/monitor`, `/dashboard/main` all respond with HTTP 200 (or 302 for `/dashboard/main`) and render their templates with no uncaught JS exception in the DevTools console at first load.

**UI Consistency Checkpoint (Phase 1):** Not applicable in Phase 1 — visual consistency is restored in Phases 2 and 4.

**Production-Readiness Checkpoint:** N/A in Phase 1.

---

#### Task 1.1 — Repair `/dashboard/main` route handler

- **Task ID:** T-1.1
- **Source Issue IDs:** ISS-003
- **Issue-to-Task Mapping Rationale:** Directly resolves the recursive `TemplateNotFound` 500 on `/dashboard/main` documented in ISS-003 / FIX-006.
- **Problem Being Solved:** `blueprints/pages.py` registers `/dashboard/main` rendering a template (`dashboard_main.html`) that does not exist. The `except` branch re-renders the same nonexistent template, producing a second `TemplateNotFound`.
- **Isolated or Architectural:** Isolated
- **Preconditions / Dependencies:** None.
- **Exact Files Involved:**
  - `web_app/blueprints/pages.py` — contains the `/dashboard/main` route to be modified.
  - `web_app/templates/dashboard_main.html` — optional create (only if redirect approach is rejected).
- **Exact Code Areas Involved:** Function `dashboard_main()` at `pages.py:290–297`.
- **Systems Affected:** Flask routing layer; no frontend module touched.
- **Architecture Impact:** Local. One route handler.
- **Reason for This Change (traced to report):** Report FIX-006: "Any request to `/dashboard/main` redirects to the working dashboard rather than throwing a 500."
- **Implementation Objective:** Requests to `/dashboard/main` return a working response (either the dashboard via redirect, or a minimal valid template).
- **Detailed Implementation Actions:**
  1. Open `web_app/blueprints/pages.py`.
  2. Confirm `redirect` and `url_for` are imported from `flask` at the top of the file. If not, add them to the existing Flask import line.
  3. Replace the body of `dashboard_main()` (lines 290–297 per the report) with:
     ```python
     @pages_bp.route("/dashboard/main")
     @require_roles("viewer")
     def dashboard_main():
         return redirect(url_for("pages.dashboard"))
     ```
  4. Save the file. Do not create `dashboard_main.html` — the redirect path is preferred by the report.
- **Expected Frontend Behavior After Fix:** Navigating to `/dashboard/main` produces a 302 redirect to `/dashboard` and renders the dashboard page.
- **Validation / Testing Instructions:**
  - Manual: Visit `http://<host>/dashboard/main`. Observe the URL becomes `/dashboard` and the dashboard renders.
  - Network panel: Confirm a 302 followed by a 200 for `/dashboard`.
  - No 500 in the Flask server logs.
- **Potential Side Effects:** Any external link/bookmark targeting `/dashboard/main` will now redirect. Acceptable.
- **Related Systems Impacted:** None on the frontend.
- **Regression Risk:** Low — single route, no shared code.
- **Rollback Considerations:** `git revert` the single commit. No build artifacts to clean.
- **Stability Checkpoint Triggered:** Yes
- **UI Consistency Checkpoint Triggered:** No
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 1.2 — Fix `GlobalState.state` → `GlobalState.data` in `alerts.page.js`

- **Task ID:** T-1.2
- **Source Issue IDs:** ISS-004
- **Issue-to-Task Mapping Rationale:** Directly resolves the four `TypeError` sites enumerated in ISS-004 / FIX-002.
- **Problem Being Solved:** `alerts.page.js` reads `GlobalState.state.alerts` at four sites. `GlobalState.state` is `undefined`; the correct slice accessor is `GlobalState.data`.
- **Isolated or Architectural:** Isolated
- **Preconditions / Dependencies:** None.
- **Exact Files Involved:**
  - `web_app/static/js/pages/alerts.page.js` — the only file modified.
- **Exact Code Areas Involved:** Lines 168, 180, 197, 285 per the report.
- **Systems Affected:** Alerts page only (filter handlers, search handler, bulk-block handler, `alert.dismissed` socket handler).
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report FIX-002: "`GlobalState.state` is undefined; the correct property is `GlobalState.data`. All filter interactions will work without TypeError."
- **Implementation Objective:** Every interaction on `/alerts` (severity filter, search input, bulk dismiss, socket-driven dismissal) completes without a `TypeError`.
- **Detailed Implementation Actions:**
  1. Open `web_app/static/js/pages/alerts.page.js`.
  2. At line 168 (severity filter handler), replace:
     ```js
     const currentAlerts = GlobalState.state.alerts || [];
     ```
     with:
     ```js
     const currentAlerts = GlobalState.data.alerts || [];
     ```
  3. At line 180 (search input handler), replace the same `GlobalState.state.alerts` reference with `GlobalState.data.alerts`.
  4. At line 197 (`bulkBlockBtn` handler), replace:
     ```js
     const alerts = GlobalState.state.alerts || [];
     ```
     with:
     ```js
     const alerts = GlobalState.data.alerts || [];
     ```
  5. At lines 285–286 (inside `Socket.on('alert.dismissed')`), replace:
     ```js
     const alerts = GlobalState.state.alerts || [];
     const updated = alerts.filter(a => a.id !== data.alert_id);
     ```
     with:
     ```js
     const alerts = GlobalState.data.alerts || [];
     const updated = alerts.filter(a => a.id !== data.alert_id);
     ```
  6. Search the entire file for any remaining `GlobalState.state` occurrence. If any are found, replace with `GlobalState.data`.
  7. Save the file.
- **Expected Frontend Behavior After Fix:** Clicking severity filter chips updates the alerts list. Typing in the search input filters alerts as you type. Bulk dismiss runs without console error. Receiving an `alert.dismissed` socket event removes the alert from the rendered list.
- **Validation / Testing Instructions:**
  - Manual: Open `/alerts` with the DevTools console open. Click each severity filter chip — no `TypeError`. Type into the search input — no error. Click "Bulk Dismiss" with at least one alert selected — no error.
  - Manual (socket): If a back-end emitter is available, emit `alert.dismissed` with a known alert id and confirm the alert disappears.
  - Console: Must be free of `TypeError: Cannot read properties of undefined (reading 'alerts')`.
- **Potential Side Effects:** None — the new property name is the documented one on the new `GlobalState`.
- **Related Systems Impacted:** None.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert` the single commit.
- **Stability Checkpoint Triggered:** Yes
- **UI Consistency Checkpoint Triggered:** No
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 1.3 — Replace `Socket.socket.on(...)` with `Socket.on(...)` in `monitor.page.js`

- **Task ID:** T-1.3
- **Source Issue IDs:** ISS-005
- **Issue-to-Task Mapping Rationale:** Directly resolves the null-socket crash documented in ISS-005 / FIX-003.
- **Problem Being Solved:** `monitor.page.js` accesses `Socket.socket.on(...)` at module top level. When no JWT is present, `SocketManager.connect()` defers connection and `this.socket` is `null`, so the property access throws.
- **Isolated or Architectural:** Isolated
- **Preconditions / Dependencies:** None.
- **Exact Files Involved:**
  - `web_app/static/js/pages/monitor.page.js` — only file modified.
- **Exact Code Areas Involved:** Lines 74–80 per the report.
- **Systems Affected:** Monitor page initialization; socket subscription on monitor only.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report FIX-003: "`Socket.on()` (the public API method on SocketManager) guards `if (this.socket)` internally. This prevents the TypeError when socket is null."
- **Implementation Objective:** `/monitor` loads cleanly with no `TypeError` even when no JWT is present, and the connection-status chip still reacts when the socket later connects.
- **Detailed Implementation Actions:**
  1. Open `web_app/static/js/pages/monitor.page.js`.
  2. Locate lines 74–80, currently:
     ```js
     Socket.socket.on('connect', () => {
         updateConnectionStatus('connected');
     });

     Socket.socket.on('disconnect', () => {
         updateConnectionStatus('disconnected');
     });
     ```
  3. Replace exactly with:
     ```js
     Socket.on('connect', () => {
         updateConnectionStatus('connected');
         console.log('%c[Monitor] Socket connected', 'color:#10b981;font-weight:bold;');
     });

     Socket.on('disconnect', () => {
         updateConnectionStatus('disconnected');
         console.warn('[Monitor] Socket disconnected');
     });
     ```
  4. Below those two listeners, add a reactive subscription so the connection chip updates when the socket later connects after login:
     ```js
     GlobalState.subscribe('socket', (state) => {
         if (state && typeof state.connected === 'boolean') {
             updateConnectionStatus(state.connected ? 'connected' : 'disconnected');
         }
     });
     ```
     If `monitor.page.js` does not currently import `GlobalState`, add the import at the top of the file using the same relative path used by other `pages/*.page.js` files (`../core/global-state.js`). The exact import location must match the project convention — REQUIRES_REPO_INSPECTION if uncertain about the existing import block layout.
  5. Save the file.
- **Expected Frontend Behavior After Fix:** Navigating to `/monitor` produces no console error at module init. The connection status chip starts as "disconnected" if no token is present, and updates to "connected" once the socket connects.
- **Validation / Testing Instructions:**
  - Manual (logged out): Visit `/monitor`. Console must be free of `TypeError: Cannot read properties of null (reading 'on')`.
  - Manual (logged in): Reload `/monitor` after login. The connection chip transitions to "connected".
  - All monitor sections (metrics, alerts, actions, engine cards) mount their containers (even if data is still pending).
- **Potential Side Effects:** If `Socket.on` does not internally guard `if (this.socket)`, listeners attached before the socket exists will be lost. Confirm with the SocketManager source (REQUIRES_REPO_INSPECTION if implementation differs from the report's described behavior). If `Socket.on` does not buffer, register the listeners again in the `GlobalState.subscribe('socket', ...)` callback once `state.connected === true`.
- **Related Systems Impacted:** Connection-status chip in the monitor topbar.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert` the single commit.
- **Stability Checkpoint Triggered:** Yes
- **UI Consistency Checkpoint Triggered:** No
- **Production-Readiness Checkpoint Triggered:** No

---

### Phase 2 — Template Repair

**Phase Objective:** Resolve incomplete-migration template defects so every page renders the intended HTML and loads the intended JS controller.

**Entry Preconditions:** Phase 1 complete (Stability Checkpoint passed).

**Exit Criteria:**
- `detection.html`, `health.html`, `realtime.html` each contain exactly one `{% block content %}`, no orphaned CSS outside any block, and their CSS appears inside `{% block extra_css %}`.
- `/detection` loads `detection.page.js` (modern ES module controller), not `detection.js`.
- Pages no longer silently drop CSS due to Jinja block-shadowing.

**Stability Checkpoint (Phase 2):** `/detection`, `/health`, `/realtime` render with both content markup and styles applied; no JS console errors at load.

**UI Consistency Checkpoint (Phase 2):** The detection page renders the modern Tailwind UI (single design, not the legacy `ds-card` version). `/health` and `/realtime` retain their CSS rules (previously orphaned).

**Production-Readiness Checkpoint:** N/A.

---

#### Task 2.1 — Repair `detection.html` dual-block structure and rewire controller

- **Task ID:** T-2.1
- **Source Issue IDs:** ISS-001, ISS-014
- **Issue-to-Task Mapping Rationale:** ISS-001 (dual `{% block content %}` and orphaned CSS) and ISS-014 (wrong controller loaded) are the same template defect; resolving the block structure also resolves the controller wiring.
- **Problem Being Solved:** `detection.html` defines `{% block content %}` twice; Jinja takes the second (legacy `ds-card`) version. CSS between the two blocks is silently discarded. The `{% block scripts %}` at the end loads legacy `detection.js`; the modern `detection.page.js` is buried inside the discarded first block.
- **Isolated or Architectural:** Architectural (template structure).
- **Preconditions / Dependencies:** T-1.1, T-1.2, T-1.3 complete (runtime stable so the page can be loaded for validation).
- **Exact Files Involved:**
  - `web_app/templates/detection.html` — the file being restructured.
- **Exact Code Areas Involved:**
  - Lines 6–118: first `{% block content %}` (Tailwind version, currently discarded).
  - Lines 119–327: CSS outside any block (currently discarded).
  - Lines 330–536: second `{% block content %}` (legacy `ds-card` version, currently rendered).
  - Lines 538–540: `{% block scripts %}` loading legacy `detection.js`.
- **Systems Affected:** Template inheritance, CSS layer for the detection page, JS module loader for the detection page.
- **Architecture Impact:** Local to one page, but resolves the migration end-state for that page.
- **Reason for This Change (traced to report):** Report FIX-005: "Remove lines 6–118 (the Tailwind version that's being discarded anyway by Jinja2). Move the CSS from lines 119–327 into a proper `{% block extra_css %}<style>...</style>{% endblock %}` before the second `{% block content %}`. Remove the `</style>` orphan tag from the CSS section. Change the `{% block scripts %}` to load `detection.page.js` instead of `detection.js`."
- **Implementation Objective:** `detection.html` has exactly one `{% block content %}` (the `ds-card` version per FIX-005), one `{% block extra_css %}` containing the previously-orphaned CSS, and one `{% block scripts %}` loading `detection.page.js`.
- **Detailed Implementation Actions:**
  1. Open `web_app/templates/detection.html`.
  2. Delete lines 6–118 inclusive (the entire first `{% block content %}` through its `{% endblock %}`), including the `<script type="module" src="...detection.page.js"></script>` line embedded inside it.
  3. The CSS region originally at lines 119–327 (which previously sat outside any block) becomes the next contiguous content. Wrap it with:
     ```jinja
     {% block extra_css %}
     <style>
     /* CSS content previously at lines 119–327 */
     </style>
     {% endblock %}
     ```
     If the existing CSS region already contains a stray `</style>` opener/closer pair, normalize so there is exactly one `<style>` and one `</style>` inside the `{% block extra_css %}`.
  4. Confirm the surviving `{% block content %}` (originally lines 330–536, now the only one) is unchanged in its inner markup.
  5. Replace the `{% block scripts %}` body at the end of the file:
     ```jinja
     {% block scripts %}
     <script src="{{ url_for('static', filename='js/detection.js') }}"></script>
     {% endblock %}
     ```
     with:
     ```jinja
     {% block scripts %}
     <script type="module" src="{{ url_for('static', filename='js/pages/detection.page.js') }}"></script>
     {% endblock %}
     ```
  6. Confirm the file now contains exactly:
     - One `{% extends "base.html" %}`
     - One `{% block title %}` (if originally present)
     - One `{% block page_title %}` (if originally present)
     - One `{% block extra_css %}`
     - One `{% block content %}`
     - One `{% block scripts %}`
  7. Save the file.
- **Expected Frontend Behavior After Fix:** `/detection` renders the legacy `ds-card` markup (the only surviving content block) styled by the recovered CSS, and its JS is provided by the modern `detection.page.js` ES module controller. No CSS is silently discarded by Jinja.
- **Validation / Testing Instructions:**
  - Manual: Load `/detection`. Inspect the network panel — confirm `detection.page.js` is fetched and `detection.js` is **not** fetched.
  - Visual: Card borders, monospace columns, the sync banner, the layout grid, and engine result panels render with the recovered styles (these styles still rely on the CSS variables fixed in T-4.1; until Phase 4 lands, some borders/colors may render with fallback values — note but do not block).
  - Console: No JS errors at load.
- **Potential Side Effects:** The modern `detection.page.js` may expect DOM hooks (IDs/classes) different from the legacy `ds-card` markup. REQUIRES_REPO_INSPECTION: cross-check the selectors `detection.page.js` queries against the elements present in the surviving content block. If selectors do not match, file a follow-up issue (do **not** rewrite the content block in this task; preserve report-defined scope).
- **Related Systems Impacted:** Legacy `detection.js` becomes unreferenced after this change (its removal is scheduled in Phase 7).
- **Regression Risk:** Medium — controller/markup contract may need follow-up if selectors don't align.
- **Rollback Considerations:** `git revert`. No build/asset cleanup required.
- **Stability Checkpoint Triggered:** Yes
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 2.2 — Wrap orphaned CSS in `health.html` inside `{% block extra_css %}`

- **Task ID:** T-2.2
- **Source Issue IDs:** ISS-001 (health.html instance)
- **Issue-to-Task Mapping Rationale:** ISS-001 identifies the same orphaned-CSS-after-content-block pattern in `health.html` (CSS after line 72).
- **Problem Being Solved:** CSS placed in `health.html` outside any `{% block ... %}` is silently dropped by Jinja2 in a child template.
- **Isolated or Architectural:** Isolated (one template).
- **Preconditions / Dependencies:** T-1.1, T-1.2, T-1.3.
- **Exact Files Involved:**
  - `web_app/templates/health.html` — the file being restructured.
- **Exact Code Areas Involved:** The CSS region beginning after line 72 (the report's marker). REQUIRES_REPO_INSPECTION to confirm exact line range; the operation is to locate the `<style>...</style>` block that is outside any `{% block %}`.
- **Systems Affected:** Health page CSS layer.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report ISS-001: "Same pattern: `health.html` (CSS orphaned after line 72)." Report FIX-005: "Apply same fix to: `health.html` (move CSS after line 72 into `{% block extra_css %}`)."
- **Implementation Objective:** All `<style>` content in `health.html` lives inside `{% block extra_css %}` and is rendered to the browser.
- **Detailed Implementation Actions:**
  1. Open `web_app/templates/health.html`.
  2. Locate the `<style>...</style>` block that sits outside any `{% block ... %}` (per the report, this begins after line 72).
  3. Wrap it as:
     ```jinja
     {% block extra_css %}
     <style>
     /* health page CSS, previously orphaned */
     </style>
     {% endblock %}
     ```
  4. Ensure there is exactly one `{% block extra_css %}` in the file. If one already exists earlier in the template, merge the orphaned CSS into the existing block rather than declaring two `{% block extra_css %}` blocks.
  5. Confirm no other top-level content sits outside a `{% block ... %}` (a child template's only valid top-level content is `{% extends %}` and `{% block %}` declarations).
  6. Save the file.
- **Expected Frontend Behavior After Fix:** `/health` renders with its custom CSS rules applied.
- **Validation / Testing Instructions:**
  - Manual: Load `/health`. Inspect any element previously expected to be styled by the orphaned CSS — its computed styles should now include rules from the recovered `<style>` block.
  - DOM check: View the rendered HTML source — the `<style>` content from `health.html` must appear inside the `<head>` (where `base.html`'s `{% block extra_css %}` slot is, per the report's snapshot).
- **Potential Side Effects:** None.
- **Related Systems Impacted:** None.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** No
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 2.3 — Wrap orphaned CSS in `realtime.html` inside `{% block extra_css %}`

- **Task ID:** T-2.3
- **Source Issue IDs:** ISS-001 (realtime.html instance)
- **Issue-to-Task Mapping Rationale:** ISS-001 identifies the same orphaned-CSS pattern in `realtime.html` (CSS after line 76).
- **Problem Being Solved:** Identical to T-2.2 but for `realtime.html`.
- **Isolated or Architectural:** Isolated.
- **Preconditions / Dependencies:** T-1.1, T-1.2, T-1.3.
- **Exact Files Involved:**
  - `web_app/templates/realtime.html`
- **Exact Code Areas Involved:** CSS region after line 76 per the report (REQUIRES_REPO_INSPECTION for exact line range).
- **Systems Affected:** Realtime page CSS layer.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report FIX-005: "Apply same fix to: ... `realtime.html` (CSS after line 76)."
- **Implementation Objective:** All `<style>` content in `realtime.html` lives inside `{% block extra_css %}`.
- **Detailed Implementation Actions:**
  1. Open `web_app/templates/realtime.html`.
  2. Locate the `<style>...</style>` block outside any `{% block ... %}`.
  3. Wrap it inside `{% block extra_css %}<style>...</style>{% endblock %}`. If a block already exists, merge.
  4. Save the file.
- **Expected Frontend Behavior After Fix:** `/realtime` renders with its CSS applied.
- **Validation / Testing Instructions:**
  - Manual: Load `/realtime`. Confirm styles previously authored in the orphaned block apply.
  - DOM check: The recovered `<style>` content appears in `<head>`.
- **Potential Side Effects:** None.
- **Related Systems Impacted:** None.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** No
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** No

---

### Phase 3 — Component Contract Fixes

**Phase Objective:** Repair the three component/controller contract mismatches that cause silent failures (block action, dashboard modules, alert flow display).

**Entry Preconditions:** Phase 2 complete.

**Exit Criteria:**
- Block IP action in `AlertCard` produces a success toast on success and updates the action history in `GlobalState`.
- Dashboard modules grid renders module cards with the correct titles (not `[object Object]`).
- Real-time alert cards display source/destination flow correctly (no `??→??` when data is present).

**Stability Checkpoint (Phase 3):** Triggering a block action does not throw; clicking the dashboard returns module cards; alert cards render flow.

**UI Consistency Checkpoint (Phase 3):** Module card titles match `module.title`; alert cards show populated `source_ip` rows.

**Production-Readiness Checkpoint:** N/A.

---

#### Task 3.1 — Fix `AlertCard` block-action response handling

- **Task ID:** T-3.1
- **Source Issue IDs:** ISS-006
- **Issue-to-Task Mapping Rationale:** Directly resolves the always-false `response.ok || response.status === 201` check.
- **Problem Being Solved:** `HttpClient.post()` returns parsed JSON body, not a `Response` object. The current code checks properties that do not exist on a JSON body and never enters its success branch.
- **Isolated or Architectural:** Isolated (single component, coupled with T-3.3 for the same `try` block).
- **Preconditions / Dependencies:** T-3.3 (field-name fix in the same component) — this task and T-3.3 may be committed together as a single coordinated change.
- **Exact Files Involved:**
  - `web_app/static/js/components/alert-card.js`
- **Exact Code Areas Involved:** Lines 89–123 per the report (`handleBlockIP` function).
- **Systems Affected:** Alert card component; `/api/actions` POST flow; `actions` slice in `GlobalState`.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report FIX-004: "HttpClient.post returns parsed JSON body, not a Response object."
- **Implementation Objective:** A successful block action produces a success toast, marks the button as Blocked, and pushes an action record into `GlobalState`. A failed block action produces an error toast.
- **Detailed Implementation Actions:**
  1. Open `web_app/static/js/components/alert-card.js`.
  2. Locate the `handleBlockIP` function (or the inline handler at lines 89–123 per the report).
  3. Remove the `if (response.ok || response.status === 201) { ... }` wrapper. Treat a non-throwing `await HttpClient.post(...)` as success.
  4. Replace the success branch body so it runs unconditionally on resolved promise:
     ```js
     AppToast.success(`Blocked IP: ${alert.source_ip}`);
     GlobalState.push("actions", {
         id: response?.id,
         type: "block",
         target: alert.source_ip,
         timestamp: new Date().toISOString(),
         status: "executed",
     });
     blockBtn.disabled = true;
     blockBtn.textContent = "✓ Blocked";
     ```
  5. Confirm the existing outer `try`/`catch` calls `AppToast.error(...)` in the `catch` branch with the error message. If the existing `catch` only logs, add:
     ```js
     AppToast.error(`Failed to block IP: ${error.message}`);
     ```
  6. Apply the loading-toast lifecycle described in report FIX-009 (this task subsumes FIX-009):
     - Before the `await`, store the loading toast id: `const loadingToastId = AppToast.loading("Blocking IP address...");`
     - In the `try` after the await success block, call `AppToast.dismiss(loadingToastId);` **before** the success toast.
     - In the `catch`, call `AppToast.dismiss(loadingToastId);` **before** the error toast.
     - If `AppToast.loading` does not exist, REQUIRES_REPO_INSPECTION to choose between (a) adding it as a thin wrapper over `AppToast.show("...", {type:"info", duration:false})` returning an id, or (b) keeping the existing `AppToast.show` call and capturing whatever handle it returns.
  7. Save the file.
- **Expected Frontend Behavior After Fix:** Clicking "Block" on an alert card shows a transient loading toast, then on success replaces it with a success toast and disables the button (label "✓ Blocked"). On failure, the loading toast is replaced by an error toast describing the failure.
- **Validation / Testing Instructions:**
  - Manual: With at least one alert displayed and a valid JWT, click the Block button. Confirm:
    - A loading toast appears immediately.
    - The loading toast is dismissed when the response returns.
    - A success toast appears with the IP.
    - The button is disabled and labeled "✓ Blocked".
    - `GlobalState.data.actions` (via console) contains the new action record.
  - Failure path: simulate by temporarily blocking `/api/actions` in DevTools → confirm error toast and that the button is not stuck in the loading state.
- **Potential Side Effects:** If other components elsewhere in the codebase relied on `HttpClient.post()` returning a `Response`-like object, they may need the same fix. REQUIRES_REPO_INSPECTION across other `HttpClient.post` callers.
- **Related Systems Impacted:** `GlobalState.actions` slice consumers (action history panels on dashboard, actions page).
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert`. No artifacts.
- **Stability Checkpoint Triggered:** Yes
- **UI Consistency Checkpoint Triggered:** No
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 3.2 — Fix `ModuleCard` call signature in `dashboard.page.js`

- **Task ID:** T-3.2
- **Source Issue IDs:** ISS-007
- **Issue-to-Task Mapping Rationale:** Directly addresses the data-object-passed-as-id mismatch identified in ISS-007.
- **Problem Being Solved:** `dashboard.page.js` calls `ModuleCard(module)` passing a data object; `ModuleCard(moduleId, config)` expects a string id plus a config object. The first argument becomes the title, rendered as `[object Object]`.
- **Isolated or Architectural:** Isolated (one call site).
- **Preconditions / Dependencies:** None.
- **Exact Files Involved:**
  - `web_app/static/js/pages/dashboard.page.js` — call site to be fixed.
  - `web_app/static/js/components/module-card.js` — read-only reference for the signature `ModuleCard(moduleId, config = {})`.
- **Exact Code Areas Involved:** Lines 60–72 per the report, inside the `GlobalState.subscribe('modules', ...)` block, specifically `modules.forEach((module, index) => { const card = ModuleCard(module); ... })`.
- **Systems Affected:** Dashboard module grid rendering.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report ISS-007: "`dashboard.page.js` treats `modules` as an array of module data objects and passes each object directly to `ModuleCard()`. But `ModuleCard` expects `(moduleId: string, config: object)`."
- **Implementation Objective:** Each module renders as a card with its real title.
- **Detailed Implementation Actions:**
  1. Open `web_app/static/js/pages/dashboard.page.js`.
  2. Locate the `modules.forEach((module, index) => { ... })` block at lines 60–72.
  3. Replace the `const card = ModuleCard(module);` call with:
     ```js
     const card = ModuleCard(module.id, {
         title: module.title || module.name || module.id,
         status: module.status,
         description: module.description,
         metrics: module.metrics,
         enabled: module.enabled,
     });
     ```
     If the module data object uses a different identifier field, REQUIRES_REPO_INSPECTION on the actual shape emitted by `/api/dashboard/metrics` or whichever endpoint populates the `modules` slice. Use the field name that exists; do not invent.
  4. Save the file.
- **Expected Frontend Behavior After Fix:** Dashboard "System Modules" grid renders one card per module with a human-readable title, not `[object Object]`.
- **Validation / Testing Instructions:**
  - Manual: Load `/dashboard` with modules data present (or seed via dev tools by calling `GlobalState.set('modules', [{id:'foo', title:'Foo Module', status:'ok'}, ...])`).
  - Confirm each card shows the real title.
- **Potential Side Effects:** If any module data lacks an `id` field, the resulting card will have an empty id; coordinate with the back-end emitter (REQUIRES_REPO_INSPECTION).
- **Related Systems Impacted:** `GlobalState.modules` slice shape (also see T-6.5 for the array-vs-object initialization issue).
- **Regression Risk:** Low–Medium.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** Yes
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 3.3 — Align `AlertCard` field names with normalized schema

- **Task ID:** T-3.3
- **Source Issue IDs:** ISS-008
- **Issue-to-Task Mapping Rationale:** Directly addresses the `src_ip` vs `source_ip` divergence between `AlertCard` and `SocketManager._normalizeAlert()`.
- **Problem Being Solved:** `AlertCard` reads `alert.src_ip` and `alert.dst_ip`; the normalized schema produced by `SocketManager` exposes `source_ip`. As a result, source/destination flow renders as `??→??` and `target_ip` in the block API call is `undefined`.
- **Isolated or Architectural:** Isolated.
- **Preconditions / Dependencies:** None. Coordinate the same commit with T-3.1 if both touch the same `handleBlockIP` block.
- **Exact Files Involved:**
  - `web_app/static/js/components/alert-card.js`
- **Exact Code Areas Involved:** Lines 37–46 (flow rendering) and line 97 (block API call target field) per the report.
- **Systems Affected:** Alert card rendering and block-action payload shape.
- **Architecture Impact:** Local; aligns to the canonical alert schema documented in the audit.
- **Reason for This Change (traced to report):** Report ISS-008: "AlertCard was written against a different alert schema than what `SocketManager._normalizeAlert()` produces."
- **Implementation Objective:** `AlertCard` reads only fields present in the normalized alert schema (`source_ip` for source). Block action sends `target_ip: alert.source_ip`.
- **Detailed Implementation Actions:**
  1. Open `web_app/static/js/components/alert-card.js`.
  2. In the flow-render section (around lines 37–46), replace the condition and references:
     - `if (alert.src_ip || alert.dst_ip) {` → `if (alert.source_ip || alert.dst_ip) {`
     - `<span>${alert.src_ip || '??'}</span>` → `<span>${alert.source_ip || '??'}</span>`
     - Leave `alert.dst_ip` as-is for now. The report explicitly states `dst_ip` is "not in normalized schema at all" — REQUIRES_REPO_INSPECTION on the actual normalized schema (the report's enumeration of normalized fields is `id, timestamp, severity, prediction, confidence, status, profile, source_ip, attack_type, reason` — it does not name a destination field). If no destination field exists, the `<span>${alert.dst_ip || '??'}</span>` will continue rendering `??`. Open a follow-up issue rather than guessing a destination field name.
  3. In the block API call body (around line 97), replace:
     ```js
     target_ip: alert.src_ip,
     ```
     with:
     ```js
     target_ip: alert.source_ip,
     ```
  4. Verify any toast/log lines that interpolate `alert.src_ip` are updated to `alert.source_ip` (search the file for `src_ip`).
  5. Save the file.
- **Expected Frontend Behavior After Fix:** Alert cards display the source IP correctly. The block action sends a defined `target_ip`.
- **Validation / Testing Instructions:**
  - Manual: With at least one alert in `GlobalState.data.alerts`, confirm the flow renders the source IP from `source_ip`.
  - Network: Click Block. Confirm the request body includes `"target_ip": "<actual ip>"` not `undefined`.
- **Potential Side Effects:** Any other component referencing `alert.src_ip` will continue to misread. REQUIRES_REPO_INSPECTION for other `src_ip` references in `static/js/`.
- **Related Systems Impacted:** Action history (uses the IP that was blocked), alerts list rendering.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** Yes
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** No

---

### Phase 4 — CSS System Reconstruction

**Phase Objective:** Define the missing design tokens and missing structural classes so templates render with the intended visual system.

**Entry Preconditions:** Phase 3 complete.

**Exit Criteria:**
- The `:root` selector in `base.html` defines every CSS variable enumerated in the audit (`--mono`, `--border`, `--border-bright`, `--bg-card`, `--bg-elevated`, `--bg-base`, `--text-primary`, `--text-secondary`, `--text-muted`, `--red`, `--red-lt`, `--green`, `--amber`, `--blue`).
- The audit-enumerated undefined classes (`.page-header`, `.page-title`, `.page-subtitle`, `.page-meta`, `.reconcile-strip`, `.rec-cell`, `.rec-label`, `.rec-val`, `.panel-header`, `.panel-title`, `.panel-kicker`, `.mono-val`, and `.page-wrapper` as a class selector) all have definitions in `base.html` or a dedicated CSS file.

**Stability Checkpoint (Phase 4):** Templates that previously rendered transparent cards or unstyled headers now render with visible borders, backgrounds, and structured headings.

**UI Consistency Checkpoint (Phase 4):** Detection, engines, health, home, index, respond, realtime, threat_intel, batch pages display borders, monospace columns, semantic colors, and card backgrounds per the design.

**Production-Readiness Checkpoint:** N/A.

---

#### Task 4.1 — Define design-token CSS variables in `base.html` `:root`

- **Task ID:** T-4.1
- **Source Issue IDs:** ISS-002
- **Issue-to-Task Mapping Rationale:** Directly implements report FIX-001.
- **Problem Being Solved:** Templates and JS files reference CSS variables that are not defined anywhere. Every `var(--border)`, `var(--mono)`, etc. silently falls back to its initial value.
- **Isolated or Architectural:** Architectural (foundation for all template CSS).
- **Preconditions / Dependencies:** None.
- **Exact Files Involved:**
  - `web_app/templates/base.html` — the `:root` selector inside the `<style>` block.
- **Exact Code Areas Involved:** The `:root { ... }` declaration after line 25 per the report.
- **Systems Affected:** CSS cascade for every template.
- **Architecture Impact:** Foundational. All Phase 4 and later visual fixes depend on this.
- **Reason for This Change (traced to report):** Report FIX-001 verbatim: "Resolves ISSUE-002. All template CSS that references these variables will now have valid values."
- **Implementation Objective:** All referenced custom properties are defined in `:root` with the exact values supplied by the audit.
- **Detailed Implementation Actions:**
  1. Open `web_app/templates/base.html`.
  2. Locate the existing `:root { ... }` rule inside the inline `<style>` block (after line 25 per the report).
  3. Replace it with the full block from FIX-001:
     ```css
     :root {
         --sidebar-w: 240px;
         --sidebar-collapsed-w: 64px;
         --topbar-h: 56px;

         /* Design tokens referenced throughout templates */
         --mono: 'JetBrains Mono', monospace;
         --border: rgba(255,255,255,0.07);
         --border-bright: rgba(255,255,255,0.14);
         --bg-card: #0f1117;
         --bg-elevated: #151922;
         --bg-base: #090c12;
         --text-primary: rgba(255,255,255,0.85);
         --text-secondary: rgba(255,255,255,0.55);
         --text-muted: rgba(255,255,255,0.35);
         --red: #ef4444;
         --red-lt: #f87171;
         --green: #10b981;
         --amber: #f59e0b;
         --blue: #3b82f6;
     }
     ```
  4. Do not remove any existing custom property already declared in `:root`. If the existing block already declares `--sidebar-w`, `--sidebar-collapsed-w`, or `--topbar-h` with different values, keep the existing values for those three and add only the new tokens below them.
  5. Save the file.
- **Expected Frontend Behavior After Fix:** Card borders are visible (`rgba(255,255,255,0.07)`). Monospace columns use JetBrains Mono. Card backgrounds are dark (`#0f1117`). Severity colors render in red/amber/green where used.
- **Validation / Testing Instructions:**
  - Visit `/detection`, `/engines`, `/health`, `/home`, `/index`, `/respond`, `/realtime`, `/threat-intel`, `/batch`. Each page that previously rendered transparent cards or default text colors now renders the design tokens.
  - DevTools → Computed → `--border` resolves to `rgba(255,255,255,0.07)` at `<html>`.
- **Potential Side Effects:** None — adding tokens that were already referenced cannot break existing rules.
- **Related Systems Impacted:** Every template using these tokens.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** No
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 4.2 — Define missing layout classes in `base.html`

- **Task ID:** T-4.2
- **Source Issue IDs:** ISS-011
- **Issue-to-Task Mapping Rationale:** Directly addresses the audit's enumerated list of undefined CSS classes that break structural layout.
- **Problem Being Solved:** Class names used across many templates are not defined in any stylesheet, leaving elements unstyled.
- **Isolated or Architectural:** Architectural (extends the design system foundation).
- **Preconditions / Dependencies:** T-4.1 (some classes rely on the tokens added there).
- **Exact Files Involved:**
  - `web_app/templates/base.html` — the inline `<style>` block.
- **Exact Code Areas Involved:** A new block of CSS rules appended at the end of the existing inline `<style>` block in `base.html`. Do not place them in `dashboard.css` — that file is `146B` per the audit and is page-scoped to `.chart-container`.
- **Systems Affected:** Visual layout for `home.html`, `404.html`, `error.html`, `index.html`, `detection.html`, `engines.html`, `batch.html`, `capture.html`, `about.html`, `investigate.html`, `learn.html`, `monitor.html`, `predict.html`, `respond.html`, `threat_intel.html`.
- **Architecture Impact:** Foundational.
- **Reason for This Change (traced to report):** Report ISS-011 enumerates the classes by name and states: "Major structural layout breaks on multiple pages. Elements render as unstyled blocks."
- **Implementation Objective:** Each class listed in ISS-011 has a definition that provides reasonable layout, typography, and spacing consistent with the design tokens from T-4.1.
- **Detailed Implementation Actions:**
  1. Open `web_app/templates/base.html`.
  2. At the end of the inline `<style>` block (before `</style>`), append:
     ```css
     /* ------------------------------------------------------------------
        Layout primitives — restored from legacy design system (ISS-011)
        ------------------------------------------------------------------ */
     .page-header {
         display: flex;
         flex-direction: column;
         gap: 6px;
         padding: 20px 24px 16px;
         border-bottom: 1px solid var(--border);
         background: var(--bg-base);
     }
     .page-title {
         font-family: 'Syne', system-ui, sans-serif;
         font-size: 22px;
         line-height: 1.2;
         font-weight: 600;
         color: var(--text-primary);
         letter-spacing: 0.2px;
     }
     .page-subtitle {
         font-size: 13px;
         color: var(--text-secondary);
     }
     .page-meta {
         display: flex;
         flex-wrap: wrap;
         gap: 16px;
         font-family: var(--mono);
         font-size: 11px;
         color: var(--text-muted);
         text-transform: uppercase;
         letter-spacing: 0.6px;
     }

     /* Reconciliation strip (dashboard summary cells) */
     .reconcile-strip {
         display: grid;
         grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
         gap: 1px;
         background: var(--border);
         border: 1px solid var(--border);
         border-radius: 6px;
         overflow: hidden;
     }
     .rec-cell {
         display: flex;
         flex-direction: column;
         gap: 4px;
         padding: 12px 16px;
         background: var(--bg-card);
     }
     .rec-label {
         font-family: var(--mono);
         font-size: 10px;
         color: var(--text-muted);
         text-transform: uppercase;
         letter-spacing: 0.8px;
     }
     .rec-val {
         font-family: var(--mono);
         font-size: 18px;
         color: var(--text-primary);
         font-weight: 500;
     }

     /* Card panel headers */
     .panel-header {
         display: flex;
         align-items: baseline;
         justify-content: space-between;
         gap: 12px;
         padding: 14px 18px;
         border-bottom: 1px solid var(--border);
         background: var(--bg-card);
     }
     .panel-title {
         font-family: 'Syne', system-ui, sans-serif;
         font-size: 14px;
         font-weight: 600;
         color: var(--text-primary);
     }
     .panel-kicker {
         font-family: var(--mono);
         font-size: 10px;
         color: var(--text-muted);
         text-transform: uppercase;
         letter-spacing: 0.6px;
     }

     /* Mono value rendering */
     .mono-val {
         font-family: var(--mono);
         color: var(--text-primary);
     }

     /* Compatibility: some legacy templates apply .page-wrapper as a class,
        while base.html exposes #page-wrapper as an id. Provide a no-op
        passthrough so the class doesn't break inheritance. */
     .page-wrapper {
         display: block;
         width: 100%;
     }
     ```
  3. Save the file.
- **Expected Frontend Behavior After Fix:** Page headers render with a title, subtitle, and meta strip. Reconciliation strips show grid-aligned label/value pairs. Panel headers have a baseline-aligned title and kicker. Elements styled with `.mono-val` use the monospace font.
- **Validation / Testing Instructions:**
  - Manual: Visit every page listed in ISS-011 (`home`, `404`, `error`, `index`, `detection`, `engines`, `batch`, `capture`, `about`, `investigate`, `learn`, `monitor`, `predict`, `respond`, `threat_intel`). Each page's header, reconciliation strip (where present), and panel headers render with visible structure.
  - DevTools: Inspect `.page-header`, `.rec-cell`, `.panel-header` on at least three pages — each has computed styles matching the rules above.
- **Potential Side Effects:** Slight visual differences for any element that previously inherited default browser styles via these class names. None should be regressions since the prior state was unstyled.
- **Related Systems Impacted:** All listed templates.
- **Regression Risk:** Low–Medium (large surface area, but additive).
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** No
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** No

---

### Phase 5 — Navigation + UX Restoration

**Phase Objective:** Make all pages reachable from the sidebar, fix the broken home-page navigation link, replace inline-`onclick` navigation with accessible `<a>` tags, and add a favicon.

**Entry Preconditions:** Phase 4 complete.

**Exit Criteria:**
- Sidebar contains links to every primary route enumerated in ISS-013.
- `/threat-intel` is reachable from the home page (no 404).
- Home-page tiles are real anchor elements supporting keyboard nav and right-click "open in new tab".
- `<link rel="icon">` is present in `base.html` and no `/favicon.ico` 404 appears in the network panel.

**Stability Checkpoint (Phase 5):** Every route in the expanded sidebar opens successfully (no 404/500).

**UI Consistency Checkpoint (Phase 5):** Active link highlighting in the sidebar matches `request.path`. Home tiles render visually identical to before but support all anchor-tag behaviors.

**Production-Readiness Checkpoint:** N/A.

---

#### Task 5.1 — Fix home-page Threat Intel tile URL

- **Task ID:** T-5.1
- **Source Issue IDs:** ISS-010
- **Issue-to-Task Mapping Rationale:** Direct fix for the `/threat_intel` vs `/threat-intel` mismatch.
- **Problem Being Solved:** Home tile points to `/threat_intel` (underscore); the registered Flask route is `/threat-intel` (hyphen). Tile produces 404.
- **Isolated or Architectural:** Isolated.
- **Preconditions / Dependencies:** Combine with T-5.3 in a single commit (both touch `home.html` navigation tiles).
- **Exact Files Involved:**
  - `web_app/templates/home.html`
- **Exact Code Areas Involved:** Line 157 per the report (the threat-intel tile).
- **Systems Affected:** Home navigation.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report FIX-007: "Resolves ISSUE-010 (404 for threat intel)."
- **Implementation Objective:** Clicking the Threat Intel home tile navigates to `/threat-intel` and renders the page.
- **Detailed Implementation Actions:**
  1. Open `web_app/templates/home.html`.
  2. Locate line 157, currently:
     ```html
     <div class="ds-card layer-panel home-tile" onclick="location.href='/threat_intel'">
     ```
  3. Per T-5.3 (combined), convert this entire tile to an anchor element with the corrected URL:
     ```html
     <a href="/threat-intel" class="ds-card layer-panel home-tile">
     ```
     Ensure the closing tag at the end of the tile is `</a>` instead of `</div>`.
  4. Save the file.
- **Expected Frontend Behavior After Fix:** Clicking the Threat Intel tile navigates to the threat-intel page; right-click → "Open in new tab" works.
- **Validation / Testing Instructions:**
  - Manual: Visit `/home`. Click the Threat Intel tile. Confirm `/threat-intel` renders.
  - Right-click → Open in new tab works.
- **Potential Side Effects:** Tile may visually change slightly (anchor element default styling). Verify `.ds-card.layer-panel.home-tile` CSS targets both `div` and `a` — it should since the selectors are class-only.
- **Related Systems Impacted:** None.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** Yes
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 5.2 — Expand sidebar navigation to all primary routes

- **Task ID:** T-5.2
- **Source Issue IDs:** ISS-013
- **Issue-to-Task Mapping Rationale:** Directly implements report FIX-010.
- **Problem Being Solved:** Sidebar exposes only 5 of 15+ routes; users cannot reach most pages.
- **Isolated or Architectural:** Architectural (cross-page navigation).
- **Preconditions / Dependencies:** None.
- **Exact Files Involved:**
  - `web_app/templates/base.html` — sidebar markup at lines 821–847 per the report.
- **Exact Code Areas Involved:** `.inids-sidebar-list` element body.
- **Systems Affected:** Sidebar component on every page.
- **Architecture Impact:** Local to a single template (but affects every page that extends it).
- **Reason for This Change (traced to report):** Report FIX-010 provides the exact replacement markup.
- **Implementation Objective:** The sidebar contains the full set of links specified in FIX-010, grouped under "Monitor", "Response", "Intelligence", and "System" section labels, with active-class highlighting based on `request.path`.
- **Detailed Implementation Actions:**
  1. Open `web_app/templates/base.html`.
  2. Locate the `.inids-sidebar-list` element (currently lines 821–847 per the report).
  3. Replace its inner content verbatim with the FIX-010 block:
     ```html
     <div class="nav-section-label">Monitor</div>
     <a href="/" title="Dashboard" class="nav-item {% if request.path == '/' %}active{% endif %}">
         <i class="bi bi-grid-1x2 nav-icon"></i><span>Dashboard</span>
     </a>
     <a href="/monitor" title="Monitor" class="nav-item {% if request.path == '/monitor' %}active{% endif %}">
         <i class="bi bi-display nav-icon"></i><span>Monitor</span>
     </a>
     <a href="/realtime" title="Realtime" class="nav-item {% if request.path == '/realtime' %}active{% endif %}">
         <i class="bi bi-broadcast nav-icon"></i><span>Realtime</span>
     </a>
     <a href="/alerts" title="Alerts" class="nav-item {% if request.path == '/alerts' %}active{% endif %}">
         <i class="bi bi-bell nav-icon"></i><span>Alerts</span>
         <span class="nav-badge">!</span>
     </a>
     <a href="/detection" title="Detection" class="nav-item {% if request.path == '/detection' %}active{% endif %}">
         <i class="bi bi-search nav-icon"></i><span>Detection</span>
     </a>

     <div class="nav-section-label">Response</div>
     <a href="/actions" title="Actions" class="nav-item {% if request.path == '/actions' %}active{% endif %}">
         <i class="bi bi-shield-lock nav-icon"></i><span>Actions</span>
     </a>
     <a href="/respond" title="Respond" class="nav-item {% if request.path == '/respond' %}active{% endif %}">
         <i class="bi bi-lightning nav-icon"></i><span>Respond</span>
     </a>
     <a href="/honeypot" title="Honeypot" class="nav-item {% if request.path == '/honeypot' %}active{% endif %}">
         <i class="bi bi-bug nav-icon"></i><span>Honeypot</span>
     </a>

     <div class="nav-section-label">Intelligence</div>
     <a href="/threat-intel" title="Threat Intel" class="nav-item {% if request.path == '/threat-intel' %}active{% endif %}">
         <i class="bi bi-globe2 nav-icon"></i><span>Threat Intel</span>
     </a>
     <a href="/investigate" title="Investigate" class="nav-item {% if request.path == '/investigate' %}active{% endif %}">
         <i class="bi bi-journal-text nav-icon"></i><span>Investigate</span>
     </a>

     <div class="nav-section-label">System</div>
     <a href="/policy" title="Policy" class="nav-item {% if request.path == '/policy' %}active{% endif %}">
         <i class="bi bi-sliders nav-icon"></i><span>Policy</span>
     </a>
     <a href="/allowlist" title="Allowlist" class="nav-item {% if request.path == '/allowlist' %}active{% endif %}">
         <i class="bi bi-list-check nav-icon"></i><span>Allowlist</span>
     </a>
     <a href="/models" title="Models" class="nav-item {% if request.path == '/models' %}active{% endif %}">
         <i class="bi bi-cpu nav-icon"></i><span>Models</span>
     </a>
     <a href="/health" title="Health" class="nav-item {% if request.path == '/health' %}active{% endif %}">
         <i class="bi bi-activity nav-icon"></i><span>Health</span>
     </a>
     <a href="/capture" title="Capture" class="nav-item {% if request.path == '/capture' %}active{% endif %}">
         <i class="bi bi-camera-video nav-icon"></i><span>Capture</span>
     </a>
     <a href="/learn" title="Learn" class="nav-item {% if request.path == '/learn' %}active{% endif %}">
         <i class="bi bi-book nav-icon"></i><span>Learn</span>
     </a>
     ```
  4. If `.nav-section-label` is not styled in `base.html`, add a minimal rule:
     ```css
     .nav-section-label {
         padding: 12px 18px 4px;
         font-family: var(--mono);
         font-size: 10px;
         text-transform: uppercase;
         letter-spacing: 0.8px;
         color: var(--text-muted);
     }
     ```
     REQUIRES_REPO_INSPECTION: check whether `.nav-section-label` already has a rule before adding.
  5. Save the file.
- **Expected Frontend Behavior After Fix:** Sidebar lists every navigation entry from FIX-010 with section labels. The current page's link has the `active` class.
- **Validation / Testing Instructions:**
  - Manual: From the dashboard, click every sidebar link. Each navigates to a renderable page.
  - Manual: Verify that the active link visually distinguishes from other links (via existing `.nav-item.active` styles).
- **Potential Side Effects:** Sidebar height may increase to the point of scrolling on small viewports. Confirm the sidebar's overflow behavior is `auto` (REQUIRES_REPO_INSPECTION if not).
- **Related Systems Impacted:** All pages.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** Yes
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 5.3 — Convert all `home.html` `onclick` tiles to `<a>` anchors

- **Task ID:** T-5.3
- **Source Issue IDs:** ISS-021
- **Issue-to-Task Mapping Rationale:** Directly implements the accessibility fix from FIX-007.
- **Problem Being Solved:** Inline `onclick="location.href='...'"` on `<div>` tiles breaks keyboard navigation, screen readers, right-click context, and requires JS.
- **Isolated or Architectural:** Isolated (single template).
- **Preconditions / Dependencies:** Combine with T-5.1 in the same commit.
- **Exact Files Involved:**
  - `web_app/templates/home.html`
- **Exact Code Areas Involved:** All `home-tile` elements at lines 107, 117, 127, 137, 147, 157, 167, 177 per the report.
- **Systems Affected:** Home page tile component.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report FIX-007.
- **Implementation Objective:** Every home tile is an `<a href="...">` with the same classes and inner content as before.
- **Detailed Implementation Actions:**
  1. Open `web_app/templates/home.html`.
  2. For each tile element at lines 107, 117, 127, 137, 147, 157, 167, 177:
     - Replace the opening tag pattern:
       ```html
       <div class="ds-card layer-panel home-tile" onclick="location.href='<URL>'">
       ```
       with:
       ```html
       <a href="<URL>" class="ds-card layer-panel home-tile">
       ```
     - Replace the corresponding closing `</div>` with `</a>`.
  3. For the tile at line 157 specifically, also apply T-5.1's URL correction: the `<URL>` value must be `/threat-intel` (hyphen), not `/threat_intel`.
  4. If the `.home-tile` CSS uses `display` rules incompatible with anchor elements (e.g., `display: block` on a `div`), no change is required — `display: block` works on anchors.
  5. Add minimal anchor reset to `.home-tile` if needed (REQUIRES_REPO_INSPECTION on current `.home-tile` styles):
     ```css
     .home-tile { text-decoration: none; color: inherit; }
     ```
  6. Save the file.
- **Expected Frontend Behavior After Fix:** All home tiles are clickable, keyboard-focusable (Tab through them), and support right-click open-in-new-tab. Visual rendering is unchanged.
- **Validation / Testing Instructions:**
  - Manual: Tab through tiles — focus ring visible on each.
  - Manual: Right-click each tile — context menu offers "Open Link in New Tab".
  - Manual: Disable JavaScript in DevTools, reload `/home`, click tiles — they still navigate.
  - Visual: Confirm tiles still render with original styling.
- **Potential Side Effects:** Slight anchor styling artifacts (underlines, default link color) if the `.home-tile` selector did not previously reset them. Add the reset in step 5 if observed.
- **Related Systems Impacted:** None.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** No
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 5.4 — Add favicon to `base.html`

- **Task ID:** T-5.4
- **Source Issue IDs:** ISS-022
- **Issue-to-Task Mapping Rationale:** Directly implements FIX-008.
- **Problem Being Solved:** Every page load triggers a 404 on `/favicon.ico`.
- **Isolated or Architectural:** Isolated.
- **Preconditions / Dependencies:** None.
- **Exact Files Involved:**
  - `web_app/templates/base.html`
- **Exact Code Areas Involved:** `<head>` block, after line 6 per the report.
- **Systems Affected:** Browser tab icon, 404 noise.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report FIX-008.
- **Implementation Objective:** Browsers no longer 404 on `/favicon.ico`.
- **Detailed Implementation Actions:**
  1. Open `web_app/templates/base.html`.
  2. In the `<head>` after the title tag, insert:
     ```html
     <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🛡</text></svg>">
     ```
  3. Save the file.
- **Expected Frontend Behavior After Fix:** Browser tab shows a shield emoji as the favicon. Network panel no longer shows a 404 for `/favicon.ico`.
- **Validation / Testing Instructions:**
  - Manual: Reload any page; check the browser tab icon (shield emoji visible).
  - Network: Confirm no 404 for `/favicon.ico` in the network panel after a hard refresh.
- **Potential Side Effects:** None.
- **Related Systems Impacted:** None.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** No
- **UI Consistency Checkpoint Triggered:** No
- **Production-Readiness Checkpoint Triggered:** No

---

### Phase 6 — Frontend Consistency Enforcement

**Phase Objective:** Address the Medium-severity polish items that affect UX consistency: native-`alert()` usage, malformed toast calls, global-DOM-lookup bugs, content-block script hygiene, and the `modules` slice type mismatch.

**Entry Preconditions:** Phases 1–5 complete.

**Exit Criteria:**
- No native `alert()` calls in module controllers.
- No malformed `AppToast.success` second-argument calls.
- Module-card DOM lookups are scoped to the element instance.
- `<script>` tags appear only inside `{% block scripts %}` in the affected templates.
- `GlobalState.modules` is initialized as an array (or the consumer is updated to handle the object shape).

**Stability Checkpoint (Phase 6):** All affected pages continue to render without JS errors after each change.

**UI Consistency Checkpoint (Phase 6):** Toasts auto-dismiss with the intended duration; module settings open a modal (not a browser alert); modules grid shows real cards rather than the "No modules configured" empty state.

**Production-Readiness Checkpoint:** N/A.

---

#### Task 6.1 — Replace native `alert()` in `base-module-controller.js` with `AppModal`

- **Task ID:** T-6.1
- **Source Issue IDs:** ISS-017, ISS-026
- **Issue-to-Task Mapping Rationale:** ISS-017 and ISS-026 describe the same defect (the audit lists it twice).
- **Problem Being Solved:** `openSettings()` calls native `alert('Module settings coming soon')`, blocking the UI thread.
- **Isolated or Architectural:** Isolated.
- **Preconditions / Dependencies:** None.
- **Exact Files Involved:**
  - `web_app/static/js/modules/base-module-controller.js`
- **Exact Code Areas Involved:** Line 164 (`openSettings()` body).
- **Systems Affected:** Module settings UX.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report ISS-017 / ISS-026: replace blocking native alert.
- **Implementation Objective:** Clicking a module's settings button opens an `AppModal` (or another non-blocking notification) instead of a browser alert.
- **Detailed Implementation Actions:**
  1. Open `web_app/static/js/modules/base-module-controller.js`.
  2. Locate the `openSettings()` method at line 164.
  3. Replace its body:
     ```js
     openSettings() {
         alert('Module settings coming soon');
     }
     ```
     with:
     ```js
     openSettings() {
         if (typeof AppModal !== 'undefined' && AppModal && AppModal.alert) {
             AppModal.alert({
                 title: 'Module settings',
                 message: 'Module settings coming soon.',
             });
         } else if (typeof AppToast !== 'undefined' && AppToast && AppToast.info) {
             AppToast.info('Module settings coming soon');
         } else {
             console.warn('[base-module-controller] Module settings UI not available');
         }
     }
     ```
  4. REQUIRES_REPO_INSPECTION on the actual `AppModal` API surface. The report references `AppModal.setContent(html)` (ISS-032) but does not explicitly document `AppModal.alert`. If `AppModal.alert` does not exist, use `AppModal.create({...}).open()` or whichever method is canonical; do not invent.
  5. Save the file.
- **Expected Frontend Behavior After Fix:** Clicking the settings button on any module card opens a non-blocking modal/toast with the "coming soon" message.
- **Validation / Testing Instructions:**
  - Manual: Click the settings button on at least one module card. Confirm no native `alert()` dialog appears; a modal or toast appears instead.
  - Manual: The page remains interactive during the message display.
- **Potential Side Effects:** If `AppModal.alert` does not exist in this codebase, the fallback path uses `AppToast.info`. Pick one canonical API and use it (REQUIRES_REPO_INSPECTION).
- **Related Systems Impacted:** Module card components.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** No
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 6.2 — Move scripts out of `{% block content %}` into `{% block scripts %}`

- **Task ID:** T-6.2
- **Source Issue IDs:** ISS-019
- **Issue-to-Task Mapping Rationale:** Directly addresses the template hygiene defect.
- **Problem Being Solved:** `<script>` tags placed inside `{% block content %}` violate template architecture; the dedicated `{% block scripts %}` exists for this purpose.
- **Isolated or Architectural:** Isolated (template hygiene).
- **Preconditions / Dependencies:** None.
- **Exact Files Involved:**
  - `web_app/templates/actions.html` — script at line 74 per the report.
  - `web_app/templates/honeypot.html` — script at line 84 per the report.
  - `web_app/templates/realtime.html` — script at line 75 per the report.
- **Exact Code Areas Involved:** The `<script>` declarations currently inside each template's `{% block content %}`.
- **Systems Affected:** Template inheritance for these three pages.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report ISS-019.
- **Implementation Objective:** In each of the three templates, all `<script>` tags appear inside `{% block scripts %}` and not inside `{% block content %}`.
- **Detailed Implementation Actions:**
  1. For `web_app/templates/actions.html`:
     - Locate the `<script>` tag near line 74 inside `{% block content %}`.
     - Cut it (including any `type="module"`, `src=...`, attributes, and closing `</script>`).
     - Paste it into the `{% block scripts %}{% endblock %}` near the end of the file. If `{% block scripts %}` is empty, populate it; if it already contains tags, append.
  2. Repeat for `web_app/templates/honeypot.html` (script near line 84).
  3. Repeat for `web_app/templates/realtime.html` (script near line 75).
  4. Save all three files.
- **Expected Frontend Behavior After Fix:** No behavioral change at the user level (scripts still load). Template structure is correct.
- **Validation / Testing Instructions:**
  - Manual: Load `/actions`, `/honeypot`, `/realtime`. Each page renders and its controller runs (open DevTools and confirm the controller's expected side effects).
  - Source view: Inspect the rendered HTML — script tags are emitted at the location of `{% block scripts %}` in `base.html`, not inside the content area.
- **Potential Side Effects:** If `{% block scripts %}` is rendered later in the document than `{% block content %}` in `base.html`, scripts execute later. This is conventional but verify no controller relied on running inside the content block's DOM context (REQUIRES_REPO_INSPECTION).
- **Related Systems Impacted:** None.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** No
- **UI Consistency Checkpoint Triggered:** No
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 6.3 — Fix `AppToast.success` second-arg type in `dashboard.page.js`

- **Task ID:** T-6.3
- **Source Issue IDs:** ISS-020
- **Issue-to-Task Mapping Rationale:** Direct fix for the malformed call.
- **Problem Being Solved:** `AppToast.success('Dashboard loaded', 'System is ready for monitoring')` — the second argument should be a duration in milliseconds (number). Passing a string coerces to `NaN` and dismisses immediately.
- **Isolated or Architectural:** Isolated.
- **Preconditions / Dependencies:** None.
- **Exact Files Involved:**
  - `web_app/static/js/pages/dashboard.page.js` — line 253 per the report.
- **Exact Code Areas Involved:** `AppToast.success(...)` call near line 253.
- **Systems Affected:** Dashboard load toast.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report ISS-020.
- **Implementation Objective:** Dashboard-loaded toast appears for a visible duration (e.g., 3000–4000 ms).
- **Detailed Implementation Actions:**
  1. Open `web_app/static/js/pages/dashboard.page.js`.
  2. Locate line 253: `AppToast.success('Dashboard loaded', 'System is ready for monitoring')`.
  3. Choose one of the two corrections based on the `AppToast.success` signature (REQUIRES_REPO_INSPECTION on the actual signature):
     - If signature is `success(message, durationMs)`:
       ```js
       AppToast.success('Dashboard loaded — System is ready for monitoring', 3500);
       ```
     - If signature is `success(message, options)` where options can include `description`/`duration`:
       ```js
       AppToast.success('Dashboard loaded', { description: 'System is ready for monitoring', duration: 3500 });
       ```
  4. Save the file.
- **Expected Frontend Behavior After Fix:** A "Dashboard loaded" toast appears for ~3.5 seconds.
- **Validation / Testing Instructions:**
  - Manual: Load `/dashboard`. Toast should appear and remain visible for several seconds before dismissing.
- **Potential Side Effects:** None.
- **Related Systems Impacted:** None.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** No
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 6.4 — Scope `base-module-controller.js` DOM lookups to module element

- **Task ID:** T-6.4
- **Source Issue IDs:** ISS-023
- **Issue-to-Task Mapping Rationale:** Direct fix for the global `document.getElementById` lookups that collide across multiple modules on the same page.
- **Problem Being Solved:** `base-module-controller.js:130` uses `document.getElementById('contextStatus')` and `document.getElementById('contextUpdated')`. With multiple modules on one page, every module updates the same DOM nodes.
- **Isolated or Architectural:** Isolated.
- **Preconditions / Dependencies:** None.
- **Exact Files Involved:**
  - `web_app/static/js/modules/base-module-controller.js` — line 130 per the report.
- **Exact Code Areas Involved:** The two `document.getElementById` calls at line 130.
- **Systems Affected:** Per-module context status and timestamp rendering.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report ISS-023.
- **Implementation Objective:** Each module instance updates only its own `contextStatus` and `contextUpdated` elements, scoped to the module's root element.
- **Detailed Implementation Actions:**
  1. Open `web_app/static/js/modules/base-module-controller.js`.
  2. Locate line 130 where:
     ```js
     document.getElementById('contextStatus')
     document.getElementById('contextUpdated')
     ```
     are used.
  3. Replace them with element-scoped queries. The base module controller has a root element reference — REQUIRES_REPO_INSPECTION to confirm its name (commonly `this.el`, `this.root`, or `this.element`). Assuming the convention used elsewhere in the file:
     ```js
     this.el.querySelector('[data-context-status]')
     this.el.querySelector('[data-context-updated]')
     ```
  4. Then update the templates `web_app/templates/modules/base_module.html` and any module-specific HTML that exposes these nodes:
     - Change `id="contextStatus"` → `data-context-status` (or keep the id but make it module-specific, e.g., `id="contextStatus-{{ module_id }}"`).
     - Change `id="contextUpdated"` → `data-context-updated`.
     REQUIRES_REPO_INSPECTION for the exact templates that emit these ids.
  5. If altering ids is too invasive, an alternative is to qualify the query: `this.el.querySelector('#contextStatus')` — this still scopes to the module's root.
  6. Save the file(s).
- **Expected Frontend Behavior After Fix:** With multiple modules on the same page, each module's context status and timestamp update independently.
- **Validation / Testing Instructions:**
  - Manual: Open a page with two or more modules. Trigger context updates on each (e.g., load events). Confirm each module's status pill updates independently.
- **Potential Side Effects:** If other code relied on `getElementById('contextStatus')` returning the first module's node, it will no longer find it. REQUIRES_REPO_INSPECTION for other consumers.
- **Related Systems Impacted:** Module template markup.
- **Regression Risk:** Medium.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** No
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 6.5 — Reconcile `GlobalState.modules` slice shape

- **Task ID:** T-6.5
- **Source Issue IDs:** ISS-031
- **Issue-to-Task Mapping Rationale:** Directly addresses the array-vs-object mismatch noted in ISS-031.
- **Problem Being Solved:** `dashboard.page.js` checks `!Array.isArray(modules)` to render an empty state; `GlobalState` initializes the `modules` slice as `{}`. Result: the empty state is shown perpetually.
- **Isolated or Architectural:** Isolated.
- **Preconditions / Dependencies:** T-3.2 (consumer call signature is already correct so changing the shape won't re-break it).
- **Exact Files Involved:**
  - `web_app/static/js/core/global-state.js` — slice initialization.
  - `web_app/static/js/pages/dashboard.page.js` — consumer (already updated in T-3.2; check that the array assumption matches).
- **Exact Code Areas Involved:** The slice-initialization block in `global-state.js`.
- **Systems Affected:** Dashboard modules grid; any other consumer of the `modules` slice.
- **Architecture Impact:** Cross-cutting if other consumers exist (REQUIRES_REPO_INSPECTION).
- **Reason for This Change (traced to report):** Report ISS-031.
- **Implementation Objective:** `GlobalState.data.modules` is an array at initialization, matching the dashboard consumer's expectation.
- **Detailed Implementation Actions:**
  1. Open `web_app/static/js/core/global-state.js`.
  2. Locate the initialization of the `modules` slice (REQUIRES_REPO_INSPECTION for exact line; look for `modules: {}` or equivalent inside a `data` initializer).
  3. Change `modules: {}` to `modules: []`.
  4. Search the rest of the codebase for other consumers reading `GlobalState.data.modules` or subscribing to `'modules'`. For each:
     - If the consumer expects array semantics (uses `.forEach`, `.length`, `.map`), no change needed.
     - If a consumer expects object semantics (uses `Object.keys`, key access), update either the consumer or coordinate a shape transformation in the producer.
  5. Save the file.
- **Expected Frontend Behavior After Fix:** Dashboard modules grid populates when modules data is set. The "No modules configured" empty state shows only when `modules.length === 0`.
- **Validation / Testing Instructions:**
  - Manual: In DevTools console, run:
    ```js
    GlobalState.set('modules', [
      {id:'m1', title:'Module 1', status:'ok'},
      {id:'m2', title:'Module 2', status:'warn'},
    ]);
    ```
    Confirm two cards render on the dashboard.
- **Potential Side Effects:** Object-shape consumers break (mitigated by step 4).
- **Related Systems Impacted:** Any code path consuming the `modules` slice.
- **Regression Risk:** Medium.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** No
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** No

---

### Phase 7 — Legacy Code Removal

**Phase Objective:** Remove or repair the orphaned/legacy files identified in the audit, only **after** their replacements are confirmed live (Phases 1–6).

**Entry Preconditions:** Phases 1–6 complete. The detection page in particular must be confirmed running `detection.page.js` end-to-end (Task 2.1 validated in production-like environment).

**Exit Criteria:**
- Duplicate `threat-intel.html` stub resolved (deleted or consolidated).
- `socket-core.js` either deleted or repaired (import path, state API).
- Orphaned legacy `state.js` and `socket.js` deleted.
- `generate_alerts.py` moved out of the public static directory.
- Dead import of `loading-spinner` in `monitor.page.js` removed.

**Stability Checkpoint (Phase 7):** No page regresses after removals; all routes and pages from Phase 5 still render.

**UI Consistency Checkpoint (Phase 7):** Threat-intel route renders one (and only one) template's content.

**Production-Readiness Checkpoint:** N/A.

---

#### Task 7.1 — Resolve duplicate threat-intel templates

- **Task ID:** T-7.1
- **Source Issue IDs:** ISS-012
- **Issue-to-Task Mapping Rationale:** Direct fix for the duplicate-template defect.
- **Problem Being Solved:** Two templates exist (`threat_intel.html`, `threat-intel.html`). The route renders the underscore version (old design). The stub serves no purpose.
- **Isolated or Architectural:** Isolated.
- **Preconditions / Dependencies:** None.
- **Exact Files Involved:**
  - `web_app/templates/threat_intel.html` (8KB, old design — currently rendered).
  - `web_app/templates/threat-intel.html` (4.2KB, stub).
  - `web_app/blueprints/pages.py` — line 127 per the report (the route handler).
- **Exact Code Areas Involved:** The `render_template("threat_intel.html")` call in `pages.py`.
- **Systems Affected:** `/threat-intel` route content.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report ISS-012.
- **Implementation Objective:** Exactly one template backs `/threat-intel`; the other is removed.
- **Detailed Implementation Actions:**
  1. Open `web_app/templates/threat-intel.html` (the 4.2KB stub) and `web_app/templates/threat_intel.html` (the 8KB old-design version). Confirm by inspection that the stub is genuinely unused.
  2. Delete `web_app/templates/threat-intel.html` (the stub).
  3. Confirm `pages.py` continues to call `render_template("threat_intel.html")` (underscore).
  4. Optional (out of scope per audit): a future task may modernize `threat_intel.html` to the new design. Do not modernize content in this task — that is not in the report's scope for this fix.
  5. Save changes; commit.
- **Expected Frontend Behavior After Fix:** `/threat-intel` continues to render the old-design page; no orphan template remains.
- **Validation / Testing Instructions:**
  - Manual: Visit `/threat-intel` — same content as before.
  - File check: `ls web_app/templates/threat-intel.html` returns no such file; `ls web_app/templates/threat_intel.html` exists.
- **Potential Side Effects:** If any code or template referenced `threat-intel.html` directly by filename, it will break. REQUIRES_REPO_INSPECTION via grep `threat-intel.html` across the repo.
- **Related Systems Impacted:** None expected.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert` restores the file; no data loss.
- **Stability Checkpoint Triggered:** Yes
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 7.2 — Fix or remove `socket-core.js`

- **Task ID:** T-7.2
- **Source Issue IDs:** ISS-016
- **Issue-to-Task Mapping Rationale:** Direct fix for the orphaned-but-broken module.
- **Problem Being Solved:** `socket-core.js` is not loaded anywhere but contains a wrong import path (`./ui_core.js` instead of `./ui-core.js`) and uses the old `GlobalState.set({...object})` API.
- **Isolated or Architectural:** Isolated.
- **Preconditions / Dependencies:** None.
- **Exact Files Involved:**
  - `web_app/static/js/core/socket-core.js`
- **Exact Code Areas Involved:** Line 6 (import) and any `GlobalState.set({...})` calls.
- **Systems Affected:** None at runtime currently; future imports of this file would break.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report ISS-016.
- **Implementation Objective:** The file is either removed (preferred since `socket-manager.js` is the live replacement) or fully repaired to be importable.
- **Detailed Implementation Actions:**
  1. Decision: per Wave 3 of the report ("Fix `socket-core.js` import path; decide to keep or remove file") and ISS-024 ("Remove orphaned `state.js`, `socket.js`"), the recommended path is **removal** because `socket-manager.js` is the canonical new implementation.
  2. Delete `web_app/static/js/core/socket-core.js`.
  3. Grep the repo for `socket-core.js` references. Confirm zero results (it is orphaned per the audit).
  4. Commit.
- **Expected Frontend Behavior After Fix:** No behavioral change. File no longer present.
- **Validation / Testing Instructions:**
  - File check: `ls web_app/static/js/core/socket-core.js` returns no such file.
  - Grep: `grep -r "socket-core" web_app` returns no matches.
  - Smoke test: load `/`, `/alerts`, `/monitor`, `/dashboard` — all still work.
- **Potential Side Effects:** None given the file is orphaned.
- **Related Systems Impacted:** None.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert` restores the file.
- **Stability Checkpoint Triggered:** Yes
- **UI Consistency Checkpoint Triggered:** No
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 7.3 — Delete orphaned legacy `state.js` and `socket.js`

- **Task ID:** T-7.3
- **Source Issue IDs:** ISS-024
- **Issue-to-Task Mapping Rationale:** Direct fix.
- **Problem Being Solved:** Orphaned legacy files (1.5KB + 6.5KB) with incompatible APIs invite future bugs.
- **Isolated or Architectural:** Isolated.
- **Preconditions / Dependencies:** Phases 1–6 complete (new `GlobalState` and `SocketManager` confirmed working in production-like behavior).
- **Exact Files Involved:**
  - `web_app/static/js/state.js`
  - `web_app/static/js/socket.js`
- **Exact Code Areas Involved:** Entire files.
- **Systems Affected:** None at runtime currently.
- **Architecture Impact:** Local; reduces footgun risk.
- **Reason for This Change (traced to report):** Report ISS-024 and Wave 3 roadmap.
- **Implementation Objective:** Both files are removed; no references remain.
- **Detailed Implementation Actions:**
  1. Grep the repo: `grep -r "static/js/state.js" web_app` and `grep -r "static/js/socket.js" web_app`. Confirm zero references in templates (`*.html`) or in any JS `import` paths. If any reference exists, do not delete — file an issue and stop.
  2. Delete `web_app/static/js/state.js`.
  3. Delete `web_app/static/js/socket.js`.
  4. Commit.
- **Expected Frontend Behavior After Fix:** No behavioral change.
- **Validation / Testing Instructions:**
  - File check: both files gone.
  - Smoke test: every primary route still renders without console errors.
- **Potential Side Effects:** None given orphaned status.
- **Related Systems Impacted:** None.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert` restores both files.
- **Stability Checkpoint Triggered:** Yes
- **UI Consistency Checkpoint Triggered:** No
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 7.4 — Move `generate_alerts.py` out of the public static directory

- **Task ID:** T-7.4
- **Source Issue IDs:** ISS-025
- **Issue-to-Task Mapping Rationale:** Direct fix.
- **Problem Being Solved:** A Python source file is exposed under `/static/sfx/generate_alerts.py`. Not a vulnerability but inappropriate.
- **Isolated or Architectural:** Isolated.
- **Preconditions / Dependencies:** None.
- **Exact Files Involved:**
  - `web_app/static/sfx/generate_alerts.py`
  - `web_app/static/sfx/README.md` (also explanatory; consider moving with the script).
- **Exact Code Areas Involved:** The file location.
- **Systems Affected:** Static asset surface.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report ISS-025.
- **Implementation Objective:** The generator script is no longer accessible at `/static/sfx/generate_alerts.py`.
- **Detailed Implementation Actions:**
  1. Create a new directory outside `static/`, e.g., `web_app/tools/sfx/` (REQUIRES_REPO_INSPECTION on existing project conventions for non-served scripts).
  2. Move `web_app/static/sfx/generate_alerts.py` to `web_app/tools/sfx/generate_alerts.py`.
  3. Move `web_app/static/sfx/README.md` to `web_app/tools/sfx/README.md`.
  4. Leave the `web_app/static/sfx/` directory in place (still used by T-9.1 for the generated `.mp3` outputs).
  5. Update any references to the old path in documentation if present.
  6. Commit.
- **Expected Frontend Behavior After Fix:** Visiting `/static/sfx/generate_alerts.py` returns 404 (because Flask's static handler will not find the file).
- **Validation / Testing Instructions:**
  - Manual: `curl http://<host>/static/sfx/generate_alerts.py` returns 404.
  - File check: file exists at the new path.
- **Potential Side Effects:** Anyone running the script needs to use the new path. Update README accordingly.
- **Related Systems Impacted:** Audio asset pipeline (T-9.1 references the script).
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** No
- **UI Consistency Checkpoint Triggered:** No
- **Production-Readiness Checkpoint Triggered:** No

---

#### Task 7.5 — Remove dead `loading-spinner` import from `monitor.page.js`

- **Task ID:** T-7.5
- **Source Issue IDs:** ISS-027
- **Issue-to-Task Mapping Rationale:** Direct fix for the dead import.
- **Problem Being Solved:** `loading-spinner.js` is imported at line 26 of `monitor.page.js` but never referenced.
- **Isolated or Architectural:** Isolated.
- **Preconditions / Dependencies:** T-1.3 (monitor page already touched).
- **Exact Files Involved:**
  - `web_app/static/js/pages/monitor.page.js`
- **Exact Code Areas Involved:** Line 26 import statement.
- **Systems Affected:** Monitor page module graph.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report ISS-027.
- **Implementation Objective:** No unused imports remain in `monitor.page.js`.
- **Detailed Implementation Actions:**
  1. Open `web_app/static/js/pages/monitor.page.js`.
  2. Locate the import at line 26 (importing the loading-spinner component).
  3. Delete the entire import line.
  4. Verify no other references to the imported binding exist in the file (search for the imported name). If references appear, do not delete — restore the import and file a follow-up.
  5. Save the file.
- **Expected Frontend Behavior After Fix:** No behavioral change.
- **Validation / Testing Instructions:**
  - Manual: Load `/monitor`. Page renders identically.
  - Network: One less module file is fetched.
- **Potential Side Effects:** None.
- **Related Systems Impacted:** None.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** No
- **UI Consistency Checkpoint Triggered:** No
- **Production-Readiness Checkpoint Triggered:** No

---

### Phase 8 — Performance Hardening

**Phase Objective:** Address the audit's three-framework CSS conflict, Tailwind CDN play-mode dependency, unused Bootstrap JS, and the JIT-arbitrary-value usage that requires play-mode to function.

**Entry Preconditions:** Phases 1–7 complete. Visual baseline established (Phase 4) so before/after diffs are meaningful.

**Exit Criteria:**
- Tailwind is delivered as a pre-built static CSS file (no `cdn.tailwindcss.com` script). FOUC eliminated.
- `bootstrap.bundle.min.js` is removed from pages that do not use Bootstrap JS components, or removed globally if no consumer requires it.
- `UICard.divider()`'s arbitrary Tailwind value is replaced with a class present in the build, or moved to plain CSS.

**Stability Checkpoint (Phase 8):** Every primary route still renders with no console error.

**UI Consistency Checkpoint (Phase 8):** Visual diff against pre-Phase-8 screenshots shows no unintended regressions (use baseline screenshots taken at end of Phase 5).

**Production-Readiness Checkpoint (Phase 8):** Initial page render no longer depends on a CDN executing JavaScript to produce CSS. Total JS payload per page reduced by removing unused Bootstrap JS.

---

#### Task 8.1 — Replace Tailwind CDN with built CSS file

- **Task ID:** T-8.1
- **Source Issue IDs:** ISS-015, ISS-028, ISS-033
- **Issue-to-Task Mapping Rationale:** All three issues share the same root cause (CDN JIT) and the same fix (move to a build step).
- **Problem Being Solved:** Tailwind CDN play executes JS in the browser to generate CSS — no SRI possible, adds ~90KB of JS, blocks render, causes FOUC, and is officially a development-only tool.
- **Isolated or Architectural:** Architectural (build pipeline introduction).
- **Preconditions / Dependencies:** All previous phases (don't change build pipeline while runtime is unstable).
- **Exact Files Involved:**
  - `web_app/templates/base.html` — line 22 currently loads `<script src="https://cdn.tailwindcss.com">`.
  - `web_app/static/js/tailwind-config.js` — current theme extension.
  - `web_app/static/css/` — new built file will land here (e.g., `static/css/tailwind.css`).
  - Project root — a new build configuration (e.g., `tailwind.config.js`, `postcss.config.js`, `package.json` script entry). REQUIRES_REPO_INSPECTION for existing tooling presence.
  - `web_app/static/js/components/ui-card.js` — for the `before:content-['']` arbitrary value (ISS-033).
- **Exact Code Areas Involved:** Tailwind CDN script tag, Tailwind config inclusion, and the `before:content-['']` class in `UICard.divider()`.
- **Systems Affected:** Build pipeline, CSS delivery, FOUC behavior.
- **Architecture Impact:** Foundational.
- **Reason for This Change (traced to report):** Report ISS-015, ISS-018, ISS-028, ISS-033 and Wave 7 ("Replace `cdn.tailwindcss.com` with Tailwind CLI build step").
- **Implementation Objective:** A pre-built `tailwind.css` is loaded from `static/css/` with SRI; the CDN script tag is removed; FOUC is eliminated; the `UICard.divider()` arbitrary value is replaced with a built-in utility or plain CSS.
- **Detailed Implementation Actions:**
  1. Add a Node-based build dependency on Tailwind CLI. Create at the project root (or under `web_app/`):
     - `package.json` with a `tailwind:build` script: `tailwindcss -i ./web_app/static/css/tailwind.src.css -o ./web_app/static/css/tailwind.css --minify`.
     - `tailwind.config.js` that mirrors the theme defined in `web_app/static/js/tailwind-config.js`. REQUIRES_REPO_INSPECTION to translate the runtime config to a Node config.
     - `web_app/static/css/tailwind.src.css` containing:
       ```css
       @tailwind base;
       @tailwind components;
       @tailwind utilities;
       ```
     - Set `content` in `tailwind.config.js` to include `./web_app/templates/**/*.html` and `./web_app/static/js/**/*.js`.
  2. Run the build locally; commit the generated `web_app/static/css/tailwind.css`. (Or wire to CI; REQUIRES_REPO_INSPECTION on deployment process.)
  3. In `web_app/templates/base.html`, at line 22:
     - Remove: `<script src="https://cdn.tailwindcss.com"></script>`
     - Remove: the inclusion of `web_app/static/js/tailwind-config.js` (the runtime theme extension is no longer needed).
     - Add (with SRI hash computed from the built file):
       ```html
       <link rel="stylesheet" href="{{ url_for('static', filename='css/tailwind.css') }}">
       ```
       SRI integrity attribute can be added once the deployment computes the hash.
  4. For ISS-033, in `web_app/static/js/components/ui-card.js`, locate the `divider()` method using `before:content-['']`. Replace with a built-in equivalent or move that styling to plain CSS:
     - Option A (preferred): replace with a Tailwind class set whose effect doesn't require arbitrary values. For a 1px divider, use `border-t border-white/10` on the divider element.
     - Option B: add a plain CSS rule in the inline `<style>` block of `base.html`:
       ```css
       .ui-card-divider { border-top: 1px solid var(--border); margin: 12px 0; }
       ```
       and make `divider()` apply the `ui-card-divider` class.
  5. Save all changes.
- **Expected Frontend Behavior After Fix:** No CDN script loads. Pages render styled at first paint (no FOUC). `UICard.divider()` renders a visible divider line.
- **Validation / Testing Instructions:**
  - Manual: Open DevTools Network panel. Reload any page. Confirm no request to `cdn.tailwindcss.com`. Confirm `/static/css/tailwind.css` is fetched.
  - Manual: First paint is already styled (no flash of unstyled content). Throttle network to "Slow 3G" to verify.
  - Manual: Pages using `UICard.divider()` still render dividers correctly.
- **Potential Side Effects:** Any Tailwind class used in templates but not detected by the `content` glob will be missing from the built CSS. Test every primary page for visual regressions.
- **Related Systems Impacted:** Every page.
- **Regression Risk:** High (large surface area, build pipeline introduction).
- **Rollback Considerations:** `git revert`. The Tailwind CDN script reappears, the built CSS link is removed. No data risk.
- **Stability Checkpoint Triggered:** Yes
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** Yes

---

#### Task 8.2 — Rationalize CSS framework stack (Bootstrap CSS scope)

- **Task ID:** T-8.2
- **Source Issue IDs:** ISS-018
- **Issue-to-Task Mapping Rationale:** Direct fix for the three-framework conflict (Bootstrap CSS portion).
- **Problem Being Solved:** Bootstrap CSS (238KB) + Tailwind + custom inline CSS coexist with overlapping class names (`.btn`, `.badge`, etc.), producing specificity wars.
- **Isolated or Architectural:** Architectural.
- **Preconditions / Dependencies:** T-8.1 complete (Tailwind on a build, so we can confirm what remains needed).
- **Exact Files Involved:**
  - `web_app/templates/base.html` — Bootstrap CSS `<link>` tag.
  - `web_app/static/bootstrap-5.2.3-dist/` — vendored Bootstrap.
- **Exact Code Areas Involved:** The Bootstrap CSS `<link>` in `base.html`.
- **Systems Affected:** Every page's CSS cascade.
- **Architecture Impact:** Cross-cutting.
- **Reason for This Change (traced to report):** Report ISS-018 and Wave 3: "Replace Tailwind CDN with build output; remove Bootstrap CSS if unused".
- **Implementation Objective:** Bootstrap CSS is removed from every page that does not require it; if no page requires it, the vendored distribution is removed entirely.
- **Detailed Implementation Actions:**
  1. Audit usage. Grep templates and components for Bootstrap class consumers (`btn-`, `col-`, `row`, `container`, `badge`, `alert`, etc.). Build a per-page list of pages that need Bootstrap CSS. REQUIRES_REPO_INSPECTION for the full list.
  2. If the audit finds zero pages truly require Bootstrap CSS to render correctly: remove the `<link>` to Bootstrap CSS from `base.html`. Then delete `web_app/static/bootstrap-5.2.3-dist/css/` (keep `js/` for now — T-8.3 handles JS).
  3. If some pages need Bootstrap (e.g., for the grid system), restructure `base.html`:
     - Remove Bootstrap CSS from the base `<head>`.
     - In the specific page templates that need it, add `{% block extra_css %}<link rel="stylesheet" href="{{ url_for('static', filename='bootstrap-5.2.3-dist/css/bootstrap.min.css') }}">{% endblock %}` (extending the existing extra_css block as needed).
  4. Per-page visual verification (every page).
  5. Commit.
- **Expected Frontend Behavior After Fix:** Pages not requiring Bootstrap render with their custom + Tailwind CSS only, with smaller payload and no specificity conflict.
- **Validation / Testing Instructions:**
  - Visual diff against baseline screenshots from end of Phase 5.
  - Network: confirm Bootstrap CSS no longer fetched on pages that don't list it in their per-page `extra_css`.
- **Potential Side Effects:** Pages that silently relied on Bootstrap utility classes (e.g., `mt-3`, `d-flex`) will lose those styles. Replace with Tailwind equivalents where found.
- **Related Systems Impacted:** Every page.
- **Regression Risk:** High.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** Yes
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** Yes

---

#### Task 8.3 — Remove Bootstrap JS bundle from pages that don't need it

- **Task ID:** T-8.3
- **Source Issue IDs:** ISS-029
- **Issue-to-Task Mapping Rationale:** Direct fix for unused 80KB JS payload per page.
- **Problem Being Solved:** `bootstrap.bundle.min.js` loads on every page but most pages do not use Bootstrap JS components (modals, dropdowns, tooltips).
- **Isolated or Architectural:** Cross-cutting.
- **Preconditions / Dependencies:** T-8.2 (Bootstrap CSS scope finalized).
- **Exact Files Involved:**
  - `web_app/templates/base.html` — `<script>` tag for Bootstrap bundle.
- **Exact Code Areas Involved:** The Bootstrap bundle script tag (per the audit, present on every page via `base.html`).
- **Systems Affected:** Every page's JS payload.
- **Architecture Impact:** Cross-cutting.
- **Reason for This Change (traced to report):** Report ISS-029.
- **Implementation Objective:** Bootstrap JS loads only on pages that need it.
- **Detailed Implementation Actions:**
  1. Audit: grep templates and JS for Bootstrap JS usage (`data-bs-toggle`, `bootstrap.Modal`, `bootstrap.Dropdown`, etc.). REQUIRES_REPO_INSPECTION to produce the per-page list.
  2. If zero pages require Bootstrap JS, remove the `<script>` from `base.html` entirely and delete `web_app/static/bootstrap-5.2.3-dist/js/`.
  3. If some pages require it, move the script tag out of `base.html`'s global `{% block scripts %}` and into `{% block scripts %}` of those specific page templates (prepend before the page's own controller).
  4. Commit.
- **Expected Frontend Behavior After Fix:** Pages that don't use Bootstrap JS no longer fetch ~80KB of JS.
- **Validation / Testing Instructions:**
  - Network: Compare requests on each primary page before/after — Bootstrap bundle absent on pages without consumers.
  - Functional: Any feature that used `data-bs-*` still works (only on pages where Bootstrap JS is kept).
- **Potential Side Effects:** Pages silently using Bootstrap JS without explicit identification will break. Test thoroughly.
- **Related Systems Impacted:** Every page.
- **Regression Risk:** Medium.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** Yes
- **UI Consistency Checkpoint Triggered:** No
- **Production-Readiness Checkpoint Triggered:** Yes

---

### Phase 9 — Production Readiness

**Phase Objective:** Generate missing assets, remove development noise, and add the modal sanitization the audit calls out.

**Entry Preconditions:** Phases 1–8 complete.

**Exit Criteria:**
- Alert audio files `alert_high.mp3`, `alert_med.mp3`, `alert_low.mp3` exist at `/static/sfx/` and `playAlertTone(...)` succeeds.
- No `console.log` statements remain in production JS at the lines named by the audit.
- `AppModal.setContent(html)` either sanitizes input or has a documented opt-in switch for raw HTML.

**Stability Checkpoint (Phase 9):** All previously-fixed surfaces continue to work.

**UI Consistency Checkpoint (Phase 9):** Audio playback occurs on triggered alerts; no console noise in production builds.

**Production-Readiness Checkpoint (Phase 9):** Assets resolve without 404. Logging is intentional. Modal content has a sanitization story.

---

#### Task 9.1 — Generate alert audio files (or guard `playAlertTone`)

- **Task ID:** T-9.1
- **Source Issue IDs:** ISS-009
- **Issue-to-Task Mapping Rationale:** Direct fix.
- **Problem Being Solved:** `playAlertTone(level)` requests `/static/sfx/alert_*.mp3` — all 404.
- **Isolated or Architectural:** Isolated.
- **Preconditions / Dependencies:** T-7.4 (generator script relocated).
- **Exact Files Involved:**
  - `web_app/tools/sfx/generate_alerts.py` (relocated in T-7.4) — the generator.
  - `web_app/static/sfx/alert_high.mp3` — output target.
  - `web_app/static/sfx/alert_med.mp3` — output target.
  - `web_app/static/sfx/alert_low.mp3` — output target.
  - `web_app/static/js/core/utils.js` — `playAlertTone` function at lines 516–529.
- **Exact Code Areas Involved:** Output `.mp3` files; optional guard in `playAlertTone`.
- **Systems Affected:** Audio notification subsystem.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report ISS-009 and Wave 2.
- **Implementation Objective:** All three audio files exist and play; no 404 occurs when `playAlertTone` is invoked.
- **Detailed Implementation Actions:**
  1. Run the generator: `python web_app/tools/sfx/generate_alerts.py`. REQUIRES_REPO_INSPECTION on the script's actual interface — confirm output paths match `/static/sfx/alert_*.mp3`. If outputs land elsewhere, copy them to `web_app/static/sfx/`.
  2. Commit the three resulting `.mp3` files into `web_app/static/sfx/`.
  3. Optionally (defensive), guard `playAlertTone` in `utils.js:516–529` so a future missing file does not produce console warnings repeatedly:
     ```js
     export function playAlertTone(level) {
         const file =
             level === "high" ? "/static/sfx/alert_high.mp3" :
             level === "medium" ? "/static/sfx/alert_med.mp3" :
             "/static/sfx/alert_low.mp3";
         try {
             const audio = new Audio(file);
             audio.play().catch(() => { /* deliberate noop */ });
         } catch (_) { /* noop */ }
     }
     ```
- **Expected Frontend Behavior After Fix:** Alerts above the configured severity threshold play the corresponding audio tone.
- **Validation / Testing Instructions:**
  - Manual: From the DevTools console: `playAlertTone('high')`. The high-severity tone plays.
  - Network: Confirm the `.mp3` returns 200.
- **Potential Side Effects:** Audio autoplay may be blocked by browser policy if user has not interacted with the page; behavior may require a user gesture before playback (browser policy, not code defect).
- **Related Systems Impacted:** None.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert` removes the audio files; `playAlertTone` returns to 404 behavior.
- **Stability Checkpoint Triggered:** No
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** Yes

---

#### Task 9.2 — Remove production `console.log` statements

- **Task ID:** T-9.2
- **Source Issue IDs:** ISS-030
- **Issue-to-Task Mapping Rationale:** Direct fix.
- **Problem Being Solved:** Stray `console.log` calls remain in production JS at five specific sites identified by the audit.
- **Isolated or Architectural:** Isolated.
- **Preconditions / Dependencies:** None.
- **Exact Files Involved:**
  - `web_app/static/js/base-modules.js` — line 9.
  - `web_app/static/js/core/global-state.js` — line 261.
  - `web_app/static/js/core/http-client.js` — line 367.
  - `web_app/static/js/core/socket-manager.js` — line 373.
  - `web_app/static/js/core/utils.js` — line 532.
- **Exact Code Areas Involved:** Each named line.
- **Systems Affected:** Console output.
- **Architecture Impact:** Local.
- **Reason for This Change (traced to report):** Report ISS-030.
- **Implementation Objective:** None of the five named log statements remain. Intentional diagnostic logs may be retained but should be wrapped behind a debug flag — see step 2.
- **Detailed Implementation Actions:**
  1. For each named file and line, examine the log statement:
     - If it is purely informational ("loaded", "initialized"), delete it.
     - If it conveys diagnostic information valuable in development, replace with:
       ```js
       if (typeof window !== 'undefined' && window.__INIDS_DEBUG__) {
           console.log(/* original args */);
       }
       ```
       so it remains togglable via `window.__INIDS_DEBUG__ = true`.
  2. Add a comment at the top of `base-modules.js`:
     ```js
     // Enable verbose logging by setting `window.__INIDS_DEBUG__ = true` before this script runs.
     ```
  3. Note: the `console.log` added by T-1.3 (Monitor socket connect) and `console.warn` (Monitor socket disconnect) are intentional and traced to the report's FIX-003. Either retain them or wrap them in the same `__INIDS_DEBUG__` guard.
  4. Save and commit.
- **Expected Frontend Behavior After Fix:** Console is clean on production page loads.
- **Validation / Testing Instructions:**
  - Manual: Open DevTools Console on each page after navigation. Confirm none of the audit-named log lines appears.
  - Set `window.__INIDS_DEBUG__ = true` and reload — diagnostic lines re-appear.
- **Potential Side Effects:** Loss of incidental debugging info. Mitigated by the debug flag.
- **Related Systems Impacted:** None.
- **Regression Risk:** Low.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** No
- **UI Consistency Checkpoint Triggered:** No
- **Production-Readiness Checkpoint Triggered:** Yes

---

#### Task 9.3 — Add sanitization to `AppModal.setContent`

- **Task ID:** T-9.3
- **Source Issue IDs:** ISS-032
- **Issue-to-Task Mapping Rationale:** Direct fix for the XSS vector.
- **Problem Being Solved:** `AppModal.setContent(html)` assigns `content.innerHTML = html` without sanitization. Any user-controlled data reaching this path is an XSS vector.
- **Isolated or Architectural:** Isolated.
- **Preconditions / Dependencies:** None.
- **Exact Files Involved:**
  - `web_app/static/js/components/app-modal.js` — REQUIRES_REPO_INSPECTION for exact location of `setContent`.
- **Exact Code Areas Involved:** The `setContent` method.
- **Systems Affected:** Modal content rendering.
- **Architecture Impact:** Local with API surface change.
- **Reason for This Change (traced to report):** Report ISS-032.
- **Implementation Objective:** `setContent` defaults to safe text/DOM-node input. Raw-HTML insertion is opt-in.
- **Detailed Implementation Actions:**
  1. Open `web_app/static/js/components/app-modal.js`.
  2. Locate `setContent` (currently of the shape `setContent(html) { this.content.innerHTML = html; }`).
  3. Replace with a two-mode API:
     ```js
     setContent(content, options = {}) {
         const { html = false } = options;
         if (content instanceof Node) {
             this.content.replaceChildren(content);
             return;
         }
         if (typeof content === 'string') {
             if (html) {
                 // Caller has opted in. Document caller-side sanitization expectation.
                 this.content.innerHTML = content;
             } else {
                 this.content.textContent = content;
             }
             return;
         }
         this.content.replaceChildren();
     }
     ```
  4. Update any callers in the codebase that pass HTML strings expecting them to render as HTML. They must now pass `{ html: true }` explicitly. REQUIRES_REPO_INSPECTION to identify these call sites.
  5. Save and commit.
- **Expected Frontend Behavior After Fix:** Modal content with plain strings renders as text (no XSS surface). Callers needing HTML insertion must opt in.
- **Validation / Testing Instructions:**
  - Manual: Call `AppModal.setContent('<script>alert(1)</script>')` — content shows the literal text, no script executes.
  - Manual: Call `AppModal.setContent('<b>bold</b>', { html: true })` — content renders as bold.
- **Potential Side Effects:** Any caller previously relying on HTML interpretation now sees literal text until updated.
- **Related Systems Impacted:** All `AppModal.setContent` consumers.
- **Regression Risk:** Medium.
- **Rollback Considerations:** `git revert`.
- **Stability Checkpoint Triggered:** No
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** Yes

---

### Phase 10 — Final QA + Validation

**Phase Objective:** Execute end-to-end verification across all phases. This phase contains no code changes — only validation.

**Entry Preconditions:** Phases 1–9 complete.

**Exit Criteria:** Every check in the Final Validation Protocol (Section 7) passes.

**Stability Checkpoint (Phase 10):** All routes render; all critical user flows complete; no `TypeError`s; no 404s on primary assets.

**UI Consistency Checkpoint (Phase 10):** Visual design system tokens are uniformly applied; sidebar covers all routes; alert cards, module cards, and dashboard tiles render correctly.

**Production-Readiness Checkpoint (Phase 10):** No CDN play-mode dependencies; intentional logging only; favicon present; alert audio plays; modal content is sanitized by default.

---

#### Task 10.1 — Execute Final Validation Protocol

- **Task ID:** T-10.1
- **Source Issue IDs:** All 33 issues, validation pass.
- **Issue-to-Task Mapping Rationale:** End-to-end verification that no issue has regressed.
- **Problem Being Solved:** N/A — verification task.
- **Isolated or Architectural:** N/A — verification.
- **Preconditions / Dependencies:** All Phase 1–9 tasks complete.
- **Exact Files Involved:** None modified; all primary routes exercised.
- **Exact Code Areas Involved:** N/A — verification only.
- **Systems Affected:** N/A.
- **Architecture Impact:** None.
- **Reason for This Change (traced to report):** Quality-bar discipline; ensures the report's stated outcomes hold.
- **Implementation Objective:** All items in Section 7 pass and are recorded.
- **Detailed Implementation Actions:**
  1. Execute every item in Section 7 (Final Validation Protocol). Record pass/fail per item in a QA log file `QA_PHASE_10.md` at the repo root.
  2. For each failure, open a follow-up issue linked to the relevant Task ID. Do not declare Phase 10 complete with unresolved failures.
  3. Capture screenshots of every primary route as a visual-regression baseline for the post-merge state.
  4. Commit the QA log.
- **Expected Frontend Behavior After Fix:** N/A.
- **Validation / Testing Instructions:** This task **is** the validation. Use Section 7's checklist verbatim.
- **Potential Side Effects:** None.
- **Related Systems Impacted:** None.
- **Regression Risk:** N/A.
- **Rollback Considerations:** N/A.
- **Stability Checkpoint Triggered:** Yes
- **UI Consistency Checkpoint Triggered:** Yes
- **Production-Readiness Checkpoint Triggered:** Yes

---

## 6. Cross-Phase Risk Register

| Risk | Phases Touched | Mitigation | Rollback Strategy |
|------|----------------|------------|-------------------|
| **Detection page controller/markup contract drift** — T-2.1 swaps `detection.js` → `detection.page.js` while keeping the legacy `ds-card` markup. Selectors may not align. | 2, 7 | Validate selectors in T-2.1 before progressing. File a follow-up rather than rewriting markup. | `git revert` T-2.1; legacy `detection.js` reloads. |
| **CSS variable definitions in T-4.1 cascade to every page** — wrong values could regress visual consistency everywhere at once. | 4, 5, 8 | Apply values from FIX-001 verbatim, then take baseline screenshots before moving on. | `git revert` T-4.1; immediate. |
| **Sidebar expansion (T-5.2) reveals routes whose pages have latent defects.** | 5, 7, 10 | Each route is loaded once during Phase 5 validation. | `git revert` T-5.2 hides routes again. |
| **Tailwind CDN → built CSS migration (T-8.1)** is the highest-risk change: missed classes vanish silently. | 8, 10 | Run a build with permissive `content` globs; visual diff every primary page against the Phase 5 baseline. | `git revert` T-8.1; Tailwind CDN script returns. |
| **Bootstrap removal (T-8.2, T-8.3) breaks pages that silently relied on Bootstrap utility classes.** | 8, 10 | Per-page visual diff and JS console check. | `git revert` the corresponding task commit. |
| **`AppModal.setContent` API change (T-9.3)** silently renders existing HTML callers as text until they opt in. | 9, 10 | Grep all `setContent` callers and update in the same change. | `git revert` T-9.3; old XSS-prone behavior returns. |
| **`GlobalState.modules` shape change (T-6.5)** can break unknown consumers expecting object semantics. | 6, 7, 10 | Grep for all `'modules'` subscribers and `data.modules` readers before flipping the shape. | `git revert` T-6.5. |
| **`HttpClient.post` return value assumption** (raised by T-3.1) applies wherever `response.ok`/`response.status` is checked. | 3 | Grep `HttpClient.post` and `HttpClient.put`/`HttpClient.patch` callers for similar response checks. File follow-ups as found. | `git revert` per call site. |

---

## 7. Final Validation Protocol

Executed only after Phase 10 begins. Every item must pass; record results in `QA_PHASE_10.md`.

1. **Build passes** — The Tailwind build (T-8.1) completes without errors. The application starts (`flask run` or the project's equivalent) with no startup exceptions.
2. **Type checks pass** — N/A — justified: report does not specify a JS type-check tooling (no TypeScript, no JSDoc enforcement).
3. **Lint passes** — Run the project's lint configuration if present. REQUIRES_REPO_INSPECTION for whether one exists. If none exists, this check is skipped (and not invented).
4. **All routes render** — Manually load each route exposed by the expanded sidebar (T-5.2): `/`, `/monitor`, `/realtime`, `/alerts`, `/detection`, `/actions`, `/respond`, `/honeypot`, `/threat-intel`, `/investigate`, `/policy`, `/allowlist`, `/models`, `/health`, `/capture`, `/learn`. Also verify `/dashboard/main` 302s to `/dashboard`. Each must return HTTP 200 (or 302 for `/dashboard/main`) and render its template.
5. **All critical user flows pass:**
   - Severity filter on `/alerts` (T-1.2) — clicking each chip filters the list with no console error.
   - Search input on `/alerts` (T-1.2) — typing filters live with no error.
   - Bulk dismiss on `/alerts` (T-1.2) — no error; alerts dismissed.
   - Monitor page boot (T-1.3) — page loads cleanly logged out and logged in.
   - Block IP on alert card (T-3.1, T-3.3) — loading toast → success/error toast; button state updates; `GlobalState.actions` populated.
   - Dashboard modules render (T-3.2, T-6.5) — cards show real titles.
   - Home tile to `/threat-intel` (T-5.1, T-5.3) — keyboard, mouse, right-click all work.
   - Module settings (T-6.1) — non-blocking modal/toast (no native alert).
6. **No console errors or warnings in target browsers per report** — DevTools Console on each primary route is free of red errors. Warnings are minimized; intentional ones (e.g., disconnect warnings) are acceptable.
7. **Bundle size targets met (if report defines them)** — N/A — justified: report defines indicative payload values (~400KB CSS, ~80KB Bootstrap JS) as problems to reduce but does not specify hard targets. Validate that T-8.1 and T-8.3 produce a measurable reduction; record before/after numbers in `QA_PHASE_10.md`.
8. **Accessibility targets met (if report defines them)** — N/A — justified: report lists accessibility as 3/10 (ISS-021, etc.) but does not define numeric targets. Validate that home tiles are anchors (Tab focus visible, right-click works) per T-5.3, that the sidebar links are anchor elements per T-5.2, and that there are no native `alert()` blocking calls per T-6.1.

---

## 8. Appendix A — Issue → Task Reverse Index

Indexed by Task ID; reciprocal to Section 1.

| Task ID | Phase | Title | Resolves Issues |
|---------|-------|-------|-----------------|
| T-1.1 | 1 | Repair `/dashboard/main` route handler | ISS-003 |
| T-1.2 | 1 | Fix `GlobalState.state` → `GlobalState.data` in `alerts.page.js` | ISS-004 |
| T-1.3 | 1 | Replace `Socket.socket.on(...)` with `Socket.on(...)` in `monitor.page.js` | ISS-005 |
| T-2.1 | 2 | Repair `detection.html` dual-block structure and rewire controller | ISS-001, ISS-014 |
| T-2.2 | 2 | Wrap orphaned CSS in `health.html` inside `{% block extra_css %}` | ISS-001 |
| T-2.3 | 2 | Wrap orphaned CSS in `realtime.html` inside `{% block extra_css %}` | ISS-001 |
| T-3.1 | 3 | Fix `AlertCard` block-action response handling | ISS-006 |
| T-3.2 | 3 | Fix `ModuleCard` call signature in `dashboard.page.js` | ISS-007 |
| T-3.3 | 3 | Align `AlertCard` field names with normalized schema | ISS-008 |
| T-4.1 | 4 | Define design-token CSS variables in `base.html` `:root` | ISS-002 |
| T-4.2 | 4 | Define missing layout classes in `base.html` | ISS-011 |
| T-5.1 | 5 | Fix home-page Threat Intel tile URL | ISS-010 |
| T-5.2 | 5 | Expand sidebar navigation to all primary routes | ISS-013 |
| T-5.3 | 5 | Convert all `home.html` `onclick` tiles to `<a>` anchors | ISS-021 |
| T-5.4 | 5 | Add favicon to `base.html` | ISS-022 |
| T-6.1 | 6 | Replace native `alert()` in `base-module-controller.js` with `AppModal` | ISS-017, ISS-026 |
| T-6.2 | 6 | Move scripts out of `{% block content %}` into `{% block scripts %}` | ISS-019 |
| T-6.3 | 6 | Fix `AppToast.success` second-arg type in `dashboard.page.js` | ISS-020 |
| T-6.4 | 6 | Scope `base-module-controller.js` DOM lookups to module element | ISS-023 |
| T-6.5 | 6 | Reconcile `GlobalState.modules` slice shape | ISS-031 |
| T-7.1 | 7 | Resolve duplicate threat-intel templates | ISS-012 |
| T-7.2 | 7 | Fix or remove `socket-core.js` | ISS-016 |
| T-7.3 | 7 | Delete orphaned legacy `state.js` and `socket.js` | ISS-024 |
| T-7.4 | 7 | Move `generate_alerts.py` out of the public static directory | ISS-025 |
| T-7.5 | 7 | Remove dead `loading-spinner` import from `monitor.page.js` | ISS-027 |
| T-8.1 | 8 | Replace Tailwind CDN with built CSS file | ISS-015, ISS-028, ISS-033 |
| T-8.2 | 8 | Rationalize CSS framework stack (Bootstrap CSS scope) | ISS-018 |
| T-8.3 | 8 | Remove Bootstrap JS bundle from pages that don't need it | ISS-029 |
| T-9.1 | 9 | Generate alert audio files (or guard `playAlertTone`) | ISS-009 |
| T-9.2 | 9 | Remove production `console.log` statements | ISS-030 |
| T-9.3 | 9 | Add sanitization to `AppModal.setContent` | ISS-032 |
| T-10.1 | 10 | Execute Final Validation Protocol | All |

Cross-check against Section 1: every Issue ID ISS-001 through ISS-033 appears in this Reverse Index at least once. Coverage confirmed.

---

## 9. Appendix B — Open Questions / REQUIRES_REPO_INSPECTION

Every place where the report was silent and a guess was refused:

1. **T-1.3** — Exact name of the `GlobalState.subscribe('socket', ...)` callback parameter shape (whether `connected` is the field). Report names the `'socket'` slice but not its shape.
2. **T-1.3** — Whether `SocketManager.on()` internally guards `if (this.socket)` for both `connect` and arbitrary event names; report asserts it does for the public `Socket.on(...)` API but does not document edge cases.
3. **T-2.2 / T-2.3** — Exact line range of the orphaned `<style>` block in `health.html` and `realtime.html`. The report names only the starting line (72 and 76 respectively).
4. **T-2.1** — Whether `detection.page.js`'s DOM selectors match the surviving `ds-card` markup in `detection.html`. The report does not enumerate either selector set.
5. **T-3.1** — Whether `AppToast.loading` exists; if not, which API to use to create a dismissable persistent toast.
6. **T-3.1** — Other `HttpClient.post`/`HttpClient.put`/`HttpClient.patch` callers that may use the same incorrect `response.ok`/`response.status` pattern.
7. **T-3.2** — Exact shape of the module data object emitted by the back end into `GlobalState.modules` (which field is `id`, which is `title`).
8. **T-3.3** — Whether the normalized alert schema includes any destination IP field (the audit lists `source_ip` but explicitly notes `dst_ip` is "not in normalized schema at all").
9. **T-3.3** — Other callers of `alert.src_ip` outside `alert-card.js`.
10. **T-5.2** — Whether `.nav-section-label` already has a CSS rule.
11. **T-5.2** — Sidebar overflow behavior (whether scroll is `auto`).
12. **T-5.3** — Current `.home-tile` CSS rules and whether anchor element default styles need explicit reset.
13. **T-6.1** — The canonical method name on `AppModal` for displaying a simple modal (`alert`, `open`, `create({...}).open()`, etc.).
14. **T-6.2** — Whether any controller in `actions.html`, `honeypot.html`, `realtime.html` depended on running inside the content block's DOM context.
15. **T-6.3** — Signature of `AppToast.success` (positional `durationMs` vs options object).
16. **T-6.4** — Name of the root-element reference on `BaseModuleController` (`this.el`, `this.root`, `this.element`, etc.) and the exact templates that emit `id="contextStatus"` / `id="contextUpdated"`.
17. **T-6.5** — Exact line of the `modules` slice initialization in `core/global-state.js`.
18. **T-6.5** — Other consumers of the `modules` slice (subscribers and direct readers).
19. **T-7.1** — Any direct filename references to `threat-intel.html` (the stub) across the repo.
20. **T-7.4** — Project convention for non-served scripts (e.g., `web_app/tools/`, `web_app/scripts/`).
21. **T-8.1** — Existing Node tooling presence (`package.json`, build steps). The report says "No build system. No bundler." but a Tailwind CLI build requires Node.
22. **T-8.1** — Translation of the runtime `tailwind-config.js` theme into a Node `tailwind.config.js`.
23. **T-8.1** — Deployment process for the built `tailwind.css` (committed vs CI-generated).
24. **T-8.2** — Per-page audit of Bootstrap CSS class usage (which pages actually need Bootstrap utility classes).
25. **T-8.3** — Per-page audit of Bootstrap JS component usage (`data-bs-*` etc.).
26. **T-9.1** — Output paths of `generate_alerts.py`.
27. **T-9.3** — Location of `AppModal.setContent` in `app-modal.js` and call sites that pass HTML strings expecting HTML rendering.
28. **Section 7 item 3 (Lint)** — Whether a lint configuration exists in the repo.