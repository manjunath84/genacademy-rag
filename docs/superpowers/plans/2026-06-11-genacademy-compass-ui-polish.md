# GenAcademy Compass UI Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the existing FastAPI/HTMX templates into a more polished demo surface, with **GenAcademy Compass** as the user-facing title and no RAG/core behavior changes.

**Architecture:** Keep the app server-rendered with Jinja templates and Tailwind CDN. Add a small shared template shell for consistent title, spacing, and navigation, then polish chat/auth/admin pages within the current route contracts. Suggested questions submit to the existing `/ask` endpoint; no new route, datastore, retriever, provider, or API behavior is introduced.

**Tech Stack:** FastAPI/Starlette Jinja templates, HTMX, Alpine.js, Tailwind CDN, pytest `TestClient`, ruff.

---

## Title Decision

Use **GenAcademy Compass** as the user-facing product title.

- Browser titles, login/signup headings, the chat header, and admin shell use `GenAcademy Compass`.
- Keep `GenAcademy RAG` as the repo/technical name in docs where architecture or implementation is discussed.
- Use the subtitle: `Evidence-first answers from the cohort materials.`

This is intentionally calmer than the Week 1 `State of the LLMs` brutalist data-story style. Borrow the interaction patterns, not the full palette: suggested chips, visible trust cues, and inspectable evidence.

## File Structure

- Create `src/genacademy_rag/web/templates/base.html`
  - Shared HTML head, Tailwind CDN, Alpine/HTMX scripts, body background, brand header, and content blocks.
  - No app logic; only template structure.
- Modify `src/genacademy_rag/web/templates/chat.html`
  - Extend `base.html`.
  - Add title/subtitle, suggested question chips, better ask form, answer card, source panel, and trust footer.
- Modify `src/genacademy_rag/web/templates/login.html`
  - Extend `base.html`.
  - Use the GenAcademy Compass title and subtitle.
- Modify `src/genacademy_rag/web/templates/signup.html`
  - Extend `base.html`.
  - Match login visual treatment.
- Modify `src/genacademy_rag/web/templates/admin_dashboard.html`
  - Extend `base.html`.
  - Add compact admin nav, improved KPI cards, safer empty states, and table overflow handling.
- Modify `src/genacademy_rag/web/templates/admin_documents.html`
  - Extend `base.html`.
  - Polish upload/re-index controls and document table.
- Modify `src/genacademy_rag/web/templates/admin_invites.html`
  - Extend `base.html`.
  - Polish invite form, success state, and invite table.
- Modify `tests/web/test_app.py`
  - Add template-level assertions for the new title, suggested question chips, admin nav, and preserved CSRF/form behavior.
- Modify `docs/deploy.md`
  - Update the live login/chat smoke text if it references the old `Ask the cohort materials` heading.
- Modify `README.md`
  - Mention `GenAcademy Compass` as the UI title while leaving `GenAcademy RAG` as the project/repo name.

## Task 1: Shared Template Shell And Title

**Files:**
- Create: `src/genacademy_rag/web/templates/base.html`
- Modify: `src/genacademy_rag/web/templates/login.html`
- Modify: `src/genacademy_rag/web/templates/signup.html`
- Test: `tests/web/test_app.py`

- [ ] **Step 1: Add failing auth page tests**

Add tests near the existing login/signup tests in `tests/web/test_app.py`:

```python
def test_login_page_uses_compass_title(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)

    page = c.get("/login")

    assert page.status_code == 200
    assert "<title>GenAcademy Compass" in page.text
    assert "GenAcademy Compass" in page.text
    assert "Evidence-first answers from the cohort materials." in page.text
    assert 'name="csrf_token"' in page.text


def test_signup_page_uses_compass_title(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)

    page = c.get("/signup")

    assert page.status_code == 200
    assert "<title>GenAcademy Compass" in page.text
    assert "Create your Compass account" in page.text
    assert 'name="code"' in page.text
    assert 'name="csrf_token"' in page.text
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
uv run pytest tests/web/test_app.py::test_login_page_uses_compass_title tests/web/test_app.py::test_signup_page_uses_compass_title -q
```

Expected: both tests fail because the current templates still say `GenAcademy RAG` / `Create account`.

- [ ] **Step 3: Create `base.html`**

Create `src/genacademy_rag/web/templates/base.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}GenAcademy Compass{% endblock %}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/htmx.org@1.9.12"></script>
  <script defer src="https://unpkg.com/alpinejs@3.14.1/dist/cdn.min.js"></script>
  <style>[x-cloak]{display:none}</style>
</head>
<body class="min-h-screen bg-slate-50 text-slate-950 antialiased">
  {% block body %}{% endblock %}
</body>
</html>
```

- [ ] **Step 4: Update login and signup templates**

Rewrite `login.html` to extend the base shell:

```html
{% extends "base.html" %}
{% block title %}GenAcademy Compass — Login{% endblock %}
{% block body %}
<main class="min-h-screen flex items-center justify-center px-4 py-10">
  <section class="w-full max-w-sm space-y-6">
    <div class="space-y-2 text-center">
      <p class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">GenAcademy</p>
      <h1 class="text-3xl font-semibold tracking-tight">GenAcademy Compass</h1>
      <p class="text-sm text-slate-600">Evidence-first answers from the cohort materials.</p>
    </div>
    <form method="post" action="/login" class="bg-white border border-slate-200 p-6 rounded-lg shadow-sm space-y-4">
      {% if error %}<p class="text-red-600 text-sm">{{ error }}</p>{% endif %}
      <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
      <label class="block text-sm font-medium text-slate-700">Email
        <input name="email" placeholder="email" class="mt-1 w-full border border-slate-300 rounded-md px-3 py-2" value="member@genacademy.local">
      </label>
      <label class="block text-sm font-medium text-slate-700">Password
        <input name="password" type="password" placeholder="password" class="mt-1 w-full border border-slate-300 rounded-md px-3 py-2">
      </label>
      <button class="w-full bg-slate-900 text-white rounded-md py-2 font-medium">Sign in</button>
      <p class="text-center text-sm text-slate-500"><a class="underline" href="/signup">Create account with invite code</a></p>
    </form>
  </section>
</main>
{% endblock %}
```

Rewrite `signup.html` similarly, with `Create your Compass account` as the H1, fields for email/password/invite code, and a link back to `/login`.

- [ ] **Step 5: Run the auth template tests**

Run:

```bash
uv run pytest tests/web/test_app.py::test_login_page_uses_compass_title tests/web/test_app.py::test_signup_page_uses_compass_title -q
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/genacademy_rag/web/templates/base.html src/genacademy_rag/web/templates/login.html src/genacademy_rag/web/templates/signup.html tests/web/test_app.py
git commit -m "feat: add compass auth shell"
```

## Task 2: Chat Workbench And Suggested Questions

**Files:**
- Modify: `src/genacademy_rag/web/templates/chat.html`
- Modify: `tests/web/test_app.py`

- [ ] **Step 1: Add failing chat page tests**

Add:

```python
def test_chat_page_has_compass_workbench_and_question_chips(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)

    page = c.get("/")

    assert page.status_code == 200
    assert "GenAcademy Compass" in page.text
    assert "Evidence-first answers from the cohort materials." in page.text
    assert "Ask a suggested question" in page.text
    assert "What is retrieval augmented generation?" in page.text
    assert "How should I evaluate a RAG system?" in page.text
    assert "What did the course say about embeddings?" in page.text
    assert 'name="question"' in page.text
    assert 'name="csrf_token"' in page.text
```

- [ ] **Step 2: Run the failing chat test**

Run:

```bash
uv run pytest tests/web/test_app.py::test_chat_page_has_compass_workbench_and_question_chips -q
```

Expected: fail because the chat page does not yet have the title/subtitle/chips.

- [ ] **Step 3: Update `chat.html`**

Make `chat.html` extend `base.html`. Keep the existing form names, CSRF fields, `/ask` action, result rendering, source rendering, copy/retry/feedback forms, and refusal behavior.

Use this structure:

```html
{% extends "base.html" %}
{% block title %}GenAcademy Compass{% endblock %}
{% block body %}
<main class="mx-auto max-w-6xl px-4 py-6 lg:px-8">
  <section class="mb-6 flex flex-col gap-4 border-b border-slate-200 pb-5 md:flex-row md:items-end md:justify-between">
    <div class="space-y-2">
      <p class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">GenAcademy</p>
      <h1 class="text-3xl font-semibold tracking-tight">GenAcademy Compass</h1>
      <p class="max-w-2xl text-sm text-slate-600">Evidence-first answers from the cohort materials.</p>
    </div>
    <div class="flex gap-2 text-sm">
      <a class="rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-700" href="/admin/documents">Documents</a>
      <a class="rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-700" href="/admin/dashboard">Dashboard</a>
    </div>
  </section>

  <section class="grid gap-6 lg:grid-cols-[minmax(0,1fr)_18rem]">
    <div class="space-y-4">
      <form method="post" action="/ask" class="bg-white border border-slate-200 rounded-lg p-3 shadow-sm">
        <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
        <div class="flex flex-col gap-3 sm:flex-row">
          <input name="question" value="{{ question or '' }}" placeholder="Ask about assignments, retrieval, embeddings, evals, or course logistics"
                 class="min-h-11 flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm">
          <button class="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white">Ask</button>
        </div>
      </form>

      <section class="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <p class="mb-3 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Ask a suggested question</p>
        <div class="flex flex-wrap gap-2">
          {% for suggested in [
            "What is retrieval augmented generation?",
            "How should I evaluate a RAG system?",
            "What did the course say about embeddings?",
            "When should the assistant refuse to answer?"
          ] %}
          <form method="post" action="/ask">
            <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
            <button name="question" value="{{ suggested }}" class="rounded-full border border-slate-300 bg-slate-50 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-100">{{ suggested }}</button>
          </form>
          {% endfor %}
        </div>
      </section>

      {# Existing result card goes here, restyled but with current conditionals preserved. #}
    </div>

    <aside class="space-y-3 text-sm">
      <div class="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <p class="font-medium text-slate-900">Trust posture</p>
        <ul class="mt-3 space-y-2 text-slate-600">
          <li>Cited answers only when supported.</li>
          <li>Honest refusal when materials are insufficient.</li>
          <li>Sources stay inspectable in the answer card.</li>
        </ul>
      </div>
    </aside>
  </section>
</main>
{% endblock %}
```

Do not copy the inline planning comment into the final file. Move the existing result card into the indicated area and restyle it with `rounded-lg`, `border border-slate-200`, `shadow-sm`, stable text sizes, and accessible button labels.

- [ ] **Step 4: Run chat tests**

Run:

```bash
uv run pytest tests/web/test_app.py::test_chat_page_has_compass_workbench_and_question_chips tests/web/test_app.py::test_answer_card_renders_badge_sources_disclaimer tests/web/test_app.py::test_refused_card_has_refusal_badge_no_copy_no_sources tests/web/test_app.py::test_ask_requires_csrf_and_writes_usage_row -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/genacademy_rag/web/templates/chat.html tests/web/test_app.py
git commit -m "feat: polish compass chat workbench"
```

## Task 3: Admin Shell And Tables

**Files:**
- Modify: `src/genacademy_rag/web/templates/admin_dashboard.html`
- Modify: `src/genacademy_rag/web/templates/admin_documents.html`
- Modify: `src/genacademy_rag/web/templates/admin_invites.html`
- Modify: `tests/web/test_app.py`

- [ ] **Step 1: Add failing admin template tests**

Add:

```python
def test_admin_pages_use_compass_admin_shell(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c, "admin@genacademy.local", "admin")

    for path, heading in [
        ("/admin/dashboard", "Operations dashboard"),
        ("/admin/documents", "Corpus documents"),
        ("/admin/invites", "Invite management"),
    ]:
        page = c.get(path)
        assert page.status_code == 200
        assert "GenAcademy Compass" in page.text
        assert heading in page.text
        assert 'href="/"' in page.text
        assert 'href="/admin/documents"' in page.text
        assert 'href="/admin/dashboard"' in page.text
        assert 'href="/admin/invites"' in page.text
```

- [ ] **Step 2: Run the failing admin shell test**

Run:

```bash
uv run pytest tests/web/test_app.py::test_admin_pages_use_compass_admin_shell -q
```

Expected: fail because admin pages do not yet include the Compass admin shell.

- [ ] **Step 3: Update admin templates**

For each admin template:

- Extend `base.html`.
- Use the title `GenAcademy Compass — Admin`.
- Add a consistent admin header:

```html
<section class="mb-6 flex flex-col gap-4 border-b border-slate-200 pb-5 md:flex-row md:items-end md:justify-between">
  <div>
    <p class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">GenAcademy Compass</p>
    <h1 class="text-3xl font-semibold tracking-tight">{% block admin_heading %}{% endblock %}</h1>
  </div>
  <nav class="flex flex-wrap gap-2 text-sm">
    <a class="rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-700" href="/">Ask</a>
    <a class="rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-700" href="/admin/documents">Documents</a>
    <a class="rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-700" href="/admin/dashboard">Dashboard</a>
    <a class="rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-700" href="/admin/invites">Invites</a>
  </nav>
</section>
```

Do not create a Jinja macro yet; duplication across three templates is acceptable for this small slice and avoids a partial/macro abstraction.

Specific headings:

- `admin_dashboard.html`: `Operations dashboard`
- `admin_documents.html`: `Corpus documents`
- `admin_invites.html`: `Invite management`

Specific polish:

- Dashboard KPI cards: use `rounded-lg border border-slate-200 bg-white p-4 shadow-sm`.
- Tables: wrap in `overflow-x-auto`, use `min-w-full`, tighter headers, and `break-words` on long questions/doc titles.
- Empty states: if `summary.top_questions`, `rows`, `documents`, or `invites` are empty, render a muted line such as `No rows yet.` inside the existing section.
- Forms: keep all existing `name`, `action`, `method`, and CSRF fields unchanged.

- [ ] **Step 4: Run admin tests**

Run:

```bash
uv run pytest tests/web/test_app.py::test_admin_pages_use_compass_admin_shell tests/web/test_app.py::test_dashboard_renders_usage_summary tests/web/test_app.py::test_dashboard_shows_feedback_counts -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/genacademy_rag/web/templates/admin_dashboard.html src/genacademy_rag/web/templates/admin_documents.html src/genacademy_rag/web/templates/admin_invites.html tests/web/test_app.py
git commit -m "feat: polish compass admin pages"
```

## Task 4: Docs And Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/deploy.md`

- [ ] **Step 1: Update docs**

In `README.md`, add one sentence near the top:

```markdown
The user-facing app title is **GenAcademy Compass**; the repository and technical docs keep the `GenAcademy RAG` name.
```

In `docs/deploy.md`, replace the old smoke expectation:

```markdown
Expected: the chat screen loads with the heading **Ask the cohort materials**.
```

with:

```markdown
Expected: the chat screen loads with the heading **GenAcademy Compass**.
```

- [ ] **Step 2: Run full verification**

Run:

```bash
git diff --check
uv run ruff check .
uv run pytest
```

Expected:

- `git diff --check`: no output, exit 0.
- `uv run ruff check .`: `All checks passed!`
- `uv run pytest`: all tests pass, with the existing warning count acceptable.

- [ ] **Step 3: Local browser check**

Run the app locally:

```bash
uv run uvicorn genacademy_rag.web.main:app --host 0.0.0.0 --port 7860
```

Open `http://127.0.0.1:7860/login`, sign in as `member@genacademy.local` / `member`, and verify:

- `GenAcademy Compass` is the visible title.
- Suggested question chips are visible and submit.
- Answer card sources remain visible.
- Admin dashboard remains readable at mobile and desktop widths.

If the local server cannot start in the implementation environment, record the exact error and use `uv run pytest tests/web/test_app.py -q` as the minimum fallback evidence.

- [ ] **Step 4: Commit docs**

```bash
git add README.md docs/deploy.md
git commit -m "docs: document compass ui title"
```

## Acceptance Criteria

- The app’s visible product title is `GenAcademy Compass`.
- Chat page first viewport communicates the trust promise without a marketing landing page.
- Suggested question chips work through the existing `/ask` form path.
- Existing answer/refusal/source/feedback behavior is preserved.
- Admin pages are more scannable without adding new metrics or backend logic.
- No React, new frontend build step, new route, new datastore table, or retrieval/core change.
- Full ruff and pytest verification is shown before completion.

## Review Handoff Prompt

Use this prompt with a fresh reviewer before implementation:

```text
Review the GenAcademy Compass UI polish implementation plan.

Target file:
- docs/superpowers/plans/2026-06-11-genacademy-compass-ui-polish.md

Context to read first:
- AGENTS.md
- README.md
- specs/roadmap.md
- docs/minimal-system-design.md
- src/genacademy_rag/web/templates/chat.html
- src/genacademy_rag/web/templates/login.html
- src/genacademy_rag/web/templates/signup.html
- src/genacademy_rag/web/templates/admin_dashboard.html
- src/genacademy_rag/web/templates/admin_documents.html
- src/genacademy_rag/web/templates/admin_invites.html
- tests/web/test_app.py

Review goal:
Evaluate whether this is a safe, minimal, demo-visible UI/UX polish slice. The user-facing title should become "GenAcademy Compass", while the repo/technical name can remain "GenAcademy RAG".

Constraints:
- Do not add React, a JS build step, a new design system, or new backend behavior.
- Do not change retrieval, eval, provider, vector store, datastore, auth, or upload semantics.
- Preserve CSRF fields, form names/actions, feedback forms, source rendering, and refusal behavior.
- Keep UI quiet and operational, not a marketing landing page.
- Suggested question chips must use the existing `/ask` POST flow.

Output format:
1. Findings first, ordered by severity.
2. Include file/line references where possible.
3. Call out any plan steps that are too broad, brittle, or likely to break existing tests.
4. State whether the plan is implementation-ready.
5. Recommend exact edits if needed.
6. Do not modify files or implement anything.
```
