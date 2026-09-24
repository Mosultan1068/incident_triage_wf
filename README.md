# IT Incident Triage Agent

An agentic workflow (built with LangGraph) that triages IT incidents: searching past incidents and policy documentation, then deciding — based on genuine, measured confidence — whether to respond, ask a clarifying question, or escalate. Built on a real professional pattern (vectorizing production incidents for similarity matching) and extended into genuine multi-step, conditional agent reasoning, a live comparison against LLM-based judgment, and real email notifications.

---

## What makes this different from a typical chatbot

Most chatbot implementations follow one fixed path: question in → retrieve → generate → answer out, every time, regardless of the question. This project genuinely branches: the code path taken depends on what's actually found at runtime, not a fixed sequence. It can confidently answer, ask for more information, decline entirely, escalate, or override its own confidence for a protected category — several structurally different outcomes, decided by the workflow itself, not a single generate-and-hope call.

---

## Architecture

**Database:** Neon (Postgres + `pgvector`) — a second Postgres provider, deliberately distinct from Supabase used in earlier projects, for genuine multi-provider experience.

**Two data types, connected by a LangGraph agent:**
- **`incidents`** — structured, with its own `embedding` column, enabling direct similarity search across past incidents (mirroring real professional incident-matching work).
- **RAG trio** (`raw_documents` → `documents`/`document_chunks`) — policy and runbook documentation, chunked and embedded, same proven pattern as earlier projects.

---

## Two graphs, one shared foundation

The project runs **two separate, independently compiled LangGraph workflows**, sharing `search_node` but differing entirely in how the decision is made:

- **`app` (deterministic)** — `decide_node` compares the top similarity score against two fixed thresholds, empirically derived by running real test queries and observing where genuine matches, borderline matches, and true non-matches actually clustered. Fast, predictable, testable.
- **`agentic_app` (agentic)** — `agentic_decide_node` sends the retrieved evidence to Claude and asks it to genuinely *judge* relevance, choosing from a richer six-outcome set (`confident_answer`, `answer_with_caveat`, `needs_clarification`, `likely_different_issue`, `escalate_recommended`, `out_of_scope`). Slower and non-deterministic, but capable of nuance a fixed number cannot — most notably `likely_different_issue`, which catches cases where a high similarity score is actually about a different underlying problem, something no threshold comparison can detect.

A **"Compare: Deterministic vs Agentic"** tab in the front end runs the same incident description through both graphs and displays the results, timing, and reasoning path side by side — turning an architectural distinction into something directly observable, not just described.

---

## The follow-up loop

A weak match doesn't just end the conversation — it triggers a bounded retry. Because a LangGraph `invoke()` call runs start to finish with no way to pause mid-execution for real human input, the loop is implemented **outside the graph, at the calling application layer**: a counter persists in `app.py` (not inside the graph's state, which resets on every fresh `invoke()`), incrementing on each weak match, capped at two attempts before the system accepts the best available answer rather than looping indefinitely. This was a genuine, hard-won design realization — not something obvious from the LangGraph documentation, but discovered by working through exactly what a graph can and cannot do across separate calls.

---

## The three-tier confidence system — derived empirically, not assumed

| Zone | Score range | Behaviour |
|---|---|---|
| Strong match | ≥ 0.5 | Confident, grounded response citing the specific evidence |
| Weak match | 0.22 – 0.5 | Asks a genuine clarifying question, looping up to twice |
| No match | < 0.22 | States it's outside the system's scope — no LLM call made |

Similarity search always returns *something*, ranked by relative closeness — it never inherently signals "nothing here is relevant." That judgment is a deliberate, separate decision layer, not built into the search itself.

---

## The unbreakable policy override

Every outcome node — strong match, weak match, no match, and both agentic paths — checks first, before doing anything else, whether the incident touches a protected category (checkout, payment, order). If so, it escalates immediately, regardless of confidence or judgment. No path through either graph can bypass this rule, because the check is duplicated at every single exit point rather than relying on a single upstream gate.

---

## Real email notifications

A shared `notify_node`, placed after every outcome node in both graphs (again, an unbypassable single point), sends the final decision and response as an email via Gmail SMTP. It fails gracefully: missing credentials or a send failure are logged and reported back through the UI (📧 sent / ⚪ skipped / ⚠️ failed), never crashing the workflow or losing the actual triage response over a notification issue.

---

## Tech stack

Python 3.11 · Neon (Postgres + `pgvector`) · `psycopg2` (direct SQL) · OpenAI (`text-embedding-3-small`) · Anthropic Claude (generation and agentic judgment) · LangGraph · Gmail SMTP (`smtplib`) · Gradio · Plotly · Docker · GitHub Actions

---

## Front end

A Gradio app with four tabs:
- **Triage** — the main interaction: decision, confidence score, the exact path taken through the graph, matched evidence (as tables), a bar chart of match scores with the threshold lines drawn on it, email notification status, and a progress indicator.
- **Compare: Deterministic vs Agentic** — the same description run through both graphs, side by side, with timing.
- **How it works** — a plain-language legend covering the decision types and the policy override.
- **History** — a running log of every incident tried in the session.

The front end calls the compiled graphs directly (`triage_app.invoke(...)`, `agentic_app.invoke(...)`) — neither graph contains any Gradio-specific code, so the same logic could sit behind a CLI, a Slack bot, or a FastAPI backend without modification.

---

## Running it

**Locally:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

**In Docker:**
```bash
docker build -t incident-triage-app .
docker run -p 7862:7862 --env-file .env incident-triage-app
```

**Populating data:**
```bash
python embed_incidents.py
python ingest_runbooks.py
```

**Email notifications** require three additional `.env` variables: `NOTIFY_EMAIL`, `NOTIFY_EMAIL_APP_PASSWORD` (a Gmail App Password, not the account password), and `NOTIFY_RECIPIENT`. Left unset, the system runs identically but skips sending, logging this clearly rather than failing.

---

## CI/CD

Every push to `main` triggers a GitHub Actions workflow: install dependencies → validate code syntax → build the Docker image. No live database or API credentials are exposed to CI — a deliberate security boundary.

---

## Real debugging stories worth keeping

Two genuinely instructive bugs came out of extending this project, beyond the ones in the original build:

- **A missing graph node registration**, twice, caused by file edits and downloads silently dropping lines during iteration — diagnosed by reading LangGraph's own validation error (`"Found edge starting at unknown node"`) literally and checking the file's actual content rather than assuming an edit had landed correctly.
- **Character corruption during copy-paste** (smart quotes, em-dashes, unterminated triple-quoted strings) repeatedly broke the file until the fix was switched to plain string concatenation and direct file downloads instead of clipboard-based paste — a small but real lesson in how invisible characters can silently break code that looks correct on screen.

---

## What I'd do differently at scale

- Add a real `ivfflat` (or `hnsw`) index once either table grows into the thousands of rows
- Move the policy-override keyword check to a proper classification step for more reliable category detection
- Add genuine authentication and per-user RLS for real (non-mock) incident data
- Extend notification routing to vary the recipient by severity or category, rather than one fixed address
- Add true streaming for the progress indicator, rather than the current two-step "before/after" approximation
