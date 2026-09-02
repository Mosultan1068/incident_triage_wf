# IT Incident Triage Agent

An agentic workflow (built with LangGraph) that triages IT incidents: searching past incidents and policy documentation, then deciding — based on genuine, measured confidence — whether to respond, ask a clarifying question, or escalate. Built on a real professional pattern (vectorizing production incidents for similarity matching) and extended into genuine multi-step, conditional agent reasoning.

---

## What makes this different from a typical chatbot

Most chatbot implementations follow one fixed path: question in → retrieve → generate → answer out, every time, regardless of the question. This project genuinely branches: the code path taken depends on what's actually found at runtime, not a fixed sequence. It can confidently answer, ask for more information, decline entirely, or override its own confidence for a protected category — four structurally different outcomes, decided by the workflow itself.

---

## Architecture

**Database:** Neon (Postgres + `pgvector`) — chosen deliberately as a second Postgres provider, distinct from Supabase used in earlier projects, for genuine multi-provider experience.

**Two data types, connected by a LangGraph agent:**
- **`incidents`** — structured, with its own `embedding` column, enabling direct similarity search across past incidents (mirroring real professional incident-matching work).
- **RAG trio** (`raw_documents` → `documents`/`document_chunks`) — policy and runbook documentation, chunked and embedded, same proven pattern as earlier projects.

**The agentic workflow** (`triage_graph.py`), built with LangGraph:
- **State** — a shared record carrying the incident description, retrieved matches, the decision made, the response, and the exact path taken through the graph.
- **Nodes** — `search` (embeds and queries both tables), `decide` (applies the confidence thresholds), and three outcome nodes: `strong_match`, `weak_match`, `no_match`.
- **Conditional edge** — after `decide`, the graph genuinely branches based on the computed confidence tier, not a fixed rule written in advance.
- **Hard policy override** — every outcome node checks, before anything else, whether the incident touches a protected category (checkout, payment, order). If so, it escalates immediately, regardless of match confidence — a rule that cannot be bypassed by any path through the graph, enforced in code rather than left to the LLM's judgment alone.

---

## The three-tier confidence system — derived empirically, not assumed

Rather than picking a similarity threshold arbitrarily, the thresholds were set by running real test queries and observing actual score clusters:

| Zone | Score range | Example | Behaviour |
|---|---|---|---|
| Strong match | ≥ 0.5 | A paraphrased version of a resolved past incident | Confident, grounded response citing the specific evidence |
| Weak match | 0.22 – 0.5 | A tangentially related question | Asks a genuine clarifying question |
| No match | < 0.22 | A genuinely unrelated question | States it's outside the system's scope — no LLM call made |

This distinction matters because similarity search always returns *something*, ranked by relative closeness — it never inherently signals "nothing here is relevant." That judgment has to be a deliberate, separate decision layer, which is exactly what the `decide` node and its thresholds provide.

---

## Tech stack

Python 3.11 · Neon (Postgres + `pgvector`) · `psycopg2` (direct SQL, no ORM/API layer) · OpenAI (`text-embedding-3-small`) · Anthropic Claude (generation) · LangGraph · Gradio · Plotly · Docker · GitHub Actions

---

## Front end

A Gradio app with three tabs:
- **Triage** — the main interaction: decision label, confidence score, the exact path taken through the graph, matched incidents and policy content (as tables), a bar chart of match scores with the threshold lines drawn on it, and the final response.
- **How it works** — a plain-language legend explaining the four decision types and why the policy override exists, so a viewer understands what they're about to see before running a query.
- **History** — a running log of every incident tried in the session.

The front end calls the compiled LangGraph app directly (`triage_app.invoke(...)`) — the agent's logic contains no Gradio-specific code at all, so the same graph could sit behind a CLI, a Slack bot, or a FastAPI backend without any changes to `triage_graph.py`.

---

## Running it

**Locally:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Opens automatically at `http://localhost:7862`.

**In Docker:**
```bash
docker build -t incident-triage-app .
docker run -p 7862:7862 --env-file .env incident-triage-app
```

**Populating data:**
```bash
python embed_incidents.py     # embeds the incidents table directly
python ingest_runbooks.py     # chunks and embeds policy/runbook content
```

---

## CI/CD

Every push to `main` triggers a GitHub Actions workflow: install dependencies → validate code syntax across all core files → build the Docker image. As with previous projects, no live database or API credentials are exposed to CI — a deliberate security boundary, not an oversight.

---

## What I'd do differently at scale

- Add a real `ivfflat` (or `hnsw`) index once either table grows into the thousands of rows — a lesson carried forward from a prior project where that index broke retrieval at small scale
- Move the policy-override keyword check to a proper classification step rather than a simple keyword match, for more reliable category detection
- Add genuine authentication and per-user RLS if this were handling real incident data rather than mock content
- Extend the graph with a loop: if `weak_match` asks a clarifying question, feed the follow-up answer back into `search` for a second pass, rather than ending the workflow

---

## What this project demonstrates

A working example of the distinction between a single LLM call and a genuine agentic workflow: state passed through multiple steps, a decision point that changes which code actually runs, and a hard safety rule enforced in code rather than left to model judgment alone.
