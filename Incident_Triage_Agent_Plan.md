# Project Plan: IT Incident Triage Agent

An agentic workflow project — a LangGraph-orchestrated agent that triages IT incidents: finding similar past incidents, checking policy/runbook documentation, and deciding whether to auto-respond or escalate. Built on a real professional pattern (incident vectorization for similarity matching), extended into genuine agentic, multi-step, conditional reasoning.

**Database:** Neon (Postgres + `pgvector`)
**Framework:** LangChain + LangGraph
**Pace:** same as previous projects — steady, one phase at a time.

---

## Why this project is a real step up

Every previous project followed one path: question in → retrieve → generate → answer out. This project introduces genuine **branching, conditional logic** — the agent decides which path to take based on what it finds, and can loop back if it doesn't have enough information. That's the real definition of "agentic," not just a new vocabulary word.

---

## Schema design

### Structured + vector-enabled: `incidents`
One row per incident, with its own embedding — enabling similarity search directly across past incidents (mirroring the real incident-matching work already done professionally).
```
id, title, description, category, severity, status,
created_at, resolved_at, resolution_notes, embedding vector(1536)
```

### RAG trio (policy/runbook knowledge base) — same proven pattern as the coffee project
- `raw_documents` — staging table for runbooks, escalation policies, SLA documentation
- `documents` — parent documents, populated by an ingestion script
- `document_chunks` — chunked, embedded content, searchable by similarity

### Similarity search functions
- `match_incidents(query_embedding, match_count)` — find similar past incidents
- `match_document_chunks(query_embedding, match_count)` — find relevant policy content

**No `ivfflat` index** — same lesson carried forward from the coffee project. Plain exact search until the dataset is genuinely large.

**RLS** — enabled with permissive policies, same reasoning as previous projects (proof of concept, no real user accounts).

---

## The LangGraph workflow — the new, central piece

**State** passed through the graph: incoming incident description, similar incidents found, relevant policy content found, severity assessment, final decision, drafted response.

**Nodes:**
1. **Search similar incidents** — embed the new incident, run `match_incidents`
2. **Search policy documentation** — embed the new incident, run `match_document_chunks`
3. **Assess severity** — combine similarity results + structured rules (e.g. category, keyword signals) to determine a severity level
4. **Conditional edge** — based on severity + whether a good historical match exists:
   - High similarity to a resolved past incident, low severity → **auto-respond node**
   - High severity, or no good match found → **escalate node**
   - Ambiguous → **request more information node** (loops back, asking a clarifying question)
5. **Draft response** — generates a grounded response or escalation summary, citing the specific similar incidents/policy content used

**Guardrails, built in from the start (not an afterthought):**
- A maximum step/loop count, so the agent can't loop indefinitely
- Every auto-response requires similarity above a defined threshold — same strict-grounding discipline as before
- Escalation is the default when uncertain, never auto-response

---

## Phased plan

**Phase 1 — Neon setup + schema**
Create Neon project, enable `pgvector`, run the schema script, insert mock incident and policy data (mirroring real incident types, in varied everyday language — same deliberate paraphrase-gap approach as the comparison project).

**Phase 2 — Ingestion**
Embedding scripts for both `incidents` and the RAG document trio — reusing the proven pattern from `ingest.py` / `embed_sleep_issues.py`.

**Phase 3 — Prove retrieval works, plain scripts**
Confirm `match_incidents` and `match_document_chunks` both return sensible results before touching LangGraph at all.

**Phase 4 — Build the LangGraph workflow**
Start with just two nodes (search + decide), prove the conditional branching works, then add the remaining nodes one at a time. This is the phase most likely to need patience and iteration — deliberately built incrementally, not all at once.

**Phase 5 — Front end**
Gradio, calling the LangGraph workflow directly — input box for a new incident description, output showing the agent's full reasoning path (which nodes it visited, why), not just the final answer. Showing the reasoning path is itself a valuable, demoable feature.

**Phase 6 — CI/CD**
Docker + GitHub Actions, same proven pattern as all three previous projects.

**Phase 7 — Write-up**
README covering the architecture, the guardrails design, and — importantly — an honest account of how the agent behaves on edge cases (ambiguous incidents, no good match found).

---

## Rules carried over from previous projects

1. Prove each phase works before adding the next.
2. Build the LangGraph workflow incrementally — two nodes working, then expand, never all nodes at once untested.
3. Guardrails (step limits, thresholds, default-to-escalate) are part of the design from day one, not bolted on after.
4. Finished and honestly scoped beats ambitious and half-working.
