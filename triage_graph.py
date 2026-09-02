"""
LangGraph workflow — first minimal version (Phase 4, step 1)

Deliberately small on purpose: two real nodes (search, decide) plus
three placeholder end nodes, just to prove the core mechanism works —
state passed through, a node making a real decision, conditional
routing actually branching to a different path depending on that
decision. The full response/escalation logic gets built in the next
step, once this proven working.
"""

import os
from typing import TypedDict
from dotenv import load_dotenv
import psycopg2
from openai import OpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

openai_client = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-3-small"

# Three-tier thresholds, based on this morning's real experiments
STRONG_MATCH_THRESHOLD = 0.5
WEAK_MATCH_THRESHOLD = 0.22


# =====================================================
# STATE — the "notebook" passed through every node
# =====================================================
class TriageState(TypedDict):
    incident_description: str
    similar_incidents: list
    policy_matches: list
    best_score: float
    decision: str  # set by the decide node
    response: str  # set by whichever end node runs
    path: list     # every node visited, in order — for display/debugging


# =====================================================
# Helper functions (same as test_retrieval.py)
# =====================================================
def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def query_db(function_name: str, embedding_str: str, match_count: int = 3):
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {function_name}(%s, %s);", (embedding_str, match_count))
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [dict(zip(columns, row)) for row in rows]


# =====================================================
# NODE 1: search — embeds the description, searches both tables
# =====================================================
def search_node(state: TriageState) -> TriageState:
    state["path"].append("search")
    print(f"[search_node] Searching for: {state['incident_description']}")

    embedding = get_embedding(state["incident_description"])
    embedding_str = str(embedding)

    incidents = query_db("match_incidents", embedding_str)
    policies = query_db("match_document_chunks", embedding_str)

    # Best score across BOTH sources — either one being strong is enough
    scores = [i["similarity"] for i in incidents] + [p["similarity"] for p in policies]
    best_score = max(scores) if scores else 0.0

    state["similar_incidents"] = incidents
    state["policy_matches"] = policies
    state["best_score"] = best_score

    print(f"[search_node] Best score found: {best_score:.3f}")
    return state


# =====================================================
# NODE 2: decide — the real decision logic, using the three tiers
# =====================================================
def decide_node(state: TriageState) -> TriageState:
    state["path"].append("decide")
    score = state["best_score"]

    if score >= STRONG_MATCH_THRESHOLD:
        state["decision"] = "strong_match"
    elif score >= WEAK_MATCH_THRESHOLD:
        state["decision"] = "weak_match"
    else:
        state["decision"] = "no_match"

    print(f"[decide_node] Decision: {state['decision']}")
    return state


# =====================================================
# Conditional routing function — reads the decision, picks the path
# =====================================================
def route_after_decision(state: TriageState) -> str:
    return state["decision"]  # must match a key in the conditional edges map below


# =====================================================
# Placeholder end nodes — just print for now, real logic comes next
# =====================================================
import anthropic

claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
CLAUDE_MODEL = "claude-sonnet-4-5"

# Categories that must ALWAYS escalate, regardless of match confidence —
# this enforces the policy rule written into raw_documents, as a hard
# rule the agent cannot override even on a strong match.
ALWAYS_ESCALATE_CATEGORIES = ["checkout", "payment", "order"]


def contains_escalation_keywords(text: str) -> bool:
    """Simple keyword check for the always-escalate policy override."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in ALWAYS_ESCALATE_CATEGORIES)


def strong_match_node(state: TriageState) -> TriageState:
    state["path"].append("strong_match")
    # Policy override: even a strong match must escalate if it touches
    # a protected category (checkout/payment/order) — this is the
    # guardrail from the plan, enforced as a hard rule, not a suggestion.
    if contains_escalation_keywords(state["incident_description"]):
        state["path"].append("policy_override")
        print("[strong_match_node] Policy override: protected category detected, escalating instead.")
        state["decision"] = "escalated_by_policy"
        state["response"] = (
            "This incident touches checkout/payment/order and is being escalated "
            "immediately per policy, regardless of match confidence."
        )
        return state

    incidents_text = "\n".join(
        f"- {i['title']} (resolution: {i['resolution_notes']})"
        for i in state["similar_incidents"] if i.get("resolution_notes")
    )
    policy_text = "\n".join(f"- {p['content']}" for p in state["policy_matches"])

    prompt = f"""You are an IT incident triage assistant. A new incident has come in:

"{state['incident_description']}"

Similar past incidents and their resolutions:
{incidents_text or "None with resolutions available."}

Relevant policy/runbook content:
{policy_text or "None found."}

Using ONLY the information above, suggest a likely cause and next step.
Be concise. If the past resolutions don't clearly apply, say so."""

    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}]
    )

    state["response"] = response.content[0].text
    print(f"[strong_match_node] Generated response:\n{state['response']}")
    return state


def weak_match_node(state: TriageState) -> TriageState:
    state["path"].append("weak_match")
    # Same policy override applies here too — even an uncertain match
    # to a protected category should escalate, not ask for clarification.
    if contains_escalation_keywords(state["incident_description"]):
        state["path"].append("policy_override")
        print("[weak_match_node] Policy override: protected category detected, escalating instead.")
        state["decision"] = "escalated_by_policy"
        state["response"] = (
            "This incident touches checkout/payment/order and is being escalated "
            "immediately per policy, regardless of match confidence."
        )
        return state

    # Build a short, honest summary of the best (but not strong) match found,
    # so the clarifying question is grounded in something real, not generic.
    best_incident = max(state["similar_incidents"], key=lambda i: i["similarity"], default=None)
    best_policy = max(state["policy_matches"], key=lambda p: p["similarity"], default=None)

    hint = ""
    if best_incident and best_incident["similarity"] >= WEAK_MATCH_THRESHOLD:
        hint = f' The closest related past incident was "{best_incident["title"]}" — is this related?'
    elif best_policy and best_policy["similarity"] >= WEAK_MATCH_THRESHOLD:
        hint = " This may be loosely related to existing guidance, but I'm not fully confident."

    state["response"] = (
        f"I found a possible but not confident match for this incident.{hint} "
        "Could you provide more detail — for example, which system or service is affected, "
        "and when it started?"
    )
    print(f"[weak_match_node] Generated response:\n{state['response']}")
    return state


def no_match_node(state: TriageState) -> TriageState:
    state["path"].append("no_match")
    # Even a "no match" case still respects the policy override — a
    # protected-category incident should never be silently dismissed
    # as out of scope, no matter how weak the similarity score is.
    if contains_escalation_keywords(state["incident_description"]):
        state["path"].append("policy_override")
        print("[no_match_node] Policy override: protected category detected, escalating instead.")
        state["decision"] = "escalated_by_policy"
        state["response"] = (
            "This incident touches checkout/payment/order and is being escalated "
            "immediately per policy, regardless of match confidence."
        )
        return state

    state["response"] = (
        "This doesn't appear to match anything in this system's known incident history "
        "or policy documentation. It may be outside the scope of this system, or may need "
        "manual review by a human."
    )
    print(f"[no_match_node] Generated response:\n{state['response']}")
    return state


# =====================================================
# Build the graph
# =====================================================
graph = StateGraph(TriageState)

graph.add_node("search", search_node)
graph.add_node("decide", decide_node)
graph.add_node("strong_match", strong_match_node)
graph.add_node("weak_match", weak_match_node)
graph.add_node("no_match", no_match_node)

graph.set_entry_point("search")
graph.add_edge("search", "decide")

# This is the conditional edge — the graph genuinely branches here,
# based on what decide_node found, not a fixed sequence.
graph.add_conditional_edges(
    "decide",
    route_after_decision,
    {
        "strong_match": "strong_match",
        "weak_match": "weak_match",
        "no_match": "no_match",
    }
)

graph.add_edge("strong_match", END)
graph.add_edge("weak_match", END)
graph.add_edge("no_match", END)

app = graph.compile()


if __name__ == "__main__":
    test_cases = [
        "Database queries are timing out for some users during busy periods",
        "The office coffee machine is broken",
        "remote control not working when changing channels",
        "Customers are getting errors when trying to pay at checkout",
    ]

    for description in test_cases:
        print(f"\n{'='*60}")
        print(f"Running workflow for: {description}")
        print('='*60)
        result = app.invoke({
            "incident_description": description,
            "similar_incidents": [],
            "policy_matches": [],
            "best_score": 0.0,
            "decision": "",
            "response": "",
            "path": []
        })
        print(f"\nFinal decision: {result['decision']}")
        print(f"Path taken: {' → '.join(result['path'])}")
        print(f"Response shown to user:\n{result['response']}")
