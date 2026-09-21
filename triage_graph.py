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
# AGENTIC DECISION NODE — the LLM-judgment alternative to decide_node
#
# Unlike decide_node (a fixed threshold comparison), this asks Claude
# to actually READ the retrieved incidents/policy content and judge
# relevance — capable of nuance a fixed number can't capture, such as
# recognising a high similarity score that's actually about a
# different underlying problem.
# =====================================================
AGENTIC_OUTCOMES = [
    "confident_answer", "answer_with_caveat", "needs_clarification",
    "likely_different_issue", "escalate_recommended", "out_of_scope"
]


def agentic_decide_node(state: TriageState) -> TriageState:
    state["path"].append("agentic_decide")

    incidents_summary = "\n".join(
        f"- {i['title']} (similarity: {i['similarity']:.2f}, status: {i['status']})"
        for i in state["similar_incidents"]
    ) or "None found."

    policy_summary = "\n".join(
        f"- {p['content'][:120]} (similarity: {p['similarity']:.2f})"
        for p in state["policy_matches"]
    ) or "None found."

    prompt = f"""A new incident was reported: "{state['incident_description']}"

Similar past incidents found:
{incidents_summary}

Relevant policy content found:
{policy_summary}

Judge this situation and choose exactly ONE outcome:
- confident_answer: the evidence clearly and directly addresses this incident
- answer_with_caveat: relevant evidence exists, but with some uncertainty
- needs_clarification: too ambiguous to answer without more detail
- likely_different_issue: the retrieved evidence LOOKS similar in wording,
  but on closer reading is actually about a different underlying problem
- escalate_recommended: this looks serious enough to warrant human review,
  regardless of how confident you are
- out_of_scope: nothing here is genuinely relevant

Respond with only the outcome name, nothing else."""

    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}]
    )
    judgment = response.content[0].text.strip()

    # Safety net: if the model returns something unexpected, don't crash —
    # fall back to the safest option.
    state["decision"] = judgment if judgment in AGENTIC_OUTCOMES else "needs_clarification"

    print(f"[agentic_decide_node] LLM judgment: {state['decision']}")
    return state


# =====================================================
# AGENTIC OUTCOME NODES — one real branch per judgment
#
# Same idea as strong/weak/no_match: the decision is not just a label
# on a single prompt. Each outcome runs different code (LLM draft vs
# clarifying question vs fixed escalation/out-of-scope text).
# =====================================================
def _agentic_policy_override(state: TriageState) -> bool:
    """Hard rule: protected categories escalate even after an LLM judgment."""
    if not contains_escalation_keywords(state["incident_description"]):
        return False
    state["path"].append("policy_override")
    state["decision"] = "escalated_by_policy"
    state["response"] = (
        "This incident touches checkout/payment/order and is being "
        "escalated immediately per policy, regardless of the agent's judgment."
    )
    print("[agentic] Policy override: protected category detected, escalating instead.")
    return True


def _retrieved_context(state: TriageState) -> tuple[str, str]:
    incidents_text = "\n".join(
        f"- {i['title']} (resolution: {i['resolution_notes']})"
        for i in state["similar_incidents"] if i.get("resolution_notes")
    ) or "None with resolutions available."
    policy_text = "\n".join(
        f"- {p['content']}" for p in state["policy_matches"]
    ) or "None found."
    return incidents_text, policy_text


def _claude_reply(prompt: str) -> str:
    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def confident_answer_node(state: TriageState) -> TriageState:
    state["path"].append("confident_answer")
    if _agentic_policy_override(state):
        return state

    incidents_text, policy_text = _retrieved_context(state)
    prompt = f"""You are an IT incident triage assistant. Evidence clearly matches this incident.

Incident: "{state['incident_description']}"

Similar past incidents and resolutions:
{incidents_text}

Relevant policy content:
{policy_text}

Using ONLY the information above, suggest a likely cause and next step.
Be concise and specific. Do not hedge unnecessarily."""

    state["response"] = _claude_reply(prompt)
    print(f"[confident_answer_node] Generated response:\n{state['response']}")
    return state


def answer_with_caveat_node(state: TriageState) -> TriageState:
    state["path"].append("answer_with_caveat")
    if _agentic_policy_override(state):
        return state

    incidents_text, policy_text = _retrieved_context(state)
    prompt = f"""You are an IT incident triage assistant. Related evidence exists, but you are not fully certain it applies.

Incident: "{state['incident_description']}"

Similar past incidents and resolutions:
{incidents_text}

Relevant policy content:
{policy_text}

Using ONLY the information above, suggest a possible cause and next step.
You MUST state the uncertainty clearly — what might not apply, and what would confirm it."""

    state["response"] = _claude_reply(prompt)
    print(f"[answer_with_caveat_node] Generated response:\n{state['response']}")
    return state


def needs_clarification_node(state: TriageState) -> TriageState:
    state["path"].append("needs_clarification")
    if _agentic_policy_override(state):
        return state

    best_incident = max(state["similar_incidents"], key=lambda i: i["similarity"], default=None)
    hint = ""
    if best_incident:
        hint = f' The closest past incident was "{best_incident["title"]}" — is this related?'

    state["response"] = (
        f"I need a bit more detail before I can triage this confidently.{hint} "
        "Could you say which system or service is affected, and when it started?"
    )
    print(f"[needs_clarification_node] Generated response:\n{state['response']}")
    return state


def likely_different_issue_node(state: TriageState) -> TriageState:
    state["path"].append("likely_different_issue")
    if _agentic_policy_override(state):
        return state

    incidents_text, policy_text = _retrieved_context(state)
    prompt = f"""You are an IT incident triage assistant. Retrieved evidence looks similar in wording, but on closer reading it is about a different underlying problem.

Incident: "{state['incident_description']}"

Retrieved past incidents and resolutions:
{incidents_text}

Retrieved policy content:
{policy_text}

Explain briefly why this is probably a different issue, and ask one focused question that would confirm it. Do not treat the retrieved resolutions as the answer."""

    state["response"] = _claude_reply(prompt)
    print(f"[likely_different_issue_node] Generated response:\n{state['response']}")
    return state


def escalate_recommended_node(state: TriageState) -> TriageState:
    state["path"].append("escalate_recommended")
    if _agentic_policy_override(state):
        return state

    best_incident = max(state["similar_incidents"], key=lambda i: i["similarity"], default=None)
    related = f' Closest past incident on file: "{best_incident["title"]}".' if best_incident else ""

    state["response"] = (
        "This incident looks serious enough to warrant human review rather than "
        f"an automated resolution.{related} Flagging it for an on-call engineer."
    )
    print(f"[escalate_recommended_node] Generated response:\n{state['response']}")
    return state


def out_of_scope_node(state: TriageState) -> TriageState:
    state["path"].append("out_of_scope")
    if _agentic_policy_override(state):
        return state

    state["response"] = (
        "This doesn't appear to match anything in this system's known incident history "
        "or policy documentation. It may be outside the scope of this system, or may need "
        "manual review by a human."
    )
    print(f"[out_of_scope_node] Generated response:\n{state['response']}")
    return state




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

# =====================================================
# SECOND GRAPH — the agentic version
#
# Reuses search_node exactly as-is. agentic_decide_node picks one of
# six outcomes, then a conditional edge runs a different node — same
# branching mechanism as the deterministic graph, with LLM judgment
# instead of a similarity threshold.
# =====================================================
AGENTIC_OUTCOME_NODES = {
    "confident_answer": confident_answer_node,
    "answer_with_caveat": answer_with_caveat_node,
    "needs_clarification": needs_clarification_node,
    "likely_different_issue": likely_different_issue_node,
    "escalate_recommended": escalate_recommended_node,
    "out_of_scope": out_of_scope_node,
}

agentic_graph = StateGraph(TriageState)

agentic_graph.add_node("search", search_node)
agentic_graph.add_node("agentic_decide", agentic_decide_node)
for name, node_fn in AGENTIC_OUTCOME_NODES.items():
    agentic_graph.add_node(name, node_fn)

agentic_graph.set_entry_point("search")
agentic_graph.add_edge("search", "agentic_decide")
agentic_graph.add_conditional_edges(
    "agentic_decide",
    route_after_decision,
    {name: name for name in AGENTIC_OUTCOME_NODES},
)
for name in AGENTIC_OUTCOME_NODES:
    agentic_graph.add_edge(name, END)

agentic_app = agentic_graph.compile()


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
