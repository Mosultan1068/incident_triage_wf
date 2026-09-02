"""
Incident Triage Agent — Gradio front end

Calls the LangGraph workflow directly (same deterministic entry-point
pattern as the coffee project's Gradio app). Organized into tabs:
- Triage: the main interaction, with the agent's decision path,
  score, considered incidents/policy, and final response.
- How it works: a legend explaining the four decision types and the
  policy override, so a viewer understands what they're about to see.
- History: a running log of every incident tried this session.
"""

import gradio as gr
import plotly.graph_objects as go
from triage_graph import app as triage_app, ALWAYS_ESCALATE_CATEGORIES, STRONG_MATCH_THRESHOLD, WEAK_MATCH_THRESHOLD

DECISION_LABELS = {
    "strong_match": "✅ Strong match — confident response",
    "weak_match": "🤔 Weak match — clarification requested",
    "no_match": "❓ No match — outside system scope",
    "escalated_by_policy": "🚨 Escalated — protected category policy override",
}

# Session-level history, feeding the History tab.
session_history = []


def build_score_chart(incidents, policies):
    labels = [f"Incident: {i['title'][:25]}" for i in incidents[:3]]
    labels += [f"Policy: {p['content'][:25]}..." for p in policies[:3]]
    scores = [i["similarity"] for i in incidents[:3]] + [p["similarity"] for p in policies[:3]]
    colors = ["#1D9E75" if s >= STRONG_MATCH_THRESHOLD else
              "#D9A441" if s >= WEAK_MATCH_THRESHOLD else
              "#D85A30" for s in scores]

    fig = go.Figure()

    if labels:
        fig.add_trace(go.Bar(x=labels, y=scores, marker_color=colors, showlegend=False))

    # Threshold reference lines, tying directly back to the three-tier system
    fig.add_hline(y=STRONG_MATCH_THRESHOLD, line_dash="dash", line_color="#1D9E75",
                  annotation_text="Strong match threshold")
    fig.add_hline(y=WEAK_MATCH_THRESHOLD, line_dash="dash", line_color="#D9A441",
                  annotation_text="Weak match threshold")

    fig.update_layout(
        title="Match confidence for this incident",
        yaxis_title="Similarity score",
        yaxis_range=[0, 1],
        margin=dict(t=60, b=100),
    )
    return fig


def run_triage(description: str):
    if not description.strip():
        empty = "Please describe an incident first."
        return empty, "", "", [], [], "", session_history_rows(), empty, go.Figure()

    result = triage_app.invoke({
        "incident_description": description,
        "similar_incidents": [],
        "policy_matches": [],
        "best_score": 0.0,
        "decision": "",
        "response": "",
        "path": []
    })

    decision_label = DECISION_LABELS.get(result["decision"], result["decision"])
    score_text = f"**Best match confidence:** {result['best_score']:.3f}"
    path_text = f"🔀 **Path taken:** `{' → '.join(result['path'])}`"

    # Explicit override note — shown only when the policy actually fired,
    # so it's visible WHY an otherwise strong/weak match got escalated.
    override_note = ""
    if result["decision"] == "escalated_by_policy":
        categories = ", ".join(ALWAYS_ESCALATE_CATEGORIES)
        override_note = (
            f"⚠️ **Policy override triggered** — this description matched a "
            f"protected category ({categories}). Auto-response was bypassed "
            f"regardless of the similarity score found."
        )

    incidents_table = [
        [f"{inc['similarity']:.3f}", inc["title"], inc["status"]]
        for inc in result["similar_incidents"][:3]
    ]
    policy_table = [
        [f"{pol['similarity']:.3f}", pol["content"][:80] + "..."]
        for pol in result["policy_matches"][:3]
    ]

    score_chart = build_score_chart(result["similar_incidents"], result["policy_matches"])

    session_history.append({
        "description": description,
        "decision": result["decision"],
        "best_score": result["best_score"],
    })

    return (
        decision_label, score_text, path_text, incidents_table, policy_table,
        result["response"], session_history_rows(), override_note, score_chart
    )


def session_history_rows():
    return [
        [h["description"][:50], DECISION_LABELS.get(h["decision"], h["decision"]), f"{h['best_score']:.3f}"]
        for h in session_history
    ]


EXAMPLE_INCIDENTS = [
    "Database queries are timing out for some users during busy periods",
    "Customers are getting errors when trying to pay at checkout",
    "The office coffee machine is broken",
]

with gr.Blocks(title="Incident Triage Agent") as demo:
    gr.Markdown("# 🤖 IT Incident Triage Agent")
    gr.Markdown(
        "An agentic workflow (LangGraph) that searches past incidents and policy "
        "documentation, then decides how to respond based on what it finds."
    )

    with gr.Tabs():
        with gr.Tab("Triage"):
            description_input = gr.Textbox(
                label="Describe the incident",
                placeholder="e.g. Users are reporting slow response times on the dashboard",
                lines=2
            )

            with gr.Row():
                for example in EXAMPLE_INCIDENTS:
                    gr.Button(example, size="sm").click(
                        fn=lambda e=example: e, outputs=description_input
                    )

            triage_button = gr.Button("Run Triage", variant="primary")

            decision_output = gr.Markdown()
            score_output = gr.Markdown()
            path_output = gr.Markdown()
            override_output = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Similar past incidents")
                    incidents_table = gr.Dataframe(
                        headers=["Score", "Title", "Status"],
                        col_count=(3, "fixed"),
                        wrap=True
                    )
                    gr.Markdown("### Relevant policy content")
                    policy_table = gr.Dataframe(
                        headers=["Score", "Content"],
                        col_count=(2, "fixed"),
                        wrap=True
                    )
                with gr.Column():
                    chart_output = gr.Plot(label="Match confidence")

            gr.Markdown("### Response")
            response_output = gr.Markdown()

        with gr.Tab("How it works"):
            gr.Markdown("""
## The four decision types

**✅ Strong match** — a similar past incident or policy content was found with high confidence (score ≥ 0.5). The agent generates a grounded response citing that specific evidence.

**🤔 Weak match** — something loosely related was found (score between 0.22 and 0.5), but not confidently enough to answer outright. The agent asks a clarifying question instead.

**❓ No match** — nothing meaningfully similar was found (score below 0.22). The agent states this is likely outside its knowledge, rather than guessing.

**🚨 Escalated by policy** — regardless of the similarity score, if the incident description touches a protected category (checkout, payment, order), the agent overrides its own confidence and escalates immediately. This rule cannot be bypassed by any other decision path.

## Why thresholds, not keywords

The similarity scores above come from comparing meaning, not exact words — a paraphrased incident description can still match a differently-worded past incident, because both are converted to embeddings and compared by closeness in meaning, not text overlap.

## Why the policy override exists

Some categories carry consequences too significant for automated confidence alone — a checkout/payment issue should always reach a human, even if the system feels confident about a similar-sounding past incident.
""")

        with gr.Tab("History"):
            gr.Markdown("Every incident tried this session, in order.")
            history_table = gr.Dataframe(
                headers=["Incident", "Decision", "Best Score"],
                col_count=(3, "fixed"),
                wrap=True
            )

    triage_button.click(
        fn=run_triage,
        inputs=description_input,
        outputs=[decision_output, score_output, path_output, incidents_table, policy_table,
                 response_output, history_table, override_output, chart_output]
    )
    description_input.submit(
        fn=run_triage,
        inputs=description_input,
        outputs=[decision_output, score_output, path_output, incidents_table, policy_table,
                 response_output, history_table, override_output, chart_output]
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True, server_name="0.0.0.0", server_port=7862)
