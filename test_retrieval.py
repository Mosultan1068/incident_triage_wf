"""
Retrieval test script — Phase 3

Proves both similarity search functions work correctly on their own,
with plain, deterministic Python — before any LangGraph agentic logic
is added on top. Same discipline as the coffee project's retrieve.py.
"""

import os
from dotenv import load_dotenv
import psycopg2
from openai import OpenAI

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

openai_client = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def search_incidents(description: str, match_count: int = 3):
    """Find past incidents similar to a new incident description."""
    embedding = get_embedding(description)
    embedding_str = str(embedding)

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM match_incidents(%s, %s);",
        (embedding_str, match_count)
    )
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return [dict(zip(columns, row)) for row in rows]


def search_runbooks(description: str, match_count: int = 3):
    """Find relevant runbook/policy content for a new incident description."""
    embedding = get_embedding(description)
    embedding_str = str(embedding)

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM match_document_chunks(%s, %s);",
        (embedding_str, match_count)
    )
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return [dict(zip(columns, row)) for row in rows]


def run_test(description: str):
    print(f"\n{'='*60}")
    print(f"New incident: {description}")
    print('='*60)

    print("\n--- Similar past incidents ---")
    incidents = search_incidents(description)
    if not incidents:
        print("No similar incidents found.")
    for inc in incidents:
        print(f"  [{inc['similarity']:.3f}] {inc['title']} "
              f"(severity: {inc['severity']}, status: {inc['status']})")
        if inc['resolution_notes']:
            print(f"      Resolution: {inc['resolution_notes']}")

    print("\n--- Relevant runbook content ---")
    docs = search_runbooks(description)
    if not docs:
        print("No relevant runbook content found.")
    for doc in docs:
        print(f"  [{doc['similarity']:.3f}] {doc['content'][:120]}...")


if __name__ == "__main__":
    # Test 1: something that should closely match an existing incident
    run_test("Database queries are timing out for some users during busy periods")

    # Test 2: something that should trigger the escalation policy content
    run_test("Customers are getting errors when trying to pay at checkout")

    # Test 3: something genuinely unrelated, to see how weak matches look
    run_test("The office coffee machine is broken")

    # Test 4: a plausible-sounding IT question, but genuinely outside
    # anything covered in our runbooks or past incidents — testing the
    # "no real answer exists" case specifically.
    run_test("Users are reporting the mobile app crashes when uploading a profile photo on Android 14")
    # my own test
    run_test("remote control not working when changing channals")
