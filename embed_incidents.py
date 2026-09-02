"""
Embedding generation script — incidents

Reads every incident missing an embedding, generates one via OpenAI
(combining title + description, same as the sleep_issues approach),
and writes it back using raw SQL through psycopg2.
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


def embed_incidents():
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    # Fetch every incident that doesn't have an embedding yet
    cursor.execute("SELECT id, title, description FROM incidents WHERE embedding IS NULL;")
    rows = cursor.fetchall()

    if not rows:
        print("No incidents need embedding. Nothing to do.")
        cursor.close()
        conn.close()
        return

    print(f"Found {len(rows)} incident(s) needing embeddings.\n")

    for incident_id, title, description in rows:
        text_to_embed = f"{title}. {description}"

        print(f"Embedding: {title}...")
        embedding = get_embedding(text_to_embed)

        # pgvector accepts a string like '[0.1,0.2,...]' for a vector value
        embedding_str = str(embedding)

        cursor.execute(
            "UPDATE incidents SET embedding = %s WHERE id = %s;",
            (embedding_str, incident_id)
        )

    conn.commit()  # writes are not saved until explicitly committed
    cursor.close()
    conn.close()

    print(f"\nDone. {len(rows)} incident(s) embedded.")


if __name__ == "__main__":
    embed_incidents()
