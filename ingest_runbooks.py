"""
Runbook ingestion script — the glue between raw_documents (staging)
and documents/document_chunks (the RAG tables).

Same shape as the coffee project's ingest.py: read unprocessed raw
content, chunk it, embed each chunk, write it into the processed
tables, mark the raw row as processed. This version uses psycopg2 and
raw SQL instead of supabase-py.
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


def chunk_text(text: str, max_words: int = 80) -> list[str]:
    """Simple sentence-grouping chunking, same approach as coffee's ingest.py."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_word_count + word_count > max_words and current_chunk:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = []
            current_word_count = 0
        current_chunk.append(sentence)
        current_word_count += word_count

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks


def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def ingest_runbooks():
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    # Step 1: fetch unprocessed raw_documents rows
    cursor.execute(
        "SELECT id, raw_content, source_system FROM raw_documents WHERE processed = false;"
    )
    raw_rows = cursor.fetchall()

    if not raw_rows:
        print("No unprocessed raw_documents found. Nothing to ingest.")
        cursor.close()
        conn.close()
        return

    print(f"Found {len(raw_rows)} unprocessed row(s) to ingest.\n")

    for raw_id, raw_content, source_system in raw_rows:
        print(f"Processing raw_documents row {raw_id}...")

        # Step 2: create the parent `documents` row
        title_preview = " ".join(raw_content.split()[:6]) + "..."
        cursor.execute(
            "INSERT INTO documents (title, source_url) VALUES (%s, %s) RETURNING id;",
            (title_preview, None)
        )
        document_id = cursor.fetchone()[0]

        # Step 3: chunk the raw content
        chunks = chunk_text(raw_content)
        print(f"  Split into {len(chunks)} chunk(s).")

        # Step 4: embed each chunk and insert it
        for index, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            embedding_str = str(embedding)

            cursor.execute(
                """
                INSERT INTO document_chunks (document_id, chunk_index, content, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s);
                """,
                (document_id, index, chunk, embedding_str, '{"source_system": "%s"}' % source_system)
            )

        # Step 5: mark the raw row as processed
        cursor.execute(
            "UPDATE raw_documents SET processed = true WHERE id = %s;",
            (raw_id,)
        )
        print(f"  Marked raw_documents row {raw_id} as processed.\n")

    conn.commit()  # commit once at the end, after all rows processed successfully
    cursor.close()
    conn.close()

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_runbooks()
