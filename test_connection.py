"""
Connection test — proves the Python <-> Neon plumbing works, before
any embedding or agentic logic is added.

Unlike Supabase (accessed via supabase-py's table/RPC methods), Neon
is plain Postgres — connected to directly using psycopg2, with raw
SQL queries.
"""

import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

# Connect directly to the Postgres database using the connection string.
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# Read back everything currently in the incidents table.
cursor.execute("SELECT id, title, category, severity, status FROM incidents;")
rows = cursor.fetchall()

print(f"Connected successfully. Found {len(rows)} row(s) in incidents:\n")

for row in rows:
    incident_id, title, category, severity, status = row
    print(f"- {title}")
    print(f"  category: {category} | severity: {severity} | status: {status}")
    print()

cursor.close()
conn.close()
