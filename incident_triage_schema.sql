-- =====================================================
-- IT Incident Triage Agent — Full Schema Setup
-- Run this in Neon's SQL Editor
--
-- Two data types, same architecture pattern as previous projects:
--   1. incidents — structured, with its OWN embedding column, enabling
--      similarity search directly across past incidents.
--   2. RAG trio — raw_documents -> documents/document_chunks, for
--      policy/runbook documentation grounding the agent's decisions.
-- =====================================================

create extension if not exists vector;

-- =====================================================
-- PART 1: Incidents — structured, with direct similarity search
-- =====================================================

create table incidents (
    id uuid primary key default gen_random_uuid(),
    title text not null,
    description text not null,
    category text,                    -- e.g. 'network', 'database', 'application'
    severity text,                    -- e.g. 'low', 'medium', 'high', 'critical'
    status text not null default 'open' check (status in ('open', 'resolved', 'escalated')),
    created_at timestamptz not null default now(),
    resolved_at timestamptz,
    resolution_notes text,
    embedding vector(1536)            -- populated by an embedding script, not by hand
);

create or replace function match_incidents (
    query_embedding vector(1536),
    match_count int default 3
)
returns table (
    id uuid,
    title text,
    description text,
    category text,
    severity text,
    status text,
    resolution_notes text,
    similarity float
)
language sql stable
as $$
    select
        incidents.id,
        incidents.title,
        incidents.description,
        incidents.category,
        incidents.severity,
        incidents.status,
        incidents.resolution_notes,
        1 - (incidents.embedding <=> query_embedding) as similarity
    from incidents
    where incidents.embedding is not null
    order by incidents.embedding <=> query_embedding
    limit match_count;
$$;

-- =====================================================
-- PART 2: RAG trio — policy / runbook documentation
-- =====================================================

create table raw_documents (
    id uuid primary key default gen_random_uuid(),
    raw_content text not null,
    source_system text,
    received_at timestamptz not null default now(),
    processed boolean not null default false
);

create table documents (
    id uuid primary key default gen_random_uuid(),
    title text not null,
    source_url text,
    created_at timestamptz not null default now()
);

create table document_chunks (
    id uuid primary key default gen_random_uuid(),
    document_id uuid not null references documents(id) on delete cascade,
    chunk_index int not null,
    content text not null,
    embedding vector(1536),
    metadata jsonb default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create or replace function match_document_chunks (
    query_embedding vector(1536),
    match_count int default 3
)
returns table (
    id uuid,
    content text,
    similarity float,
    document_id uuid
)
language sql stable
as $$
    select
        document_chunks.id,
        document_chunks.content,
        1 - (document_chunks.embedding <=> query_embedding) as similarity,
        document_chunks.document_id
    from document_chunks
    where document_chunks.embedding is not null
    order by document_chunks.embedding <=> query_embedding
    limit match_count;
$$;

-- No ivfflat index on either similarity function — same lesson carried
-- forward from the coffee project. Add one only once a table holds
-- thousands+ of rows, and tune probes/lists deliberately at that point.

-- =====================================================
-- Row Level Security — permissive policies, same reasoning as
-- previous projects (proof of concept, no real user accounts).
-- =====================================================

alter table incidents enable row level security;
alter table raw_documents enable row level security;
alter table documents enable row level security;
alter table document_chunks enable row level security;

create policy "Allow all access" on incidents for all using (true) with check (true);
create policy "Allow all access" on raw_documents for all using (true) with check (true);
create policy "Allow all access" on documents for all using (true) with check (true);
create policy "Allow all access" on document_chunks for all using (true) with check (true);

-- =====================================================
-- Sample incident data — deliberately varied wording, mirroring
-- real incident descriptions, to give similarity search something
-- genuine to work with once embedded.
-- =====================================================

insert into incidents (title, description, category, severity, status, resolution_notes) values
('Database connection timeouts', 'Application reporting intermittent timeouts when connecting to the primary database. Started around 09:00, affecting roughly 5% of requests.', 'database', 'medium', 'resolved', 'Connection pool size was too small for peak load. Increased pool size from 20 to 50, timeouts stopped.'),
('Users cannot log in', 'Multiple users reporting they cannot authenticate — login page loads but submitting credentials just spins and eventually errors.', 'application', 'high', 'resolved', 'Auth service was pointing at an expired certificate. Renewed the certificate and restarted the service.'),
('Slow page load times', 'Customer-facing dashboard taking 8-10 seconds to load, previously under 2 seconds. No recent deployments.', 'application', 'medium', 'resolved', 'A missing index on a frequently-queried table was causing full table scans. Added the index, load time back to under 2 seconds.'),
('Network packet loss between data centres', 'Intermittent packet loss observed on the link between primary and DR data centres, causing replication lag.', 'network', 'high', 'resolved', 'Faulty transceiver on one end of the link. Replaced hardware, packet loss resolved.'),
('Disk space critical on production server', 'Alert fired for disk usage above 95% on the main application server.', 'infrastructure', 'critical', 'resolved', 'Log rotation had silently stopped working weeks ago. Cleared old logs and fixed the rotation config.'),
('Email notifications not sending', 'Users report they are not receiving password reset or notification emails, no errors visible in the app logs.', 'application', 'medium', 'open', null),
('Scheduled batch job failed overnight', 'The nightly data reconciliation job did not complete, leaving downstream reports out of date this morning.', 'database', 'high', 'open', null),
('Intermittent 500 errors on checkout', 'A small percentage of checkout attempts are failing with a generic server error, no clear pattern yet.', 'application', 'high', 'escalated', null);

-- =====================================================
-- Sample policy/runbook content — inserted into raw_documents,
-- to be embedded by a Python ingestion script next.
-- =====================================================

insert into raw_documents (raw_content, source_system, processed) values
('When database connection timeouts occur, first check current connection pool utilisation against configured maximum. If consistently near the limit during peak hours, increasing pool size is the standard first remediation before investigating query performance.', 'runbook', false),
('Authentication failures affecting multiple users simultaneously should be treated as high severity by default, since they block all user access. Check certificate expiry and auth service health before investigating individual account issues.', 'runbook', false),
('Any incident affecting checkout, payment, or order processing should be escalated immediately regardless of apparent severity, given direct revenue impact. Do not attempt silent auto-resolution for these categories.', 'escalation_policy', false),
('Disk space alerts above 90% should be treated as high severity; above 95% as critical. Common root causes include stopped log rotation, orphaned temp files, and unbounded cache growth.', 'runbook', false);
