-- NoteCrate sandbox schema (mirrors the production shape on Supabase)
-- Run once:  psql "$DATABASE_URL" -f schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS chunks;

CREATE TABLE chunks (
    id               bigserial PRIMARY KEY,
    text             text NOT NULL,
    embedding        vector(768) NOT NULL,   -- nomic-embed-text-v1.5

    -- member-model fields: promoted to real columns because they are
    -- query filters, not just payload
    source_id        text NOT NULL,          -- de-indexing key
    permission_scope text NOT NULL,          -- hard filter, pre-rank
    origin           text,
    author           text,
    created_at       timestamptz,            -- content's own timestamp

    -- provenance / pipeline metadata
    source           text,
    source_type      text,
    section          text,
    subsection       text,
    chunk_strategy   text,
    owner            text,
    ingested         date,
    n_tokens         int,

    -- keyword side for Week 4 hybrid search: maintained automatically
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

-- vector similarity (cosine)
CREATE INDEX chunks_embedding_hnsw ON chunks
    USING hnsw (embedding vector_cosine_ops);

-- the two filter paths the product depends on
CREATE INDEX chunks_source_id_idx        ON chunks (source_id);          -- revocation: DELETE WHERE source_id = ...
CREATE INDEX chunks_permission_scope_idx ON chunks (permission_scope);   -- permission filter before rank

-- keyword retrieval (Week 4)
CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);
