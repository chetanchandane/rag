-- ─────────────────────────────────────────────────────────────────────────────
-- Read-only role for the ClinRAG text-to-SQL query path (governance).
--
-- Reference DDL. The runnable path is `python -m src.sql.setup_readonly_role`,
-- which applies the equivalent statements via psycopg (no psql needed) and
-- prints the DATABASE_URL_RO to add to your .env.
--
-- Run as the database OWNER. Replace <dbname> and the password before running
-- by hand. The app connects with THIS role only; ingestion uses the owner role.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE ROLE clinrag_readonly LOGIN PASSWORD 'CHANGE_ME';

GRANT CONNECT ON DATABASE <dbname>        TO clinrag_readonly;
GRANT USAGE   ON SCHEMA public            TO clinrag_readonly;
GRANT SELECT  ON ALL TABLES IN SCHEMA public TO clinrag_readonly;

-- Future tables are readable too, without re-granting.
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO clinrag_readonly;

-- Defense in depth: never allow writes, even if a future GRANT slips through.
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public FROM clinrag_readonly;

-- DB-enforced guardrails on the role itself:
ALTER ROLE clinrag_readonly SET statement_timeout = 5000;             -- kill slow queries (ms)
ALTER ROLE clinrag_readonly SET default_transaction_read_only = on;  -- reject writes at the session level
