-- ─────────────────────────────────────────────────────────────────────────────
-- ClinRAG relational backend — Day 1 schema (option 2)
--
-- Two independent tables, NO cross-feed join. Each text-to-SQL query stays
-- single-table, which keeps generation accurate and avoids openFDA's fragile
-- join keys. A curated bridge table can be added deliberately later.
--
-- This file is idempotent: safe to run repeatedly.
-- ─────────────────────────────────────────────────────────────────────────────

-- ── approvals ────────────────────────────────────────────────────────────────
-- Source: openFDA drugsfda (https://api.fda.gov/drug/drugsfda.json)
-- Grain:  one row per product within an application (application_number, product_number)
CREATE TABLE IF NOT EXISTS approvals (
    application_number  TEXT NOT NULL,
    product_number      TEXT NOT NULL,
    brand_name          TEXT,
    active_ingredient   TEXT,
    dosage_form         TEXT,
    route               TEXT,
    marketing_status    TEXT,          -- Prescription / OTC / Discontinued / None
    sponsor             TEXT,
    approval_date       DATE,          -- original (ORIG/AP) submission approval date
    PRIMARY KEY (application_number, product_number)
);

CREATE INDEX IF NOT EXISTS idx_approvals_brand_name       ON approvals (brand_name);
CREATE INDEX IF NOT EXISTS idx_approvals_marketing_status ON approvals (marketing_status);
CREATE INDEX IF NOT EXISTS idx_approvals_approval_date    ON approvals (approval_date);


-- ── recalls ──────────────────────────────────────────────────────────────────
-- Source: openFDA drug/enforcement (https://api.fda.gov/drug/enforcement.json)
-- Grain:  one row per recall (recall_number)
CREATE TABLE IF NOT EXISTS recalls (
    recall_number           TEXT PRIMARY KEY,
    event_id                TEXT,
    product_description     TEXT,          -- free-text product/drug name + details
    brand_name              TEXT,          -- openfda.brand_name[0] when present
    generic_name            TEXT,          -- openfda.generic_name[0] when present
    recalling_firm          TEXT,
    reason_for_recall       TEXT,
    classification          TEXT,          -- Class I / II / III (severity)
    status                  TEXT,          -- Ongoing / Completed / Terminated
    voluntary_mandated      TEXT,
    distribution_pattern    TEXT,
    city                    TEXT,
    state                   TEXT,
    country                 TEXT,
    recall_initiation_date  DATE,
    report_date             DATE,
    termination_date        DATE
);

CREATE INDEX IF NOT EXISTS idx_recalls_classification ON recalls (classification);
CREATE INDEX IF NOT EXISTS idx_recalls_status         ON recalls (status);
CREATE INDEX IF NOT EXISTS idx_recalls_init_date      ON recalls (recall_initiation_date);
CREATE INDEX IF NOT EXISTS idx_recalls_brand_name     ON recalls (brand_name);
