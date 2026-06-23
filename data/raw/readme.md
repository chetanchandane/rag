# Raw source documents

This folder holds the **source corpus** for the Clinical RAG system: FDA and ICH
clinical-trial guidance documents (PDFs). The ingestion pipeline reads everything
in here, chunks it, embeds it, and upserts it into Qdrant.

> [!IMPORTANT]
> **Do not chunk this `readme.md`.** It is documentation, not corpus content —
> including it would pollute the vector store with metadata text.
> The indexer's `--dir` mode already globs `*.pdf` only
> (see [`src/ingestion/indexer.py`](../../src/ingestion/indexer.py)), so this file
> is excluded automatically. If you change the loader to ingest other file types,
> make sure `readme.md` (and any non-corpus files) stay excluded.

## Where to download the files

All documents are publicly available from the FDA and ICH:

- **FDA — ICH guidance documents:**
  <https://www.fda.gov/science-research/clinical-trials-and-human-subject-protection/ich-guidance-documents>
- **FDA — clinical trials guidance documents:**
  <https://www.fda.gov/science-research/clinical-trials-and-human-subject-protection/clinical-trials-guidance-documents>
- **ICH E6(R3) (direct PDF):**
  <https://database.ich.org/sites/default/files/ICH_E6(R3)_Step4_FinalGuideline_2025_0106.pdf>

