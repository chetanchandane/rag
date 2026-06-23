# DEPRECATED — superseded by the split modules below.
# This file is kept only for backward compatibility.
# The pipeline is now orchestrated in src/api/main.py::run_rag()
# using the following modules:
#
#   src/retrieval/search.py       → Searcher.search()
#   src/retrieval/reranker.py     → Reranker.rerank()
#   src/generation/llm_client.py  → ClaudeClient.generate()
