# Shopping Copilot — Submission

A multi-turn conversational shopping agent for the TechJam Conversational E-Commerce
Search Challenge. See `REPORT.md` for method, model choice, cost/latency disclosure,
and known limitations.

## Setup

Python 3.9 (tested version; not pinned strictly, but this is what was verified).

```bash
pip install -r requirements.txt
```

**Local LLM (required, one-time setup):**

```bash
# Install Ollama: https://ollama.com
ollama pull qwen3:8b
ollama serve   # must be running on localhost:11434 before calling Agent.respond()
```

**Embedding model:** `BAAI/bge-small-en-v1.5` is loaded via `transformers` and downloads
automatically from the Hugging Face Hub on first use (~130MB) unless already cached
locally (`~/.cache/huggingface`). All subsequent runs use the local cache — no network
needed for embeddings after the first run. See `REPORT.md`'s Network & Offline Behavior
section for the full disclosure.

## Reproduction

```python
from agent import Agent

agent = Agent(catalog_path="path/to/catalog.jsonl")
agent.reset(session_id="s1", user_profile={...})
response = agent.respond(session_id="s1", user_message="I'm looking for running shoes.", turn=1, top_k=10)
```

`Agent(catalog_path=...)` builds an in-memory SQLite FTS5 index over the catalog and
computes (or loads a cached copy of) BGE embeddings for all products on construction —
this is a one-time cost per catalog file; embeddings are cached to
`<catalog_dir>/<catalog_stem>.bge_embeddings.npy` (+ a matching `_ids.json`) next to
the catalog path given, and reused on subsequent runs against the same catalog file.

`Agent()` is designed to be instantiated **once** for a whole evaluation run — `reset()`
is called per-session, but some internal state (a long-term per-user profile store, and
a small adaptive-orchestration policy — see `REPORT.md`) intentionally persists across
sessions within one `Agent()` instance's lifetime.

## Package Layout

```
submission/
  agent.py            <- entry point, re-exports Agent
  requirements.txt
  README.md            (this file)
  REPORT.md            <- method, model choice, cost/latency, limitations
  starter/
    agent.py            orchestrator: Agent.__init__/reset/respond
    text_utils.py        tokenization helpers
    slot_schema.py        slot vocabulary + pydantic schema
    catalog_index.py      catalog-side metadata extraction
    slot_state.py          free-text slot extraction + state application
    ollama_client.py       shared Ollama HTTP call
    stage2.py               context formatting + the combined LLM call
    fusion.py                keyword+dense retrieval fusion, magnitude/spread signals
    rankllm.py                final semantic re-rank
    clarify.py                 clarifying-question attribute selection
    policy.py                   adaptive clarify-strategy bandit
    profile_store.py             long-term profile distillation
    embeddings.py                 embedding model plumbing
```
