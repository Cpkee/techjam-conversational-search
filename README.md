# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Local Setup

`starter/agent.py` in this repo is no longer the bare BM25 stub — it depends on a
locally-served LLM (via [Ollama](https://ollama.com)) and a Hugging Face embedding
model. Set both up before running the evaluator.

### 1. Python dependencies

Python 3.10+ is recommended.

```bash
pip install pydantic numpy torch transformers
```

Everything else (`sqlite3`, `urllib`, etc.) is standard library — no vector DB or
LLM-orchestration framework required.

### 2. Local LLM via Ollama

Install Ollama, then pull and serve the model the agent calls (`qwen3:8b`, see
`starter/agent.py`'s `OLLAMA_MODEL`):

```bash
brew install ollama        # or download from https://ollama.com/download
ollama serve                # starts the local server on http://localhost:11434
ollama pull qwen3:8b        # in a second terminal, one-time download
```

`ollama serve` must stay running (or be running as a background service) for the
whole time you run the evaluator — `agent.py` calls `http://localhost:11434/api/chat`
directly and silently falls back to a no-op response for any turn where that call
fails, so a stopped server won't crash a run, it will just quietly tank your score.
Verify it's up before a real run:

```bash
curl -s http://localhost:11434/api/tags   # should list qwen3:8b
```

### 3. Embedding model (dense retrieval / BROWSE)

The dense-retrieval track (`starter/embeddings.py`) uses `BAAI/bge-small-en-v1.5` by
default (`blair-roberta-base` is available as a documented, non-default alternative —
see the module docstring and `CLAUDE.md` 2.5 for why BGE won out). The first run
downloads the model from Hugging Face automatically and then embeds the full 50,000-
product catalog, caching the result to `data/catalog.bge_embeddings.npy` /
`data/catalog.bge_embeddings_ids.json`; subsequent runs reuse that cache and skip
re-embedding. The first run can take a while — let it finish uninterrupted.

If an Apple Silicon Mac is available, embeddings automatically run on `mps`;
otherwise they fall back to CPU.

## Run the Starter

```bash
python3 -m evaluator.local_evaluator
```

Run from the repo root (it imports `evaluator` and `starter` as packages). Edit
`starter/agent.py` to implement your system. Do not edit the evaluator or public
labels when reporting your local score. The command writes per-session results and
aggregate metrics to `results.json`.

The included weak BM25 starter (i.e. `agent.py` before any local edits) scores Hit
Rate@10 `0.125`, MRR `0.068034`, and MTTC `9.81` on the released public set. See
`docs/baseline_results.json`.

To run against the smaller 39-session dev subset instead of the full 200-session
public set (useful for faster local iteration given local LLM inference is the
bottleneck, not the evaluator):

```bash
python3 -m evaluator.local_evaluator --dataset data/dev_subset.jsonl --output dev_results.json
```

### Comparison / calibration scripts

Two paired-bootstrap comparison scripts (used to settle the `erasure_mode` question
documented in `CLAUDE.md`) are also runnable directly, and follow the same Ollama/
embedding-model setup above:

```bash
python3 erase_vs_accumulate_comparison.py          # erase vs. accumulate
python3 erase_accumulate_gated_comparison.py        # + the "gated" DELETE-validation mode
```

### Tests

```bash
python3 -m unittest discover
```

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
