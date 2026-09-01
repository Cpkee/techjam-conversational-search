# Shopping Copilot — Method Report

## Architecture Summary

A multi-stage pipeline, one Ollama call's worth of LLM reasoning per turn plus a
second for final re-ranking:

1. **Slot extraction & state** (`starter/slot_state.py`) — a lightweight regex
   heuristic seeds known slots from the raw message; `CATCMP` compares leaf category
   against the session's prior category to detect a hard conflict.
2. **Stage 2 combined LLM call** (`starter/stage2.py`) — one Qwen3 8B call per turn
   producing five things at once: a precise `rewrite`, a broader `expansion`, per-slot
   CARRYOVER/UPDATE/DELETE operations, an intent+confidence read, and a
   `no_preference_signal` flag.
3. **Dual-track retrieval + fusion** (`starter/fusion.py`) — keyword (BM25 over an
   in-memory SQLite FTS5 index) and dense (BGE embeddings) tracks, normalized against
   fixed reference ranges and combined via weighted fusion. A confidence `GATE` skips
   the dense track entirely on a high-confidence buying turn. `MAG`/`SPREAD` read the
   fused pool's magnitude and flatness; a `MAG` floor miss triggers one capped query
   revision (`REVISE`) before ranking proceeds regardless of outcome.
4. **RANKLLM** (`starter/rankllm.py`) — a second Qwen3 8B call re-ranks the fused
   candidate pool using semantic judgment on top of the fusion prior.
5. **CLARIFY** (`starter/clarify.py`) — picks which attribute (if any) to ask about
   next, by information gain (Shannon entropy) over the current candidate pool's
   catalog metadata, restricted to attributes the grading harness can actually route a
   customer answer back through.
6. **Adaptive orchestration** (`starter/policy.py`) — an epsilon-greedy bandit picks
   between two named clarify strategies (`clarify_aggressive`: up to 5 questions per
   session; `clarify_sparing`: at most 1) based on cross-session mean turns-consumed,
   gating CLARIFY via a `REALIGN` check (also suppresses clarification once fewer than
   2 turns remain in the session regardless of strategy).
7. **Long-term profile distillation** (`starter/profile_store.py`) — at the start of
   each `reset()`, distills the just-finished session's disclosed facts into an updated
   profile, stored locally and available to seed a future session for the same user.
   Recommendations are always returned every turn, independent of whether a clarifying
   question is also asked — the harness scores hits every turn regardless.

## Model Choice

- **LLM: Qwen3 8B, served locally via Ollama.** Runs fully offline once pulled — no
  API keys, no per-token cost, no network dependency for inference itself. Chosen
  because the competition provides no hosted model access or API credits.
- **Embeddings: `BAAI/bge-small-en-v1.5`** (via `transformers`), not the initially
  favored `blair-roberta-base` (pretrained specifically on the Amazon Reviews 2023
  dataset this catalog is drawn from — the better a priori choice). BLaIR's raw
  embeddings showed severe anisotropy (cosine-to-centroid mean 0.895 across the full
  50k catalog); even after whitening reduced that aggregate statistic to 0.006, 2-3
  shared "hub" items still polluted the top-5 results for unrelated queries. BGE
  showed zero hub overlap on the same test — chosen on empirical grounds despite being
  domain-generic rather than domain-trained.

## Measured Results

`recommended_technical_score` (0.50·HitRate@10 + 0.30·MRR + 0.20·Efficiency) at each
build step, from a stateless BM25 baseline through the current build:

| Step | Score | Note |
|---|---|---|
| Baseline (stateless BM25, no LLM) | 0.1067 | |
| + real `ask_attribute` | 0.4459 | single highest-leverage fix in the project |
| + turn accumulation | 0.5955 | |
| + Stage 2 combined LLM call | 0.6031 | |
| + dual-track retrieval + `RANKLLM` | 0.6209 | |
| + real `CLARIFY` (entropy-based) | 0.6383 | dev_subset (39 sessions), pre-calibration |
| + calibration (FUSION + MAG resolved) | **0.6717** | dev_subset (39 sessions); `GATE_CONFIDENCE_THRESHOLD` calibration still open |

**These are dev_subset (39-session) numbers, not the full 200-session public set.**
CLAUDE.md's own build discipline requires a full 200-session confirmation run before
any figure is treated as final — that run has not been executed as of this report. Run
`python3 evaluator/local_evaluator.py` against `data/public_set.jsonl` before citing a
final number.

Calibrated defaults, both resolved via paired-bootstrap comparison on
`data/dev_subset.jsonl` (see `starter/fusion.py` for full derivation notes):
`FUSION_KW_WEIGHT=0.65` / `FUSION_BROWSE_WEIGHT=0.35`, `MAG_FLOOR=0.5`.

## Known Limitations

- **`GATE_CONFIDENCE_THRESHOLD` (currently `0.75`) is an uncalibrated placeholder.**
  Calibration was started but not completed — see `starter/fusion.py`'s `# TUNE`
  marker. This is the one remaining open scoring-relevant parameter.
- **`CATCMP` (leaf-category conflict detection) is structurally inert on this
  dataset** — `intent_override` sessions never actually change category mid-session,
  only micro-tier facts (material, style, etc.) change even at the override turn. As a
  result `erasure_mode="erase"` vs. `"accumulate"` are empirically indistinguishable
  here (confirmed via paired-bootstrap, CI exactly `[0.0000, 0.0000]`); `accumulate` is
  the default.
- **Personalization (profile distillation, `starter/profile_store.py`) cannot move
  `recommended_technical_score`, by construction** — no session in this dataset shares
  a `user_id` with any other, so no session's distilled profile can ever be read back
  by a later session in the same evaluation run. It is exercised and validated via a
  standalone round-trip test (`profile_distillation_roundtrip.py`), not the official
  scoring path.
- **The adaptive clarify-strategy bandit (`starter/policy.py`) is similarly
  structurally unable to affect `recommended_technical_score`** for the same reason —
  its cross-session learning has nothing to learn from within a single evaluation run
  where every session is a different, unrelated user.
- **`MAG`/`REVISE` calibration is condition-dependent, not a fixed constant.** An
  earlier measurement (different `FUSION` weights, before an unrelated seeding bug was
  fixed) found `REVISE` net-negative; re-measured under the current calibrated
  `FUSION` weights, the opposite config (`MAG_FLOOR=0.5`, `REVISE` left on) won. Noted
  explicitly because it's a real example of parameters interacting rather than each
  having one universally-correct value — recalibrate if any of `FUSION`/`MAG`/`GATE`
  changes again.
- **The free-text slot-extraction heuristic (`starter/slot_state.py`) is a regex
  heuristic, not a model call** — cheap and fast, but will miss phrasing outside its
  patterns; Stage 2's LLM call is the primary source of truth for slot state, this is
  only a fallback seed.

## Cost & Latency Disclosure

- **No per-token API cost.** Both the LLM (Qwen3 8B) and the embedding model
  (`bge-small-en-v1.5`) run locally — cost is local compute time only.
- **Latency was not precisely instrumented** during development (no per-call timing
  capture was built). Qualitatively: each turn makes two local LLM calls (Stage 2
  analysis + RANKLLM re-rank) plus one embedding call when `GATE` doesn't skip it;
  on CPU-only inference this was noticeably slower than a hosted API would be. This is
  a genuine gap — measuring and reporting actual per-turn latency (e.g. mean/p95 turn
  latency from a `local_evaluator.py` run) should happen before final submission if
  latency is part of the scored disclosure.
- **`usage` (`prompt_tokens`/`completion_tokens`) is reported per turn**, summed across
  both LLM calls, read directly from Ollama's own `prompt_eval_count`/`eval_count`
  response fields — not estimated.

## Network & Offline Behavior

- **LLM inference requires no network access** once Ollama is running locally with
  `qwen3:8b` pulled (`ollama pull qwen3:8b`, one-time, requires network).
- **Embedding model download requires network access once**, on first use, to fetch
  `BAAI/bge-small-en-v1.5` (~130MB) from the Hugging Face Hub; subsequent runs use the
  local cache and require no network.
- **Graceful degradation is already built in, not just planned**: `starter/
  ollama_client.py`'s `_ollama_chat` catches connection/timeout/parse failures and
  returns `None`, which `starter/stage2.py` and `starter/rankllm.py` both handle by
  falling back to a default (unanalyzed rewrite/no slot changes, and the incoming
  fusion order respectively) rather than raising. The agent does not crash if Ollama is
  unreachable mid-run — it degrades to weaker (but still functional) behavior.

## Team Contributions

_Fill in before submitting — not populated here since this report was generated from
the working repository's build history, which does not track individual authorship._

## One Demonstrated Multi-Turn Session

See `demo_scenarios.py` in the working repository (not part of this submission bundle)
for a script that plays back real sessions from `public_set.jsonl` through the live
agent turn-by-turn. Run it and capture a transcript (or record a short screen capture)
for the final submission's required demonstrated session.
