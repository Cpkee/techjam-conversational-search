# CLAUDE.md — Shopping Copilot Build Instructions

This file is the accumulated result of extensive research against this project's real
files (`local_evaluator.py`, `catalog.jsonl`, `public_set.jsonl`) and real competition
rules — not assumptions. Follow it in order. **Measure after every step by running
`local_evaluator.py` against all 200 public sessions before starting the next step.**
Do not skip the measurement step, even when a change seems obviously good — one single
one-line fix in this project changed the technical score by 4.2x, and a change that
seemed obviously bad turned out to matter less than expected. Intuition here has
already been wrong before. Trust the number.

---

## PART 1 — INTUITION (read this before touching architecture)

Picture a salesperson in a 50,000-item store, texting back and forth with a customer.
Every message: (1) re-read the whole conversation so far, dropping parts that don't
matter anymore, (2) in one AI call, produce five things at once — a clean rewrite of
the request, a looser/broader guess in case they're vague, an updated fact checklist,
a read on how serious they sound, and a flag for "they just said they don't care about
something," (3) if they totally changed what they want, wipe the checklist; if they
just added a detail, update only that line, (4) search two ways — exact-match and
meaning-based — and blend the results, (5) if results look bad, try rephrasing once,
(6) always show the current best guesses, every single turn, no exceptions, (7) only
ask a clarifying question if there's something real left worth asking about that
hasn't been asked or answered or declined already, (8) get scored on: did the right
item show up in the top 10, how near the top, and how many messages it took.

---

## PART 2 — ARCHITECTURE (confirmed facts, not assumptions)

### 2.1 The Agent interface contract (from reading `local_evaluator.py` directly)

```python
class Agent:
    def __init__(self, catalog_path: str, ...): ...
    def reset(self, session_id: str, user_profile: dict) -> None: ...
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": str,
            "ask_attribute": str | None,   # one of ALLOWED_ATTRIBUTES below, or None
            "recommendations": list[dict], # [{"parent_asin": "..."}], up to top_k
            "usage": {"prompt_tokens": int, "completion_tokens": int},  # optional
        }
```

**Confirmed, not assumed:**
- `MAX_TURNS = 10`, `TOP_K = 10`.
- The evaluator checks `recommendations` for a hit **every turn**, regardless of
  whether `ask_attribute` was also set. **NEVER withhold recommendations because
  you're also asking a question — this is proven, not theoretical: adding ANY
  non-null `ask_attribute` (even a dumb rotating one, no real logic) took the overall
  technical score from 0.107 to 0.446 on the real starter baseline, purely because it
  stopped `customer_reply()` from returning a content-free nag message every turn
  after turn 1.**
- `ALLOWED_ATTRIBUTES = {"category","material","color","size","style","brand","budget","feature","use_case","other"}`.
  This is the *complete, closed* vocabulary for `ask_attribute`. Anything else gets
  silently coerced to `"other"` (a wildcard — matches any undisclosed constraint, but
  loses targeting precision). **`"department"` is NOT in this set** —
  `classify_constraint()` folds the literal word "department" into `"style"`. Use
  `department` freely as a *filter* field (it's reliable, see 2.3), never as an
  `ask_attribute` value.
- `intent_override` sessions: hits are gated off (`override_applied=False`) until the
  exact turn the override message is delivered (turn 3 or 4, randomly chosen). No
  agent, however good, can score a hit before that turn on this scenario type. This is
  a structural floor, not a bug to fix.
- Repeatedly asking about an attribute that's already been asked (answered or not)
  wastes a turn for nothing — `customer_reply()` either returns nothing new or the
  literal string `"I don't have an additional preference for X"`. Track what's been
  asked; never re-ask it.
- The `boundary` scenario's first-ever `ask_attribute` call gets a special
  `"I don't have a preference for X; please use your judgment"` response
  (`boundary_used` flips True after that). Treat this as a strong signal to stop
  probing *that specific attribute* — not a ban on asking about anything else, since
  every ask after the first goes through completely normal matching and can still
  surface a real constraint.
- **`respond()` can never know whether a hit occurred** — that check happens entirely
  inside `evaluate()`, never reported back. The only place cross-session bookkeeping
  (e.g. "how many turns did the last session take") can correctly live is at the START
  of the NEXT `reset()` call, looking backward at the session that just ended. Do not
  attempt this inside `respond()` — it will silently never fire correctly.
- `Agent()` is instantiated **once** for the whole evaluator run (see `main()`).
  `reset()` is called per-session but does not clear whatever you deliberately keep as
  instance-level state outside of it — this is the only reason any run-level learning
  (e.g. a bandit over strategies) is possible at all.

### 2.2 Metric formulas (confirmed from `metric_summary()`)

```
hit_rate_at_10 = fraction of sessions where target ever entered top_k
mrr            = mean(1/rank) across sessions, 0 if never hit
mttc           = mean(first_hit_turn, or 11 if never hit)      # miss = MAX_TURNS+1
efficiency     = max(0, min(1, (11 - mttc) / 10))
technical_score = 0.50*hit_rate_at_10 + 0.30*mrr + 0.20*efficiency
```
Token usage is reported but **never scored**. Do not optimize for it.

### 2.3 Catalog facts (from profiling the real 50,000-row `catalog.jsonl` directly)

- `parent_asin`: fully unique, never null.
- `categories`: array length varies 2-8. The SAME array index holds different semantic
  content in different branches (demographic labels recur at multiple depths). **Do
  not use a fixed depth for category-conflict detection.** Use the **leaf node**
  (`categories[-1]`) — 800 distinct values, only 5.5% land on a generic bucket
  (Clothing/Women/Men/Accessories/etc). Fall back to `categories[-2]` when the leaf is
  one of that small generic-bucket set.
- Structured-field coverage (this is the single most important catalog finding):
  `details.Department` 87.2% · `store` 99.4% · `details.Color` 4.9% · `details.Brand`
  4.7% · `details.Material` 4.1% · `details.Style` 3.5% · `details.Size` 1.8%.
  **`department` and `store` are the only two attributes reliable enough for a
  structured dict-lookup filter.** Everything else (color/size/material/brand/style)
  must resolve via keyword/text search over `title`+`features`+`description`, never a
  hard structured filter — the field is simply absent for 95%+ of the catalog.
- `department` needs lowercasing before use — `"womens"` (12,740 items) and
  `"Womens"` (12,711 items) are currently separate raw string values for the same
  thing.
- `price`: 78.9% null, plus 0.2% non-numeric placeholder strings (mostly literal `"—"`,
  some `"from X.XX"` range strings needing a prefix-strip before float-casting). **Soft
  score only. Never a hard filter, ever.**
- `store` (not `details.Brand`) is the practical brand signal — 99.4% coverage, and
  87.5% of the time the store name appears verbatim in the product title.
- `average_rating` / `rating_number`: never null, safe to use directly.
- `features` empty 10.4% of the time; `description` empty 47.8% — prefer `title` and
  `features` over `description` when both are available.

### 2.4 Slot schema

```
MACRO tier (drives a full session-state reset):
  category — leaf node of `categories`, per 2.3 above

MICRO tier — reliable structured filter (dict lookup OK):
  department (lowercased), store

MICRO tier — text-matched only, NEVER a hard filter:
  color, size, material, brand, style
  price_band — a {min, max} range, not a scalar
```

### 2.5 Model choice

- **LLM: `Qwen3 8B` via Ollama.** Confirmed viable per current competition rules (local
  models explicitly supported; no hosted model access, API keys, or GPU credits are
  provided, so free/local is the practical default, not a compromise). Use Ollama's
  `format` parameter with a JSON schema for every structured call — this makes
  malformed JSON syntax mechanically impossible, not just less likely. Validate the
  result with `pydantic` on top regardless.
  
- **Embedding model for dense retrieval: `blair-roberta-base` as the primary choice**,
  `bge-small-en-v1.5` as a documented fallback/comparison, not a coin flip — BLaIR was
  pretrained specifically on the Amazon Reviews 2023 dataset (the same source family
  this catalog is a slice of) and its own paper defines and evaluates exactly the
  "complex product search" task shape your `expansion` field produces. BGE is more
  broadly retrieval-optimized but domain-generic. **Build this as a swappable choice
  and actually test both against real dev sessions before locking it in** — reasoning
  alone shouldn't be the final word here, the same way it wasn't for `ask_attribute`.

### 2.6 Rules confirmed (do not re-litigate these)

- LangChain / LlamaIndex / FAISS are **not prohibited**. Not required either — at
  50,000 vectors, brute-force numpy matmul is well within reason and FAISS's
  approximate-search machinery isn't earning its complexity at this scale. Default to
  the simple option; reach for a framework only if a specific, demonstrated need
  arises.
- No training or fine-tuning of any base model. Calibrating plain scalar weights
  (fusion weights, thresholds) against dev sessions is NOT training — no gradients, no
  weight updates, just picking the best of a handful of named configurations. Call
  this "calibrated," not "trained," in code comments and any write-up.

---

## PART 3 — BUILD INSTRUCTIONS (do these in order; measure after each)

**Step 0 — Baseline sanity check.**
Run `local_evaluator.py` against the current `agent.py` (the stateless BM25 starter,
zero LLM calls, `ask_attribute` hardcoded to `None`) exactly as-is. Confirm you get
`hit_rate_at_10=0.125, mrr=0.068034, mttc=9.81, recommended_technical_score=0.10671`.
If these don't match, stop and report — something differs in the environment before
any of the steps below are meaningful.

**Step 1 — Real `ask_attribute`, no LLM yet.**
Track two things per session: `known_slots` (dict) and `asked_attributes` (set).
Each turn, if there's an `ALLOWED_ATTRIBUTES` value not yet in either set, ask about
it (any fixed priority order is fine to start — this does not need to be smart yet).
Parse the customer's reply text with a simple heuristic to fill `known_slots`. This
is the highest-leverage single step in the whole project — expect a large jump.
Measure.

**Step 2 — Turn-to-turn accumulation.**
Stop re-searching from scratch on only the current turn's raw text. Carry forward
everything disclosed so far in the session into each search. Measure.

**Step 3 — Stage 2, the combined LLM call.**
One Ollama call (Qwen3 8B, schema-constrained JSON, pydantic-validated) producing
five fields: `rewrite`, `expansion`, `slot_operations` (CARRYOVER/UPDATE/DELETE per
slot, per 2.4's schema), `intent`+`confidence`, `no_preference_signal`. Few-shot
examples MUST contrast accumulation language ("also", "in size X") against override
language ("actually", "instead") AND include raw/ungrammatical fragments (e.g.
`"Material:alloy"`, `"Buckle closure"` with no connecting grammar) — real generated
customer messages look like this, not like clean sentences. Measure.

**Step 4 — State management, macro/micro, WITH A TOGGLE.**
Build `CATCMP` (leaf-category compare, per 2.3) and per-slot `CARRYOVER`/`UPDATE`/
`DELETE` application. **Do not hardcode erase-on-conflict as the only behavior.**
Expose an `erasure_mode` constructor argument (`"erase"` or `"accumulate"`) — this is
a genuinely open, untested question, not a settled design choice. Wire it so
`erase_vs_accumulate_comparison.py` (already built, in this repo) can run both modes
and report a paired-bootstrap comparison. Run it. Report both the point estimate and
the confidence interval — an interval that includes zero means "inconclusive," not
"no difference found favors either."

**Step 5 — Dual-track retrieval.**
`BROWSE` (dense, embedding model per 2.5) + `KW` (BM25 — extend the existing SQLite
FTS5 approach already in `agent.py` — plus the field-tier logic from 2.4: dict-lookup
on department/store, text search on color/size/material/brand/style, soft score only
on price). Add a confidence-threshold short-circuit on `BROWSE` (placeholder
threshold, mark clearly as `# TUNE`) for high-confidence buying turns. Measure.

**Step 6 — Merge and rank.**
`POOL`: min-max normalize scores per track, truncate to top-N per track (placeholder
N, mark `# TUNE`), merge. `FUSION`: placeholder equal weights, mark `# TUNE`.
`RANKLLM`: one more Ollama call, fuses keyword+category+vector scores into the final
top-10. Measure.

**Step 7 — Magnitude, spread, revise — always rank regardless.**
`MAG` (placeholder floor, `# TUNE`) gates only whether one query revision is
attempted — it must NEVER gate whether `RANKLLM`/`TOPK` runs. Ranking runs
unconditionally once magnitude clears (first check, or forced through after the one
allowed revision, regardless of outcome). `SPREAD` (placeholder flatness threshold,
`# TUNE`) is a separate, parallel signal that only decides whether `CLARIFY`
*additionally* fires — never whether ranking fires. Measure.

**Step 8 — Real `CLARIFY`.**
Replace Step 1's placeholder rotation. Candidate set = `ALLOWED_ATTRIBUTES` minus
known slots minus `asked_attributes`. Pick by information-gain over the current
`POOL` as the primary signal. Wire `no_preference_signal` from Step 3: hard-exclude
that attribute immediately; soft-raise the bar for asking anything else this session
(not an absolute ban). If the candidate set is empty, suppress clarification
regardless of what `SPREAD` says. Measure — this is where the two MTTC fixes actually
land structurally.

**Step 9 — Calibrate.**
Named-configuration search first (3-4 options per parameter, not a continuous sweep —
200 sessions is a small enough sample that fine-grained search risks overfitting to
dev-set noise), Optuna refinement only if time remains and the first pass shows real
sensitivity. Order: fusion weights first (isolated, against Hit Rate/MRR only), then
`MAG`/`SPREAD`/`GATE` together (against full `technical_score`, since they trade Hit
Rate against MTTC). Every comparison gets the same paired-bootstrap treatment as
Step 4 — a config "winning" on a point estimate with a CI that includes zero is noise,
not a finding.

**Step 10 — Pillar III (lowest priority; only with time to spare after Step 9).**
`RESETHOOK` inside `reset()` (per 2.1, not `respond()`) → `distill_session_to_profile()`
(only update `preference_tags`/`purchase_frequency`/`summary` — one conversation gives
no real signal on `average_prior_rating`/`rating_style`, don't fabricate one) →
`ProfileStore` (local JSON file, no network) → validate via a standalone round-trip
test (real profile → real session → distill → synthetic second session seeded with
the distilled profile → confirm turn-1 behavior actually changed), since no session in
this competition's data shares a user ID with any other and this cannot be exercised
by the real eval. Separately: `POLICYSTATS`/`STRATEGY`/`REALIGN` — define the named
strategies concretely (e.g. exact difference in `SPREAD` threshold or max-questions
cap between `clarify_aggressive` and `clarify_sparing`) before writing the bandit
itself; choose epsilon via Monte Carlo simulation using a real measured strategy-gap
estimate, not a generic default — a static replay of 200 sessions cannot validate an
exploration parameter, only a simulation can.

---

## OPEN QUESTIONS — build as configurable and report back, do not decide silently

- Erase vs. accumulate on intent-override (Step 4) — run the comparison, report the
  confidence interval, do not pick one without it.
- BLaIR vs. BGE (2.5) — test against real dev sessions, do not assume BLaIR wins just
  because it's better-reasoned; verify.
- Every `# TUNE` placeholder from Steps 5-9 — these are real unknowns, not values to
  guess confidently and move on from.
- Whether the private 800-session grading run is one continuous process or
  parallelized across workers — materially changes how much Step 10's bandit is worth
  building. Reasonable working assumption is "one continuous process," but this is not
  confirmed — ask if there's any channel to check.