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

## PROGRESS LOG — current confirmed state (update this after every step)

| Step | recommended_technical_score | Status |
|---|---|---|
| 0 (baseline) | 0.1067 | done |
| 1 (real `ask_attribute`) | 0.4459 | done — the single highest-leverage fix in the project |
| 2 (turn accumulation) | 0.5955 | done |
| 3 (Stage 2 LLM call) | 0.6031 | done |
| 4 (state / erasure) | ~0.60 (dev_subset) | done — see finding below |
| 5 (dual-track retrieval) | 0.6017 | done — see finding below |
| 6 (fusion + `RANKLLM`) | 0.6209 | done |
| 7 (`MAG`/`SPREAD`/`REVISE`) | 0.6034 (MAG fixed, not yet calibrated) | in progress — see finding below |
| 8 (real `CLARIFY`) | — | not started |
| 9 (calibration) | — | not started |

**Every step so far has surfaced at least one real bug that reasoning alone did not
catch.** Budget Steps 8-9 assuming this pattern continues, not assuming a clean run.

- **`CATCMP` is structurally inert on this dataset.** Traced directly: `intent_override`
  sessions never change category mid-session — only micro-tier facts (material, care
  instructions, etc.) change, even at the override turn. The macro/micro split's macro
  half has zero opportunity to fire here. `erasure_mode="erase"` and `"accumulate"` are
  therefore deterministically identical outside a path that never triggers — confirmed
  via `erase_vs_accumulate_comparison.py`, CI exactly `[0.0000, 0.0000]`.
- A third mode, `"gated"` (only honor a per-slot `DELETE` if some OTHER slot got a real
  `UPDATE` the same turn — NOT the same slot, that condition is impossible given the
  schema), was built to test whether per-slot `DELETE` needed validation. It engages
  often (58% of turns had a `DELETE`-with-zero-`UPDATE`s pattern; 95% of raw `DELETE`s
  got downgraded) but produced a CI of exactly `[0.0000, 0.0000]` against `accumulate` on
  hit/miss outcome — zero sessions flipped. **Final decision: default
  `erasure_mode = "accumulate"`. Do not reopen this.** A real, separate bug was found
  alongside this investigation instead: "I don't have an additional preference for X"
  was being misread as license to `DELETE` an already-known value for X, rather than
  "nothing NEW to add." Fixed at the Stage 2 prompt level (folded into Step 5), not via
  `erasure_mode`.
- **`GATE`'s branch was inverted for a real stretch of build time.** Designed meaning:
  "high buying-confidence → trust `KW`, skip `BROWSE`." First implementation did the
  reverse — discarded `KW`'s results and used `BROWSE` alone whenever confidence was
  high, silently dropping a confirmed rank-1 `KW` hit on ~19% of all turns (measured, not
  an edge case). Fixed: confidence check now happens BEFORE `browse_ids` is computed;
  high-confidence turns skip the embedding call entirely, restoring `GATE`'s original
  latency-saving intent too.
- **Raw BLaIR embeddings have severe anisotropy/hubness.** Cosine-to-centroid mean=0.895
  across the full 50k catalog; unrelated queries returned nearly identical top-5 results.
  Whitening fixed the aggregate stat (mean 0.895→0.006) but did NOT eliminate practical
  hub pollution (2-3 shared junk items persisted across unrelated queries even after
  whitening). **Switched to `bge-small-en-v1.5`**, zero hub overlap on the same test.
  **This supersedes 2.5 below — BGE is the model actually in use for `BROWSE`**, despite
  being domain-generic rather than Amazon-trained, because it's the one that empirically
  works. Don't revert to BLaIR without re-solving hubness first.
- **`MAG`'s per-track min-max normalization made it structurally unable to fire.** Each
  track's own #1 candidate always normalized to 1.0 regardless of underlying match
  quality, so the fused top score could only fall below any reasonable floor if BOTH
  tracks returned zero candidates. Fixed via percentile-based fixed-reference
  normalization (KW p10/p90≈[14,37], BROWSE p10/p90≈[0.65,0.82] — computed from
  `dev_subset.jsonl`; re-derive against the full 200 before final submission). Confirmed
  genuinely discriminating (weak case→0.0, strong case→1.0). **BUT: with the current
  placeholder floor, net effect on `technical_score` is slightly negative
  (0.6209→0.6034), driven by `mttc` getting worse.** Of 38 revision events (14.4% of
  turns), only 6 coincided with a hit; 10 of the 38 fired on turn 1, where `REVISE`'s
  strategy (broaden `rewrite`→`expansion` terms) has little to work with since almost
  nothing has been disclosed yet. **`MAG_FLOOR`, and possibly a turn-aware `REVISE`
  strategy, are real open calibration targets for Step 9 — not values expected to just
  work once picked reasonably. Watch `mttc` specifically, and check whether turn-1
  revisions should be suppressed or handled differently from later-turn revisions.**
- **Step 10a (long-term profile distillation) is DONE — a real, working artifact, not a
  design doc.** `RESETHOOK` inside `reset()` → `distill_session_to_profile()` →
  `ProfileStore` (all in `starter/profile_store.py`) built and validated via
  `profile_distillation_roundtrip.py`, a standalone round-trip test (real session →
  distill → seed a synthetic second session with the distilled profile → compare turn
  1). Confirmed working: turn 1 `ask_attribute` flipped `color`→`material` and the full
  recommendation set changed, same opening message, same everything except the profile.
  **The mechanism was traced, not inferred from the before/after outputs — and it's a
  four-hop chain, which turned out to be a better story than a direct read would have
  been.** The entropy-based `CLARIFY` selection (`_information_gain`/`_attribute_signal`)
  never reads `notable_facts` at all — it only ever sees `catalog_meta` for whatever
  `candidate_asins` it's handed. Instrumenting `_analyze_turn`/`_select_ask_attribute`
  directly (not eyeballing outputs) showed what actually happened: (1) the distilled
  profile's `preference_tags` seeded `notable_facts` at `reset()`, reusing the existing
  "Other confirmed facts" prompt slot — zero new selection/retrieval code involved; (2)
  that changed the Stage 2 LLM's `rewrite`/`expansion` — it pulled "polyester" into the
  query, which the un-personalized run's output never had; (3) that changed KW/BROWSE's
  actual queries, pulling a substantially different 20-candidate fused pool (12/20
  candidates differ from the un-personalized run); (4) recomputing entropy over that
  *different pool*'s real catalog metadata — same formula, different inputs — `material`
  entropy rose 1.054→1.533 while `color` fell 1.857→1.154, flipping the arg-max from
  `color` to `material`. Every one of those four numbers was read directly off the
  instrumented run, not assumed. **Caveat, stated plainly rather than treated as
  something to go fix:** this was traced on exactly one session (`public_0006`). The
  chain has four separate places the effect could wash out on a different session — the
  LLM might not fold a given tag into `rewrite`/`expansion` at all, the retrieval delta
  might be too small to change the top-20 pool, the entropy flip might land on an
  attribute `known_slots`/`asked_attributes` had already excluded, or SPREAD might not
  have been flat enough for CLARIFY to fire at all that turn. This is a demonstrated
  real mechanism on n=1, not a general result, and re-running it across more sessions is
  future work, not a gap in this write-up. `technical_score` cannot be affected either
  way — no session in this dataset shares a `user_id` with any other (see Step 10's
  framing below) — so this is purely a rubric-credit artifact, not a scoring lever.
- `notable_facts` entries changed shape from a bare string to `{"text": ...,
  "source": "profile"|"session"}`, so a fact seeded from a distilled profile stays
  distinguishable from one the customer actually said this session. Pure tag, no
  behavior change: `_format_transcript` and `distill_session_to_profile` were updated to
  read `entry["text"]` and treat both sources identically; nothing branches on `source`
  yet. The round-trip test was re-run after this change and reproduced the exact same
  result (`color`→`material`, same recommendation sets) — confirming the tag is inert.

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
- **Embedding model for dense retrieval: `bge-small-en-v1.5` — SUPERSEDES the original
  guidance below, confirmed by real testing, see PROGRESS LOG.** `blair-roberta-base` was
  the original reasoned choice (pretrained specifically on the Amazon Reviews 2023
  dataset this catalog is a slice of, evaluated on exactly the "complex product search"
  task shape `expansion` produces) — good reasoning, wrong empirical outcome. BLaIR's raw
  embeddings showed severe anisotropy/hubness (cosine-to-centroid mean=0.895 across the
  full catalog); whitening reduced the aggregate statistic but did not eliminate practical
  hub pollution. BGE showed zero hub overlap on the same test and is what's actually
  running. Keep the embedding model swappable in code regardless — this is exactly the
  kind of decision that should stay easy to revisit if the input data or model choices
  change later.

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

**Step 4 — State management, macro/micro. DONE — see PROGRESS LOG for full detail.**
`CATCMP` (leaf-category compare) and per-slot `CARRYOVER`/`UPDATE`/`DELETE` are built.
**Resolved: `erasure_mode` defaults to `"accumulate"`.** `CATCMP` never fires on this
dataset (category doesn't change mid-`intent_override`), so erase/accumulate are
provably identical here; a `"gated"` per-slot-`DELETE`-validation mode was tested and
showed no measurable outcome difference either. Do not reopen this question. A real,
separate bug (no-preference replies misread as license to delete a known value) was
found and fixed at the Stage 2 prompt level instead — see Step 5.

**Step 5 — Dual-track retrieval. DONE — see PROGRESS LOG for full detail.**
`BROWSE` (dense, BGE — see 2.5's superseding note) + `KW` (BM25, plus the 2.3 field-tier
logic) both built. `GATE`'s confidence-threshold short-circuit is built AND was found and
fixed after shipping inverted (was discarding `KW`'s results on high confidence instead
of `BROWSE`'s — the opposite of its design intent — on ~19% of all turns before the fix).
Also bundled into this step: a Stage 2 prompt fix so "no additional preference for X"
is not misread as grounds to delete an already-known value for X.

**Step 6 — Merge and rank. DONE.** `POOL` (min-max normalize per track, top-N truncate,
mark N `# TUNE`), `FUSION` (equal weights, mark `# TUNE`), `RANKLLM` (Ollama call fusing
keyword+category+vector scores into the final top-10) all built and measured — improved
every metric with no regressions (0.6017 → 0.6209).

**Step 7 — Magnitude, spread, revise — always rank regardless. IN PROGRESS, see
PROGRESS LOG.** `MAG` shipped structurally broken (per-track min-max normalization made
it unable to ever fire on anything short of a total blank), then fixed via fixed-reference
normalization — confirmed genuinely discriminating between weak and strong results now.
Ranking still correctly runs unconditionally regardless of `MAG`'s outcome — that
constraint has held throughout. **What's still open: `MAG_FLOOR` itself and the `REVISE`
strategy are real calibration targets for Step 9, not placeholders expected to work once
picked reasonably** — the current placeholder floor produces a net-negative effect on
`technical_score`, and turn-1 revisions specifically look unhelpful (see PROGRESS LOG).
`SPREAD` is computed and stored every turn but deliberately NOT yet wired to any
behavior — that's Step 8's job, not this one. Do not wire `SPREAD` into `ask_attribute`
before Step 8; doing so risks reintroducing the exact regression Step 1 fixed.

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

**`MAG_FLOOR` specifically needs real attention here, not a quick pick** — per the
Step 7 finding, the current placeholder makes `technical_score` net-negative, and 10 of
38 revision events fired on turn 1 where `REVISE` (broadening `rewrite`→`expansion`
terms) has little to work with. Test at minimum: a stricter floor (fewer, more
targeted revisions), and whether suppressing `REVISE` on turn 1 specifically (or using
a different revision strategy there) recovers the `mttc` cost without losing the `mrr`
gain the fix already produced. Also re-derive the p10/p90 reference bounds against the
full 200-session `public_set.jsonl` before finalizing — they were computed against
`dev_subset.jsonl`'s 39 sessions and should be confirmed, not assumed to transfer.

**Step 10 — Pillar III. Only after Step 9 closes. Two halves, in this order, not in
parallel — 10a before 10b, for a real dependency reason, not just priority.**

**Important context for both halves:** `technical_score` (`hit_rate_at_10`/`mrr`/`mttc`)
cannot be affected by either half, structurally — no session in this competition's data
shares a user ID with any other, and this is graded under the human-scored rubric
categories (Innovation & Problem Insight, Feasibility & Practicality), not the automated
score. If time runs out before either half, document the design as a reasoned plan in
the written submission instead of building it — that still earns credit under those
categories. Do not let either half compete with Steps 8-9 for time; those move the
actual graded score and these do not.

**Step 10a — long-term profile distillation. DONE — see PROGRESS LOG for full detail.**
`RESETHOOK` inside `reset()` (per 2.1, NOT `respond()` — `respond()` structurally cannot
know whether a hit occurred, only a later `reset()` can infer it retrospectively) →
`distill_session_to_profile()` (updates only `preference_tags`/`purchase_frequency`/
`summary` — `average_prior_rating`/`rating_style` are explicitly left unchanged, with the
reasoning in a comment, since one conversation gives no real signal on either) →
`ProfileStore` (local JSON file, no network) — all built in `starter/profile_store.py`
and wired into `Agent.reset()`. Validated via `profile_distillation_roundtrip.py`, a
standalone round-trip test: real profile in → run a real session → distill → seed a
SYNTHETIC second session with the distilled profile → confirmed turn-1 behavior actually
changed (`ask_attribute` flipped `color`→`material`, full recommendation set changed).
The mechanism behind that flip was separately traced end-to-end (real instrumentation,
not inference) — see the PROGRESS LOG entry for the four-hop chain and its
one-session-only caveat. `notable_facts` entries also now carry a `source:
"profile"|"session"` tag so profile-seeded facts stay distinguishable from
session-disclosed ones, with no behavior change yet riding on that tag.

**Step 10b — Adaptive Orchestration. Build this only after 10a, and only after Step 8
is real — it depends on Step 8's actual `CLARIFY` logic to have anything meaningful to
optimize.** `POLICYSTATS` (a dict on the `Agent` instance, persists across the whole
run since `Agent()` is only instantiated once) → `STRATEGY` (epsilon-greedy pick between
2 named, CONCRETELY DEFINED strategies — e.g. an exact different `SPREAD_FLATNESS_
THRESHOLD` value or a max-questions-per-session cap between `clarify_aggressive` and
`clarify_sparing`; do not leave these as placeholder names) → `REALIGN` (reads `SPREAD`'s
result plus the active `STRATEGY` to decide whether `CLARIFY` fires this turn, absorbing
the turn-budget guardrail: suppress clarification once `turns_remaining ≤ 2` regardless
of strategy). Reward = `-turns_consumed`, updated inside `RESETHOOK`, looking backward at
the session that just finished. **Warm-start the bandit**: run both named strategies once
on `dev_subset.jsonl` (same shape as `erase_vs_accumulate_comparison.py`) to get real
measured average turn-counts, and use those as the bandit's initial `Q` values rather
than starting blind — local dev runs never carry into the real graded run, so this is the
only way prior measurement helps at all. Choose epsilon via Monte Carlo simulation using
that real measured strategy-gap estimate, not a generic default — a static replay of 200
sessions cannot validate an exploration parameter, only a simulation can. Validate by
confirming mean session length in the back half of a `dev_subset.jsonl` run is lower than
the front half — expect this to be a weak signal at only 39 sessions, not a dramatic one;
that's a sample-size fact, not evidence the mechanism is broken.

---

## RESOLVED — do not reopen these

- Erase vs. accumulate on intent-override: **`accumulate`, settled.** `CATCMP` never
  fires on this dataset; `"gated"` mode tested and showed no measurable difference.
- BLaIR vs. BGE: **`bge-small-en-v1.5`, settled.** BLaIR's hubness survived whitening;
  BGE didn't have the problem at all. Domain-training reasoning was good; empirical
  result went the other way.

## OPEN QUESTIONS — build as configurable and report back, do not decide silently

- **`MAG_FLOOR` and the `REVISE` strategy (Step 9)** — the newest and most concrete open
  item. Current placeholder is net-negative on `technical_score`; turn-1 revisions look
  specifically unhelpful. See Step 9 and the PROGRESS LOG for what to test.
- Every other `# TUNE` placeholder from Steps 5-9 (fusion weights, `SPREAD`'s flatness
  threshold, `GATE`'s confidence threshold, pool top-N) — real unknowns, not values to
  guess confidently and move on from.
- Whether the private 800-session grading run is one continuous process or parallelized
  across workers — materially changes how much Step 10b's bandit is worth building.
  Reasonable working assumption is "one continuous process," but this is not confirmed —
  ask if there's any channel to check.