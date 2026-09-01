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
| 7 (`MAG`/`SPREAD`/`REVISE`) | 0.6034 (MAG fixed, not yet calibrated) | done — calibrated in Step 9 |
| 8 (real `CLARIFY`) | 0.6383 (dev_subset, SPREAD-gate off + brand-fix, pre-Step-10a) | done — see finding below |
| 9 (calibration) | 0.596816 (dev_subset, corrected baseline) → 0.671749 (FUSION+MAG calibrated, `GATE` still open) | in progress — see findings below |

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
- **Step 8 (real `CLARIFY`) is DONE.** Entropy-based `_select_ask_attribute` over
  `POOL`'s current candidates, gated by `no_preference_signal` exclusions and (at the
  time) `SPREAD`. Found and fixed a real bug during this build: entropy ranking was
  picking `brand` first in 28/29 sessions that asked anything, because its catalog
  signal (`store`) has near-maximal diversity (2.3: 99.4% coverage, almost every
  product a distinct store) — but `local_evaluator.classify_constraint()` has **no
  branch that ever returns `"brand"`**, confirmed by extracting every literal in its
  `return "..."` statements. `ask_attribute="brand"` was therefore a guaranteed
  content-free turn no matter what the customer knew. Fixed via
  `CONSTRAINT_ROUTABLE_ATTRIBUTES` (the exhaustive traced-reachable set:
  `material/color/size/style/budget/use_case/feature`) gating entropy ranking, and
  `CLARIFY_UNREACHABLE_ATTRIBUTES = {"category", "brand"}` excluding both dead ends
  (`category` is separately owned by `known_slots`/`CATCMP`, not `CLARIFY`, on its own
  architectural grounds). Deliberately hardcoded rather than imported live from
  `evaluator.local_evaluator` — that module is dev-only tooling a real submitted agent
  can't assume is importable.
- **SPREAD-gating `CLARIFY` is RESOLVED: gate permanently OFF (`clarify_spread_gate`
  now defaults to `False`), do not reopen.** Measured on `dev_subset.jsonl` (39
  sessions), same code otherwise (brand-fix already in): gate ON (the original Step 8
  design — only ask when the pool looks genuinely flat) scored
  `hit_rate=0.359/mrr=0.241/technical_score=0.2949`, with only 122/320 turns (~38%)
  asking anything — badly under-asking. Gate OFF (ask whenever `attr_candidates` is
  non-empty, ignoring `SPREAD` entirely) scored
  `hit_rate=0.795/mrr=0.565/technical_score=0.6383`, 250/282 turns (~89%) asking. A
  >2x swing on `technical_score` from one boolean — this is now closed, matching the
  same pattern as `erasure_mode`: a plausible-sounding gate that empirically hurts.
  `SPREAD` itself is still computed and stored every turn (diagnostic value, and
  `Step 10b`/future work may still read it) — only its use as a `CLARIFY` gate is
  dead.
- **Two real bugs found while re-verifying the `0.6383` baseline at the start of Step
  9 — neither is a rehash of anything above.**
  1. **Step 10a's personalization seed was folding *bare attribute-name words* into
     `notable_facts`**, not just genuine values. A profile's `preference_tags` can
     literally be the string `"material"` or `"fit"` (real values from this dataset,
     not hypothetical) with no accompanying value — e.g. no fabric named for
     `"material"`. Seeding these verbatim let the Stage 2 LLM fold the bare word
     itself into `rewrite`/`expansion` (confirmed directly on `public_0012`: seeding
     `"material"` alone produced `rewrite="women dresses, material"`), which
     discriminates nothing since it names the SLOT, not a value for it. **Fixed, but
     not via a curated word list** (a first pass using a hand-picked "does this sound
     descriptive" list was explicitly rejected as exactly the kind of dataset-specific
     judgment call this project avoids) — the final `_BARE_ATTRIBUTE_NAME_TOKENS` is
     the union of `ALLOWED_ATTRIBUTES`, `SLOT_NAMES`, `CONSTRAINT_ROUTABLE_ATTRIBUTES`,
     and `_CLASSIFY_CONSTRAINT_NAME_ALIASES = {"fit", "sizing"}` — the last two
     traced directly to `evaluator.local_evaluator.classify_constraint()`'s own
     source: its `"style"` branch is
     `any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck"))`,
     i.e. the evaluator's own code already treats `"fit"` as interchangeable with
     `"style"` — the identical relationship 2.1 documents for `"department"`, just
     one more word in that literal tuple. `"sizing"` gets the same treatment from the
     `"size"` branch. Verified: `"material"`/`"fit"` excluded, `"polyester"`/`"black"`/
     `"hiking"`/`"wide"`/`"narrow"`/`"comfort"`/`"durability"` (genuine values and this
     dataset's other quality-descriptor tags) all still pass through.
  2. **Applying that fix made the aggregate score go *down*, not up**
     (`0.627949→0.596816` on `dev_subset.jsonl`, hit_rate 31/39→30/39). Traced directly
     (controlled A/B, identical code, seed on vs. off, `public_0006`/`public_0081`/
     `public_0012`): all three showed real, confirmed divergence from the seed, mixed
     direction (`public_0012` miss→hit, `public_0081` hit→miss). For `public_0081`/
     `public_0006`, `rewrite`/`expansion` actually *converge* to nearly identical text
     within 2-3 turns, yet outcomes still diverge by turn 4-8 — the mechanism is a
     **cascade**, not a sustained different query: turn 1's small difference nudges
     `CLARIFY`'s `ask_attribute` choice, which changes what the customer discloses
     next, compounding over the session. **This is a real, inherent characteristic of
     the profile-seeding mechanism, not a bug to chase further** — flagged as a known
     limitation of Step 10a for the written submission. A structurally-correct fix
     does not guarantee a monotonic score improvement on a 39-session sample; it just
     removes that one bug's effect, whichever direction it happened to net out to —
     the same lesson Step 7's `MAG` fix already taught once. **`0.596816` (dev_subset)
     is the correct Step 9 reference baseline** — the `0.6383` figure predates the
     material/fit fix and Step 10a's interaction with it, and should not be cited as
     current.
- **Step 9, parameter 1/3 — FUSION weights: RESOLVED, `FUSION_KW_WEIGHT=0.65` /
  `FUSION_BROWSE_WEIGHT=0.35`.** Tested twice, because the first pass predated the
  material/fit fix above and the ranking of configs **completely reversed** once it
  landed — cited here only as a worked example of why re-measuring after a fix isn't
  optional: pre-fix, 0.65/0.35 scored *worse* than equal weights
  (`hit_rate=0.769/technical_score=0.6095` vs. equal-weight's `0.6279`); post-fix, on
  `dev_subset.jsonl` against the corrected baseline: 0.5/0.5 →
  `hit_rate=0.7692/mrr=0.4697/technical_score=0.5968`; 0.65/0.35 (KW-favored) →
  `hit_rate=0.8205/mrr=0.5159/technical_score=0.6389`; 0.35/0.65 (BROWSE-favored) →
  `hit_rate=0.8205/mrr=0.5078/technical_score=0.6365`. Paired-bootstrap on the
  designated decision metric (`hit_rate_at_10`/`mrr` only, per this project's own
  rule that fusion weights shouldn't be tuned against a metric they don't directly
  target) was **inconclusive for all three pairwise comparisons** at n=39 (every CI
  included zero) — the two "heavy" variants aren't even distinguishable from each
  other. The one non-noise signal: KW-heavy's full `technical_score` CI vs. equal
  weights barely excluded zero (`[+0.0029, +0.0974]`) — weak, and not the specified
  gating metric, but real and never pointing the other direction. Chosen on that
  basis (explicit call, not a silent default).
- **Cross-session file collision corrupted (then was cleanly recovered from) two
  Step 9 `MAG_FLOOR` calibration runs — a real operational risk for this project's
  process, not an architecture finding, but worth recording so it isn't repeated.**
  A second, independent Claude Code session was concurrently developing Step 10b
  (`POLICYSTATS`/`STRATEGY`/`REALIGN`) directly inside `starter/agent.py`,
  uncommitted, and wired its `REALIGN` gate straight into the shared `respond()`
  `CLARIFY` logic (a per-session question cap as low as 1 under its
  `clarify_sparing` strategy). Two `MAG_FLOOR`/`REVISE` calibration runs
  (`MAG_FLOOR=0.5` alone, and `suppress_revise_turn1` alone — two unrelated levers)
  both silently ran against this and collapsed to `~0.25-0.27`, nearly identically.
  **Caught because of that similarity, not because anything crashed**: two different
  parameters producing nearly the same catastrophic number is a tell that a third,
  shared factor is actually driving the result. Root cause, confirmed directly by
  the other session: its bandit's strategy-picker used a fixed `random.Random(0)`
  seed, so every fresh `Agent()` (exactly what a calibration script creates per
  config) got an *identical* deterministic strategy sequence regardless of which
  `MAG_FLOOR`/`REVISE` value was under test, and its two strategies' placeholder
  warm-start values differed by under 3% — fragile enough that the bandit kept
  locking onto the 1-question-cap strategy either way. **Fix for the rest of the
  session**: built an isolated copy of `starter/`/`evaluator/` pinned to the last
  clean commit plus only this project's own verified Step 9 fixes, symlinked to the
  shared (read-only, low-risk) `data/` directory, and ran all subsequent Step 9
  calibration from there — fully decoupled from whatever either session edits in the
  live working directory. Re-running the identical two configs from the clean copy
  flipped both from catastrophic collapse to the best scores measured in the whole
  calibration (see below) — confirming the contamination diagnosis directly, not
  just plausibly. **Process takeaway for future work on this repo: check
  `git status`/`git diff` on shared files before trusting a surprising calibration
  result, especially if multiple sessions might be active** — this cost two ~30-40
  minute runs before being caught.
- **Step 9, parameter 2/3 — `MAG_FLOOR`/`REVISE`: RESOLVED, `MAG_FLOOR=0.5`,
  `suppress_revise_turn1` stays `False`.** All numbers below are from the isolated,
  contamination-free re-run (see above), on top of the now-locked `FUSION_KW_WEIGHT
  =0.65`. Baseline (`floor=0.3`, no suppression) = `hit_rate=0.8205/mrr=0.5159/
  technical_score=0.6389`. `floor=0.5` alone → `0.8462/0.5520/0.6717`.
  `suppress_revise_turn1` alone (`floor` stays `0.3`) → `0.8205/0.5496/0.6562`.
  `floor=0.5` + suppression combined → `0.8462/0.5513/0.6710`. All three beat
  baseline on every point estimate, but none individually cleared a 95% CI excluding
  zero at n=39 against baseline. **The one comparison that WAS conclusive**:
  `floor=0.5` alone vs. the combined config are statistically indistinguishable from
  each other (`diff=+0.0007`, CI `[-0.0160, +0.0182]`) — turn-1 `REVISE` suppression
  adds no measurable benefit once the floor is already raised, so it isn't worth the
  extra parameter. Chose `floor=0.5` alone: best point estimate, CI closest to
  significance (`[-0.0060, +0.0931]`), no evidence the simpler default costs
  anything. **This reverses Step 7's own finding that `REVISE` was net-negative** —
  that conclusion was measured under different conditions (equal `FUSION` weights,
  the seeding bug still present); these parameters interact, they don't each have one
  fixed correct value measured in isolation. `suppress_revise_turn1` stays available
  as a constructor param (default `False`) in case a different combination at full
  200-session scale reopens the question.
- **Step 9, parameter 3/3 — `GATE_CONFIDENCE_THRESHOLD`: DEFERRED, kept at the
  original `0.75` placeholder.** A `threshold=0.6` test was launched (against the
  now-locked `FUSION_KW_WEIGHT=0.65`/`MAG_FLOOR=0.5`) but stopped before completion
  and never compared — an explicit, time-constrained call, not a finding. **`0.75` is
  untested by Step 9, not confirmed-good by it** — this is real remaining calibration
  work, not a closed question, if time allows a return to it. Do not read the current
  default as validated the way `FUSION`/`MAG_FLOOR` are.

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

**Step 7 — Magnitude, spread, revise — always rank regardless. DONE, calibrated in
Step 9.** `MAG` shipped structurally broken (per-track min-max normalization made it
unable to ever fire on anything short of a total blank), then fixed via fixed-reference
normalization — confirmed genuinely discriminating between weak and strong results.
Ranking still correctly runs unconditionally regardless of `MAG`'s outcome throughout.
`MAG_FLOOR` and `REVISE`'s turn-1 behavior were real, open calibration targets exactly
as flagged here — see Step 9 and the PROGRESS LOG for the resolved values
(`MAG_FLOOR=0.5`, `suppress_revise_turn1=False`) and why the original "REVISE is
net-negative" reading of this step turned out to be conditional on FUSION weights that
Step 9 later changed. `SPREAD` is computed and stored every turn; its use as a
`CLARIFY` gate was tested in Step 8/9 and resolved OFF (see below) — it is not wired to
any other behavior currently.

**Step 8 — Real `CLARIFY`. DONE — see PROGRESS LOG for full detail.**
Entropy-based `_select_ask_attribute` over `POOL`'s current candidates replaces Step 1's
placeholder rotation, gated by `no_preference_signal` exclusions. Found and fixed a real
bug: entropy ranking was picking `brand` first almost every time because its catalog
signal (`store`) has near-maximal diversity, but `classify_constraint()` can never
actually route a match to `"brand"` — a structurally dead attribute regardless of
calibration. Fixed via `CONSTRAINT_ROUTABLE_ATTRIBUTES`/`CLARIFY_UNREACHABLE_ATTRIBUTES`.
**SPREAD-gating `CLARIFY` (only ask when the pool looks flat) was tested and is
RESOLVED OFF, do not reopen** — gated scored `technical_score=0.2949` vs. always-ask's
`0.6383` on `dev_subset.jsonl`, a >2x swing from one boolean. `clarify_spread_gate`
defaults to `False` permanently now.

**Step 9 — Calibrate. IN PROGRESS, see PROGRESS LOG for full detail.**
Named-configuration search (3-4 options per parameter), paired-bootstrap treatment on
every comparison — a config "winning" on a point estimate with a CI that includes zero
is noise, not a finding, and this showed up repeatedly in practice, not just as a
disclaimer. **Two real bugs were found and fixed before any calibration numbers could
be trusted**: (1) Step 10a's personalization seed was folding bare attribute-name words
(`"material"`, `"fit"`) into `notable_facts` as if they were disclosed values — fixed
via `_BARE_ATTRIBUTE_NAME_TOKENS`, traced to `classify_constraint()`'s own source, not a
curated list; (2) a second Claude Code session concurrently developing Step 10b inside
the same `starter/agent.py`, uncommitted, corrupted two `MAG_FLOOR` calibration runs via
a fixed-seed bandit bug in its own in-progress code — recovered by running the rest of
Step 9's calibration from an isolated copy of the source pinned to the last clean
commit. See PROGRESS LOG for both incidents in full, including the exact numbers.

**Parameter 1/3, FUSION weights: RESOLVED.** `FUSION_KW_WEIGHT=0.65`/
`FUSION_BROWSE_WEIGHT=0.35`. Tested twice — the pre-material/fit-fix and post-fix
rankings of the same three configs completely reversed, which is itself the concrete
argument for re-measuring after any fix rather than assuming a prior comparison still
holds. See PROGRESS LOG for full numbers and CIs.

**Parameter 2/3, `MAG_FLOOR`/`REVISE`: RESOLVED.** `MAG_FLOOR=0.5`,
`suppress_revise_turn1` stays `False` (tested combined with the floor change; added no
measurable benefit on top of it — the one fully conclusive comparison in this
parameter's testing). This reverses Step 7's "REVISE is net-negative" finding, measured
under the old, now-changed FUSION weights — the two parameters interact. Also
re-derive the `MAG`/`SPREAD` p10/p90 reference bounds against the full 200-session
`public_set.jsonl` before treating this as final — they were computed against
`dev_subset.jsonl`'s 39 sessions and have not yet been reconfirmed at full scale.

**Parameter 3/3, `GATE_CONFIDENCE_THRESHOLD`: DEFERRED under time pressure**, kept at
the original `0.75` placeholder — untested by Step 9, not validated by it. If picking
this back up: named-config comparison against the now-locked `FUSION_KW_WEIGHT=0.65`/
`MAG_FLOOR=0.5`, full `technical_score` with buying/browsing scenario breakdowns
specifically (per this parameter's own scenario-specific design), same paired-bootstrap
discipline as the other two parameters.

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

**Step 11 — Dynamic Context Engineering. Proposed, not started. Sequence AFTER Steps
8-9 close and Step 10b's own pending verification (real warm-start numbers, front/back-
half validation) lands — do not start this concurrently with either, for the same
reason Step 10b had to be kept isolated from Step 9: this project's LLM-in-the-loop
prompt construction is exactly where a small, well-intentioned change silently moves
`technical_score` (see the Step 10b RNG-seed incident, PROGRESS LOG).**

Unlike Step 10a/10b (structurally inert on `technical_score` — no session shares a
`user_id`, so cross-session state can't move the number), this DOES touch the live
single-session `rewrite`/`expansion`/RANKLLM path — so it's a real scoring lever, not
just rubric credit, and must go through the same paired-bootstrap measurement
discipline as Step 9, not a "seems obviously better" judgment call.

**Mechanism 1 — Stage 2 prompt construction (`_format_transcript`).**
`notable_facts` is currently dumped in full, unconditionally; `_recent_content_turns`
keeps a fixed `RECENT_TURN_WINDOW=2` regardless of relevance to the turn being
processed. Replace both with relevance-scored selection: rank `notable_facts` by
lexical overlap with the current message (`_fact_relevance`/`_select_relevant_facts` —
see chat sketch, not yet in code) and cap at `MAX_FACTS_IN_PROMPT` (`# TUNE`), always
keeping at least the most recent fact so the line never goes empty. Constraint: this
call happens INSIDE `_analyze_turn`, BEFORE KW/BROWSE/POOL/MAG/SPREAD run this turn —
no retrieval-confidence signal is available yet, only the fact text and the current
message. Do not design this mechanism as if MAG/SPREAD were available here; they
aren't.

**Mechanism 2 — RANKLLM candidate context (`RANKLLM_CANDIDATE_LIMIT`).**
Currently a flat top-20 cutoff fed to `_rank_llm`, unconditional every turn. This call
site runs AFTER `MAG`/`SPREAD` are computed, so — unlike Mechanism 1 — it legitimately
can use that signal: fewer/terser candidates when the pool is confidently peaked (MAG
high, SPREAD not flat, POOL's own ranking is probably already close), more when flat
(RANKLLM has more real work to do disambiguating). Not sketched yet; scope only, same
as Mechanism 1 was before this write-up.

**Measurement plan**: paired-bootstrap on `hit_rate_at_10`/`mrr`/`mttc` against
`dev_subset.jsonl` first (fast iteration), full 200-session `public_set.jsonl`
confirmation before calling either mechanism done — same bar as every other step in
this file. Expect real risk of a null or negative result, same as MAG_FLOOR's Step 7
finding — a relevance-scored fact selector is not obviously an improvement over
"just send everything" until measured; an LLM with irrelevant-but-present context is
not always worse than one with a smaller, wrongly-filtered context.

---

## RESOLVED — do not reopen these

- Erase vs. accumulate on intent-override: **`accumulate`, settled.** `CATCMP` never
  fires on this dataset; `"gated"` mode tested and showed no measurable difference.
- BLaIR vs. BGE: **`bge-small-en-v1.5`, settled.** BLaIR's hubness survived whitening;
  BGE didn't have the problem at all. Domain-training reasoning was good; empirical
  result went the other way.
- SPREAD-gating `CLARIFY`: **OFF, permanently, settled.** `clarify_spread_gate` defaults
  to `False`. Gated scored `technical_score=0.2949`; always-ask scored `0.6383` — a
  >2x swing. Kept as a constructor param only for reproducing the comparison.
- FUSION weights: **`FUSION_KW_WEIGHT=0.65`/`FUSION_BROWSE_WEIGHT=0.35`, settled** (Step
  9, parameter 1/3). See PROGRESS LOG for the full pre/post-fix numbers and CIs.
- `MAG_FLOOR`/`REVISE`: **`MAG_FLOOR=0.5`, `suppress_revise_turn1=False`, settled**
  (Step 9, parameter 2/3). Combining suppression with the floor change was tested and
  conclusively added nothing — don't re-add it without new evidence.

## OPEN QUESTIONS — build as configurable and report back, do not decide silently

- **`GATE_CONFIDENCE_THRESHOLD` (Step 9, parameter 3/3)** — the current concrete open
  item, calibration in progress as of this writing. See Step 9 and the PROGRESS LOG.
- Every other `# TUNE` placeholder not yet calibrated (pool top-N, `SPREAD`'s flatness
  threshold — resolved OFF as a `CLARIFY` gate but its threshold value itself is
  untested) — real unknowns, not values to guess confidently and move on from.
- The `MAG`/`SPREAD` p10/p90 reference bounds were profiled against `dev_subset.jsonl`
  (39 sessions) and have not yet been re-derived against the full 200-session
  `public_set.jsonl` — do the full-200 confirmation run (Step 9's last action) before
  treating any of Step 9's dev_subset-based choices as final.
- Whether the private 800-session grading run is one continuous process or parallelized
  across workers — materially changes how much Step 10b's bandit is worth building.
  Reasonable working assumption is "one continuous process," but this is not confirmed —
  ask if there's any channel to check.
- **New, process-level (not architecture): multiple Claude Code sessions can be
  concurrently active on this same repo, editing the same files, uncommitted.** This
  actually happened during Step 9 and corrupted two calibration runs before being
  caught (see PROGRESS LOG). If picking this project back up with another session
  potentially active, check `git status`/`git diff` on `starter/agent.py` before
  trusting a surprising result, and consider working from an isolated copy of the
  source for any multi-run calibration work rather than the shared working directory.