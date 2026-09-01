"""Agent orchestrator. The pipeline stages themselves live in dedicated modules
(split out for readability -- see each module's own docstring for its exact
responsibility and dependencies):

  starter/text_utils.py    -- tokenization/text helpers, no internal deps
  starter/slot_schema.py   -- CLAUDE.md 2.4 slot schema, ALLOWED_ATTRIBUTES, word lists
  starter/catalog_index.py -- catalog-side metadata extraction (CLAUDE.md 2.3)
  starter/slot_state.py    -- CATCMP, free-text slot extraction, CARRYOVER/UPDATE/DELETE
  starter/ollama_client.py -- shared low-level Ollama HTTP call
  starter/stage2.py        -- Stage 1/2: context formatting + the combined LLM call
  starter/fusion.py        -- KW/BROWSE/GATE/POOL/FUSION/MAG/SPREAD
  starter/rankllm.py       -- RANKLLM final semantic re-rank
  starter/clarify.py       -- CLARIFY attribute selection (Step 8)
  starter/policy.py        -- Step 10b's epsilon-greedy CLARIFY-strategy bandit
  starter/profile_store.py -- Step 10a profile distillation (pre-existing, untouched)
  starter/embeddings.py    -- BROWSE embedding model plumbing (pre-existing, untouched)

This file keeps only the Agent class itself: construction, the catalog FTS index,
reset()'s RESETHOOK, and respond()'s per-turn wiring across all of the above.
"""
from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

from starter.catalog_index import (
    _catalog_leaf_category,
    _department_value,
    _parse_price,
    _price_bucket,
    _store_value,
)
from starter.clarify import (
    CLARIFY_UNREACHABLE_ATTRIBUTES,
    CONSTRAINT_ROUTABLE_ATTRIBUTES,
    _clarify_spread_ok,
    _select_ask_attribute,
)
from starter.embeddings import (
    apply_whitening,
    build_or_load_catalog_embeddings,
    build_or_load_whitening,
    embed_query,
    top_k_by_cosine,
)
from starter.fusion import (
    BROWSE_TOP_N,
    GATE_CONFIDENCE_THRESHOLD,
    KW_TOP_N,
    _fuse_pool,
    _mag_ok,
    _spread_is_flat,
    _spread_value,
)
from starter.policy import CLARIFY_STRATEGIES, MAX_TURNS, _select_clarify_strategy
from starter.profile_store import ProfileStore, distill_session_to_profile
from starter.rankllm import RANKLLM_CANDIDATE_LIMIT, _rank_llm
from starter.slot_schema import ALLOWED_ATTRIBUTES, SLOT_NAMES, SLOT_TO_ASK_ATTRIBUTE
from starter.slot_state import _apply_slot_operations, _update_known_slots
from starter.stage2 import _analyze_turn
from starter.text_utils import _terms, _text

# Step 9 fix: a profile's preference_tags can be a bare attribute-name word (e.g.
# "material", "style") rather than an actual value -- confirmed directly on
# public_0012, where seeding the tag "material" alone (no real material named) let
# the Stage 2 LLM fold that literal word into rewrite/expansion ("women dresses,
# material"), which doesn't discriminate anything since it names the SLOT, not a
# value for it. Excluded from the RESETHOOK personalization seed below; a genuine
# descriptive term (e.g. "polyester") still seeds normally.
#
# v1 of this fix (ALLOWED_ATTRIBUTES/SLOT_NAMES/CONSTRAINT_ROUTABLE_ATTRIBUTES
# below) catches "material" but not "fit" -- "fit" isn't literally one of those
# strings, yet behaves identically in practice: a bare dimension name, not a
# value within it (no fit-type named, same as "material" naming no fabric).
# Not a curated guess -- confirmed against evaluator/local_evaluator.py's
# classify_constraint() source, the evaluator's OWN grouping of constraint text
# into these same seven buckets. Its "style" branch is:
#     any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck"))
# i.e. classify_constraint() itself already treats "fit" as interchangeable
# with "style" -- the identical relationship CLAUDE.md 2.1 documents for
# "department" ("classify_constraint() folds the literal word 'department'
# into 'style'"), just one more word in that same literal tuple ("department"
# is already covered via SLOT_NAMES). "sizing" gets the same treatment from
# the "size" branch (`("size", "sizing", "width", "wide", "narrow")`) -- a
# grammatical variant of "size" itself; "wide"/"narrow" are genuine values and
# stay allowed. Mirrored here as a calibrated constant rather than imported
# live, same reasoning as CONSTRAINT_ROUTABLE_ATTRIBUTES: local_evaluator is
# dev-only tooling an agent submitted for real grading can't assume is
# importable.
_CLASSIFY_CONSTRAINT_NAME_ALIASES = {"fit", "sizing"}
_BARE_ATTRIBUTE_NAME_TOKENS = (
    {value.lower() for value in ALLOWED_ATTRIBUTES}
    | {value.lower() for value in SLOT_NAMES}
    | CONSTRAINT_ROUTABLE_ATTRIBUTES
    | _CLASSIFY_CONSTRAINT_NAME_ALIASES
)


class Agent:
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        erasure_mode: str = "accumulate",
        embedding_model: str = "bge",
        clarify_spread_gate: bool = False,
        suppress_revise_turn1: bool = False,
        profile_store_path: str | Path = "profiles.json",
        forced_clarify_strategy: str | None = None,
        policy_rng_seed: int | None = None,
    ) -> None:
        if erasure_mode not in ("erase", "accumulate", "gated"):
            raise ValueError('erasure_mode must be "erase", "accumulate", or "gated"')
        if embedding_model not in ("bge", "blair"):
            raise ValueError('embedding_model must be "bge" or "blair"')
        if forced_clarify_strategy is not None and forced_clarify_strategy not in CLARIFY_STRATEGIES:
            raise ValueError(f"forced_clarify_strategy must be one of {sorted(CLARIFY_STRATEGIES)} or None")
        self.erasure_mode = erasure_mode
        self.embedding_model = embedding_model
        # Step 9: SPREAD-gating CLARIFY is RESOLVED, not an open ablation any more --
        # dev_subset (39 sessions, same code otherwise): clarify_spread_gate=True
        # (SPREAD-gated) scored recommended_technical_score=0.2949 (turns_with_ask=
        # 122/320, ~38% -- badly under-asking); clarify_spread_gate=False (always-ask
        # whenever attr_candidates is non-empty) scored 0.6383 (turns_with_ask=250/282,
        # ~89%). Default is now False permanently. Kept as a constructor parameter
        # only for reproducing that comparison, not because the question is open.
        self.clarify_spread_gate = clarify_spread_gate
        # Step 9 MAG/REVISE calibration knob: when True, a turn-1 MAG-floor miss falls
        # straight through to ranking instead of attempting REVISE. Per CLAUDE.md's
        # Step 7 finding, turn 1 has almost nothing disclosed yet, so REVISE's strategy
        # (broadening rewrite->expansion terms) has little to work with there --
        # candidate for recovering the mttc cost MAG's fix introduced. Default False
        # (REVISE fires on any turn, including turn 1) until Step 9 picks a winner.
        self.suppress_revise_turn1 = suppress_revise_turn1
        self.gated_counters = {
            "delete_proposed": 0,
            "delete_downgraded": 0,
            "turns_with_delete_and_no_update": 0,
            "total_turns": 0,
        }
        self.mag_counters = {"revision_triggered": 0}
        self.call_log: list[dict] = []  # per-respond() diagnostic trail (Step 7 MAG analysis)
        # Step 10a: RESETHOOK support. `Agent()` is instantiated once for a whole
        # evaluator run (per CLAUDE.md 2.1), so profile_store and _last_session_id
        # both persist across every session in that run -- the only reason any
        # cross-session learning is possible at all.
        self.profile_store = ProfileStore(profile_store_path)
        self._last_session_id: str | None = None

        # Step 10b: POLICYSTATS. Same lifetime as gated_counters/mag_counters above --
        # set once here, persists across the whole run since Agent() is instantiated
        # once (CLAUDE.md 2.1) -- but the running sums inside it are only ever mutated
        # by RESETHOOK (see reset()), never by reset() reassigning this dict wholesale.
        #
        # Warm-started from a real measured run of each strategy against
        # data/dev_subset.jsonl (see clarify_strategy_warmstart.py, same shape as
        # erase_vs_accumulate_comparison.py) instead of starting at n=0/sum_turns=0:
        # local dev runs never carry into the real graded run, so seeding the bandit's
        # initial Q-values with this measurement is the only way that prior
        # calibration work helps at all once grading starts a fresh Agent(). Measured
        # 2026-09-01, dev_subset.jsonl (39 sessions), turns_consumed = respond() calls
        # actually made per session (first_hit_turn on a hit, MAX_TURNS on a miss):
        #   clarify_aggressive: n=39, mean turns_consumed = 8.5641
        #   clarify_sparing:    n=39, mean turns_consumed = 8.7949
        self._policy_stats: dict[str, dict[str, int]] = {
            "clarify_aggressive": {"n": 39, "sum_turns": 334},
            "clarify_sparing": {"n": 39, "sum_turns": 343},
        }
        # Seeded from real entropy by default (random.Random(None) draws from the
        # OS's urandom) so distinct Agent() instances get INDEPENDENT explore/exploit
        # sequences -- a fixed seed here previously meant every fresh Agent() (e.g.
        # one per named config in a calibration script, same pattern as
        # erase_vs_accumulate_comparison.py's per-mode Agent()) replayed an identical
        # deterministic strategy sequence regardless of what was actually being
        # calibrated. Confirmed as the root cause of a real incident: it silently
        # corrupted an unrelated peer session's MAG_FLOOR/REVISE dev_subset
        # comparison -- both configs collapsed to nearly the same score because both
        # got dealt the identical corrupted sequence, washing out the MAG signal
        # being measured. Pass an explicit int only when reproducibility across runs
        # of the SAME config is deliberately wanted (e.g. debugging one run twice).
        self._policy_rng = random.Random(policy_rng_seed)
        # Calibration/warm-start override: force every session in this Agent's whole
        # run onto one named strategy instead of letting the bandit pick. None (the
        # default) means "let epsilon-greedy choose" -- the real behavior.
        self._forced_clarify_strategy = forced_clarify_strategy

        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._catalog_meta: dict[str, dict] = {}
        self._build_index()
        raw_vectors, self._browse_ids = build_or_load_catalog_embeddings(self.catalog_path, model_key=embedding_model)
        if embedding_model == "blair":
            # blair-roberta-base needs the ZCA whitening fix (see starter/embeddings.py
            # module docstring) -- still doesn't fully resolve hub-item pollution, kept
            # available for comparison/extension, not the default path.
            self._whitening_mu, self._whitening_W = build_or_load_whitening(
                self.catalog_path, raw_vectors, model_key=embedding_model
            )
            self._browse_vectors = apply_whitening(raw_vectors, self._whitening_mu, self._whitening_W)
        else:
            self._whitening_mu, self._whitening_W = None, None
            self._browse_vectors = raw_vectors

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                batch.append(
                    (
                        parent_asin,
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                # Step 8: lightweight per-product metadata for CLARIFY's information-gain
                # scoring -- kept separate from the FTS table since it needs price/
                # department/leaf-category as discrete values, not indexed text.
                self._catalog_meta[parent_asin] = {
                    "leaf_category": _catalog_leaf_category(product.get("categories") or []),
                    "department": _department_value(product.get("details")),
                    "store": _store_value(product.get("store")),
                    "price_bucket": _price_bucket(_parse_price(product.get("price"))),
                    "text": " ".join([_text(product.get("title")), _text(product.get("features"))]),
                }
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def _lookup_products(self, asins: list[str]) -> dict[str, dict]:
        """title/features text for a candidate set, for RANKLLM's prompt."""
        if not asins:
            return {}
        placeholders = ",".join("?" for _ in asins)
        rows = self.connection.execute(
            f"SELECT parent_asin, title, features FROM products WHERE parent_asin IN ({placeholders})",
            asins,
        ).fetchall()
        return {str(row[0]): {"title": row[1], "features": row[2]} for row in rows}

    def reset(self, session_id: str, user_profile: dict) -> None:
        # RESETHOOK (Step 10a): respond() can never know whether a hit occurred --
        # that check happens entirely inside evaluate(), never reported back (see
        # CLAUDE.md 2.1). The only place cross-session learning can correctly live
        # is here, at the START of the NEXT reset() call, looking BACKWARD at the
        # session that just finished. Skip on the very first reset() of a run --
        # there is nothing yet to look back on.
        if self._last_session_id is not None:
            finished_state = self._sessions.get(self._last_session_id)
            if finished_state is not None:
                prior_profile = finished_state.get("user_profile") or {}
                distilled = distill_session_to_profile(prior_profile, finished_state)
                self.profile_store.set(self._last_session_id, distilled)

                # Step 10b RESETHOOK extension: update the just-finished session's
                # strategy stats BEFORE picking the new session's strategy below.
                # turns_consumed = how many respond() calls actually happened this
                # session -- len(state["turns"]) already equals exactly that, since
                # evaluate() stops calling respond() the turn a hit lands and runs the
                # full MAX_TURNS on a miss (see local_evaluator.py's evaluate() loop),
                # so no separate bookkeeping is needed to get "reward = -turns_consumed".
                finished_strategy = finished_state.get("clarify_strategy")
                if finished_strategy in self._policy_stats:
                    turns_consumed = len(finished_state["turns"])
                    self._policy_stats[finished_strategy]["n"] += 1
                    self._policy_stats[finished_strategy]["sum_turns"] += turns_consumed
        self._last_session_id = session_id

        # Step 10b: pick this session's clarify strategy now (epsilon-greedy over the
        # POLICYSTATS just updated above), or use the forced override if one was
        # passed to __init__ for calibration/warm-start runs.
        active_strategy = self._forced_clarify_strategy or _select_clarify_strategy(
            self._policy_stats, self._policy_rng
        )

        # The profile is anonymized and may be used for personalization.
        self._sessions[session_id] = {
            "known_slots": {},
            "asked_attributes": set(),
            "last_ask_attribute": None,
            "initial_message": None,
            "turns": [],
            "last_expansion": None,
            "last_rewrite": None,
            "slot_state": {slot: None for slot in SLOT_NAMES},
            "last_spread_flat": None,
            "last_spread_value": None,
            "no_preference_count": 0,
            "notable_facts": [],
            "user_profile": user_profile,
            "clarify_strategy": active_strategy,
            "questions_asked": 0,
        }

        # Personalization seed: fold the incoming profile's preference_tags into
        # notable_facts up front, as if they'd been disclosed on an (imaginary)
        # turn 0. This reuses the existing "Other confirmed facts" prompt slot
        # (see _format_transcript / _analyze_turn) rather than adding any new
        # selection, retrieval, or ranking logic -- CLARIFY/POOL/SPREAD/MAG code
        # itself is untouched; only the STATE they read from starts non-empty.
        #
        # Each entry is tagged {"text": ..., "source": "profile"|"session"} so a
        # profile-seeded fact stays distinguishable from one the customer actually
        # said this session (see the matching append in respond()) -- purely a
        # tag for now, no downstream logic branches on `source` yet.
        existing_texts = {entry["text"] for entry in self._sessions[session_id]["notable_facts"]}
        for tag in (user_profile or {}).get("preference_tags") or []:
            cleaned = str(tag).strip()
            if not cleaned or cleaned.lower() in _BARE_ATTRIBUTE_NAME_TOKENS:
                continue
            if cleaned not in existing_texts:
                self._sessions[session_id]["notable_facts"].append({"text": cleaned, "source": "profile"})
                existing_texts.add(cleaned)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]
        if state["initial_message"] is None:
            state["initial_message"] = user_message

        # Step 8 fix: snapshot known_slots/notable_facts before processing this turn,
        # so we can tell afterward whether this turn actually disclosed anything new --
        # see _recent_content_turns (starter/stage2.py) for why that (not string-
        # matching wording) is what determines whether a turn counts toward
        # RECENT_TURN_WINDOW.
        known_slots_before = dict(state["known_slots"])
        notable_facts_before = list(state["notable_facts"])

        _update_known_slots(state, user_message)

        analysis, usage = _analyze_turn(state, user_message)
        _apply_slot_operations(state, analysis.slot_operations, self.erasure_mode, self.gated_counters)
        state["last_expansion"] = analysis.expansion
        state["last_rewrite"] = analysis.rewrite

        # Step 8 fix: persist notable_facts (real disclosed facts with no matching
        # structured slot, e.g. "Imported") so they keep showing up in every future
        # turn's prompt via _format_transcript's "Other confirmed facts" line,
        # independent of RECENT_TURN_WINDOW -- this is what actually keeps them out of
        # rewrite/expansion once the raw turn that mentioned them scrolls out of view.
        # Step 10a tag: source="session" distinguishes these from the profile-
        # seeded entries reset() adds (source="profile") -- see that append site.
        existing_fact_texts = {entry["text"] for entry in state["notable_facts"]}
        for fact in analysis.notable_facts:
            cleaned = fact.strip()
            if cleaned and cleaned not in existing_fact_texts:
                state["notable_facts"].append({"text": cleaned, "source": "session"})
                existing_fact_texts.add(cleaned)

        content_free_turn = (
            state["known_slots"] == known_slots_before and state["notable_facts"] == notable_facts_before
        )

        # Step 8: wire no_preference_signal -- hard-exclude the slot the instant it
        # fires (permanent for the session, same as any other asked_attributes entry)
        # and count how often it happens, so CLARIFY below can soft-raise its SPREAD
        # bar after repeated signals instead of banning clarification outright.
        no_preference_fired = bool(analysis.no_preference_signal)
        if no_preference_fired:
            excluded = SLOT_TO_ASK_ATTRIBUTE.get(analysis.no_preference_signal)
            if excluded:
                state["asked_attributes"].add(excluded)
            state["no_preference_count"] += 1

        fallback_text = " ".join([state["initial_message"], *state["known_slots"].values()])
        query_text = analysis.rewrite.strip() or fallback_text
        unique_terms = list(dict.fromkeys(_terms(query_text)))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        kw_scores: dict[str, float] = {}
        if expression:
            rows = self.connection.execute(
                "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS score "
                "FROM products WHERE products MATCH ? ORDER BY score LIMIT ?",
                (expression, KW_TOP_N),
            ).fetchall()
            # sqlite's bm25(): smaller (more negative) = better match. Negate so
            # higher = better, matching BROWSE's cosine-similarity convention, before
            # FUSION min-max normalizes both tracks.
            kw_scores = {str(asin): -float(raw_score) for asin, raw_score in rows}

        # GATE: high-confidence buying turn -> trust KW alone, skip BROWSE entirely.
        # (Checked BEFORE computing browse scores: a high-confidence specific_purchase
        # read says Stage 2 is sure what the customer wants, not that BROWSE's dense
        # retrieval will find it -- discarding a strong KW hit for an unverified
        # BROWSE-only result was the confirmed cause of the Step 5 regression.)
        high_confidence_buying = analysis.intent == "specific_purchase" and analysis.confidence >= GATE_CONFIDENCE_THRESHOLD
        if high_confidence_buying:
            browse_scores: dict[str, float] = {}
        else:
            # BROWSE: dense retrieval over `expansion` (falls back to `rewrite`/
            # query_text if the model left expansion blank).
            browse_query = analysis.expansion.strip() or query_text
            if browse_query:
                query_vector = embed_query(browse_query, model_key=self.embedding_model)
                if self.embedding_model == "blair":
                    query_vector = apply_whitening(query_vector, self._whitening_mu, self._whitening_W)
                browse_scored = top_k_by_cosine(query_vector, self._browse_vectors, self._browse_ids, BROWSE_TOP_N)
                browse_scores = {asin: score for asin, score in browse_scored}
            else:
                browse_scores = {}

        # POOL/FUSION: min-max normalize each track independently, weighted-combine.
        # When GATE fired, browse_scores is empty, so this reduces to a KW-only
        # ranking (order-preserving) -- GATE controls *inputs*, not whether RANKLLM
        # gets a final semantic pass over them.
        fused = _fuse_pool(kw_scores, browse_scores, RANKLLM_CANDIDATE_LIMIT)

        # MAG: does POOL clear a minimum confidence floor? If not, REVISE once --
        # broaden the KW query to `expansion`'s looser terms (BROWSE already draws
        # from `expansion`, so only KW needs re-running) -- then fall through to
        # ranking regardless of whether the revision actually improved anything. MAG
        # gates only this one attempt; it never gates RANKLLM/TOPK below.
        # suppress_revise_turn1 (Step 9 calibration knob, see __init__): on turn 1
        # specifically, skip the REVISE attempt even on a MAG-floor miss.
        revised = False
        if not _mag_ok(fused) and not (self.suppress_revise_turn1 and turn == 1):
            revised = True
            self.mag_counters["revision_triggered"] += 1
            revised_terms = list(dict.fromkeys(_terms(analysis.expansion)))[:40]
            revised_expression = " OR ".join(f'"{term}"' for term in revised_terms)
            if revised_expression:
                rows = self.connection.execute(
                    "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS score "
                    "FROM products WHERE products MATCH ? ORDER BY score LIMIT ?",
                    (revised_expression, KW_TOP_N),
                ).fetchall()
                kw_scores = {str(asin): -float(raw_score) for asin, raw_score in rows}
                fused = _fuse_pool(kw_scores, browse_scores, RANKLLM_CANDIDATE_LIMIT)
            # No second attempt, no gating below -- forced through either way.

        # SPREAD: never gates ranking below -- only whether CLARIFY additionally fires.
        state["last_spread_value"] = _spread_value(fused)
        state["last_spread_flat"] = _spread_is_flat(fused)

        # CLARIFY (Step 8): replaces Step 1's fixed-priority rotation. Candidate set is
        # ALLOWED_ATTRIBUTES minus known_slots minus asked_attributes (including this
        # turn's no_preference_signal exclusion above) minus CLARIFY_UNREACHABLE_
        # ATTRIBUTES (category/brand -- structurally dead turns on this harness, see
        # that constant's definition above); if empty, suppress regardless of SPREAD --
        # nothing left worth asking. Otherwise only ask when SPREAD says the pool is
        # genuinely flat, and pick by information gain over POOL's fused candidates.
        # recommendations are still returned every turn either way -- that constraint
        # from Step 6/7 is unchanged below.
        attr_candidates = {
            attribute for attribute in ALLOWED_ATTRIBUTES
            if attribute not in state["known_slots"]
            and attribute not in state["asked_attributes"]
            and attribute not in CLARIFY_UNREACHABLE_ATTRIBUTES
        }
        ask_attribute: str | None = None
        spread_ok = not self.clarify_spread_gate or _clarify_spread_ok(fused, state["no_preference_count"])

        # REALIGN (Step 10b): the gate CLARIFY checks before asking -- decides ONLY
        # whether to ask, never which attribute (that stays Step 8's information-gain
        # _select_ask_attribute below). Suppresses if turns_remaining <= 2 regardless
        # of strategy (no point spending a question with too few turns left to act on
        # the answer), OR if the active strategy's per-session question cap
        # (CLARIFY_STRATEGIES[...]["max_questions"]) has already been reached.
        turns_remaining = MAX_TURNS - turn
        strategy_cap = CLARIFY_STRATEGIES[state["clarify_strategy"]]["max_questions"]
        realign_ok = turns_remaining > 2 and state["questions_asked"] < strategy_cap

        if attr_candidates and spread_ok and realign_ok:
            candidate_asins = [asin for asin, _score in fused]
            ask_attribute = _select_ask_attribute(attr_candidates, self._catalog_meta, candidate_asins)
            if ask_attribute:
                state["asked_attributes"].add(ask_attribute)
                state["questions_asked"] += 1
        state["last_ask_attribute"] = ask_attribute
        state["turns"].append({"customer": user_message, "asked": ask_attribute, "content_free": content_free_turn})

        candidate_products = self._lookup_products([asin for asin, _score in fused])
        candidates = [
            {"parent_asin": asin, "score": score, **candidate_products.get(asin, {})}
            for asin, score in fused
        ]

        # RANKLLM: final semantic re-rank over POOL's fused candidates.
        ranked_ids, rankllm_usage = _rank_llm(analysis.rewrite, analysis.expansion, candidates)
        merged_ids = ranked_ids[:top_k]
        recommendations = [{"parent_asin": asin} for asin in merged_ids]
        self.call_log.append({
            "revised": revised,
            "recommendations": list(merged_ids),
            "ask_attribute": ask_attribute,
            "no_preference_fired": no_preference_fired,
        })
        combined_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0) + rankllm_usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0) + rankllm_usage.get("completion_tokens", 0),
        }
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": combined_usage,
        }
