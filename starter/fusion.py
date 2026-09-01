"""POOL/FUSION (Step 6/7) and MAG/SPREAD (Step 7): merges the KW (BM25) and BROWSE
(dense) tracks computed in Agent.respond() into one ranked candidate list, and the two
signals (MAG magnitude, SPREAD flatness) read off that merged pool. No dependency on
any other starter module -- pure numeric/dict operations over caller-supplied scores.

Critical constraint, preserved from the original design: MAG only ever gates whether
REVISE is attempted in Agent.respond() -- RANKLLM/TOPK always runs afterward regardless
of outcome. SPREAD is a separate, parallel signal that never gates ranking either; it
only feeds CLARIFY's decision (see starter/clarify.py's _clarify_spread_ok)."""
from __future__ import annotations

# --- Step 5: BROWSE (dense retrieval) + minimal POOL merge with the existing KW (BM25)
# track. Full normalize/truncate/weighted FUSION/RANKLLM is Step 6 -- this is a simple,
# clearly-provisional round-robin merge to wire the two tracks together for now.
KW_TOP_N = 20  # TUNE: KW candidates pulled before merging
BROWSE_TOP_N = 20  # TUNE: BROWSE candidates pulled before merging

# GATE: on a high-confidence buying turn (Stage 2's own intent+confidence output),
# trust KW alone and skip BROWSE entirely -- a confident specific_purchase read means
# Stage 2 is sure what the customer wants, not that BROWSE's dense retrieval will find
# it; discarding a strong KW hit for an unverified BROWSE-only result was the confirmed
# cause of the original Step 5 regression (see git history around Step 5/6).
GATE_CONFIDENCE_THRESHOLD = 0.75  # TUNE

# --- Step 6/7: real FUSION (replaces Step 5's round-robin _merge_pool placeholder).
# Step 9 calibration: named-config comparison on dev_subset.jsonl (39 sessions,
# paired bootstrap, current code as of the material/fit seeding fix) --
# 0.5/0.5 scored hit_rate=0.7692/mrr=0.4697/technical_score=0.5968; 0.65/0.35
# (KW-favored) scored hit_rate=0.8205/mrr=0.5159/technical_score=0.6389; 0.35/0.65
# (BROWSE-favored) scored hit_rate=0.8205/mrr=0.5078/technical_score=0.6365. Per
# CLAUDE.md's own standard, a CI including zero is noise, not a finding -- on the
# designated decision metric (hit_rate_at_10/mrr), none of the three were
# statistically distinguishable at n=39 (all three pairwise CIs included zero).
# KW-heavy's full technical_score CI barely excluded zero ([+0.0029, +0.0974]),
# a weak but real signal even though technical_score wasn't the specified gating
# metric for this comparison. Chosen as the new default on that basis. Kept as
# module constants (not constructor args) so they stay the easy-to-revisit knob
# CLAUDE.md 2.5 calls for if the picture changes at full 200-session scale.
FUSION_KW_WEIGHT = 0.65  # TUNE: Step 9 calibrated default, see comment above
FUSION_BROWSE_WEIGHT = 0.35  # TUNE

# Step 7 fix: per-turn min-max normalization made a track's own #1 candidate always
# normalize to exactly 1.0, *regardless of underlying match quality* -- a weak query
# with a mediocre best score and a strong query with an excellent best score both
# collapsed to the same post-normalization ceiling, making MAG_FLOOR structurally
# unable to tell them apart (confirmed: the fused top score could never fall below
# max(FUSION_KW_WEIGHT, FUSION_BROWSE_WEIGHT)=0.5 whenever either track was non-empty).
# Fix: normalize each track against a FIXED reference range instead of that turn's own
# min/max, so a genuinely weak raw score actually normalizes low. Reference bounds
# below are the 10th/90th percentile of each track's per-query top score, profiled
# directly from dev_subset.jsonl (39 turn-1 queries, no LLM needed -- KW via BM25,
# BROWSE via BGE cosine): KW top-score p10=14.2, p90=36.9 (min=12.2, median=22.3,
# max=119.5 -- one outlier query, percentiles avoid letting it distort the range);
# BROWSE top-score p10=0.655, p90=0.819 (min=0.619, median=0.770, max=0.856).
# Percentile-of-top-score-per-query (not a pooled all-candidate range) was chosen
# because MAG only ever inspects the TOP fused score, so the reference range should
# characterize "what does a typical best candidate look like," not "what does a
# typical candidate at any rank look like." # TUNE: all four bounds are placeholders
# from this one profiling pass on 39 samples, not a calibrated final choice.
KW_REFERENCE_LO = 14.0  # TUNE
KW_REFERENCE_HI = 37.0  # TUNE
BROWSE_REFERENCE_LO = 0.65  # TUNE
BROWSE_REFERENCE_HI = 0.82  # TUNE


def _normalize_against_reference(scores: dict[str, float], lo: float, hi: float) -> dict[str, float]:
    """Normalizes one track's raw scores against a FIXED reference range (not that
    turn's own min/max), clipped to [0,1] -- a score at or below `lo` normalizes to 0,
    at or above `hi` normalizes to 1, so genuinely weak and genuinely strong retrieval
    produce meaningfully different fused scores instead of both hitting the same
    per-turn ceiling."""
    span = hi - lo
    return {asin: max(0.0, min(1.0, (value - lo) / span)) for asin, value in scores.items()}


def _fuse_pool(kw_scores: dict[str, float], browse_scores: dict[str, float], top_k: int) -> list[tuple[str, float]]:
    """POOL (Step 6/7): each track already truncated to its own top-N (KW_TOP_N/
    BROWSE_TOP_N) by the caller before scores reach here. Normalize each track against
    its own fixed reference range (see above), then combine via weighted FUSION
    (placeholder equal weights). Returns (parent_asin, fused_score) pairs, best first."""
    kw_norm = _normalize_against_reference(kw_scores, KW_REFERENCE_LO, KW_REFERENCE_HI)
    browse_norm = _normalize_against_reference(browse_scores, BROWSE_REFERENCE_LO, BROWSE_REFERENCE_HI)
    all_ids = set(kw_norm) | set(browse_norm)
    fused = {
        asin: FUSION_KW_WEIGHT * kw_norm.get(asin, 0.0) + FUSION_BROWSE_WEIGHT * browse_norm.get(asin, 0.0)
        for asin in all_ids
    }
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


# --- Step 7: MAG (magnitude), SPREAD (flatness), REVISE (one capped query revision).
# Critical constraint: MAG only gates whether REVISE is attempted -- RANKLLM/TOPK below
# always runs, on the first check or forced through after the one allowed revision,
# regardless of outcome. SPREAD is a separate, parallel signal and never gates ranking
# either.
# Step 9 calibration: named-config comparison on dev_subset.jsonl (39 sessions,
# paired bootstrap, on top of FUSION_KW_WEIGHT=0.65 above), all vs. the
# 0.3/no-suppression baseline (technical_score=0.6389): floor=0.5 alone ->
# 0.6717 (hit_rate=0.8462/mrr=0.5520); suppress_revise_turn1 alone (floor stays
# 0.3) -> 0.6562 (hit_rate=0.8205/mrr=0.5496); floor=0.5 + suppression combined
# -> 0.6710 (hit_rate=0.8462/mrr=0.5513). All three point estimates beat baseline
# on every metric, but none individually cleared a 95% CI excluding zero at
# n=39. The one thing that WAS conclusive: floor=0.5 alone vs. the combined
# config are statistically indistinguishable from each other (diff +0.0007, CI
# [-0.0160, +0.0182]) -- turn-1 REVISE suppression adds no measurable benefit
# once the floor is already raised, so it's not worth the added parameter.
# Chose floor=0.5 alone: best point estimate, CI closest to significance
# ([-0.0060, +0.0931]), and no evidence the simpler suppress_revise_turn1=False
# default is actually costing anything.
MAG_FLOOR = 0.5  # TUNE: Step 9 calibrated default, see comment above
# TUNE: placeholder flatness cutoff on top-N fused scores. Recalibrated from a
# turn-1-only, LLM-free profiling pass over dev_subset.jsonl's 39 sessions (same style
# as MAG's p10/p90 fix): min=0.0000, p10=0.0089, p25=0.0442, p50=0.0688, p60=0.0961,
# p75=0.1405, p90=0.2291, max=0.3614, mean=0.1043. The original 0.05 sat at only ~p28
# even on this best-case (turn-1) profiling, and the real per-turn ask rate it produced
# collapsed to 17.6% (vs. ~94% before CLARIFY was gated on SPREAD at all) -- confirmed
# via a full dev_subset run, recommended_technical_score 0.6209->0.1019. 0.10 (~p61,
# ~62% turn-1 ask rate) is a sane starting point given a skipped turn now carries real
# downside risk (see the notable_facts/RECENT_TURN_WINDOW fix above), not Step 9's real
# calibration -- re-derive against the full 200-session public_set.jsonl, and note this
# was profiled from turn-1 queries only, so later-turn spread was not directly measured.
SPREAD_FLATNESS_THRESHOLD = 0.10


def _mag_ok(fused: list[tuple[str, float]]) -> bool:
    """MAG: does POOL's top fused score clear a minimum confidence floor?"""
    if not fused:
        return False
    return fused[0][1] >= MAG_FLOOR


def _spread_value(fused: list[tuple[str, float]], top_n: int = 5) -> float:
    """Raw SPREAD magnitude (max-min of the top-N fused scores), factored out of
    `_spread_is_flat` purely so callers can inspect/log the actual number for
    calibration diagnostics -- does not change any behavior."""
    top_scores = [score for _asin, score in fused[:top_n]]
    if len(top_scores) < 2:
        return 0.0
    return max(top_scores) - min(top_scores)


def _spread_is_flat(
    fused: list[tuple[str, float]], top_n: int = 5, threshold: float = SPREAD_FLATNESS_THRESHOLD
) -> bool:
    """SPREAD: are the top-N fused scores too close together to confidently favor one
    candidate over the rest? A separate, parallel signal from MAG. `threshold` defaults
    to the module placeholder but Step 8's CLARIFY gate can pass a stricter value (see
    `_clarify_spread_ok` in starter/clarify.py) to soft-raise the bar after repeated
    no_preference_signal fires this session."""
    if len(fused[:top_n]) < 2:
        return False
    return _spread_value(fused, top_n) <= threshold
