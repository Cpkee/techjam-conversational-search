from __future__ import annotations

import json
import re
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ValidationError

from starter.embeddings import (
    apply_whitening,
    build_or_load_catalog_embeddings,
    build_or_load_whitening,
    embed_query,
    top_k_by_cosine,
)


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

ALLOWED_ATTRIBUTES: list[str] = [
    "category", "material", "color", "size", "style",
    "brand", "budget", "use_case", "feature", "other",
]

COLOR_WORDS = (
    "black", "white", "blue", "navy", "red", "pink", "green", "brown",
    "gray", "grey", "purple", "yellow", "orange", "beige", "tan",
    "gold", "silver", "maroon", "teal", "cream", "khaki", "multicolor",
)
MATERIAL_WORDS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "denim", "linen", "cashmere", "suede", "canvas",
    "mesh", "fleece", "velvet", "elastane", "viscose",
)
USE_CASE_WORDS = (
    "hiking", "running", "gym", "winter", "outdoor", "work", "summer",
    "travel", "everyday", "casual",
)
SIZE_WORDS = ("xxs", "xs", "small", "medium", "large", "extra large", "extra small")

COLOR_RE = re.compile(r"\bcolor\s*[:\-]?\s*([a-z]+)\b", re.I)
MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIAL_WORDS) + r")\b", re.I)
BUDGET_RE = re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)")
SIZE_TAG_RE = re.compile(r"\bsize\s*[:\-]?\s*([a-z0-9]+)\b", re.I)
SIZE_TOKEN_RE = re.compile(r"(?<![A-Za-z])(XXS|XS|S|M|L|XL|XXL|XXXL)(?![A-Za-z])")
STYLE_RE = re.compile(r"\b(style|fit|neck|sleeve|department)\s*[:\-]?\s*([a-z0-9 \-]{2,24})", re.I)
BRAND_RE = re.compile(r"\bbrand\s*[:\-]?\s*([a-z0-9&' \-]{2,30})", re.I)
CATEGORY_RE = re.compile(r"looking for ([^.,]+)", re.I)
USE_CASE_RE = re.compile(r"\b(" + "|".join(USE_CASE_WORDS) + r")\b", re.I)
NO_PREFERENCE_RE = re.compile(r"don't have|do not have|no preference|not quite right", re.I)
REPLY_VALUE_RE = re.compile(r"matters is:\s*(.+)", re.I)

# --- Stage 2 LLM call (Step 3): one Ollama call per turn producing rewrite/expansion/
# slot_operations/intent+confidence/no_preference_signal. slot_operations follows the
# CLAUDE.md 2.4 schema (category/department/store/color/size/material/brand/style/
# price_band) -- a distinct, wider vocabulary than ALLOWED_ATTRIBUTES above. Only
# `rewrite`/`expansion` are wired into behavior this step; known_slots/asked_attributes
# keep using the Step 1-2 heuristic untouched.
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_TIMEOUT = 45

SLOT_NAMES = (
    "category", "department", "store", "color", "size",
    "material", "brand", "style", "price_band",
)

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

# CATCMP: leaf-category compare (Step 4, per CLAUDE.md 2.3). 2.3's leaf-node/generic-
# bucket-fallback rule was defined over a catalog product's `categories` array; here it
# is applied to the free-text `category` slot value the LLM extracts turn to turn (no
# catalog categories array exists for live customer intent), using the last non-generic
# token as a stand-in "leaf".
GENERIC_CATEGORY_WORDS = {
    "clothing", "shoes", "jewelry", "accessories", "men", "women",
    "womens", "mens", "kids", "boys", "girls", "item", "items",
    "product", "products",
}


def _category_leaf(category_text: str) -> str:
    tokens = [token for token in _terms(category_text) if token not in GENERIC_CATEGORY_WORDS]
    if tokens:
        return tokens[-1]
    all_tokens = _terms(category_text)
    return all_tokens[-1] if all_tokens else ""


def catcmp(previous_category: str | None, new_category: str | None) -> bool:
    """True when new_category is a different macro-tier (leaf) category than
    previous_category -- signals the kind of conflict that 2.4 says "drives a full
    session-state reset"."""
    if not previous_category or not new_category:
        return False
    prev_leaf = _category_leaf(previous_category)
    new_leaf = _category_leaf(new_category)
    if not prev_leaf or not new_leaf:
        return False
    return prev_leaf != new_leaf


class SlotOperation(BaseModel):
    operation: Literal["CARRYOVER", "UPDATE", "DELETE"]
    value: Optional[str] = None


class SlotOperations(BaseModel):
    category: SlotOperation
    department: SlotOperation
    store: SlotOperation
    color: SlotOperation
    size: SlotOperation
    material: SlotOperation
    brand: SlotOperation
    style: SlotOperation
    price_band: SlotOperation


class TurnAnalysis(BaseModel):
    rewrite: str
    expansion: str
    slot_operations: SlotOperations
    intent: Literal["specific_purchase", "exploring", "override", "indifferent"]
    confidence: float
    no_preference_signal: Optional[str] = None


TURN_ANALYSIS_SCHEMA = TurnAnalysis.model_json_schema()

_TURN_ANALYSIS_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are the turn-analysis stage of a shopping copilot. Given the customer's "
        "conversation so far in an online store, respond with exactly one JSON object "
        "with five fields: `rewrite` (a clean, complete restatement of everything the "
        "customer currently wants, suitable as a keyword search query), `expansion` "
        "(a broader/looser paraphrase of the same intent for a semantic search "
        "fallback), `slot_operations` (for every one of these slots: category, "
        "department, store, color, size, material, brand, style, price_band -- decide "
        "CARRYOVER (keep any previously known value untouched), UPDATE (set/replace the "
        "value using something in the latest message), or DELETE (the customer no "
        "longer wants any value for this slot); only set `value` when the operation is "
        "UPDATE), `intent` (one of specific_purchase / exploring / override / "
        "indifferent), `confidence` (0 to 1), and `no_preference_signal` (the name of "
        "the one slot the customer just explicitly said they don't care about, or null "
        "if none this turn). Only use information actually present in the conversation "
        "-- never invent slot values. Customer messages are sometimes terse, "
        "fragment-like, or missing normal sentence grammar (e.g. raw catalog phrases "
        "like 'Material:alloy'); treat these the same as full sentences. IMPORTANT: a "
        "reply of the form 'I don't have an additional preference for X' means only "
        "that there is nothing NEW to add about X this turn -- it is NOT grounds to "
        "DELETE an already-established value for X. If a slot already has a known "
        "value, only an explicit contradiction or correction (e.g. 'actually, X should "
        "be Y instead') justifies a DELETE or UPDATE on that slot; a bare no-new-"
        "preference reply about it should be CARRYOVER."
    ),
}

# Few-shot pair 1 (accumulation language: "also", "size 10") vs pair 2 (override
# language: "actually", "instead") -- required contrast. Pair 3 is a raw, ungrammatical
# fragment input.
_TURN_ANALYSIS_FEW_SHOTS = [
    {
        "role": "user",
        "content": (
            'Known slots so far: {"category": "running shoes", "color": "black"}\n'
            "Conversation so far:\n"
            "1. Customer: I'm looking for running shoes. A key requirement is: black.\n"
            "2. Assistant asked about: size\n"
            "3. Customer: Also, I need it in size 10, and it should be waterproof too."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "rewrite": "running shoes, black, size 10, waterproof",
                "expansion": "black waterproof athletic running shoes for size 10 feet",
                "slot_operations": {
                    "category": {"operation": "CARRYOVER", "value": None},
                    "department": {"operation": "CARRYOVER", "value": None},
                    "store": {"operation": "CARRYOVER", "value": None},
                    "color": {"operation": "CARRYOVER", "value": None},
                    "size": {"operation": "UPDATE", "value": "10"},
                    "material": {"operation": "CARRYOVER", "value": None},
                    "brand": {"operation": "CARRYOVER", "value": None},
                    "style": {"operation": "UPDATE", "value": "waterproof"},
                    "price_band": {"operation": "CARRYOVER", "value": None},
                },
                "intent": "specific_purchase",
                "confidence": 0.85,
                "no_preference_signal": None,
            }
        ),
    },
    {
        "role": "user",
        "content": (
            'Known slots so far: {"category": "running shoes", "color": "black", "size": "10"}\n'
            "Conversation so far:\n"
            "1. Customer: I'm looking for running shoes. A key requirement is: black.\n"
            "2. Assistant asked about: size\n"
            "3. Customer: Size 10 please.\n"
            "4. Assistant asked about: material\n"
            "5. Customer: Actually, ignore my earlier preference. What I need is: "
            "leather boots instead."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "rewrite": "leather boots, size 10",
                "expansion": "leather boots or shoes in size 10",
                "slot_operations": {
                    "category": {"operation": "UPDATE", "value": "boots"},
                    "department": {"operation": "CARRYOVER", "value": None},
                    "store": {"operation": "CARRYOVER", "value": None},
                    "color": {"operation": "DELETE", "value": None},
                    "size": {"operation": "CARRYOVER", "value": None},
                    "material": {"operation": "UPDATE", "value": "leather"},
                    "brand": {"operation": "CARRYOVER", "value": None},
                    "style": {"operation": "CARRYOVER", "value": None},
                    "price_band": {"operation": "CARRYOVER", "value": None},
                },
                "intent": "override",
                "confidence": 0.9,
                "no_preference_signal": None,
            }
        ),
    },
    {
        "role": "user",
        "content": (
            'Known slots so far: {"category": "belts"}\n'
            "Conversation so far:\n"
            "1. Customer: I'm looking for belts, but I'm still exploring.\n"
            "2. Assistant asked about: material\n"
            "3. Customer: Material:alloy. Buckle closure. No preference on color."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "rewrite": "belts, alloy buckle closure",
                "expansion": "belts with metal alloy buckle hardware",
                "slot_operations": {
                    "category": {"operation": "CARRYOVER", "value": None},
                    "department": {"operation": "CARRYOVER", "value": None},
                    "store": {"operation": "CARRYOVER", "value": None},
                    "color": {"operation": "DELETE", "value": None},
                    "size": {"operation": "CARRYOVER", "value": None},
                    "material": {"operation": "UPDATE", "value": "alloy"},
                    "brand": {"operation": "CARRYOVER", "value": None},
                    "style": {"operation": "UPDATE", "value": "buckle closure"},
                    "price_band": {"operation": "CARRYOVER", "value": None},
                },
                "intent": "specific_purchase",
                "confidence": 0.6,
                "no_preference_signal": "color",
            }
        ),
    },
    # Pair 4: "no additional preference for X" on an ALREADY-KNOWN slot must CARRYOVER,
    # not DELETE -- it means "nothing new to add", not "remove what we already know".
    {
        "role": "user",
        "content": (
            'Known slots so far: {"category": "basketball", "material": "polyester", "style": "men"}\n'
            "Conversation so far:\n"
            "1. Customer: I'm looking for basketball. A key requirement is: polyester.\n"
            "2. Assistant asked about: style\n"
            "3. Customer: I don't have an additional preference for style."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "rewrite": "basketball, polyester, men",
                "expansion": "men's basketball equipment made of polyester",
                "slot_operations": {
                    "category": {"operation": "CARRYOVER", "value": None},
                    "department": {"operation": "CARRYOVER", "value": None},
                    "store": {"operation": "CARRYOVER", "value": None},
                    "color": {"operation": "CARRYOVER", "value": None},
                    "size": {"operation": "CARRYOVER", "value": None},
                    "material": {"operation": "CARRYOVER", "value": None},
                    "brand": {"operation": "CARRYOVER", "value": None},
                    "style": {"operation": "CARRYOVER", "value": None},
                    "price_band": {"operation": "CARRYOVER", "value": None},
                },
                "intent": "exploring",
                "confidence": 0.5,
                "no_preference_signal": "style",
            }
        ),
    },
]


RECENT_TURN_WINDOW = 2  # Step 4: feed Stage 2 a compact STATE summary + only the last
                        # N raw turns, instead of the full turn-by-turn history.


def _format_transcript(slot_state: dict, recent_turns: list[dict], current_message: str) -> str:
    known = {slot: value for slot, value in slot_state.items() if value}
    lines = [f"Known slots so far: {json.dumps(known)}", "Conversation so far:"]
    index = 1
    for entry in recent_turns:
        lines.append(f"{index}. Customer: {entry['customer']}")
        index += 1
        if entry["asked"]:
            lines.append(f"{index}. Assistant asked about: {entry['asked']}")
            index += 1
    lines.append(f"{index}. Customer: {current_message}")
    return "\n".join(lines)


def _ollama_chat(messages: list[dict], schema: dict) -> tuple[dict | None, dict]:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "format": schema,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0},
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None, {"prompt_tokens": 0, "completion_tokens": 0}
    usage = {
        "prompt_tokens": int(body.get("prompt_eval_count") or 0),
        "completion_tokens": int(body.get("eval_count") or 0),
    }
    content = body.get("message", {}).get("content")
    if not content:
        return None, usage
    try:
        return json.loads(content), usage
    except json.JSONDecodeError:
        return None, usage


def _default_turn_analysis(current_message: str) -> TurnAnalysis:
    empty_op = SlotOperation(operation="CARRYOVER", value=None)
    return TurnAnalysis(
        rewrite=current_message,
        expansion=current_message,
        slot_operations=SlotOperations(**{slot: empty_op for slot in SLOT_NAMES}),
        intent="exploring",
        confidence=0.0,
        no_preference_signal=None,
    )


def _analyze_turn(state: dict, current_message: str) -> tuple[TurnAnalysis, dict]:
    recent_turns = state["turns"][-RECENT_TURN_WINDOW:]
    transcript = _format_transcript(state["slot_state"], recent_turns, current_message)
    messages = [_TURN_ANALYSIS_SYSTEM_MESSAGE, *_TURN_ANALYSIS_FEW_SHOTS, {"role": "user", "content": transcript}]
    raw, usage = _ollama_chat(messages, TURN_ANALYSIS_SCHEMA)
    if raw is not None:
        try:
            return TurnAnalysis.model_validate(raw), usage
        except ValidationError:
            pass
    return _default_turn_analysis(current_message), usage


def _apply_slot_operations(
    state: dict,
    slot_operations: SlotOperations,
    erasure_mode: str,
    gated_counters: dict | None = None,
) -> None:
    """Apply Stage 2's per-slot CARRYOVER/UPDATE/DELETE decisions to state["slot_state"]
    (the CLAUDE.md 2.4 9-slot STATE).

    MACRO tier: CATCMP gates a category conflict. In "erase" mode that forces a full
    state reset (2.4: category "drives a full session-state reset"); in "accumulate"
    and "gated" modes no forced wipe happens -- every slot, including category, is
    governed purely by its own operation. This MACRO-tier behavior is identical across
    "accumulate" and "gated" (both simply skip the erase-only branch below) and is
    untouched by the "gated" extension.

    MICRO tier: "gated" (an extension alongside "erase"/"accumulate") only honors a
    DELETE this turn if AT LEAST ONE OTHER slot carries a genuine UPDATE in that SAME
    turn's slot_operations output -- i.e. the customer is actively revising something,
    not just dropping a preference in isolation. (The original rule required the SAME
    slot to carry both DELETE and UPDATE, which our schema makes structurally
    impossible -- exactly one operation per slot per turn -- so every DELETE silently
    downgraded to CARRYOVER regardless of the rest of the turn. Since a slot proposing
    DELETE can never itself be the slot with a genuine UPDATE, checking for a genuine
    UPDATE anywhere across the turn's 9 slots is equivalent to checking "at least one
    OTHER slot".) If the turn has zero genuine UPDATEs anywhere, every DELETE proposed
    that turn downgrades to CARRYOVER instead. gated_counters (if given) tallies total
    DELETEs proposed vs. how many were downgraded under this corrected condition, plus
    how many TURNS (not slot-ops) had >=1 DELETE proposed with zero UPDATEs anywhere --
    i.e. how often this rule actually has anything to engage with."""
    slot_state = state["slot_state"]
    if gated_counters is not None and erasure_mode == "gated":
        gated_counters["total_turns"] += 1

    category_op = slot_operations.category
    if category_op.operation == "UPDATE" and category_op.value:
        conflict = catcmp(slot_state.get("category"), category_op.value)
        if conflict and erasure_mode == "erase":
            for slot in SLOT_NAMES:
                slot_state[slot] = None

    turn_has_genuine_update = any(
        getattr(slot_operations, slot).operation == "UPDATE" and getattr(slot_operations, slot).value
        for slot in SLOT_NAMES
    )
    turn_has_delete = any(getattr(slot_operations, slot).operation == "DELETE" for slot in SLOT_NAMES)
    if gated_counters is not None and erasure_mode == "gated" and turn_has_delete and not turn_has_genuine_update:
        gated_counters["turns_with_delete_and_no_update"] += 1

    for slot in SLOT_NAMES:
        op = getattr(slot_operations, slot)
        operation = op.operation
        if operation == "DELETE" and erasure_mode == "gated":
            if gated_counters is not None:
                gated_counters["delete_proposed"] += 1
            if not turn_has_genuine_update:
                if gated_counters is not None:
                    gated_counters["delete_downgraded"] += 1
                operation = "CARRYOVER"
        if operation == "UPDATE" and op.value:
            slot_state[slot] = op.value
        elif operation == "DELETE":
            slot_state[slot] = None
        # CARRYOVER: leave the existing value untouched.


# --- Step 6: real FUSION (replaces Step 5's round-robin _merge_pool placeholder).
FUSION_KW_WEIGHT = 0.5  # TUNE: calibration is Step 9, not here
FUSION_BROWSE_WEIGHT = 0.5  # TUNE


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normalizes one track's scores to [0,1] independently -- KW (BM25) and BROWSE
    (cosine similarity) are on different, non-comparable scales, so combining raw
    scores directly would silently let whichever track happens to have larger raw
    magnitudes dominate FUSION regardless of the 0.5/0.5 weights."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {asin: 1.0 for asin in scores}
    return {asin: (value - lo) / (hi - lo) for asin, value in scores.items()}


def _fuse_pool(kw_scores: dict[str, float], browse_scores: dict[str, float], top_k: int) -> list[tuple[str, float]]:
    """POOL (Step 6): each track already truncated to its own top-N (KW_TOP_N/
    BROWSE_TOP_N) by the caller before scores reach here. Min-max normalize each track
    independently, then combine via weighted FUSION (placeholder equal weights).
    Returns (parent_asin, fused_score) pairs, best first."""
    kw_norm = _min_max_normalize(kw_scores)
    browse_norm = _min_max_normalize(browse_scores)
    all_ids = set(kw_norm) | set(browse_norm)
    fused = {
        asin: FUSION_KW_WEIGHT * kw_norm.get(asin, 0.0) + FUSION_BROWSE_WEIGHT * browse_norm.get(asin, 0.0)
        for asin in all_ids
    }
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


# --- Step 6: RANKLLM -- one more Ollama call, takes POOL's fused candidates (with
# their fusion score and title/features text) plus the turn's query context, and
# produces the final top-10 ranking using semantic judgment on top of the fusion prior.
RANKLLM_CANDIDATE_LIMIT = 20  # TUNE: how many fused POOL candidates get sent to RANKLLM


class RankLLMOutput(BaseModel):
    ranking: list[str]


RANKLLM_SCHEMA = RankLLMOutput.model_json_schema()

_RANKLLM_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are the final ranking stage of a shopping copilot. You are given the "
        "customer's current search intent and a pooled list of candidate products, "
        "each already carrying a fusion_score that blends keyword-match and semantic-"
        "similarity signals. Re-rank these candidates using fusion_score as a "
        "reasonable prior, adjusted by your own judgment of how well each candidate's "
        "title and features actually match what the customer wants. Respond with "
        "exactly one JSON object: `ranking`, a list of up to 10 parent_asin strings, "
        "best match first, drawn ONLY from the candidate list given below -- never "
        "invent an id that is not present in the candidates, and never return more "
        "candidates than were given to you."
    ),
}


def _format_rankllm_prompt(rewrite: str, expansion: str, candidates: list[dict]) -> str:
    lines = [f"Customer wants: {rewrite}", f"Broader intent: {expansion}", "Candidates:"]
    for candidate in candidates:
        title = (candidate.get("title") or "(no title)").strip()
        features = (candidate.get("features") or "").strip()
        line = f"- {candidate['parent_asin']} [fusion_score={candidate['score']:.3f}] {title} | {features}"
        lines.append(line[:300])
    return "\n".join(lines)


def _rank_llm(rewrite: str, expansion: str, candidates: list[dict]) -> tuple[list[str], dict]:
    """Returns (ranked parent_asins, usage). Falls back to the incoming fusion order
    (i.e. no reranking) on any call failure, invalid/empty output, or a returned id
    that isn't in the candidate set -- RANKLLM only ever reorders/truncates candidates
    POOL already vetted, never invents recommendations of its own."""
    if not candidates:
        return [], {"prompt_tokens": 0, "completion_tokens": 0}
    fallback_order = [candidate["parent_asin"] for candidate in candidates]
    prompt = _format_rankllm_prompt(rewrite, expansion, candidates)
    messages = [_RANKLLM_SYSTEM_MESSAGE, {"role": "user", "content": prompt}]
    raw, usage = _ollama_chat(messages, RANKLLM_SCHEMA)
    if raw is None:
        return fallback_order, usage
    try:
        parsed = RankLLMOutput.model_validate(raw)
    except ValidationError:
        return fallback_order, usage
    valid_ids = {candidate["parent_asin"] for candidate in candidates}
    ranking = [asin for asin in parsed.ranking if asin in valid_ids]
    if not ranking:
        return fallback_order, usage
    # Append any candidates RANKLLM omitted, in their original fusion order, so a
    # partial/short LLM ranking still fills out to the full candidate list.
    seen = set(ranking)
    ranking.extend(asin for asin in fallback_order if asin not in seen)
    return ranking, usage


# --- Step 7: MAG (magnitude), SPREAD (flatness), REVISE (one capped query revision).
# Critical constraint: MAG only gates whether REVISE is attempted -- RANKLLM/TOPK below
# always runs, on the first check or forced through after the one allowed revision,
# regardless of outcome. SPREAD is a separate, parallel signal and never gates ranking
# either.
MAG_FLOOR = 0.3  # TUNE: placeholder magnitude floor on POOL's top fused score
SPREAD_FLATNESS_THRESHOLD = 0.05  # TUNE: placeholder flatness cutoff on top-N fused scores


def _mag_ok(fused: list[tuple[str, float]]) -> bool:
    """MAG: does POOL's top fused score clear a minimum confidence floor?"""
    if not fused:
        return False
    return fused[0][1] >= MAG_FLOOR


def _spread_is_flat(fused: list[tuple[str, float]], top_n: int = 5) -> bool:
    """SPREAD: are the top-N fused scores too close together to confidently favor one
    candidate over the rest? A separate, parallel signal from MAG -- computed and
    stored here (state["last_spread_flat"]) for Step 8's CLARIFY redesign to consume;
    Step 8 explicitly owns replacing the Step 1 ask_attribute rotation with an
    information-gain selection that SPREAD feeds into, so this step computes the
    signal without changing which attribute gets asked or whether asking happens."""
    top_scores = [score for _asin, score in fused[:top_n]]
    if len(top_scores) < 2:
        return False
    return (max(top_scores) - min(top_scores)) <= SPREAD_FLATNESS_THRESHOLD


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _extract_slots(message: str) -> dict[str, str]:
    slots: dict[str, str] = {}
    lowered = message.lower()

    category_match = CATEGORY_RE.search(message)
    if category_match:
        slots["category"] = category_match.group(1).strip()

    material_match = MATERIAL_RE.search(message)
    if material_match:
        slots["material"] = material_match.group(1).lower()

    color_match = COLOR_RE.search(message)
    if color_match:
        slots["color"] = color_match.group(1).lower()
    else:
        for word in COLOR_WORDS:
            if re.search(rf"\b{word}\b", lowered):
                slots["color"] = word
                break

    size_match = SIZE_TAG_RE.search(message) or SIZE_TOKEN_RE.search(message)
    if size_match:
        slots["size"] = size_match.group(1)
    else:
        for word in SIZE_WORDS:
            if word in lowered:
                slots["size"] = word
                break

    style_match = STYLE_RE.search(message)
    if style_match:
        slots["style"] = style_match.group(2).strip()

    brand_match = BRAND_RE.search(message)
    if brand_match:
        slots["brand"] = brand_match.group(1).strip()

    budget_match = BUDGET_RE.search(message)
    if budget_match:
        slots["budget"] = budget_match.group(1).replace(",", "")

    use_case_match = USE_CASE_RE.search(message)
    if use_case_match:
        slots["use_case"] = use_case_match.group(1).lower()

    return slots


def _update_known_slots(state: dict, message: str) -> None:
    known = state["known_slots"]
    for attribute, value in _extract_slots(message).items():
        known.setdefault(attribute, value)

    last_asked = state["last_ask_attribute"]
    if last_asked and last_asked not in known and not NO_PREFERENCE_RE.search(message):
        marker = REPLY_VALUE_RE.search(message)
        fallback_value = marker.group(1) if marker else message
        fallback_value = fallback_value.strip().rstrip(".")
        if fallback_value:
            known[last_asked] = fallback_value


class Agent:
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        erasure_mode: str = "accumulate",
        embedding_model: str = "bge",
    ) -> None:
        if erasure_mode not in ("erase", "accumulate", "gated"):
            raise ValueError('erasure_mode must be "erase", "accumulate", or "gated"')
        if embedding_model not in ("bge", "blair"):
            raise ValueError('embedding_model must be "bge" or "blair"')
        self.erasure_mode = erasure_mode
        self.embedding_model = embedding_model
        self.gated_counters = {
            "delete_proposed": 0,
            "delete_downgraded": 0,
            "turns_with_delete_and_no_update": 0,
            "total_turns": 0,
        }
        self.mag_counters = {"revision_triggered": 0}
        self.call_log: list[dict] = []  # per-respond() diagnostic trail (Step 7 MAG analysis)
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
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
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
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
        }

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
        _update_known_slots(state, user_message)

        ask_attribute: str | None = None
        for attribute in ALLOWED_ATTRIBUTES:
            if attribute not in state["known_slots"] and attribute not in state["asked_attributes"]:
                ask_attribute = attribute
                state["asked_attributes"].add(attribute)
                break
        state["last_ask_attribute"] = ask_attribute

        analysis, usage = _analyze_turn(state, user_message)
        _apply_slot_operations(state, analysis.slot_operations, self.erasure_mode, self.gated_counters)
        state["turns"].append({"customer": user_message, "asked": ask_attribute})
        state["last_expansion"] = analysis.expansion
        state["last_rewrite"] = analysis.rewrite

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
        revised = False
        if not _mag_ok(fused):
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

        # SPREAD: computed + stored for Step 8's CLARIFY redesign to consume; never
        # gates ranking, and doesn't change ask_attribute selection in this step.
        state["last_spread_flat"] = _spread_is_flat(fused)

        candidate_products = self._lookup_products([asin for asin, _score in fused])
        candidates = [
            {"parent_asin": asin, "score": score, **candidate_products.get(asin, {})}
            for asin, score in fused
        ]

        # RANKLLM: final semantic re-rank over POOL's fused candidates.
        ranked_ids, rankllm_usage = _rank_llm(analysis.rewrite, analysis.expansion, candidates)
        merged_ids = ranked_ids[:top_k]
        recommendations = [{"parent_asin": asin} for asin in merged_ids]
        self.call_log.append({"revised": revised, "recommendations": list(merged_ids)})
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
