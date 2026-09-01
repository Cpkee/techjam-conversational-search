"""STATE management (CLAUDE.md 2.4): CATCMP (macro-tier leaf-category compare), the
Step 1/2 free-text slot-extraction heuristic, and Step 4's per-slot CARRYOVER/UPDATE/
DELETE application (including the "gated" erasure-mode extension). Depends on
text_utils (tokenization) and slot_schema (SLOT_NAMES, word lists, SlotOperations
type) only."""
from __future__ import annotations

import re

from starter.slot_schema import COLOR_WORDS, MATERIAL_WORDS, SIZE_WORDS, SLOT_NAMES, USE_CASE_WORDS, SlotOperations
from starter.text_utils import _terms

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
