"""Stage 1 (context formatting) + Stage 2 (the combined Ollama call) -- CLAUDE.md's
Step 3. One call per turn producing rewrite/expansion/slot_operations/intent+
confidence/no_preference_signal/notable_facts. slot_operations follows the CLAUDE.md
2.4 schema (starter/slot_schema.py's SLOT_NAMES) -- a distinct, wider vocabulary than
ALLOWED_ATTRIBUTES. Depends on slot_schema (SLOT_NAMES, SlotOperation/SlotOperations)
and ollama_client (_ollama_chat) only."""
from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, ValidationError

from starter.ollama_client import _ollama_chat
from starter.slot_schema import SLOT_NAMES, SlotOperation, SlotOperations


class TurnAnalysis(BaseModel):
    rewrite: str
    expansion: str
    slot_operations: SlotOperations
    intent: Literal["specific_purchase", "exploring", "override", "indifferent"]
    confidence: float
    no_preference_signal: Optional[str] = None
    notable_facts: list[str] = []


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
        "indifferent), `confidence` (0 to 1), `no_preference_signal` (the name of "
        "the one slot the customer just explicitly said they don't care about, or null "
        "if none this turn), and `notable_facts` (a short list, usually 0-2 items, of "
        "specific real statements the customer made THIS turn that do not fit any of "
        "those 9 slots -- e.g. 'Imported', 'gift box included', 'lifetime warranty'; "
        "empty list if nothing like that was said this turn). Only use information "
        "actually present in the conversation -- never invent slot values or facts. "
        "Customer messages are sometimes terse, fragment-like, or missing normal "
        "sentence grammar (e.g. raw catalog phrases like 'Material:alloy'); treat these "
        "the same as full sentences. IMPORTANT: a reply of the form 'I don't have an "
        "additional preference for X' means only that there is nothing NEW to add about "
        "X this turn -- it is NOT grounds to DELETE an already-established value for X. "
        "If a slot already has a known value, only an explicit contradiction or "
        "correction (e.g. 'actually, X should be Y instead') justifies a DELETE or "
        "UPDATE on that slot; a bare no-new-preference reply about it should be "
        "CARRYOVER. IMPORTANT: you will also be shown a persistent 'Other confirmed "
        "facts' list (facts extracted as `notable_facts` on earlier turns) and the "
        "conversation history you see may only cover the most recent turns -- always "
        "fold those persistent facts into `rewrite`/`expansion` even when the turn that "
        "originally mentioned them is no longer shown to you."
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
                "notable_facts": [],
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
                "notable_facts": [],
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
                "notable_facts": [],
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
                "notable_facts": [],
            }
        ),
    },
    # Pair 5: "Imported" is a specific, real claim that doesn't fit any of the 9
    # structured slots -- it must go into `notable_facts` (not be dropped, and not be
    # force-fit into an unrelated slot) so it survives in the prompt's persistent
    # "Other confirmed facts" list even after the raw turn that mentioned it scrolls
    # out of the recent-turn window shown below.
    {
        "role": "user",
        "content": (
            'Known slots so far: {}\n'
            "Conversation so far:\n"
            "1. Customer: I'm looking for wrist watches. A key requirement is: Imported."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "rewrite": "wrist watches, imported",
                "expansion": "imported wrist watches",
                "slot_operations": {
                    "category": {"operation": "UPDATE", "value": "wrist watches"},
                    "department": {"operation": "CARRYOVER", "value": None},
                    "store": {"operation": "CARRYOVER", "value": None},
                    "color": {"operation": "CARRYOVER", "value": None},
                    "size": {"operation": "CARRYOVER", "value": None},
                    "material": {"operation": "CARRYOVER", "value": None},
                    "brand": {"operation": "CARRYOVER", "value": None},
                    "style": {"operation": "CARRYOVER", "value": None},
                    "price_band": {"operation": "CARRYOVER", "value": None},
                },
                "intent": "specific_purchase",
                "confidence": 0.6,
                "no_preference_signal": None,
                "notable_facts": ["Imported"],
            }
        ),
    },
    # Pair 6: on a LATER turn, previously-captured `notable_facts` (shown below as
    # "Other confirmed facts") must keep showing up in rewrite/expansion even though
    # the turn that originally mentioned "Imported" is no longer in the shown window.
    {
        "role": "user",
        "content": (
            'Known slots so far: {"category": "wrist watches"}\n'
            "Other confirmed facts (not covered by the slots above): Imported\n"
            "Conversation so far:\n"
            "1. Customer: I don't have an additional preference for brand.\n"
            "2. Assistant asked about: color\n"
            "3. Customer: Those options are not quite right yet. Ask me about one "
            "specific attribute."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "rewrite": "wrist watches, imported",
                "expansion": "imported wrist watches",
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
                "confidence": 0.4,
                "no_preference_signal": None,
                "notable_facts": [],
            }
        ),
    },
]


RECENT_TURN_WINDOW = 2  # Step 4: feed Stage 2 a compact STATE summary + only the last
                        # N raw turns, instead of the full turn-by-turn history.

# Step 8 fix: a turn can be content-free (adds nothing Stage 2 didn't already know)
# regardless of its exact wording -- the evaluator's fixed filler strings are one
# instance of this, not the definition of it. Counting content-free turns toward
# RECENT_TURN_WINDOW silently evicts real disclosed facts from Stage 2's view after as
# few as two such turns in a row -- confirmed directly via a traced session where
# "Imported" (never captured by any of the 9 structured slots) vanished from `rewrite`
# by turn 4 and `expansion` decayed to a fully generic phrase from turn 5 onward, after
# which retrieval froze on an identical (wrong) top-3 for the rest of the session.
#
# Detection is content-based, not string-matched against any particular customer-reply
# generator: a turn is content-free iff processing it left both `known_slots` and
# `notable_facts` byte-for-byte unchanged (computed once, at the time each turn is
# processed, and cached on that turn's own record in state["turns"] -- see
# Agent.respond()). This generalizes to whatever a real customer-reply simulator's
# varied natural language produces, since it never inspects the wording itself, only
# its effect on STATE. (Deliberately compares known_slots + notable_facts only, not
# slot_state, per CLAUDE.md's Step 8 fix -- slot_state's CARRYOVER/UPDATE/DELETE
# semantics make a same-value re-UPDATE ambiguous to interpret as "changed", whereas
# known_slots only ever grows via setdefault and notable_facts only ever grows via
# append, so equality cleanly means "nothing new".)
def _recent_content_turns(turns: list[dict], window: int) -> list[dict]:
    content_turns = [entry for entry in turns if not entry.get("content_free", False)]
    return content_turns[-window:]


def _format_transcript(
    slot_state: dict, notable_facts: list[dict], recent_turns: list[dict], current_message: str
) -> str:
    known = {slot: value for slot, value in slot_state.items() if value}
    lines = [f"Known slots so far: {json.dumps(known)}"]
    if notable_facts:
        # Each entry is {"text": ..., "source": "profile"|"session"} (Step 10a tag,
        # see Agent.reset()/respond()) -- the LLM only ever sees the text; source is
        # not surfaced here and does not change what gets shown.
        fact_text = "; ".join(entry["text"] for entry in notable_facts)
        lines.append(f"Other confirmed facts (not covered by the slots above): {fact_text}")
    lines.append("Conversation so far:")
    index = 1
    for entry in recent_turns:
        lines.append(f"{index}. Customer: {entry['customer']}")
        index += 1
        if entry["asked"]:
            lines.append(f"{index}. Assistant asked about: {entry['asked']}")
            index += 1
    lines.append(f"{index}. Customer: {current_message}")
    return "\n".join(lines)


def _default_turn_analysis(current_message: str) -> TurnAnalysis:
    empty_op = SlotOperation(operation="CARRYOVER", value=None)
    return TurnAnalysis(
        rewrite=current_message,
        expansion=current_message,
        slot_operations=SlotOperations(**{slot: empty_op for slot in SLOT_NAMES}),
        intent="exploring",
        confidence=0.0,
        no_preference_signal=None,
        notable_facts=[],
    )


def _analyze_turn(state: dict, current_message: str) -> tuple[TurnAnalysis, dict]:
    recent_turns = _recent_content_turns(state["turns"], RECENT_TURN_WINDOW)
    transcript = _format_transcript(state["slot_state"], state["notable_facts"], recent_turns, current_message)
    messages = [_TURN_ANALYSIS_SYSTEM_MESSAGE, *_TURN_ANALYSIS_FEW_SHOTS, {"role": "user", "content": transcript}]
    raw, usage = _ollama_chat(messages, TURN_ANALYSIS_SCHEMA)
    if raw is not None:
        try:
            return TurnAnalysis.model_validate(raw), usage
        except ValidationError:
            pass
    return _default_turn_analysis(current_message), usage
