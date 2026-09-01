"""CLAUDE.md 2.4's slot schema: the closed ALLOWED_ATTRIBUTES vocabulary (what
`ask_attribute` is allowed to be), the wider 9-slot SLOT_NAMES vocabulary Stage 2's
`slot_operations` field actually operates over, the pydantic shape of a single
CARRYOVER/UPDATE/DELETE decision, the catalog-signal word lists CLARIFY and the
free-text slot-extraction heuristic both key off of, and the mapping that bridges
Stage 2's wider vocabulary down to ALLOWED_ATTRIBUTES. No dependency on any other
starter module (pydantic only) -- this is pure declarative schema data, imported by
slot_state.py, stage2.py, clarify.py, and agent.py's orchestrator."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

ALLOWED_ATTRIBUTES: list[str] = [
    "category", "material", "color", "size", "style",
    "brand", "budget", "use_case", "feature", "other",
]

SLOT_NAMES = (
    "category", "department", "store", "color", "size",
    "material", "brand", "style", "price_band",
)

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


# --- Step 8: catalog-side signal extraction for CLARIFY's information-gain scoring.
# Candidate set = ALLOWED_ATTRIBUTES minus known_slots minus asked_attributes (the
# latter now also gets a hard exclusion the instant no_preference_signal fires for a
# slot, see SLOT_TO_ASK_ATTRIBUTE below).
SLOT_TO_ASK_ATTRIBUTE = {
    # Stage 2's wider 2.4 slot vocabulary -> the closed ALLOWED_ATTRIBUTES vocabulary.
    # `department` folds into `style` (per CLAUDE.md 2.1 -- classify_constraint does
    # the same fold on the evaluator side); `store` maps to `brand` (2.3: store is the
    # practical brand signal); `price_band` maps to `budget`.
    "category": "category",
    "department": "style",
    "store": "brand",
    "color": "color",
    "size": "size",
    "material": "material",
    "brand": "brand",
    "style": "style",
    "price_band": "budget",
}
