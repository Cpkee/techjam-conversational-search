"""Step 10a: long-term profile distillation (CLAUDE.md Part 3, Step 10a).

Two small, self-contained pieces used by `Agent`'s RESETHOOK (see agent.py's
`reset()`):

- `distill_session_to_profile()`: turns one finished session's STATE into an
  updated `user_profile`-shaped dict.
- `ProfileStore`: a plain local JSON file mapping an id -> the most recently
  distilled profile for it. No network, no database.

Neither of these touches retrieval/POOL/SPREAD/MAG/CLARIFY -- they only read
and write the same `user_profile` shape the evaluator already passes into
`Agent.reset()`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Cap on how many preference_tags accumulate across a customer's history --
# real disclosed values only (drawn from known_slots/notable_facts), but
# unbounded growth over many sessions would eventually make the tag list
# useless as a signal.
MAX_PREFERENCE_TAGS = 10

_RANGE_RE = re.compile(r"(\d+)\s*-\s*(\d+)")
_SINGLE_NUMBER_RE = re.compile(r"(\d+)")


def _bump_purchase_frequency(value: str) -> str:
    """Increments the leading number(s) in a purchase_frequency string, e.g.
    "3-4 prior purchases" -> "4-5 prior purchases". Leaves the string
    untouched if it doesn't contain a recognizable number -- guessing a
    replacement would be fabricating a signal, not reading one."""
    range_match = _RANGE_RE.search(value)
    if range_match:
        lo, hi = int(range_match.group(1)), int(range_match.group(2))
        return _RANGE_RE.sub(f"{lo + 1}-{hi + 1}", value, count=1)
    single_match = _SINGLE_NUMBER_RE.search(value)
    if single_match:
        n = int(single_match.group(1))
        return _SINGLE_NUMBER_RE.sub(str(n + 1), value, count=1)
    return value


def _derive_new_tags(known_slots: dict, notable_facts: list) -> list[str]:
    """Real disclosed values from this session only -- known_slots (the
    Step 1/2 heuristic's parsed customer-stated values, e.g. material="alloy")
    and notable_facts (Step 3's catch-all for real statements that don't fit
    any of the 9 slots, e.g. "Imported"). No inference beyond what the
    customer actually said this session.

    notable_facts entries are {"text": ..., "source": "profile"|"session"}
    (Step 10a tag, see Agent.reset()/respond()). Both sources feed tags here
    identically -- the tag exists to make the distinction available, not to
    change this behavior; a fact re-derived from an already-distilled profile
    is deduplicated the same way any repeated tag already is, below."""
    tags: list[str] = []
    for value in known_slots.values():
        cleaned = str(value).strip()
        if cleaned:
            tags.append(cleaned)
    for fact in notable_facts:
        text = fact["text"] if isinstance(fact, dict) else fact
        cleaned = str(text).strip()
        if cleaned:
            tags.append(cleaned)
    return tags


def _build_summary(tags: list[str], rating_style: str) -> str:
    tag_clause = ", ".join(tags) if tags else "no specific preferences yet"
    rating_clause = rating_style or "unknown"
    return f"Prior purchases emphasize {tag_clause}; ratings are {rating_clause}."


def distill_session_to_profile(prior_profile: dict, final_state: dict) -> dict:
    """Returns a new dict in the exact `user_profile` shape (average_prior_rating,
    preference_tags, purchase_frequency, rating_style, summary).

    Updates ONLY preference_tags / purchase_frequency / summary, from what
    `final_state` (a session's STATE dict, as built by `Agent.reset`/`respond`)
    actually contains -- `known_slots` and `notable_facts`.

    average_prior_rating and rating_style are deliberately left unchanged:
    they describe how the customer rates products AFTER buying them, and one
    shopping conversation -- which never even tells `respond()` whether a hit
    occurred, let alone a post-purchase rating -- gives no real signal about
    that. Carrying them forward untouched is the honest choice; updating them
    from conversation content would fabricate a learning signal that isn't
    there.
    """
    known_slots = final_state.get("known_slots") or {}
    notable_facts = final_state.get("notable_facts") or []
    new_tags = _derive_new_tags(known_slots, notable_facts)

    merged_tags = list(prior_profile.get("preference_tags") or [])
    seen = {tag.lower() for tag in merged_tags}
    for tag in new_tags:
        if tag.lower() not in seen:
            merged_tags.append(tag)
            seen.add(tag.lower())
    merged_tags = merged_tags[:MAX_PREFERENCE_TAGS]

    rating_style = prior_profile.get("rating_style")

    return {
        "average_prior_rating": prior_profile.get("average_prior_rating"),
        "preference_tags": merged_tags,
        "purchase_frequency": _bump_purchase_frequency(str(prior_profile.get("purchase_frequency", ""))),
        "rating_style": rating_style,
        "summary": _build_summary(merged_tags, rating_style or ""),
    }


class ProfileStore:
    """Plain local JSON file mapping id -> most recently distilled profile.
    No network calls, no database -- just a dict that persists to disk."""

    def __init__(self, path: str | Path = "profiles.json") -> None:
        self.path = Path(path)
        if self.path.exists():
            self._profiles: dict[str, dict] = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self._profiles = {}

    def get(self, user_id: str) -> dict | None:
        return self._profiles.get(user_id)

    def set(self, user_id: str, profile: dict) -> None:
        self._profiles[user_id] = profile
        self.path.write_text(json.dumps(self._profiles, indent=2) + "\n", encoding="utf-8")
