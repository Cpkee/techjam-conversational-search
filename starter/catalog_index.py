"""Catalog-side metadata extraction (CLAUDE.md 2.3), used by Agent._build_index to
populate `_catalog_meta` -- CLARIFY's information-gain scoring reads that dict's
fields, but never calls these extraction functions directly. No dependency on any
other starter module."""
from __future__ import annotations

import re

# --- Step 8: catalog-side signal extraction for CLARIFY's information-gain scoring.
# Per CLAUDE.md 2.3: department/price/store are reliable structured fields (direct
# lookup); color/material/size/style/brand-detail are sparse (<5% coverage) so we reuse
# the same keyword-list text signals (COLOR_WORDS etc., see slot_schema.py) instead
# of a hard structured lookup.
GENERIC_CATALOG_CATEGORIES = {
    "clothing", "shoes & jewelry", "clothing, shoes & jewelry", "women", "men",
    "women's", "men's", "accessories", "kids", "boys", "girls",
}
PRICE_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _catalog_leaf_category(categories: list) -> str | None:
    """2.3's leaf-node rule (categories[-1], fallback to [-2] on a generic bucket),
    applied to a catalog product's actual `categories` array."""
    cleaned = [str(value).strip() for value in categories if str(value).strip()]
    if not cleaned:
        return None
    leaf = cleaned[-1]
    if leaf.lower() in GENERIC_CATALOG_CATEGORIES and len(cleaned) >= 2:
        leaf = cleaned[-2]
    return leaf.lower()


def _department_value(details: object) -> str | None:
    if isinstance(details, dict):
        value = details.get("Department")
        if value:
            return str(value).strip().lower()
    return None


def _store_value(store: object) -> str | None:
    if store:
        return str(store).strip().lower()
    return None


def _parse_price(value: object) -> float | None:
    """2.3: price is 78.9% null plus placeholder strings ("—", "from X.XX") --
    extract the first numeric token, or None if there isn't one."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = PRICE_RE.search(str(value))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _price_bucket(price: float | None) -> str | None:
    if price is None:
        return None
    if price < 15:
        return "under_15"
    if price < 30:
        return "15_30"
    if price < 50:
        return "30_50"
    if price < 100:
        return "50_100"
    return "100_plus"
