"""Demo transcripts for a screen recording: runs a small, curated set of real
public_set.jsonl sessions through the live Agent turn-by-turn, printing a readable
conversation (customer message, what the agent asked, its top recommendations by
title, hit/miss) for each, and optionally dumping the same data as structured JSON
(--json-out) for a UI to render (see demo_viewer.html).

This is presentation tooling, not a scoring tool -- it replicates evaluate()'s
exact per-turn loop from evaluator/local_evaluator.py (same initial_message /
customer_reply / override logic), so what's shown is real agent behavior on real
dataset sessions, not a scripted mock.

Default picks -- one per scenario_type actually present in public_set.jsonl:
  public_0001 (buying, easy)          -- straightforward slot-filling to a hit
  public_0002 (intent_override, hard) -- customer changes their mind mid-session;
                                          hits are gated off until the override
                                          turn regardless of agent quality (see
                                          CLAUDE.md 2.1)
  public_0006 (browsing, medium)      -- vague opening, CLARIFY-driven; also the
                                          exact session CLAUDE.md's Step 10a
                                          profile-distillation trace used
  public_0035 (boundary, medium)      -- "I don't have a preference" handling

Override which sessions run with --samples (comma-separated sample_ids from
--dataset). Note a fresh Agent() here has no warm-started POLICYSTATS/profile
history -- it behaves like the very first sessions of a real graded run, not a
run that's been going for a while.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent
from starter.policy import CLARIFY_STRATEGIES

# Demo-only patch, not touching the shared scored pipeline: Step 10b's REALIGN caps
# how many questions a session may ask (5 for "clarify_aggressive", 1 for
# "clarify_sparing"), picked per-session by an epsilon-greedy bandit that -- as of
# tonight's fix -- uses real RNG entropy, not a fixed seed. For a recording that
# means (a) a session can hit its cap mid-conversation and go silent for the
# remaining turns even though CLARIFY still has real attributes left to ask about
# (confirmed directly on public_0002: assigned "clarify_aggressive", asked exactly
# 5 questions, then ask_attribute=None for turns 6-10 despite "feature"/"other"
# still being unasked), and (b) re-running this script can assign a different
# strategy to the same sample_id and show different behavior on camera. Neither is
# desirable for a demo, whose job is to show the calibrated core pipeline's real,
# repeatable behavior. Raise both caps above MAX_TURNS so REALIGN's artificial cap
# never fires before a session's natural ceiling (running out of real attributes to
# ask) would -- that natural ceiling is still real agent behavior and stays visible.
for _strategy in CLARIFY_STRATEGIES.values():
    _strategy["max_questions"] = MAX_TURNS

DEFAULT_SAMPLES = ["public_0001", "public_0002", "public_0006", "public_0035"]
RECS_KEPT = 5  # how many ranked recommendations to keep per turn in the JSON output


def _title(products: dict, asin: str) -> str:
    product = products.get(asin) or {}
    return str(product.get("title") or asin)


def run_session(agent: Agent, sample: dict, catalog_ids: set, categories: dict, products: dict) -> dict:
    session_id = f"demo_{sample['sample_id']}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    category = coarse_category(categories.get(target, []))
    user_message = initial_message(effective_sample, category, disclosed)

    record: dict = {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "difficulty": sample.get("difficulty_bucket"),
        "target_asin": target,
        "target_title": _title(products, target),
        "turns": [],
        "outcome": {"hit": False, "turn": None, "rank": None},
    }

    print(f"\n{'=' * 78}")
    print(
        f"SESSION {sample['sample_id']}  |  scenario={sample['scenario_type']}  "
        f"|  difficulty={sample.get('difficulty_bucket', '?')}  |  target={target} "
        f'"{record["target_title"][:70]}"'
    )
    print("=" * 78)

    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(session_id, user_message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        ask = response.get("ask_attribute")

        print(f"\n-- Turn {turn} --", flush=True)
        print(f'  Customer: "{user_message}"')
        print(f"  Agent asks about: {ask if ask else '(none)'}")
        top_titles = [f"#{i + 1} {_title(products, asin)[:70]}" for i, asin in enumerate(ranked[:3])]
        print("  Top recommendations: " + (" | ".join(top_titles) if top_titles else "(none)"))

        is_hit_turn = override_applied and target in ranked
        hit_rank = ranked.index(target) + 1 if is_hit_turn else None

        turn_record = {
            "turn": turn,
            "customer_message": user_message,
            "ask_attribute": ask,
            "recommendations": [
                {"rank": i + 1, "asin": asin, "title": _title(products, asin)}
                for i, asin in enumerate(ranked[:RECS_KEPT])
            ],
            "is_hit_turn": is_hit_turn,
            "hit_rank": hit_rank,
            "event": None,
        }

        if is_hit_turn:
            print(f"\n  >>> HIT on turn {turn}, rank {hit_rank} <<<")
            record["turns"].append(turn_record)
            record["outcome"] = {"hit": True, "turn": turn, "rank": hit_rank}
            return record

        if turn == MAX_TURNS:
            print("\n  >>> MISS -- target never entered the top 10 across all 10 turns <<<")
            record["turns"].append(turn_record)
            return record

        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            turn_record["event"] = "override_next"
            print("\n  [[ customer changes their mind -- override lands next turn ]]")
        else:
            user_message, boundary_used = customer_reply(
                effective_sample, response.get("ask_attribute"), disclosed, boundary_used
            )

        record["turns"].append(turn_record)

    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Print readable turn-by-turn demo transcripts for a screen recording")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--samples", default=",".join(DEFAULT_SAMPLES))
    parser.add_argument("--json-out", default=None, help="optional path to dump structured session data as JSON")
    args = parser.parse_args()

    sample_ids = args.samples.split(",")
    all_samples = {s["sample_id"]: s for s in load_jsonl(args.dataset)}
    missing = [sid for sid in sample_ids if sid not in all_samples]
    if missing:
        raise SystemExit(f"sample_id(s) not found in {args.dataset}: {missing}")

    catalog_ids, categories, products = catalog_index(args.catalog)
    # forced_clarify_strategy makes strategy assignment deterministic (no bandit
    # RNG involved at all) -- combined with the CLARIFY_STRATEGIES patch above,
    # this means every run of this script shows the same behavior for the same
    # sample_ids, which a recording needs and the real scored pipeline doesn't.
    agent = Agent(args.catalog, forced_clarify_strategy="clarify_aggressive")

    results = [run_session(agent, all_samples[sid], catalog_ids, categories, products) for sid in sample_ids]

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for result in results:
        outcome = result["outcome"]
        if outcome["hit"]:
            print(f"  {result['sample_id']}: HIT  turn={outcome['turn']}  rank={outcome['rank']}")
        else:
            print(f"  {result['sample_id']}: MISS")

    if args.json_out:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset,
            "sessions": results,
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nWrote structured transcript JSON to {args.json_out}")


if __name__ == "__main__":
    main()
