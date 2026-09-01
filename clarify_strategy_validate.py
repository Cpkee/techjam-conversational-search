"""Step 10b validation: run dev_subset.jsonl sequentially through ONE live Agent
instance with the bandit active (no forced_clarify_strategy), and report mean
turns-per-session in the front half of the run vs. the back half.

This is a plumbing/correctness check, not a claim of a dramatic learning curve --
CLAUDE.md's Step 10b instructions call this out explicitly: "expect this to be a
weak signal at only 39 sessions." A single Agent() run means POLICYSTATS updates in
real time across these 39 sessions (on top of whatever warm-start values
Agent.__init__ already seeded it with), so the bandit has the chance to shift
strategy choice as it goes -- but 39 sessions total split into two ~19-session
halves is a small sample to expect a dramatic delta from.

Same shape as erase_vs_accumulate_comparison.py / clarify_strategy_warmstart.py:
dev-time tooling, not part of the organizer's scoring path. Does not touch or claim
anything about recommended_technical_score -- Step 10b is ungraded by
local_evaluator.py (see CLAUDE.md's Step 10 framing: no session in this dataset
shares a user_id with any other, so no session-to-session bandit state read here
can affect any other session's hit/miss outcome; the only real effect measured is
CLARIFY's own turn-count behavior).
"""
from __future__ import annotations

import argparse
import statistics

from evaluator.local_evaluator import MAX_TURNS, catalog_index, evaluate, load_jsonl
from starter.agent import Agent


def _turns_consumed(session: dict) -> int:
    return session["first_hit_turn"] if session["hit"] else MAX_TURNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 10b front-half vs back-half validation")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/dev_subset.jsonl")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)  # bandit active: forced_clarify_strategy=None (default)

    result = evaluate(agent, samples, catalog_ids, categories, products)
    sessions = result["sessions"]
    turns = [_turns_consumed(session) for session in sessions]

    mid = len(turns) // 2
    front, back = turns[:mid], turns[mid:]

    print(f"n_sessions = {len(turns)}")
    print(f"front half (n={len(front)}): mean turns_consumed = {statistics.fmean(front):.4f}")
    print(f"back half  (n={len(back)}): mean turns_consumed = {statistics.fmean(back):.4f}")
    print(f"delta (front - back) = {statistics.fmean(front) - statistics.fmean(back):+.4f}")
    print()
    print("Final POLICYSTATS state (includes warm-start seed + all 39 live sessions):")
    for name, stats in agent._policy_stats.items():
        mean = stats["sum_turns"] / stats["n"] if stats["n"] else float("nan")
        print(f'  "{name}": n={stats["n"]}, sum_turns={stats["sum_turns"]}, mean={mean:.4f}')


if __name__ == "__main__":
    main()
