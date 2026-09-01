"""Step 10b warm-start measurement: run each CLARIFY_STRATEGIES arm once,
independently, on dev_subset.jsonl to get a real measured mean turns-consumed per
strategy (CLAUDE.md Part 3, Step 10b).

Same shape as erase_vs_accumulate_comparison.py: a dev-time calibration tool, not
part of the organizer's scoring path. Its output seeds POLICYSTATS's initial values
in Agent.__init__ (starter/agent.py) -- local dev runs never carry into the real
graded run, so this is the only way this measurement helps at all once grading
starts a fresh Agent().

turns_consumed per session = however many respond() calls the evaluator actually
made: first_hit_turn on a hit, MAX_TURNS on a miss (evaluate() stops calling
respond() the turn a hit lands, and runs the full MAX_TURNS otherwise -- see
local_evaluator.py's evaluate() loop). This is exactly what RESETHOOK's
len(state["turns"]) reads inside the live Agent, so the number measured here is
directly comparable to what POLICYSTATS accumulates during a real run.
"""
from __future__ import annotations

import argparse
import statistics

from evaluator.local_evaluator import MAX_TURNS, catalog_index, evaluate, load_jsonl
from starter.agent import Agent, CLARIFY_STRATEGIES


def _turns_consumed(session: dict) -> int:
    return session["first_hit_turn"] if session["hit"] else MAX_TURNS


def _measure(catalog_path: str, dataset_path: str, strategy: str) -> list[int]:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = Agent(catalog_path, forced_clarify_strategy=strategy)
    result = evaluate(agent, samples, catalog_ids, categories, products)
    return [_turns_consumed(session) for session in result["sessions"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start measurement for Step 10b's clarify strategies")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/dev_subset.jsonl")
    args = parser.parse_args()

    print(f"Measuring CLARIFY_STRATEGIES against {args.dataset} ...")
    for strategy in CLARIFY_STRATEGIES:
        print(f"  running {strategy} (max_questions={CLARIFY_STRATEGIES[strategy]['max_questions']}) ...")
        turns = _measure(args.catalog, args.dataset, strategy)
        n = len(turns)
        mean_turns = statistics.fmean(turns)
        print(
            f'    "{strategy}": {{"n": {n}, "sum_turns": {sum(turns)}}},  '
            f"# mean turns_consumed = {mean_turns:.4f}"
        )


if __name__ == "__main__":
    main()
