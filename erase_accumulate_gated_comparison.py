"""Three-way comparison of Agent erasure_mode: erase vs. accumulate vs. gated.

"gated" (Step 4 extension) changes only MICRO-tier (per-slot) DELETE handling: a
DELETE is honored only if the same slot also carries a genuine UPDATE in the same
turn's slot_operations output. CATCMP and the MACRO-tier category-conflict reset are
untouched and identical across all three modes -- see starter/agent.py's
_apply_slot_operations docstring.

This is a dev-time calibration tool, not part of the organizer's scoring path.
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from erase_vs_accumulate_comparison import paired_bootstrap_ci


def _run(catalog_path: str, dataset_path: str, erasure_mode: str):
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = Agent(catalog_path, erasure_mode=erasure_mode)
    result = evaluate(agent, samples, catalog_ids, categories, products)
    return samples, result, agent


def main() -> None:
    parser = argparse.ArgumentParser(description="erase vs. accumulate vs. gated comparison")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/dev_subset.jsonl")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    print(f"Running erasure_mode='accumulate' on {args.dataset} ...")
    samples, result_accum, _ = _run(args.catalog, args.dataset, "accumulate")
    print(f"Running erasure_mode='erase' on {args.dataset} ...")
    _, result_erase, _ = _run(args.catalog, args.dataset, "erase")
    print(f"Running erasure_mode='gated' on {args.dataset} ...")
    _, result_gated, gated_agent = _run(args.catalog, args.dataset, "gated")

    accumulate_hits = [int(s["hit"]) for s in result_accum["sessions"]]
    erase_hits = [int(s["hit"]) for s in result_erase["sessions"]]
    gated_hits = [int(s["hit"]) for s in result_gated["sessions"]]

    print("\n=== Point metrics (hit_rate_at_10 / mrr / mttc) ===")
    for label, result in (("accumulate", result_accum), ("erase", result_erase), ("gated", result_gated)):
        print(
            f"  {label:11s} hit_rate_at_10={result['hit_rate_at_10']:.4f}  "
            f"mrr={result['mrr']:.4f}  mttc={result['mttc']:.4f}"
        )

    ci_gated_vs_accum = paired_bootstrap_ci(gated_hits, accumulate_hits, n_bootstrap=args.bootstrap, seed=1)
    ci_gated_vs_erase = paired_bootstrap_ci(gated_hits, erase_hits, n_bootstrap=args.bootstrap, seed=0)

    print("\n=== gated - accumulate (hit_rate_at_10 difference) -- THE comparison that matters ===")
    print(f"  point estimate = {ci_gated_vs_accum['point_estimate']:+.4f}")
    print(f"  95% CI         = [{ci_gated_vs_accum['ci_95'][0]:+.4f}, {ci_gated_vs_accum['ci_95'][1]:+.4f}]")
    inconclusive = ci_gated_vs_accum["ci_95"][0] <= 0 <= ci_gated_vs_accum["ci_95"][1]
    print("  -> " + ("INCONCLUSIVE (CI includes zero)" if inconclusive else "SIGNIFICANT (CI excludes zero)"))

    print("\n=== gated - erase (hit_rate_at_10 difference) -- secondary, erase≈accumulate here ===")
    print(f"  point estimate = {ci_gated_vs_erase['point_estimate']:+.4f}")
    print(f"  95% CI         = [{ci_gated_vs_erase['ci_95'][0]:+.4f}, {ci_gated_vs_erase['ci_95'][1]:+.4f}]")
    inconclusive = ci_gated_vs_erase["ci_95"][0] <= 0 <= ci_gated_vs_erase["ci_95"][1]
    print("  -> " + ("INCONCLUSIVE (CI includes zero)" if inconclusive else "SIGNIFICANT (CI excludes zero)"))

    total_turns = gated_agent.gated_counters["total_turns"]
    turns_engaged = gated_agent.gated_counters["turns_with_delete_and_no_update"]
    print("\n=== Gated DELETE-downgrade counters (across this dataset) ===")
    print(f"  Total turns processed (gated mode):                 {total_turns}")
    print(f"  DELETE ops proposed by Stage 2 (slot-level):        {gated_agent.gated_counters['delete_proposed']}")
    print(f"  DELETE ops downgraded to CARRYOVER (slot-level):    {gated_agent.gated_counters['delete_downgraded']}")
    print(
        f"  TURNS with >=1 DELETE and 0 UPDATEs anywhere:       {turns_engaged} "
        f"({turns_engaged}/{total_turns} = {turns_engaged / total_turns:.1%} of all turns)"
        if total_turns else "  TURNS with >=1 DELETE and 0 UPDATEs anywhere:       0"
    )

    print("\n=== Sanity: erase vs. accumulate agreement (expect ~identical, per CATCMP finding) ===")
    identical = sum(1 for a, b in zip(erase_hits, accumulate_hits) if a == b)
    print(f"  agree on {identical}/{len(samples)} sessions")

    print("\n=== Sessions where gated differed from BOTH erase and accumulate ===")
    scenario_counts: Counter = Counter()
    differing = []
    for i, sample in enumerate(samples):
        if gated_hits[i] != erase_hits[i] and gated_hits[i] != accumulate_hits[i]:
            differing.append((sample["sample_id"], sample["scenario_type"]))
            scenario_counts[sample["scenario_type"]] += 1
    print(f"  total differing sessions: {len(differing)} / {len(samples)}")
    for sample_id, scenario_type in differing:
        print(f"    {sample_id} ({scenario_type})")
    print(f"  scenario_type breakdown: {dict(scenario_counts)}")


if __name__ == "__main__":
    main()
