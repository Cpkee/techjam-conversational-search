"""Paired-bootstrap comparison of Agent(erasure_mode="erase") vs
Agent(erasure_mode="accumulate") on hit_rate_at_10 (CLAUDE.md Step 4's open question).

This is a dev-time calibration tool, not part of the organizer's scoring path: it runs
the real evaluate() pipeline twice (once per mode) against the same session set and
reports whether the difference in hit_rate_at_10 is distinguishable from noise.
"""
from __future__ import annotations

import argparse
import random
import statistics

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent

N_BOOTSTRAP = 2000


def _hits(catalog_path: str, dataset_path: str, erasure_mode: str) -> list[int]:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = Agent(catalog_path, erasure_mode=erasure_mode)
    result = evaluate(agent, samples, catalog_ids, categories, products)
    return [int(session["hit"]) for session in result["sessions"]]


def paired_bootstrap_ci(a: list[int], b: list[int], n_bootstrap: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """95% CI on mean(a) - mean(b), resampling session indices in lockstep (paired)."""
    assert len(a) == len(b)
    n = len(a)
    rng = random.Random(seed)
    point_estimate = statistics.fmean(a) - statistics.fmean(b)
    diffs = []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        sample_a = statistics.fmean(a[i] for i in idx)
        sample_b = statistics.fmean(b[i] for i in idx)
        diffs.append(sample_a - sample_b)
    diffs.sort()
    lo = diffs[int(0.025 * n_bootstrap)]
    hi = diffs[min(int(0.975 * n_bootstrap), n_bootstrap - 1)]
    return {"point_estimate": point_estimate, "ci_95": [lo, hi], "n_bootstrap": n_bootstrap, "n_sessions": n}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired-bootstrap erase-vs-accumulate comparison")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/dev_subset.jsonl")
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    print(f"Running erasure_mode='accumulate' on {args.dataset} ...")
    accumulate_hits = _hits(args.catalog, args.dataset, "accumulate")
    print(f"Running erasure_mode='erase' on {args.dataset} ...")
    erase_hits = _hits(args.catalog, args.dataset, "erase")

    result = paired_bootstrap_ci(accumulate_hits, erase_hits, n_bootstrap=args.bootstrap)
    result["accumulate_hit_rate"] = statistics.fmean(accumulate_hits)
    result["erase_hit_rate"] = statistics.fmean(erase_hits)

    print("\nhit_rate_at_10 difference (accumulate - erase):")
    print(f"  accumulate hit_rate_at_10 = {result['accumulate_hit_rate']:.4f}")
    print(f"  erase hit_rate_at_10      = {result['erase_hit_rate']:.4f}")
    print(f"  point estimate            = {result['point_estimate']:+.4f}")
    print(f"  95% CI                    = [{result['ci_95'][0]:+.4f}, {result['ci_95'][1]:+.4f}]")
    if result["ci_95"][0] <= 0 <= result["ci_95"][1]:
        print("  -> CI includes zero: INCONCLUSIVE, no significant difference detected.")
    else:
        winner = "accumulate" if result["point_estimate"] > 0 else "erase"
        print(f"  -> CI excludes zero: {winner} mode significantly better on this sample.")


if __name__ == "__main__":
    main()
