"""Paired-bootstrap comparison between two step9_calibration_runner.py result
files, same treatment as erase_vs_accumulate_comparison.py: resample session
indices in lockstep (paired), report 95% CIs, and call a difference
"inconclusive" whenever its CI includes zero.

Recomputes hit_rate_at_10 / mrr / mttc / efficiency / recommended_technical_score
fresh on every bootstrap resample (all four are decomposable from each
session's own hit / reciprocal_rank / first_hit_turn-or-11), so the same
paired resample simultaneously yields valid CIs for all of them, not just one
metric at a time.

Usage:
  python3 step9_bootstrap_compare.py results_a.json results_b.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics

N_BOOTSTRAP = 2000


def _load_sessions(path: str) -> list[dict]:
    data = json.load(open(path, encoding="utf-8"))
    sessions = data["all_sessions"]
    sessions = sorted(sessions, key=lambda s: s["sample_id"])
    return sessions


def _metrics(sessions: list[dict]) -> dict:
    hit = [1.0 if s["hit"] else 0.0 for s in sessions]
    rr = [s["reciprocal_rank"] for s in sessions]
    ttc = [s["first_hit_turn"] if s["first_hit_turn"] is not None else 11 for s in sessions]
    hit_rate = statistics.fmean(hit)
    mrr = statistics.fmean(rr)
    mttc = statistics.fmean(ttc)
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    technical_score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
    return {
        "hit_rate_at_10": hit_rate, "mrr": mrr, "mttc": mttc,
        "efficiency": efficiency, "recommended_technical_score": technical_score,
    }


def paired_bootstrap(sessions_a: list[dict], sessions_b: list[dict], n_bootstrap: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    ids_a = [s["sample_id"] for s in sessions_a]
    ids_b = [s["sample_id"] for s in sessions_b]
    assert ids_a == ids_b, f"session sets/order differ: {set(ids_a) ^ set(ids_b)}"
    n = len(sessions_a)

    hit_a = [1.0 if s["hit"] else 0.0 for s in sessions_a]
    rr_a = [s["reciprocal_rank"] for s in sessions_a]
    ttc_a = [s["first_hit_turn"] if s["first_hit_turn"] is not None else 11 for s in sessions_a]
    hit_b = [1.0 if s["hit"] else 0.0 for s in sessions_b]
    rr_b = [s["reciprocal_rank"] for s in sessions_b]
    ttc_b = [s["first_hit_turn"] if s["first_hit_turn"] is not None else 11 for s in sessions_b]

    point_a = _metrics(sessions_a)
    point_b = _metrics(sessions_b)

    rng = random.Random(seed)
    diffs = {"hit_rate_at_10": [], "mrr": [], "recommended_technical_score": []}
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        ha = statistics.fmean(hit_a[i] for i in idx)
        ra = statistics.fmean(rr_a[i] for i in idx)
        ta = statistics.fmean(ttc_a[i] for i in idx)
        ea = max(0.0, min(1.0, (11.0 - ta) / 10.0))
        sa = 0.50 * ha + 0.30 * ra + 0.20 * ea

        hb = statistics.fmean(hit_b[i] for i in idx)
        rb = statistics.fmean(rr_b[i] for i in idx)
        tb = statistics.fmean(ttc_b[i] for i in idx)
        eb = max(0.0, min(1.0, (11.0 - tb) / 10.0))
        sb = 0.50 * hb + 0.30 * rb + 0.20 * eb

        diffs["hit_rate_at_10"].append(ha - hb)
        diffs["mrr"].append(ra - rb)
        diffs["recommended_technical_score"].append(sa - sb)

    result = {"n_sessions": n, "n_bootstrap": n_bootstrap, "point_a": point_a, "point_b": point_b, "metrics": {}}
    for name, values in diffs.items():
        values.sort()
        lo = values[int(0.025 * n_bootstrap)]
        hi = values[min(int(0.975 * n_bootstrap), n_bootstrap - 1)]
        point_diff = point_a[name] - point_b[name]
        result["metrics"][name] = {
            "point_estimate_diff": point_diff,
            "ci_95": [lo, hi],
            "conclusive": not (lo <= 0.0 <= hi),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired-bootstrap comparison of two Step 9 config result files")
    parser.add_argument("result_a")
    parser.add_argument("result_b")
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    sessions_a = _load_sessions(args.result_a)
    sessions_b = _load_sessions(args.result_b)
    result = paired_bootstrap(sessions_a, sessions_b, n_bootstrap=args.bootstrap)

    print(f"A = {args.result_a}")
    print(f"B = {args.result_b}")
    print(f"n_sessions = {result['n_sessions']}\n")
    for name, m in result["metrics"].items():
        a_val = result["point_a"][name]
        b_val = result["point_b"][name]
        print(f"{name}:")
        print(f"  A = {a_val:.4f}   B = {b_val:.4f}   diff (A-B) = {m['point_estimate_diff']:+.4f}")
        print(f"  95% CI = [{m['ci_95'][0]:+.4f}, {m['ci_95'][1]:+.4f}]")
        if m["conclusive"]:
            winner = "A" if m["point_estimate_diff"] > 0 else "B"
            print(f"  -> CI excludes zero: {winner} significantly better on this sample.")
        else:
            print("  -> CI includes zero: INCONCLUSIVE.")
        print()


if __name__ == "__main__":
    main()
