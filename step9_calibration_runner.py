"""Step 9 calibration: runs ONE named Agent configuration against a dataset
(dev_subset.jsonl by default) and saves per-session results to a JSON file.

Generalizes the pattern already used by erase_vs_accumulate_comparison.py /
the earlier ablation scripts (run_ablation_*.py) to the three parameters
CLAUDE.md's Step 9 calls out: FUSION_KW_WEIGHT/FUSION_BROWSE_WEIGHT, MAG_FLOOR
(+ suppress_revise_turn1), and GATE_CONFIDENCE_THRESHOLD. Those are plain
module-level `# TUNE` constants in starter/agent.py (not constructor args),
so this script overrides them on starter.agent's module namespace inside each
worker process, before that worker constructs its own Agent instance --
exactly the same pattern the earlier SPREAD-gating ablation scripts used.

Usage:
  python3 step9_calibration_runner.py --config-name fusion_kw_heavy \
      --fusion-kw-weight 0.65 --fusion-browse-weight 0.35 \
      --output step9_results/fusion_kw_heavy.json
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)

CATALOG_PATH = str(Path(__file__).parent / "data" / "catalog.jsonl")
WORKERS = 4  # matches Ollama's OLLAMA_NUM_PARALLEL / -np 4

_WORKER: dict = {}
_CONFIG: dict = {}


def _init_worker() -> None:
    import starter.agent as agent_mod

    if _CONFIG.get("fusion_kw_weight") is not None:
        agent_mod.FUSION_KW_WEIGHT = _CONFIG["fusion_kw_weight"]
    if _CONFIG.get("fusion_browse_weight") is not None:
        agent_mod.FUSION_BROWSE_WEIGHT = _CONFIG["fusion_browse_weight"]
    if _CONFIG.get("mag_floor") is not None:
        agent_mod.MAG_FLOOR = _CONFIG["mag_floor"]
    if _CONFIG.get("gate_confidence_threshold") is not None:
        agent_mod.GATE_CONFIDENCE_THRESHOLD = _CONFIG["gate_confidence_threshold"]

    _WORKER["agent"] = agent_mod.Agent(
        CATALOG_PATH,
        clarify_spread_gate=False,  # settled per Step 9's opening baseline, not reopened
        suppress_revise_turn1=_CONFIG.get("suppress_revise_turn1", False),
    )
    _WORKER["catalog_ids"], _WORKER["categories"], _WORKER["products"] = catalog_index(CATALOG_PATH)


def _run_one_session(sample: dict) -> dict:
    agent = _WORKER["agent"]
    catalog_ids = _WORKER["catalog_ids"]
    categories = _WORKER["categories"]
    products = _WORKER["products"]

    session_id = f"s9_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"

    user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)

    turn_log: list[dict] = []
    hit_turn = None
    best_rank = None
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for turn in range(1, MAX_TURNS + 1):
        try:
            response = agent.respond(session_id, user_message, turn, TOP_K)
        except Exception:
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        usage = response.get("usage") or {}
        total_prompt_tokens += usage.get("prompt_tokens", 0) or 0
        total_completion_tokens += usage.get("completion_tokens", 0) or 0

        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        hit = override_applied and target in ranked
        rank = ranked.index(target) + 1 if hit else None
        turn_log.append({"turn": turn, "ask_attribute": response.get("ask_attribute")})

        if hit:
            hit_turn = turn
            best_rank = rank
            break
        if turn == MAX_TURNS:
            break

        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(
                effective_sample, response.get("ask_attribute"), disclosed, boundary_used
            )

    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "best_rank": best_rank,
        "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        "num_turns": len(turn_log),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }


def run(dataset_path: str, workers: int = WORKERS) -> list[dict]:
    samples = load_jsonl(dataset_path)
    sessions: list[dict] = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
        futures = {executor.submit(_run_one_session, sample): sample for sample in samples}
        done = 0
        for future in as_completed(futures):
            sample = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[ERROR] {sample['sample_id']}: {exc}", file=sys.stderr)
                continue
            sessions.append(result)
            done += 1
            print(
                f"[{done}/{len(samples)}] {result['sample_id']} hit={result['hit']} "
                f"turn={result['first_hit_turn']} elapsed={time.time() - start:.0f}s",
                file=sys.stderr,
            )
    sessions.sort(key=lambda s: s["sample_id"])
    return sessions


def summarize(sessions: list[dict]) -> dict:
    metric_input = [
        {"hit": s["hit"], "reciprocal_rank": s["reciprocal_rank"], "first_hit_turn": s["first_hit_turn"]}
        for s in sessions
    ]
    overall = metric_summary(metric_input)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical_score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency

    scenario_groups: dict[str, list[dict]] = {}
    for s in sessions:
        scenario_groups.setdefault(s["scenario_type"], []).append(
            {"hit": s["hit"], "reciprocal_rank": s["reciprocal_rank"], "first_hit_turn": s["first_hit_turn"]}
        )
    scenario_metrics = {name: metric_summary(items) for name, items in sorted(scenario_groups.items())}

    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "scenario_metrics": scenario_metrics,
        "reported_token_usage": {
            "prompt_tokens": sum(s["prompt_tokens"] for s in sessions),
            "completion_tokens": sum(s["completion_tokens"] for s in sessions),
        },
    }


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description="Step 9 named-configuration calibration runner")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--dataset", default=str(Path(__file__).parent / "data" / "dev_subset.jsonl"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--fusion-kw-weight", type=float, default=None)
    parser.add_argument("--fusion-browse-weight", type=float, default=None)
    parser.add_argument("--mag-floor", type=float, default=None)
    parser.add_argument("--gate-confidence-threshold", type=float, default=None)
    parser.add_argument("--suppress-revise-turn1", action="store_true")
    args = parser.parse_args()

    _CONFIG.update({
        "fusion_kw_weight": args.fusion_kw_weight,
        "fusion_browse_weight": args.fusion_browse_weight,
        "mag_floor": args.mag_floor,
        "gate_confidence_threshold": args.gate_confidence_threshold,
        "suppress_revise_turn1": args.suppress_revise_turn1,
    })

    start = time.time()
    print(f"=== Step 9 config '{args.config_name}' on {args.dataset} ===", file=sys.stderr)
    print(f"    overrides: {_CONFIG}", file=sys.stderr)
    sessions = run(args.dataset, workers=args.workers)
    summary = summarize(sessions)
    summary["config_name"] = args.config_name
    summary["overrides"] = dict(_CONFIG)
    summary["wall_clock_seconds"] = round(time.time() - start, 1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"summary": summary, "all_sessions": sessions}, indent=2), encoding="utf-8")

    print("=== SUMMARY ===", file=sys.stderr)
    print(json.dumps(summary, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
