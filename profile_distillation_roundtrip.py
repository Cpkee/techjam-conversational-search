"""Step 10a round-trip validation: RESETHOOK -> distill_session_to_profile() ->
ProfileStore -> a personalized reset() actually changes turn-1 behavior.

Standalone demonstration artifact, NOT part of local_evaluator.py's scoring path
and NOT exercised by it: no session in this competition's dataset shares a
user_id with any other, so RESETHOOK's cross-session profile carry-forward can
never fire during real grading (see CLAUDE.md Step 10, "Important context for
both halves"). This script exercises the mechanism directly instead:

  1. Run one real session from dev_subset.jsonl with its real "before" profile,
     all the way through -- this is the SAME Agent, doing nothing different from
     a normal evaluate() run.
  2. Start a throwaway next session purely to trigger RESETHOOK -- exactly what
     happens automatically the moment any next real session begins.
  3. Read back what distill_session_to_profile() produced (the "after" profile)
     from ProfileStore.
  4. Reset a SECOND, synthetic session -- same category/target/opening message
     as the original -- but seeded with the "after" profile instead of "before".
  5. Compare that synthetic session's turn-1 respond() output against the
     original session's own turn 1 (captured live in step 1), and report the
     difference plainly.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    evaluate,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.agent import Agent

CATALOG_PATH = "data/catalog.jsonl"
DATASET_PATH = "data/dev_subset.jsonl"
# Isolated from the real run's profiles.json -- this script's throwaway
# sessions have no business polluting the actual profile store.
PROFILE_STORE_PATH = Path(__file__).parent / "results_roundtrip_profiles.json"


def main() -> None:
    samples = load_jsonl(DATASET_PATH)
    catalog_ids, categories, products = catalog_index(CATALOG_PATH)
    sample = samples[0]

    before_profile = sample["user_profile"]
    target = str(sample["ground_truth"]["parent_asin"])
    category_text = coarse_category(categories.get(target, []))
    effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
    turn1_message = initial_message(effective_sample, category_text, set())

    PROFILE_STORE_PATH.unlink(missing_ok=True)
    agent = Agent(CATALOG_PATH, profile_store_path=PROFILE_STORE_PATH)

    print(f"[1/5] Running real session {sample['sample_id']} ({sample['scenario_type']}, "
          f"target={target}) with the ORIGINAL ('before') profile...")
    print(f"      before profile: {json.dumps(before_profile)}")
    result = evaluate(agent, [sample], catalog_ids, categories, products)
    session_info = result["sessions"][0]
    print(f"      session result: hit={session_info['hit']} first_hit_turn={session_info['first_hit_turn']}")

    # evaluate() generated its own session_id internally; the agent instance
    # tracks it as _last_session_id since this was the only session run so far.
    session_1_id = agent._last_session_id
    response_before = agent.call_log[0]  # this session's own turn 1, captured live above
    final_state = agent._sessions[session_1_id]
    print(f"      turn-1 (before) ask_attribute = {response_before['ask_attribute']!r}")
    print(f"      final known_slots   = {final_state['known_slots']}")
    print(f"      final notable_facts = {final_state['notable_facts']}")

    print("\n[2/5] Starting a throwaway next session to trigger RESETHOOK "
          "(exactly what happens automatically when any next real session begins)...")
    agent.reset(f"roundtrip_flush_{uuid.uuid4().hex}", before_profile)

    after_profile = agent.profile_store.get(session_1_id)
    assert after_profile is not None, "RESETHOOK did not distill/store a profile for session_1"
    print("\n[3/5] distill_session_to_profile() output ('after' profile):")
    print(f"      {json.dumps(after_profile)}")

    session_2_id = f"roundtrip_after_{uuid.uuid4().hex}"
    print(f"\n[4/5] Resetting a SYNTHETIC second session -- same category/target/opening "
          f"message as session 1 -- seeded with the 'after' profile instead of 'before'...")
    agent.reset(session_2_id, after_profile)
    response_after = agent.respond(session_2_id, turn1_message, 1, 10)
    seeded_facts = agent._sessions[session_2_id]["notable_facts"]
    print(f"      turn-1 (after) ask_attribute = {response_after['ask_attribute']!r}")
    print(f"      notable_facts seeded from profile at reset() = {seeded_facts}")

    print("\n[5/5] BEFORE vs AFTER, both turn 1, identical opening message:")
    print(f"      opening message      : {turn1_message!r}")
    print(f"      before ask_attribute : {response_before['ask_attribute']!r}")
    print(f"      after  ask_attribute : {response_after['ask_attribute']!r}")
    print(f"      before recommendations : {response_before['recommendations']}")
    print(f"      after  recommendations : {[r['parent_asin'] for r in response_after['recommendations']]}")

    ask_attribute_changed = response_before["ask_attribute"] != response_after["ask_attribute"]
    recs_changed = response_before["recommendations"] != [r["parent_asin"] for r in response_after["recommendations"]]
    print(f"\n      ask_attribute changed  : {ask_attribute_changed}")
    print(f"      recommendations changed: {recs_changed}")
    if ask_attribute_changed or recs_changed:
        print("      -> Turn-1 behavior IS observably different under the distilled profile.")
    else:
        print("      -> No observable difference on this particular session -- see report for why.")


if __name__ == "__main__":
    main()
