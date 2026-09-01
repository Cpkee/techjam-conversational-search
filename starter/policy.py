"""Step 10b: Adaptive Orchestration (CLAUDE.md Part 3, Step 10b) -- the epsilon-greedy
bandit over CLARIFY_STRATEGIES. No dependency on any other starter module (random
only). POLICYSTATS itself (the running per-strategy stats dict) lives on the Agent
instance, not here -- this module only holds the strategy definitions and the pure
selection function that reads whatever stats dict it's handed."""
from __future__ import annotations

import random

MAX_TURNS = 10  # confirmed via CLAUDE.md 2.1 / evaluator/local_evaluator.py

# Two CONCRETE, SPREAD-independent clarify strategies. SPREAD-gating CLARIFY was
# already measured net-negative and disabled by default in Step 9 (see
# clarify_spread_gate's docstring in Agent.__init__), so these two strategies must
# differ on something else -- the one picked here is a hard cap on how many
# clarifying questions get asked in a single session before REALIGN forces every
# remaining turn straight to ranking, regardless of pool state.
#
# clarify_aggressive: up to 5 questions per session -- most, not all, of the 8
#   CLARIFY-reachable attributes (ALLOWED_ATTRIBUTES minus
#   CLARIFY_UNREACHABLE_ATTRIBUTES's category/brand; see starter/clarify.py).
# clarify_sparing: at most 1 question per session, then rank on every remaining turn
#   no matter how flat POOL looks.
CLARIFY_STRATEGIES = {
    "clarify_aggressive": {"max_questions": 5},
    "clarify_sparing": {"max_questions": 1},
}
CLARIFY_EPSILON = 0.1  # TUNE: starting placeholder per CLAUDE.md Step 10b


def _select_clarify_strategy(policy_stats: dict, rng: random.Random) -> str:
    """Epsilon-greedy pick between CLARIFY_STRATEGIES. Reward = -turns_consumed (see
    RESETHOOK in Agent.reset()), so the greedy arm is whichever strategy has the
    LOWER mean turns-per-session-so-far -- an unstarted strategy (n=0) is treated as
    infinitely costly so it always loses to any strategy with real data, which in
    practice only matters before POLICYSTATS is warm-started (see Agent.__init__)."""
    if rng.random() < CLARIFY_EPSILON:
        return rng.choice(list(CLARIFY_STRATEGIES))

    def mean_turns(name: str) -> float:
        stats = policy_stats[name]
        return stats["sum_turns"] / stats["n"] if stats["n"] > 0 else float("inf")

    return min(CLARIFY_STRATEGIES, key=mean_turns)
