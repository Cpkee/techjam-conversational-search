"""RANKLLM (Step 6): one more Ollama call, takes POOL's fused candidates (with their
fusion score and title/features text) plus the turn's query context, and produces the
final top-10 ranking using semantic judgment on top of the fusion prior. Depends on
ollama_client (_ollama_chat) only."""
from __future__ import annotations

from pydantic import BaseModel, ValidationError

from starter.ollama_client import _ollama_chat

RANKLLM_CANDIDATE_LIMIT = 20  # TUNE: how many fused POOL candidates get sent to RANKLLM


class RankLLMOutput(BaseModel):
    ranking: list[str]


RANKLLM_SCHEMA = RankLLMOutput.model_json_schema()

_RANKLLM_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are the final ranking stage of a shopping copilot. You are given the "
        "customer's current search intent and a pooled list of candidate products, "
        "each already carrying a fusion_score that blends keyword-match and semantic-"
        "similarity signals. Re-rank these candidates using fusion_score as a "
        "reasonable prior, adjusted by your own judgment of how well each candidate's "
        "title and features actually match what the customer wants. Respond with "
        "exactly one JSON object: `ranking`, a list of up to 10 parent_asin strings, "
        "best match first, drawn ONLY from the candidate list given below -- never "
        "invent an id that is not present in the candidates, and never return more "
        "candidates than were given to you."
    ),
}


def _format_rankllm_prompt(rewrite: str, expansion: str, candidates: list[dict]) -> str:
    lines = [f"Customer wants: {rewrite}", f"Broader intent: {expansion}", "Candidates:"]
    for candidate in candidates:
        title = (candidate.get("title") or "(no title)").strip()
        features = (candidate.get("features") or "").strip()
        line = f"- {candidate['parent_asin']} [fusion_score={candidate['score']:.3f}] {title} | {features}"
        lines.append(line[:300])
    return "\n".join(lines)


def _rank_llm(rewrite: str, expansion: str, candidates: list[dict]) -> tuple[list[str], dict]:
    """Returns (ranked parent_asins, usage). Falls back to the incoming fusion order
    (i.e. no reranking) on any call failure, invalid/empty output, or a returned id
    that isn't in the candidate set -- RANKLLM only ever reorders/truncates candidates
    POOL already vetted, never invents recommendations of its own."""
    if not candidates:
        return [], {"prompt_tokens": 0, "completion_tokens": 0}
    fallback_order = [candidate["parent_asin"] for candidate in candidates]
    prompt = _format_rankllm_prompt(rewrite, expansion, candidates)
    messages = [_RANKLLM_SYSTEM_MESSAGE, {"role": "user", "content": prompt}]
    raw, usage = _ollama_chat(messages, RANKLLM_SCHEMA)
    if raw is None:
        return fallback_order, usage
    try:
        parsed = RankLLMOutput.model_validate(raw)
    except ValidationError:
        return fallback_order, usage
    valid_ids = {candidate["parent_asin"] for candidate in candidates}
    ranking = [asin for asin in parsed.ranking if asin in valid_ids]
    if not ranking:
        return fallback_order, usage
    # Append any candidates RANKLLM omitted, in their original fusion order, so a
    # partial/short LLM ranking still fills out to the full candidate list.
    seen = set(ranking)
    ranking.extend(asin for asin in fallback_order if asin not in seen)
    return ranking, usage
