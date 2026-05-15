from __future__ import annotations

from collections import Counter, defaultdict

from ..schemas.agent import AgentOpinion, Direction, Vote
from ..schemas.decision import ConsensusBranch, ConsensusScore


def compute_consensus(votes: list[Vote]) -> float:
    eligible = [vote for vote in votes if not vote.informational]
    if not eligible:
        return 0.0
    direction_map = {
        Direction.INCREASE: 1.0,
        Direction.HOLD: 0.0,
        Direction.DECREASE: -1.0,
    }
    numerator = sum(direction_map[vote.direction] * vote.confidence for vote in eligible)
    denominator = sum(vote.confidence for vote in eligible)
    return round(numerator / denominator, 4) if denominator > 0 else 0.0


def classify_branch(votes: list[Vote]) -> ConsensusBranch:
    eligible = [vote for vote in votes if not vote.informational]
    confidence_sum = sum(vote.confidence for vote in eligible)
    if len(eligible) < 2 or confidence_sum < 1.0:
        return ConsensusBranch.INSUFFICIENT_VOTES

    hold_ratio = sum(1 for vote in eligible if vote.direction == Direction.HOLD) / len(eligible)
    if hold_ratio >= 0.6:
        return ConsensusBranch.STRONG_HOLD

    score = abs(compute_consensus(votes))
    if score >= 0.6:
        return ConsensusBranch.STRONG_CONSENSUS
    if score >= 0.3:
        return ConsensusBranch.WEAK_CONSENSUS
    return ConsensusBranch.CONFLICT


def consensus_by_subject(opinions: list[AgentOpinion]) -> dict[str, ConsensusScore]:
    grouped: dict[str, list[Vote]] = defaultdict(list)
    for opinion in opinions:
        for vote in opinion.votes:
            grouped[vote.subject].append(vote)

    result: dict[str, ConsensusScore] = {}
    for subject, votes in grouped.items():
        eligible = [vote for vote in votes if not vote.informational]
        distribution = Counter(vote.direction for vote in eligible)
        result[subject] = ConsensusScore(
            subject=subject,
            weighted_score=compute_consensus(votes),
            confidence_sum=round(sum(vote.confidence for vote in eligible), 4),
            vote_distribution={
                direction: distribution.get(direction, 0) for direction in Direction
            },
            branch=classify_branch(votes),
        )
    return result


def select_targets(
    consensus_per_subject: dict[str, ConsensusScore],
    round1_opinions: list[AgentOpinion],
    *,
    max_targets: int = 4,
    min_confidence: float = 0.4,
) -> list[str]:
    conflict_subjects = {
        subject
        for subject, score in consensus_per_subject.items()
        if score.branch == ConsensusBranch.CONFLICT
    }
    if not conflict_subjects:
        return []

    candidates: list[tuple[str, float]] = []
    for opinion in round1_opinions:
        if opinion.agent.casefold() == "compliance":
            continue
        for vote in opinion.votes:
            if (
                vote.informational
                or vote.subject not in conflict_subjects
                or vote.confidence < min_confidence
            ):
                continue
            candidates.append((opinion.agent, vote.confidence))

    seen: set[str] = set()
    targets: list[str] = []
    for agent, _confidence in sorted(candidates, key=lambda item: item[1], reverse=True):
        if agent in seen:
            continue
        seen.add(agent)
        targets.append(agent)
        if len(targets) >= max_targets:
            break
    return targets
