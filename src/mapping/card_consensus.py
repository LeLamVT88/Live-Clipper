from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median

from mapping.schema import PlayerObservation
from mapping.tracking import center, source_weight


def _number_evidence(track: list[PlayerObservation]) -> dict[str, object]:
    votes: dict[int, set[float]] = defaultdict(set)
    vote_scores: Counter[int] = Counter()
    reliable_scores: Counter[int] = Counter()
    candidate_times: dict[int, set[float]] = defaultdict(set)
    sole_times: dict[int, set[float]] = defaultdict(set)
    candidate_scores: Counter[int] = Counter()
    sources: dict[int, Counter[str]] = defaultdict(Counter)
    candidates: set[int] = set()
    for item in track:
        candidates.update(item.number_candidates)
        for candidate in set(item.number_candidates):
            candidate_times[candidate].add(item.timestamp)
            candidate_scores[candidate] += max(item.number_candidate_scores.get(candidate, 0.0), .25)
        if len(set(item.number_candidates)) == 1:
            sole_times[item.number_candidates[0]].add(item.timestamp)
        if item.jersey_number is None:
            continue
        weight = source_weight(item) * (.5 + .5 * item.pair_confidence)
        votes[item.jersey_number].add(item.timestamp)
        vote_scores[item.jersey_number] += weight
        candidate_scores[item.jersey_number] += weight
        if item.number_source != "local_preprocessed_ocr":
            reliable_scores[item.jersey_number] += source_weight(item)
        sources[item.jersey_number][item.number_source] += 1
    return {
        "votes": votes, "vote_scores": vote_scores, "reliable_scores": reliable_scores,
        "candidate_times": candidate_times, "sole_times": sole_times,
        "candidate_scores": candidate_scores, "sources": sources, "candidates": candidates,
    }


def _choose_number(evidence: dict[str, object]) -> tuple[int | None, str | None, bool]:
    votes, scores = evidence["votes"], evidence["vote_scores"]
    strong = sorted((scores[number], len(times), number)
                    for number, times in votes.items() if len(times) >= 2)
    if strong:
        winner = strong[-1][2]
        if len(strong) == 1 or (
            evidence["reliable_scores"][winner] > 0
            and strong[-1][0] >= strong[-2][0] * 1.6
        ):
            source = evidence["sources"][winner].most_common(1)[0][0]
            return winner, source, False

    candidates = sorted((evidence["candidate_scores"][candidate], len(times),
                         len(evidence["sole_times"][candidate]), candidate)
                        for candidate, times in evidence["candidate_times"].items())
    if candidates:
        score, seen, sole, winner = candidates[-1]
        runner = candidates[-2][0] if len(candidates) > 1 else 0.0
        dominant = score >= max(.01, runner) * 1.6
        accepted_once = bool(votes[winner])
        repeated = seen >= 3 or (seen >= 2 and sole >= 2) or (
            seen >= 2 and accepted_once and score >= max(.01, runner) * 2.0
        )
        if dominant and repeated:
            return winner, "local_candidate_consensus", False
    conflict = len(votes) > 1
    return None, None, conflict


def resolve_track(track: list[PlayerObservation]) -> dict[str, object] | None:
    timestamps = sorted({item.timestamp for item in track})
    if len(timestamps) < 2:
        return None
    evidence = _number_evidence(track)
    number, number_source, conflict = _choose_number(evidence)
    best_name = max(track, key=lambda item: item.name_confidence)
    x, y = median(center(item)[0] for item in track), median(center(item)[1] for item in track)
    candidate_values = sorted(evidence["candidates"] | set(evidence["votes"]))
    return {
        "name": best_name.name,
        "shirt_number": number,
        "name_confirmed": True,
        "number_confirmed": number is not None,
        "confirmed": number is not None,
        "name_confidence": round(sum(item.name_confidence for item in track) / len(track), 4),
        "pair_confidence": round(sum(item.pair_confidence for item in track) / len(track), 4),
        "number_status": "conflict" if conflict else (
            "weighted_consensus" if number is not None else "insufficient_evidence"),
        "number_candidates": candidate_values,
        "number_candidate_scores": {str(value): round(float(evidence["candidate_scores"][value]), 4)
                                    for value in candidate_values},
        "number_candidate_timestamp_counts": {
            str(value): len(evidence["candidate_times"][value] | evidence["votes"][value])
            for value in candidate_values
        },
        "number_source": number_source,
        "evidence_timestamps": [round(value, 3) for value in timestamps],
        "normalized_center": [round(x, 5), round(y, 5)],
        "observations": [item.to_evidence() for item in track],
    }


def recover_goalkeeper_one(players: list[dict[str, object]]) -> None:
    if not players or any(player["shirt_number"] == 1 for player in players):
        return
    xs = [float(player["normalized_center"][0]) for player in players]
    ys = [float(player["normalized_center"][1]) for player in players]
    for player in players:
        if player["number_confirmed"] or 1 not in player["number_candidates"]:
            continue
        x, y = map(float, player["normalized_center"])
        edge = abs(y - min(ys)) <= .045 or abs(y - max(ys)) <= .045
        if not edge or abs(x - median(xs)) > .15:
            continue
        scores, times = player["number_candidate_scores"], player["number_candidate_timestamp_counts"]
        one_score = float(scores.get("1", 0.0)) * 1.6
        runner = max((float(score) for key, score in scores.items() if key != "1"), default=0.0)
        if int(times.get("1", 0)) >= 2 and one_score >= max(.01, runner) * 1.15:
            player.update(shirt_number=1, number_confirmed=True, confirmed=True,
                          number_status="goalkeeper_one_consensus",
                          number_source="formation_prior+local_candidates")


def resolve_duplicate_numbers(players: list[dict[str, object]]) -> bool:
    numbers = [player["shirt_number"] for player in players if player["shirt_number"] is not None]
    duplicates = {number for number, count in Counter(numbers).items() if count > 1}
    for duplicate in duplicates:
        claimants = [player for player in players if player["shirt_number"] == duplicate]
        ranked = sorted(claimants, key=lambda player: float(
            player["number_candidate_scores"].get(str(duplicate), 0.0)), reverse=True)
        first = float(ranked[0]["number_candidate_scores"].get(str(duplicate), 0.0))
        second = float(ranked[1]["number_candidate_scores"].get(str(duplicate), 0.0))
        winner = ranked[0] if first > 0 and first >= second * 1.35 else None
        for player in claimants:
            if player is winner:
                player["number_status"] = "duplicate_resolved_by_evidence"
            else:
                player.update(shirt_number=None, number_confirmed=False, confirmed=False,
                              number_status="duplicate_conflict")
    return bool(duplicates)
