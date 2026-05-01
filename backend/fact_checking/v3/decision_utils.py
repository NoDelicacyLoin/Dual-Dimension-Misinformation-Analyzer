from api_contract import EachFactChecking, EachFactualClaim
from shared_constants import NEUTRAL_SCORE


MAX_SCORE_SHIFT = 0.4
NEUTRAL_DISTANCE_LOW_CONFIDENCE = 0.15
GEMINI_TRUTH_SCORE_WEIGHT = 0.15


def set_verdict_from_truth_score(fact_check_result: EachFactualClaim | EachFactChecking) -> None:
    truth_score = fact_check_result.truth_score

    if truth_score is None:
        fact_check_result.verdict = None
    elif truth_score >= 0.85:
        fact_check_result.verdict = "True"
    elif truth_score >= 0.65:
        fact_check_result.verdict = "Mostly True"
    elif truth_score >= 0.45:
        fact_check_result.verdict = "Neutral"
    elif truth_score >= 0.25:
        fact_check_result.verdict = "Mostly False"
    else:
        fact_check_result.verdict = "False"


def count_decision_usable_sources(factual_claim: EachFactualClaim) -> int:
    usable_count = 0

    for evidence_item in factual_claim.evidence:
        if evidence_item.stance == "background":
            continue
        if evidence_item.evidence_quality in {"strong", "usable"}:
            usable_count += 1

    return usable_count


def aggregate_truth_score(factual_claim: EachFactualClaim) -> None:
    support_score = 0
    contradiction_score = 0

    for evidence_item in factual_claim.evidence:
        if evidence_item.evidence_quality == "strong":
            evidence_score = 2
        elif evidence_item.evidence_quality == "usable":
            evidence_score = 1
        else:
            evidence_score = 0

        if evidence_item.stance == "supports":
            support_score += evidence_score
        elif evidence_item.stance == "contradicts":
            contradiction_score += evidence_score

    total_direction_score = support_score + contradiction_score
    if total_direction_score <= 0:
        backend_truth_score = NEUTRAL_SCORE
    else:
        balance_score = (support_score - contradiction_score) / total_direction_score
        backend_truth_score = NEUTRAL_SCORE + (MAX_SCORE_SHIFT * balance_score)

    truth_score = backend_truth_score

    gemini_truth_score = factual_claim.metadata.gemini_truth_score
    if gemini_truth_score is not None:
        if gemini_truth_score < 0.0:
            gemini_truth_score = 0.0
        elif gemini_truth_score > 1.0:
            gemini_truth_score = 1.0

        truth_score = (
            ((1 - GEMINI_TRUTH_SCORE_WEIGHT) * truth_score)
            + (GEMINI_TRUTH_SCORE_WEIGHT * gemini_truth_score)
        )

    if truth_score < 0.0:
        factual_claim.truth_score = 0.0
    elif truth_score > 1.0:
        factual_claim.truth_score = 1.0
    else:
        factual_claim.truth_score = round(truth_score, 4)


def summarize_selected_evidence(factual_claim: EachFactualClaim) -> None:
    if not factual_claim.evidence:
        factual_claim.evidence_sufficiency = ""
        return

    evidence_score = 0

    for evidence_item in factual_claim.evidence:
        if evidence_item.evidence_quality == "strong":
            evidence_score += 2
        elif evidence_item.evidence_quality == "usable":
            evidence_score += 1

    if evidence_score >= 3:
        factual_claim.evidence_sufficiency = "sufficient"
    elif evidence_score >= 2:
        factual_claim.evidence_sufficiency = "limited"
    else:
        factual_claim.evidence_sufficiency = "insufficient"


def calculate_decision_confidence(factual_claim: EachFactualClaim) -> None:
    if factual_claim.status != "success" or factual_claim.truth_score is None:
        factual_claim.decision_confidence = "low"
        return

    usable_count = count_decision_usable_sources(factual_claim)
    if usable_count <= 1:
        factual_claim.decision_confidence = "low"
    elif usable_count == 2:
        factual_claim.decision_confidence = "medium"
    else:
        factual_claim.decision_confidence = "high"

    distance_from_neutral = abs(factual_claim.truth_score - NEUTRAL_SCORE)
    if distance_from_neutral < NEUTRAL_DISTANCE_LOW_CONFIDENCE:
        if factual_claim.decision_confidence == "high":
            factual_claim.decision_confidence = "medium"
        else:
            factual_claim.decision_confidence = "low"
