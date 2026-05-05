from api_contract import EachFactualClaim
from fact_checking.decision_utils import aggregate_truth_score
from fact_checking.gemini_agent import apply_gemini_verdict_to_factual_claim
from shared_constants import (
    BOUNDARY_MARGIN,
    LARGE_STABILIZATION_DELTA,
    NEAR_NEUTRAL_WINDOW,
    NEUTRAL_SCORE,
    SMALL_STABILIZATION_DELTA,
    VERDICT_BOUNDARIES,
)


def should_trigger_selective_stabilization(
    factual_claim: EachFactualClaim,
    use_selective_stabilization: bool,
) -> bool:
    if not use_selective_stabilization:
        return False

    truth_score = factual_claim.truth_score if factual_claim.truth_score is not None else NEUTRAL_SCORE
    decision_confidence = factual_claim.decision_confidence

    if decision_confidence == "low":
        return True
    if decision_confidence != "medium":
        return False

    distance_from_neutral = abs(truth_score - NEUTRAL_SCORE)
    near_verdict_boundary = any(abs(truth_score - boundary) < BOUNDARY_MARGIN for boundary in VERDICT_BOUNDARIES)

    return distance_from_neutral < NEAR_NEUTRAL_WINDOW or near_verdict_boundary


def stabilize_result(
    claim_for_verdict: str,
    factual_claim: EachFactualClaim,
    use_selective_stabilization: bool = True,
) -> None:
    first_truth_score = factual_claim.truth_score if factual_claim.truth_score is not None else NEUTRAL_SCORE
    first_explanation = factual_claim.explanation

    if not should_trigger_selective_stabilization(
        factual_claim=factual_claim,
        use_selective_stabilization=use_selective_stabilization,
    ):
        return

    second_factual_claim = factual_claim.model_copy(deep=True)
    apply_gemini_verdict_to_factual_claim(claim_for_verdict, second_factual_claim)
    aggregate_truth_score(second_factual_claim)

    second_truth_score = second_factual_claim.truth_score if second_factual_claim.truth_score is not None else NEUTRAL_SCORE
    second_explanation = second_factual_claim.explanation or first_explanation
    stabilization_delta = abs(first_truth_score - second_truth_score)

    crossed_neutral_boundary = (
        (first_truth_score < NEUTRAL_SCORE < second_truth_score)
        or (second_truth_score < NEUTRAL_SCORE < first_truth_score)
    )
    if crossed_neutral_boundary or stabilization_delta >= LARGE_STABILIZATION_DELTA:
        factual_claim.truth_score = NEUTRAL_SCORE
        factual_claim.explanation = (
            f"{first_explanation} The result was re-checked because the case was not fully stable. "
            "The second scoring pass pointed in a meaningfully different direction, so the final score was reset to a neutral value."
        ).strip()
        return

    factual_claim.truth_score = (first_truth_score + second_truth_score) / 2
    if stabilization_delta < SMALL_STABILIZATION_DELTA:
        factual_claim.explanation = first_explanation
        return

    factual_claim.explanation = f"{first_explanation} The result was re-checked because the case was borderline.".strip()
    if second_explanation and second_explanation != first_explanation:
        factual_claim.explanation = (
            f"{factual_claim.explanation} A second scoring pass produced a meaningfully different score, "
            "so the final score was stabilized toward the middle."
        )
