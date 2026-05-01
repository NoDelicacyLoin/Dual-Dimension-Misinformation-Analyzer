from api_contract import AnalysisOptions, EachEvidence, EachFactualClaim
from fact_checking.decision_utils import aggregate_truth_score
from fact_checking.gemini_agent import (
    apply_gemini_verdict_to_factual_claim,
    prepare_claim_for_fact_checking,
)
from fact_checking.retrieval_service import (
    build_frame_search_query,
    has_enough_evidence,
    normalize_search_query,
)
from shared_constants import (
    BOUNDARY_MARGIN,
    LARGE_STABILIZATION_DELTA,
    NEAR_NEUTRAL_WINDOW,
    NEUTRAL_SCORE,
    SMALL_STABILIZATION_DELTA,
    VERDICT_BOUNDARIES,
)


def clear_fact_check_judgment(factual_claim: EachFactualClaim) -> None:
    factual_claim.status = "pending"
    factual_claim.truth_score = None
    factual_claim.verdict = None
    factual_claim.explanation = ""
    factual_claim.decision_confidence = ""
    factual_claim.evidence_sufficiency = ""


def should_use_fallback(
    selected_evidence: list[EachEvidence],
    options: AnalysisOptions,
) -> bool:
    return not has_enough_evidence(
        selected_evidence,
        top_k=options.top_k,
        use_all_eligible_evidence=options.use_all_eligible_evidence,
    )


def add_rewrite_fallback_queries(search_queries: list[str], atomic_claim: str) -> list[str]:
    rewrite_check = prepare_claim_for_fact_checking(
        atomic_claim,
        use_query_rewrite=True,
    )

    if not rewrite_check.is_valid_claim:
        return search_queries

    backup_queries = [
        rewrite_check.search_query,
        build_frame_search_query(rewrite_check),
    ]

    clean_queries = list(search_queries)
    seen_queries = {normalize_search_query(query) for query in clean_queries}
    for query in backup_queries:
        clean_query = str(query or "").strip()
        query_key = normalize_search_query(clean_query)
        if clean_query and query_key not in seen_queries:
            clean_queries.append(clean_query)
            seen_queries.add(query_key)

    return clean_queries


def apply_evidence_to_claim(
    factual_claim: EachFactualClaim,
    retrieval,
    selection,
    queries_tried: list[str],
    primary_query: str,
) -> None:
    factual_claim.evidence = selection.selected_evidence
    factual_claim.metadata.retrieval_query_used = retrieval.search_query
    factual_claim.metadata.retrieval_queries_tried = queries_tried
    factual_claim.metadata.fallback_used = (
        normalize_search_query(retrieval.search_query) != normalize_search_query(primary_query)
    )
    factual_claim.metadata.search_raw_evidence_count = retrieval.search_raw_count
    factual_claim.metadata.selected_evidence_count = len(factual_claim.evidence)


# Verdict stabilization


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
