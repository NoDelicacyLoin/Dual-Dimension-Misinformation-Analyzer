from dataclasses import dataclass

from api_contract import EachEvidence
from fact_checking.nli_filter import filter_top_evidence
from fact_checking.search import fetch_oversampled_evidence
from shared_constants import (
    DEFAULT_RETRIEVAL_RESULTS,
    MIN_RESULTS_BEFORE_RETRY,
    OVERSAMPLING_EXTRA_RESULTS,
)


@dataclass
class RetrievalResult:
    raw_evidence: list[dict]
    final_claim: str
    search_raw_count: int
    error_type: str = ""
    error_message: str = ""


@dataclass
class SelectionResult:
    selected_evidence: list[EachEvidence]
    filter_debug: dict
    final_claim: str
    search_raw_count: int


def explain_empty_selection(raw_evidence: list[dict], filter_debug: dict) -> tuple[str, str]:
    if not raw_evidence:
        return (
            "no_retrieved_evidence",
            "The search step did not return any usable evidence for this claim.",
        )

    scored_evidence = filter_debug.get("scored_evidence", []) if filter_debug else []
    if not scored_evidence:
        return (
            "no_scored_evidence",
            "Retrieved pages did not contain enough text for evidence scoring.",
        )

    filter_reason_counts: dict[str, int] = {}
    for scored_item in scored_evidence:
        filter_reason = scored_item.get("filter_reason", "unknown")
        if filter_reason == "passed":
            continue
        filter_reason_counts[filter_reason] = filter_reason_counts.get(filter_reason, 0) + 1

    if not filter_reason_counts:
        return (
            "no_selected_evidence",
            "The system could not retain any evidence for final judging.",
        )

    dominant_reason = max(filter_reason_counts, key=filter_reason_counts.get)
    if dominant_reason == "below_relevance_threshold":
        return (
            "all_evidence_below_relevance_threshold",
            "Retrieved evidence was found, but it did not look directly relevant enough to the claim.",
        )
    if dominant_reason == "low_claim_match":
        return (
            "all_evidence_low_claim_match",
            "Retrieved evidence was found, but it did not line up closely enough with the specific claim to support a verdict.",
        )
    if dominant_reason == "missing_claim_anchor":
        return (
            "all_evidence_missing_claim_anchor",
            "Retrieved evidence was found, but it did not cover the main entity or title needed to check the claim.",
        )
    if dominant_reason == "conflicting_claim_details":
        return (
            "all_evidence_conflicting_claim_details",
            "Retrieved evidence discussed the topic, but key claim details pointed in a different direction.",
        )

    return (
        "no_selected_evidence",
        "The system could not retain any evidence for final judging.",
    )


def run_retrieval_with_oversampling_retry(
    final_claim: str,
    retrieval_results: int,
    use_oversampling_retry: bool,
) -> RetrievalResult:
    first_pass = fetch_oversampled_evidence(final_claim, max_results=retrieval_results)

    if (
        first_pass.error_type
        or not use_oversampling_retry
        or len(first_pass.evidence) > MIN_RESULTS_BEFORE_RETRY
    ):
        return RetrievalResult(
            raw_evidence=first_pass.evidence,
            final_claim=final_claim,
            search_raw_count=len(first_pass.evidence),
            error_type=first_pass.error_type,
            error_message=first_pass.error_message,
        )

    oversampled_result_count = retrieval_results + OVERSAMPLING_EXTRA_RESULTS
    second_pass = fetch_oversampled_evidence(final_claim, max_results=oversampled_result_count)
    if second_pass.error_type or len(second_pass.evidence) <= len(first_pass.evidence):
        return RetrievalResult(
            raw_evidence=first_pass.evidence,
            final_claim=final_claim,
            search_raw_count=len(first_pass.evidence),
            error_type=first_pass.error_type,
            error_message=first_pass.error_message,
        )

    return RetrievalResult(
        raw_evidence=second_pass.evidence,
        final_claim=final_claim,
        search_raw_count=len(second_pass.evidence),
        error_type=second_pass.error_type,
        error_message=second_pass.error_message,
    )


def build_evidence_objects(selected_evidence: list[dict]) -> list[EachEvidence]:
    evidence_objects: list[EachEvidence] = []
    for evidence_item in selected_evidence:
        evidence_objects.append(EachEvidence(**evidence_item))
    return evidence_objects


def get_selected_evidence_quality(selected_evidence: list[EachEvidence]) -> str:
    if not selected_evidence:
        return "weak"

    evidence_score = 0
    evidence_quality = "weak"

    for evidence_item in selected_evidence:
        item_quality = evidence_item.evidence_quality
        if item_quality == "strong":
            evidence_score += 2
            evidence_quality = "strong"
        elif item_quality == "usable":
            evidence_score += 1
            if evidence_quality != "strong":
                evidence_quality = "mixed"

    if evidence_score <= 0:
        return "weak"
    return evidence_quality


def retrieve_evidence(
    search_query: str,
    retrieval_results: int = DEFAULT_RETRIEVAL_RESULTS,
    use_oversampling_retry: bool = True,
) -> RetrievalResult:
    retrieval_attempt = run_retrieval_with_oversampling_retry(
        search_query,
        retrieval_results=retrieval_results,
        use_oversampling_retry=use_oversampling_retry,
    )

    return RetrievalResult(
        raw_evidence=retrieval_attempt.raw_evidence,
        final_claim=search_query,
        search_raw_count=retrieval_attempt.search_raw_count,
        error_type=retrieval_attempt.error_type,
        error_message=retrieval_attempt.error_message,
    )


def choose_evidence(
    final_claim: str,
    raw_evidence: list[dict],
    relevance_threshold: float,
    top_k: int,
    use_all_eligible_evidence: bool,
) -> SelectionResult:
    selected_evidence_dicts, filter_debug = filter_top_evidence(
        final_claim,
        raw_evidence,
        relevance_threshold=relevance_threshold,
        top_k=top_k,
        use_all_eligible_evidence=use_all_eligible_evidence,
        return_debug_info=True,
    )
    selected_evidence = build_evidence_objects(selected_evidence_dicts)

    return SelectionResult(
        selected_evidence=selected_evidence,
        filter_debug=filter_debug,
        final_claim=final_claim,
        search_raw_count=len(raw_evidence),
    )
