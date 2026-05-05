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
    fallback_used: bool = False
    error_type: str = ""
    error_message: str = ""


@dataclass
class SelectionResult:
    selected_evidence: list[EachEvidence]
    filter_debug: dict
    final_claim: str
    search_raw_count: int
    fallback_used: bool


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


def get_average_source_quality(raw_evidence: list[dict]) -> float:
    if not raw_evidence:
        return 0.0

    return sum(
        evidence_item.get("source_quality_score", 0.65)
        for evidence_item in raw_evidence
    ) / len(raw_evidence)


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
    original_claim: str,
    final_claim: str,
    retrieval_results: int = DEFAULT_RETRIEVAL_RESULTS,
    use_oversampling_retry: bool = True,
) -> RetrievalResult:
    retrieval_attempt = run_retrieval_with_oversampling_retry(
        final_claim,
        retrieval_results=retrieval_results,
        use_oversampling_retry=use_oversampling_retry,
    )
    raw_evidence = retrieval_attempt.raw_evidence
    active_final_claim = final_claim
    fallback_used = False

    search_failed = bool(retrieval_attempt.error_type)
    current_average_quality = get_average_source_quality(retrieval_attempt.raw_evidence)
    search_too_weak = retrieval_attempt.search_raw_count < 2 or current_average_quality < 0.72

    if (search_failed or search_too_weak) and final_claim != original_claim:
        original_attempt = run_retrieval_with_oversampling_retry(
            original_claim,
            retrieval_results=retrieval_results,
            use_oversampling_retry=use_oversampling_retry,
        )
        original_average_quality = get_average_source_quality(original_attempt.raw_evidence)

        original_is_better = (
            original_average_quality > current_average_quality
            or (
                original_average_quality == current_average_quality
                and original_attempt.search_raw_count > retrieval_attempt.search_raw_count
            )
        )
        if not original_attempt.error_type and original_is_better:
            raw_evidence = original_attempt.raw_evidence
            active_final_claim = original_claim
            retrieval_attempt = original_attempt
            fallback_used = True

    return RetrievalResult(
        raw_evidence=raw_evidence,
        final_claim=active_final_claim,
        search_raw_count=retrieval_attempt.search_raw_count,
        fallback_used=fallback_used,
        error_type=retrieval_attempt.error_type,
        error_message=retrieval_attempt.error_message,
    )


def choose_evidence(
    original_claim: str,
    final_claim: str,
    raw_evidence: list[dict],
    relevance_threshold: float,
    top_k: int,
    use_all_eligible_evidence: bool,
    retrieval_results: int,
    use_oversampling_retry: bool,
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

    evidence_quality = get_selected_evidence_quality(selected_evidence)
    should_check_original_claim = (
        final_claim != original_claim
        and (
            not selected_evidence
            or (
                not use_all_eligible_evidence
                and len(selected_evidence) < top_k
                and evidence_quality != "strong"
            )
        )
    )

    if not should_check_original_claim:
        return SelectionResult(
            selected_evidence=selected_evidence,
            filter_debug=filter_debug,
            final_claim=final_claim,
            search_raw_count=len(raw_evidence),
            fallback_used=False,
        )

    original_attempt = run_retrieval_with_oversampling_retry(
        original_claim,
        retrieval_results=retrieval_results,
        use_oversampling_retry=use_oversampling_retry,
    )

    if original_attempt.error_type:
        return SelectionResult(
            selected_evidence=selected_evidence,
            filter_debug=filter_debug,
            final_claim=final_claim,
            search_raw_count=len(raw_evidence),
            fallback_used=False,
        )

    fallback_selected_evidence_dicts, fallback_filter_debug = filter_top_evidence(
        original_claim,
        original_attempt.raw_evidence,
        relevance_threshold=relevance_threshold,
        top_k=top_k,
        use_all_eligible_evidence=use_all_eligible_evidence,
        return_debug_info=True,
    )
    fallback_selected_evidence = build_evidence_objects(fallback_selected_evidence_dicts)

    if not fallback_selected_evidence:
        return SelectionResult(
            selected_evidence=selected_evidence,
            filter_debug=filter_debug,
            final_claim=final_claim,
            search_raw_count=len(raw_evidence),
            fallback_used=False,
        )

    quality_rank = {
        "weak": 1,
        "mixed": 2,
        "strong": 3,
    }
    current_rank = (
        quality_rank.get(get_selected_evidence_quality(selected_evidence), 0) if selected_evidence else 0,
        len(selected_evidence),
    )
    fallback_rank = (
        quality_rank.get(get_selected_evidence_quality(fallback_selected_evidence), 0),
        len(fallback_selected_evidence),
    )

    if fallback_rank <= current_rank:
        return SelectionResult(
            selected_evidence=selected_evidence,
            filter_debug=filter_debug,
            final_claim=final_claim,
            search_raw_count=len(raw_evidence),
            fallback_used=False,
        )

    return SelectionResult(
        selected_evidence=fallback_selected_evidence,
        filter_debug=fallback_filter_debug,
        final_claim=original_claim,
        search_raw_count=original_attempt.search_raw_count,
        fallback_used=True,
    )
