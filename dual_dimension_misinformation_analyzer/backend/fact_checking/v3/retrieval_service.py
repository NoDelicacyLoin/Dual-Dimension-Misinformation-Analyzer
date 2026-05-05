from dataclasses import dataclass

from api_contract import AnalysisOptions, EachEvidence
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
    search_query: str
    search_raw_count: int
    error_type: str = ""
    error_message: str = ""


@dataclass
class SelectionResult:
    selected_evidence: list[EachEvidence]
    filter_debug: dict
    claim_for_verdict: str
    search_raw_count: int


@dataclass
class SearchPlanResult:
    retrieval: RetrievalResult | None
    selection: SelectionResult | None
    queries_tried: list[str]
    next_query_index: int


def explain_empty_selection(raw_evidence: list[dict], filter_debug: dict) -> str:
    if not raw_evidence:
        return "The search step did not return any usable evidence for this claim."

    scored_evidence = filter_debug.get("scored_evidence", []) if filter_debug else []
    if not scored_evidence:
        return "Retrieved pages did not contain enough text for evidence scoring."

    filter_reason_counts: dict[str, int] = {}
    for scored_item in scored_evidence:
        filter_reason = scored_item.get("filter_reason", "unknown")
        if filter_reason == "passed":
            continue
        filter_reason_counts[filter_reason] = filter_reason_counts.get(filter_reason, 0) + 1

    if not filter_reason_counts:
        return "The system could not retain any evidence for final judging."

    dominant_reason = max(filter_reason_counts, key=filter_reason_counts.get)
    if dominant_reason == "below_relevance_threshold":
        return "Retrieved evidence was found, but it did not look directly relevant enough to the claim."
    if dominant_reason == "low_claim_match":
        return "Retrieved evidence was found, but it did not line up closely enough with the specific claim to support a verdict."
    if dominant_reason == "missing_claim_anchor":
        return "Retrieved evidence was found, but it did not cover the main entity or title needed to check the claim."
    return "The system could not retain any evidence for final judging."


def run_retrieval_with_oversampling_retry(
    search_query: str,
    retrieval_results: int,
    use_oversampling_retry: bool,
) -> RetrievalResult:
    first_pass = fetch_oversampled_evidence(search_query, max_results=retrieval_results)

    if (
        first_pass.error_type
        or not use_oversampling_retry
        or len(first_pass.evidence) > MIN_RESULTS_BEFORE_RETRY
    ):
        return RetrievalResult(
            raw_evidence=first_pass.evidence,
            search_query=search_query,
            search_raw_count=len(first_pass.evidence),
            error_type=first_pass.error_type,
            error_message=first_pass.error_message,
        )

    oversampled_result_count = retrieval_results + OVERSAMPLING_EXTRA_RESULTS
    second_pass = fetch_oversampled_evidence(search_query, max_results=oversampled_result_count)
    if second_pass.error_type or len(second_pass.evidence) <= len(first_pass.evidence):
        return RetrievalResult(
            raw_evidence=first_pass.evidence,
            search_query=search_query,
            search_raw_count=len(first_pass.evidence),
            error_type=first_pass.error_type,
            error_message=first_pass.error_message,
        )

    return RetrievalResult(
        raw_evidence=second_pass.evidence,
        search_query=search_query,
        search_raw_count=len(second_pass.evidence),
        error_type=second_pass.error_type,
        error_message=second_pass.error_message,
    )


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


def get_selection_rank(selected_evidence: list[EachEvidence]) -> tuple[int, int]:
    quality_rank = {
        "weak": 1,
        "mixed": 2,
        "strong": 3,
    }
    evidence_quality = get_selected_evidence_quality(selected_evidence)
    return quality_rank.get(evidence_quality, 0), len(selected_evidence)


def normalize_search_query(search_query: str) -> str:
    return " ".join(str(search_query or "").replace('"', "").lower().split())


def has_enough_evidence(
    selected_evidence: list[EachEvidence],
    top_k: int,
    use_all_eligible_evidence: bool,
) -> bool:
    return (
        not use_all_eligible_evidence
        and len(selected_evidence) >= top_k
        and get_selected_evidence_quality(selected_evidence) == "strong"
    )


def quote_search_part(text: str) -> str:
    cleaned_text = text.strip()
    if not cleaned_text:
        return ""
    if " " in cleaned_text and not (cleaned_text.startswith('"') and cleaned_text.endswith('"')):
        return f'"{cleaned_text}"'
    return cleaned_text


def build_frame_search_query(claim_check) -> str:
    query_parts = []

    for entity in claim_check.main_entities:
        query_part = quote_search_part(entity)
        if query_part:
            query_parts.append(query_part)

    if claim_check.relation:
        query_parts.append(claim_check.relation)

    for constraint in claim_check.constraints:
        if constraint:
            query_parts.append(constraint)

    return " ".join(query_parts).strip()


def build_search_queries(atomic_claim: str, claim_check) -> list[str]:
    queries = [
        atomic_claim,
        claim_check.search_query,
        build_frame_search_query(claim_check),
    ]

    clean_queries = []
    seen_queries = set()
    for query in queries:
        clean_query = str(query or "").strip()
        query_key = normalize_search_query(clean_query)
        if clean_query and query_key not in seen_queries:
            clean_queries.append(clean_query)
            seen_queries.add(query_key)

    return clean_queries


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
        search_query=search_query,
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
    selected_evidence = [EachEvidence(**evidence_item) for evidence_item in selected_evidence_dicts]

    return SelectionResult(
        selected_evidence=selected_evidence,
        filter_debug=filter_debug,
        claim_for_verdict=final_claim,
        search_raw_count=len(raw_evidence),
    )


def search_for_evidence(
    search_queries: list[str],
    claim_for_verdict: str,
    options: AnalysisOptions,
    current_result: SearchPlanResult | None = None,
) -> SearchPlanResult:
    if current_result is None:
        chosen_retrieval = None
        chosen_selection = None
        queries_tried = []
        current_search_position = 0
    else:
        chosen_retrieval = current_result.retrieval
        chosen_selection = current_result.selection
        queries_tried = list(current_result.queries_tried)
        current_search_position = current_result.next_query_index

    next_query_index = current_search_position

    for query_index in range(current_search_position, len(search_queries)):
        search_query = search_queries[query_index]
        next_query_index = query_index + 1
        queries_tried.append(search_query)

        retrieval = retrieve_evidence(
            search_query=search_query,
            retrieval_results=options.retrieval_results,
            use_oversampling_retry=options.use_oversampling_retry,
        )

        if retrieval.error_type:
            if chosen_retrieval is None:
                chosen_retrieval = retrieval
            continue

        selection = choose_evidence(
            final_claim=claim_for_verdict,
            raw_evidence=retrieval.raw_evidence,
            relevance_threshold=options.relevance_threshold,
            top_k=options.top_k,
            use_all_eligible_evidence=options.use_all_eligible_evidence,
        )

        if chosen_selection is None:
            chosen_retrieval = retrieval
            chosen_selection = selection
        elif get_selection_rank(selection.selected_evidence) > get_selection_rank(chosen_selection.selected_evidence):
            chosen_retrieval = retrieval
            chosen_selection = selection

        if has_enough_evidence(
            chosen_selection.selected_evidence,
            options.top_k,
            options.use_all_eligible_evidence,
        ):
            break

    return SearchPlanResult(
        retrieval=chosen_retrieval,
        selection=chosen_selection,
        queries_tried=queries_tried,
        next_query_index=next_query_index,
    )
