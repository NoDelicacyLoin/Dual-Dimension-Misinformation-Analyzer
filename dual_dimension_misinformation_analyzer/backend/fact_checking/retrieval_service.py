from dataclasses import dataclass

from api_contract import AnalysisOptions, EachEvidence
from fact_checking.nli_filter import filter_top_evidence
from fact_checking.search import fetch_search_evidence
from shared_constants import DEFAULT_RETRIEVAL_RESULTS


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
    search_raw_count: int


@dataclass
class EvidenceSearchResult:
    retrieval: RetrievalResult | None
    selection: SelectionResult | None


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


def normalize_search_query(search_query: str) -> str:
    return " ".join(str(search_query or "").replace('"', "").lower().split())


def build_fact_check_target(checkable_claim: str, constraints: list[str] | None = None) -> str:
    clean_claim = " ".join(str(checkable_claim or "").split())
    if not clean_claim:
        return ""

    target_parts = [clean_claim.rstrip(" .;:,")]
    target_key = normalize_search_query(clean_claim)
    seen_constraint_keys = set()

    for raw_constraint in constraints or []:
        constraint = " ".join(str(raw_constraint or "").split())
        constraint_key = normalize_search_query(constraint)
        if not constraint or not constraint_key:
            continue
        if constraint_key in target_key or constraint_key in seen_constraint_keys:
            continue
        target_parts.append(constraint)
        seen_constraint_keys.add(constraint_key)

    if len(target_parts) == 1:
        return clean_claim
    return " ".join(target_parts)


def build_search_queries(checkable_claim: str, constraints: list[str] | None = None) -> list[str]:
    clean_query = build_fact_check_target(checkable_claim, constraints)
    return [clean_query] if clean_query else []


def retrieve_evidence(
    search_query: str,
    retrieval_results: int = DEFAULT_RETRIEVAL_RESULTS,
) -> RetrievalResult:
    search_result = fetch_search_evidence(search_query, max_results=retrieval_results)
    return RetrievalResult(
        raw_evidence=search_result.evidence,
        search_query=search_query,
        search_raw_count=len(search_result.evidence),
        error_type=search_result.error_type,
        error_message=search_result.error_message,
    )


def choose_evidence(
    checkable_claim: str,
    raw_evidence: list[dict],
    relevance_threshold: float,
    top_k: int,
    use_all_eligible_evidence: bool,
) -> SelectionResult:
    selected_evidence_dicts, filter_debug = filter_top_evidence(
        checkable_claim,
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
        search_raw_count=len(raw_evidence),
    )


def search_for_evidence(
    search_query: str,
    checkable_claim: str,
    options: AnalysisOptions,
) -> EvidenceSearchResult:
    retrieval = retrieve_evidence(
        search_query=search_query,
        retrieval_results=options.retrieval_results,
    )

    if retrieval.error_type:
        return EvidenceSearchResult(retrieval=retrieval, selection=None)

    selection = choose_evidence(
        checkable_claim=checkable_claim,
        raw_evidence=retrieval.raw_evidence,
        relevance_threshold=options.relevance_threshold,
        top_k=options.top_k,
        use_all_eligible_evidence=options.use_all_eligible_evidence,
    )

    return EvidenceSearchResult(
        retrieval=retrieval,
        selection=selection,
    )
