from api_contract import EachFactualClaim
from fact_checking.gemini_agent import (
    prepare_rewrite_query_for_fallback,
)
from fact_checking.retrieval_service import normalize_search_query


def should_use_rewrite_fallback(search_result) -> bool:
    retrieval = search_result.retrieval if search_result else None
    if retrieval is None:
        return True
    if retrieval.error_type and retrieval.error_type != "no_results":
        return False

    selection = search_result.selection
    return selection is None or len(selection.selected_evidence) == 0


def add_rewrite_fallback_queries(search_queries: list[str], checkable_claim: str) -> list[str]:
    rewrite_check = prepare_rewrite_query_for_fallback(checkable_claim)

    if not rewrite_check.is_valid_claim:
        return search_queries

    clean_queries = list(search_queries)
    seen_queries = {normalize_search_query(query) for query in clean_queries}
    clean_query = str(rewrite_check.search_query or "").strip()
    query_key = normalize_search_query(clean_query)
    if clean_query and query_key not in seen_queries:
        clean_queries.append(clean_query)

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
