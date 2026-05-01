from api_contract import (
    AnalysisOptions,
    EachFactChecking,
    EachFactualClaim,
    EachFactualClaimMetadata,
)
from fact_checking.decision_utils import (
    aggregate_truth_score,
    calculate_decision_confidence,
    count_decision_usable_sources,
    set_verdict_from_truth_score,
    summarize_selected_evidence,
)
from fact_checking.gemini_agent import (
    apply_gemini_verdict_to_factual_claim,
    is_gemini_available,
    prepare_claim_for_fact_checking,
)
from fact_checking.recovery import (
    add_rewrite_fallback_queries,
    apply_evidence_to_claim,
    clear_fact_check_judgment,
    should_use_fallback,
    stabilize_result,
)
from fact_checking.retrieval_service import (
    build_search_queries,
    explain_empty_selection,
    normalize_search_query,
    search_for_evidence,
)


def normalize_analysis_options(raw_options: AnalysisOptions) -> AnalysisOptions:
    return AnalysisOptions(
        use_query_rewrite=bool(raw_options.use_query_rewrite),
        relevance_threshold=min(max(raw_options.relevance_threshold, 0.0), 1.0),
        use_oversampling_retry=bool(raw_options.use_oversampling_retry),
        use_selective_stabilization=bool(raw_options.use_selective_stabilization),
        top_k=min(max(raw_options.top_k, 1), 10),
        use_all_eligible_evidence=bool(raw_options.use_all_eligible_evidence),
        retrieval_results=min(max(raw_options.retrieval_results, 1), 20),
    )


def build_fact_checking_summary(fact_checking: EachFactChecking) -> None:
    truth_scores = []
    available_scores = []

    for factual_claim in fact_checking.factual_claims:
        if factual_claim.truth_score is not None:
            available_scores.append(factual_claim.truth_score)
        if factual_claim.truth_score is not None and factual_claim.status == "success":
            truth_scores.append(factual_claim.truth_score)

    if truth_scores:
        fact_checking.truth_score = sum(truth_scores) / len(truth_scores)
        fact_checking.explanation = (
            f"Aggregated mean truth score over {len(truth_scores)} successful factual claim(s)."
        )
    elif available_scores:
        fact_checking.truth_score = sum(available_scores) / len(available_scores)
        fact_checking.explanation = "Aggregated mean truth score including degraded or partial factual-claim runs."
    else:
        fact_checking.truth_score = None
        fact_checking.explanation = "No numeric truth score was available for the evidence-based branch."

    set_verdict_from_truth_score(fact_checking)


def mark_all_evidence_as_background(factual_claim: EachFactualClaim) -> None:
    for evidence_item in factual_claim.evidence:
        evidence_item.stance = "background"
        evidence_item.ai_analysis = "No specific analysis was generated for this source."


def set_unsuccessful_fact_check(
    factual_claim: EachFactualClaim,
    status: str,
    explanation: str,
) -> None:
    factual_claim.status = status
    factual_claim.explanation = explanation
    factual_claim.decision_confidence = "low"
    if status == "insufficient_evidence":
        factual_claim.evidence_sufficiency = "insufficient"


def finalize_successful_fact_check(
    final_claim: str,
    factual_claim: EachFactualClaim,
    use_selective_stabilization: bool,
) -> None:
    factual_claim.status = "success"
    aggregate_truth_score(factual_claim)
    calculate_decision_confidence(factual_claim)
    stabilize_result(
        claim_for_verdict=final_claim,
        factual_claim=factual_claim,
        use_selective_stabilization=use_selective_stabilization,
    )
    calculate_decision_confidence(factual_claim)
    set_verdict_from_truth_score(factual_claim)


def finish_fact_check_with_selected_evidence(
    final_claim: str,
    factual_claim: EachFactualClaim,
    use_selective_stabilization: bool,
) -> EachFactualClaim:
    if not is_gemini_available():
        mark_all_evidence_as_background(factual_claim)
        set_unsuccessful_fact_check(
            factual_claim,
            status="degraded",
            explanation="Gemini API key is missing.",
        )
        return factual_claim

    applied_count = apply_gemini_verdict_to_factual_claim(final_claim, factual_claim)

    if applied_count < len(factual_claim.evidence):
        set_unsuccessful_fact_check(
            factual_claim,
            status="degraded",
            explanation=factual_claim.explanation or "No explanation was generated.",
        )
        return factual_claim

    summarize_selected_evidence(factual_claim)

    if count_decision_usable_sources(factual_claim) == 0:
        set_unsuccessful_fact_check(
            factual_claim,
            status="insufficient_evidence",
            explanation=(
                "Selected evidence was mostly background context and did not provide enough direct "
                "supporting or contradicting signal for a verdict."
            ),
        )
        return factual_claim

    finalize_successful_fact_check(
        final_claim=final_claim,
        factual_claim=factual_claim,
        use_selective_stabilization=use_selective_stabilization,
    )
    return factual_claim


def try_more_evidence_after_background_judgment(
    finished_claim: EachFactualClaim,
    search_result,
    search_queries: list[str],
    claim_for_verdict: str,
    options: AnalysisOptions,
    primary_query: str,
) -> EachFactualClaim:
    should_continue_search = (
        finished_claim.status == "insufficient_evidence"
        and count_decision_usable_sources(finished_claim) == 0
        and search_result.next_query_index < len(search_queries)
    )
    if not should_continue_search:
        return finished_claim

    previous_search_query = finished_claim.metadata.retrieval_query_used
    search_result = search_for_evidence(
        search_queries=search_queries,
        claim_for_verdict=claim_for_verdict,
        options=options,
        current_result=search_result,
    )

    retrieval = search_result.retrieval
    selection = search_result.selection

    if not retrieval or not selection or not selection.selected_evidence:
        return finished_claim
    if normalize_search_query(retrieval.search_query) == normalize_search_query(previous_search_query):
        return finished_claim

    clear_fact_check_judgment(finished_claim)
    apply_evidence_to_claim(
        factual_claim=finished_claim,
        retrieval=retrieval,
        selection=selection,
        queries_tried=search_result.queries_tried,
        primary_query=primary_query,
    )
    return finish_fact_check_with_selected_evidence(
        final_claim=selection.claim_for_verdict,
        factual_claim=finished_claim,
        use_selective_stabilization=options.use_selective_stabilization,
    )


def run_fact_check_for_atomic_claim(
    atomic_claim: str,
    claim_group_id: int,
    fact_claim_id: int,
    original_sentence: str,
    text_feature_text: str,
    options: AnalysisOptions,
) -> EachFactualClaim:
    factual_claim = EachFactualClaim(
        claim_group_id=claim_group_id,
        fact_claim_id=fact_claim_id,
        original_sentence=original_sentence,
        text_feature_text=text_feature_text,
        claim=atomic_claim,
        status="invalid_request",
        metadata=EachFactualClaimMetadata(),
    )

    claim_check = prepare_claim_for_fact_checking(
        atomic_claim,
        use_query_rewrite=options.use_query_rewrite,
    )
    if not claim_check.is_valid_claim:
        set_unsuccessful_fact_check(
            factual_claim,
            status="invalid_request",
            explanation=(
                "This atomic claim does not look factual enough to run evidence-based fact-checking."
            ),
        )
        return factual_claim

    claim_for_verdict = claim_check.final_claim

    search_queries = build_search_queries(atomic_claim, claim_check)
    primary_query = search_queries[0] if search_queries else atomic_claim
    search_result = search_for_evidence(
        search_queries=search_queries,
        claim_for_verdict=claim_for_verdict,
        options=options,
    )

    if (
        not options.use_query_rewrite
        and should_use_fallback(
            search_result.selection.selected_evidence if search_result.selection else [],
            options=options,
        )
    ):
        search_queries = add_rewrite_fallback_queries(search_queries, atomic_claim)
        search_result = search_for_evidence(
            search_queries=search_queries,
            claim_for_verdict=claim_for_verdict,
            options=options,
            current_result=search_result,
        )

    retrieval = search_result.retrieval
    selection = search_result.selection

    if retrieval is None:
        set_unsuccessful_fact_check(
            factual_claim,
            status="no_evidence",
            explanation="No reliable evidence found for this claim.",
        )
        return factual_claim

    factual_claim.metadata = EachFactualClaimMetadata(
        retrieval_query_used=retrieval.search_query,
        retrieval_queries_tried=search_result.queries_tried,
        fallback_used=False,
        search_raw_evidence_count=retrieval.search_raw_count,
        claim_entities=claim_check.main_entities,
        claim_relation=claim_check.relation,
        claim_constraints=claim_check.constraints,
    )

    if selection is not None:
        apply_evidence_to_claim(
            factual_claim=factual_claim,
            retrieval=retrieval,
            selection=selection,
            queries_tried=search_result.queries_tried,
            primary_query=primary_query,
        )

    if retrieval.error_type and retrieval.error_type != "no_results":
        set_unsuccessful_fact_check(
            factual_claim,
            status="system_error",
            explanation=retrieval.error_message or "Search failed.",
        )
        return factual_claim

    if selection is None:
        set_unsuccessful_fact_check(
            factual_claim,
            status="no_evidence",
            explanation=retrieval.error_message or "No reliable evidence found for this claim.",
        )
        return factual_claim

    claim_for_verdict = selection.claim_for_verdict

    if not factual_claim.evidence:
        explanation = explain_empty_selection(retrieval.raw_evidence, selection.filter_debug)
        set_unsuccessful_fact_check(
            factual_claim,
            status="no_evidence",
            explanation=explanation,
        )
        return factual_claim

    finished_claim = finish_fact_check_with_selected_evidence(
        final_claim=claim_for_verdict,
        factual_claim=factual_claim,
        use_selective_stabilization=options.use_selective_stabilization,
    )

    return try_more_evidence_after_background_judgment(
        finished_claim=finished_claim,
        search_result=search_result,
        search_queries=search_queries,
        claim_for_verdict=claim_for_verdict,
        options=options,
        primary_query=primary_query,
    )


def analyze_fact_check_claims(
    claim_groups,
    raw_options: AnalysisOptions,
) -> EachFactChecking:
    options = normalize_analysis_options(raw_options)
    factual_claims: list[EachFactualClaim] = []

    for claim_group in claim_groups:
        for factual_claim in claim_group["factual_claims"]:
            factual_claims.append(
                run_fact_check_for_atomic_claim(
                    atomic_claim=factual_claim["claim"],
                    claim_group_id=claim_group["claim_group_id"],
                    fact_claim_id=factual_claim["fact_claim_id"],
                    original_sentence=claim_group["original_sentence"],
                    text_feature_text=claim_group["text_feature_text"],
                    options=options,
                )
            )

    if factual_claims and all(item.status == "success" for item in factual_claims):
        overall_status = "success"
    elif any(item.status == "success" for item in factual_claims):
        overall_status = "partial_success"
    elif any(item.status == "degraded" for item in factual_claims):
        overall_status = "degraded"
    elif any(item.status == "insufficient_evidence" for item in factual_claims):
        overall_status = "insufficient_evidence"
    elif any(item.status == "no_evidence" for item in factual_claims):
        overall_status = "no_evidence"
    else:
        overall_status = "failed"

    fact_checking = EachFactChecking(
        status=overall_status,
        factual_claims=factual_claims,
    )
    build_fact_checking_summary(fact_checking)
    return fact_checking
