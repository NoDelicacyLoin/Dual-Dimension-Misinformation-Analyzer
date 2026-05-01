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
    map_truth_score_to_verdict,
    summarize_selected_evidence,
)
from fact_checking.gemini_agent import (
    apply_gemini_verdict_to_factual_claim,
    is_gemini_available,
    prepare_claim_for_fact_checking,
)
from fact_checking.retrieval_service import (
    choose_evidence,
    explain_empty_selection,
    get_selected_evidence_quality,
    retrieve_evidence,
)
from fact_checking.stabilization import stabilize_result


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


def get_fact_checking_verdict_label(truth_score: float | None) -> str | None:
    if truth_score is None:
        return None
    if truth_score >= 0.85:
        return "True"
    if truth_score >= 0.65:
        return "Mostly True"
    if truth_score >= 0.45:
        return "Neutral"
    if truth_score >= 0.25:
        return "Mostly False"
    return "False"


def build_fact_checking_summary(fact_checking: EachFactChecking) -> None:
    truth_scores = []
    fallback_scores = []

    for factual_claim in fact_checking.factual_claims:
        if factual_claim.truth_score is not None:
            fallback_scores.append(factual_claim.truth_score)
        if factual_claim.truth_score is not None and factual_claim.status == "success":
            truth_scores.append(factual_claim.truth_score)

    if truth_scores:
        fact_checking.truth_score = sum(truth_scores) / len(truth_scores)
        fact_checking.explanation = (
            f"Aggregated mean truth score over {len(truth_scores)} successful factual claim(s)."
        )
    elif fallback_scores:
        fact_checking.truth_score = sum(fallback_scores) / len(fallback_scores)
        fact_checking.explanation = "Aggregated mean truth score including degraded or partial factual-claim runs."
    else:
        fact_checking.truth_score = None
        fact_checking.explanation = "No numeric truth score was available for the evidence-based branch."

    fact_checking.verdict = get_fact_checking_verdict_label(fact_checking.truth_score)


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


def get_selection_rank(selected_evidence) -> tuple[int, int]:
    quality_rank = {
        "weak": 1,
        "mixed": 2,
        "strong": 3,
    }
    evidence_quality = get_selected_evidence_quality(selected_evidence)
    return quality_rank.get(evidence_quality, 0), len(selected_evidence)


def should_try_rewrite_fallback(
    selected_evidence,
    top_k: int,
    use_all_eligible_evidence: bool,
) -> bool:
    evidence_quality = get_selected_evidence_quality(selected_evidence)
    return (
        not selected_evidence
        or (
            not use_all_eligible_evidence
            and len(selected_evidence) < top_k
            and evidence_quality != "strong"
        )
    )


def find_rewrite_fallback_evidence(
    atomic_claim: str,
    claim_for_verdict: str,
    options: AnalysisOptions,
):
    rewrite_check = prepare_claim_for_fact_checking(
        atomic_claim,
        use_query_rewrite=True,
    )
    rewrite_query = rewrite_check.search_query or rewrite_check.final_claim

    if not rewrite_check.is_valid_claim or not rewrite_query or rewrite_query == atomic_claim:
        return None, None

    fallback_retrieval = retrieve_evidence(
        search_query=rewrite_query,
        retrieval_results=options.retrieval_results,
        use_oversampling_retry=options.use_oversampling_retry,
    )
    if fallback_retrieval.error_type:
        return None, None

    fallback_selection = choose_evidence(
        final_claim=claim_for_verdict,
        raw_evidence=fallback_retrieval.raw_evidence,
        relevance_threshold=options.relevance_threshold,
        top_k=options.top_k,
        use_all_eligible_evidence=options.use_all_eligible_evidence,
    )

    if not fallback_selection.selected_evidence:
        return None, None

    return fallback_retrieval, fallback_selection


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
    map_truth_score_to_verdict(factual_claim)


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
        use_query_rewrite=False,
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

    retrieval = retrieve_evidence(
        search_query=atomic_claim,
        retrieval_results=options.retrieval_results,
        use_oversampling_retry=options.use_oversampling_retry,
    )
    raw_evidence = retrieval.raw_evidence

    factual_claim.metadata = EachFactualClaimMetadata(
        retrieval_query_used=retrieval.final_claim,
        fallback_used=False,
        search_raw_evidence_count=retrieval.search_raw_count,
    )

    if retrieval.error_type and retrieval.error_type != "no_results":
        set_unsuccessful_fact_check(
            factual_claim,
            status="system_error",
            explanation=retrieval.error_message or "Search failed.",
        )
        return factual_claim

    if retrieval.error_type == "no_results":
        selection = None
        filter_debug = {}
    else:
        selection = choose_evidence(
            final_claim=claim_for_verdict,
            raw_evidence=raw_evidence,
            relevance_threshold=options.relevance_threshold,
            top_k=options.top_k,
            use_all_eligible_evidence=options.use_all_eligible_evidence,
        )
        filter_debug = selection.filter_debug

    should_try_fallback = (
        retrieval.error_type == "no_results"
        or should_try_rewrite_fallback(
            selection.selected_evidence if selection else [],
            top_k=options.top_k,
            use_all_eligible_evidence=options.use_all_eligible_evidence,
        )
    )

    if should_try_fallback:
        fallback_retrieval, fallback_selection = find_rewrite_fallback_evidence(
            atomic_claim=atomic_claim,
            claim_for_verdict=claim_for_verdict,
            options=options,
        )

        if fallback_retrieval and fallback_selection:
            current_rank = get_selection_rank(selection.selected_evidence if selection else [])
            fallback_rank = get_selection_rank(fallback_selection.selected_evidence)
            if fallback_rank > current_rank:
                retrieval = fallback_retrieval
                raw_evidence = fallback_retrieval.raw_evidence
                selection = fallback_selection
                filter_debug = fallback_selection.filter_debug
                factual_claim.metadata.retrieval_query_used = fallback_retrieval.final_claim
                factual_claim.metadata.search_raw_evidence_count = fallback_retrieval.search_raw_count
                factual_claim.metadata.fallback_used = True

    if selection is None:
        set_unsuccessful_fact_check(
            factual_claim,
            status="no_evidence",
            explanation=retrieval.error_message or "No reliable evidence found for this claim.",
        )
        return factual_claim

    claim_for_verdict = selection.final_claim
    selected_evidence = selection.selected_evidence

    factual_claim.evidence = selected_evidence
    factual_claim.metadata.selected_evidence_count = len(factual_claim.evidence)

    if not factual_claim.evidence:
        _, explanation = explain_empty_selection(raw_evidence, filter_debug)
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

    should_try_post_gemini_fallback = (
        finished_claim.status == "insufficient_evidence"
        and not finished_claim.metadata.fallback_used
        and count_decision_usable_sources(finished_claim) == 0
    )
    if not should_try_post_gemini_fallback:
        return finished_claim

    fallback_retrieval, fallback_selection = find_rewrite_fallback_evidence(
        atomic_claim=atomic_claim,
        claim_for_verdict=claim_for_verdict,
        options=options,
    )
    if not fallback_retrieval or not fallback_selection:
        finished_claim.status = "pending"
        finished_claim.truth_score = None
        finished_claim.verdict = None
        finished_claim.explanation = ""
        finished_claim.decision_confidence = ""
        finished_claim.evidence_sufficiency = ""
        return finish_fact_check_with_selected_evidence(
            final_claim=claim_for_verdict,
            factual_claim=finished_claim,
            use_selective_stabilization=options.use_selective_stabilization,
        )

    finished_claim.status = "pending"
    finished_claim.truth_score = None
    finished_claim.verdict = None
    finished_claim.explanation = ""
    finished_claim.decision_confidence = ""
    finished_claim.evidence_sufficiency = ""
    finished_claim.evidence = fallback_selection.selected_evidence
    finished_claim.metadata.retrieval_query_used = fallback_retrieval.final_claim
    finished_claim.metadata.search_raw_evidence_count = fallback_retrieval.search_raw_count
    finished_claim.metadata.selected_evidence_count = len(finished_claim.evidence)
    finished_claim.metadata.fallback_used = True

    return finish_fact_check_with_selected_evidence(
        final_claim=claim_for_verdict,
        factual_claim=finished_claim,
        use_selective_stabilization=options.use_selective_stabilization,
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
