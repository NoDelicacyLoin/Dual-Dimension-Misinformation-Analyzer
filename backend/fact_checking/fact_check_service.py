import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from api_contract import (
    AnalysisOptions,
    AtomizedClaimGroup,
    EachFactChecking,
    EachFactualClaim,
    EachFactualClaimMetadata,
)
from fact_checking.decision_utils import (
    aggregate_truth_score,
    calculate_decision_confidence,
    set_verdict_from_truth_score,
    summarize_selected_evidence,
)
from fact_checking.gemini_agent import (
    apply_gemini_verdict_to_factual_claim,
    is_too_empty_for_fact_checking,
    is_gemini_available,
)
from fact_checking.recovery import (
    add_rewrite_fallback_queries,
    apply_evidence_to_claim,
    should_use_rewrite_fallback,
)
from fact_checking.retrieval_service import (
    build_fact_check_target,
    build_search_queries,
    explain_empty_selection,
    search_for_evidence,
)
from shared_constants import PROGRESS_STAGE_LLM_EVIDENCE_PROGRESS, PROGRESS_STAGE_TAVILY_NLI_PROGRESS


MAX_FACT_CHECK_WORKERS = 2


def normalize_analysis_options(raw_options: AnalysisOptions) -> AnalysisOptions:
    return AnalysisOptions(
        use_query_rewrite=bool(raw_options.use_query_rewrite),
        relevance_threshold=min(max(raw_options.relevance_threshold, 0.0), 1.0),
        top_k=min(max(raw_options.top_k, 1), 10),
        use_all_eligible_evidence=bool(raw_options.use_all_eligible_evidence),
        retrieval_results=min(max(raw_options.retrieval_results, 1), 20),
    )


def derive_fact_check_status(factual_claims: list[EachFactualClaim]) -> str:
    if not factual_claims:
        return "failed"
    if all(item.status == "success" for item in factual_claims):
        return "success"
    if any(item.status == "success" for item in factual_claims):
        return "partial_success"
    if any(item.status in {"degraded", "system_error"} for item in factual_claims):
        return "degraded"
    if any(item.status == "no_evidence" for item in factual_claims):
        return "no_evidence"
    return "failed"


def build_fact_checking_summary(fact_checking: EachFactChecking) -> None:
    truth_scores = []
    for factual_claim in fact_checking.factual_claims:
        if factual_claim.truth_score is not None:
            truth_scores.append(factual_claim.truth_score)

    if truth_scores:
        fact_checking.truth_score = sum(truth_scores) / len(truth_scores)
        fact_checking.explanation = (
            f"Aggregated mean truth score over {len(truth_scores)} factual claim(s) with numeric score(s)."
        )
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


def finalize_successful_fact_check(factual_claim: EachFactualClaim) -> None:
    factual_claim.status = "success"
    aggregate_truth_score(factual_claim)
    calculate_decision_confidence(factual_claim)
    set_verdict_from_truth_score(factual_claim)


def finish_fact_check_with_selected_evidence(
    checkable_claim: str,
    factual_claim: EachFactualClaim,
) -> EachFactualClaim:
    if not is_gemini_available():
        mark_all_evidence_as_background(factual_claim)
        set_unsuccessful_fact_check(
            factual_claim,
            status="degraded",
            explanation="Gemini API key is missing.",
        )
        return factual_claim

    applied_count = apply_gemini_verdict_to_factual_claim(checkable_claim, factual_claim)

    if applied_count < len(factual_claim.evidence):
        set_unsuccessful_fact_check(
            factual_claim,
            status="degraded",
            explanation=factual_claim.explanation or "No explanation was generated.",
        )
        return factual_claim

    summarize_selected_evidence(factual_claim)

    finalize_successful_fact_check(factual_claim)
    return factual_claim


def run_fact_check_for_checkable_claim(
    checkable_claim: str,
    claim_group_id: int,
    fact_claim_id: int,
    original_sentence: str,
    text_feature_text: str,
    entities: list[str] | None,
    relation: str,
    constraints: list[str] | None,
    options: AnalysisOptions,
    retrieval_progress_callback=None,
) -> EachFactualClaim:
    checkable_claim = checkable_claim.strip()
    claim_constraints = constraints or []
    fact_check_target = build_fact_check_target(checkable_claim, claim_constraints)

    factual_claim = EachFactualClaim(
        claim_group_id=claim_group_id,
        fact_claim_id=fact_claim_id,
        original_sentence=original_sentence,
        text_feature_text=text_feature_text,
        claim=checkable_claim,
        entities=entities or [],
        relation=relation,
        constraints=claim_constraints,
        status="invalid_request",
        metadata=EachFactualClaimMetadata(),
    )
    retrieval_progress_sent = False

    def mark_retrieval_done() -> None:
        nonlocal retrieval_progress_sent
        if retrieval_progress_sent:
            return
        retrieval_progress_sent = True
        if retrieval_progress_callback is not None:
            retrieval_progress_callback(factual_claim)

    if is_too_empty_for_fact_checking(fact_check_target):
        set_unsuccessful_fact_check(
            factual_claim,
            status="invalid_request",
            explanation=(
                "This checkable claim does not look factual enough to run evidence-based fact-checking."
            ),
        )
        mark_retrieval_done()
        return factual_claim

    search_queries = build_search_queries(checkable_claim, claim_constraints)
    primary_query = search_queries[0] if search_queries else fact_check_target
    queries_tried = [primary_query] if primary_query else []
    search_result = search_for_evidence(
        search_query=primary_query,
        checkable_claim=fact_check_target,
        options=options,
    )

    if (
        options.use_query_rewrite
        and should_use_rewrite_fallback(search_result)
    ):
        fallback_queries = add_rewrite_fallback_queries(search_queries, fact_check_target)
        if len(fallback_queries) > len(search_queries):
            fallback_query = fallback_queries[-1]
            queries_tried.append(fallback_query)
            fallback_result = search_for_evidence(
                search_query=fallback_query,
                checkable_claim=fact_check_target,
                options=options,
            )
            if (
                fallback_result.retrieval is not None
                and not (
                    fallback_result.retrieval.error_type
                    and fallback_result.retrieval.error_type != "no_results"
                )
            ):
                search_result = fallback_result

    retrieval = search_result.retrieval
    selection = search_result.selection
    mark_retrieval_done()

    if retrieval is None:
        set_unsuccessful_fact_check(
            factual_claim,
            status="no_evidence",
            explanation="No reliable evidence found for this claim.",
        )
        return factual_claim

    factual_claim.metadata = EachFactualClaimMetadata(
        retrieval_query_used=retrieval.search_query,
        retrieval_queries_tried=queries_tried,
        fallback_used=False,
        search_raw_evidence_count=retrieval.search_raw_count,
    )

    if selection is not None:
        apply_evidence_to_claim(
            factual_claim=factual_claim,
            retrieval=retrieval,
            selection=selection,
            queries_tried=queries_tried,
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

    if not factual_claim.evidence:
        explanation = explain_empty_selection(retrieval.raw_evidence, selection.filter_debug)
        set_unsuccessful_fact_check(
            factual_claim,
            status="no_evidence",
            explanation=explanation,
        )
        return factual_claim

    finished_claim = finish_fact_check_with_selected_evidence(
        checkable_claim=fact_check_target,
        factual_claim=factual_claim,
    )

    return finished_claim


def analyze_fact_check_claims(
    claim_groups: list[AtomizedClaimGroup],
    raw_options: AnalysisOptions,
    progress_callback=None,
) -> EachFactChecking:
    options = normalize_analysis_options(raw_options)
    claim_jobs = []
    job_index = 0

    for claim_group in claim_groups:
        for raw_factual_claim in claim_group.fact_check_claims:
            claim_jobs.append((job_index, claim_group, raw_factual_claim))
            job_index += 1

    total_fact_claims = len(claim_jobs)
    retrieval_done_count = 0
    llm_done_count = 0
    progress_lock = threading.Lock()

    def emit_retrieval_progress(checked_claim: EachFactualClaim) -> None:
        nonlocal retrieval_done_count

        with progress_lock:
            retrieval_done_count += 1
            completed_count = retrieval_done_count

        if progress_callback is not None:
            progress_callback(
                {
                    "stage": PROGRESS_STAGE_TAVILY_NLI_PROGRESS,
                    "status": "running",
                    "message": f"Tavily and NLI finished {completed_count} of {total_fact_claims} factual claim(s).",
                    "claim_group_id": checked_claim.claim_group_id,
                    "fact_claim_id": checked_claim.fact_claim_id,
                    "completed_tavily_nli_count": completed_count,
                    "fact_check_claim_count": total_fact_claims,
                }
            )

    def run_fact_check_job(job):
        current_job_index, claim_group, raw_factual_claim = job
        checked_claim = run_fact_check_for_checkable_claim(
            checkable_claim=raw_factual_claim.claim,
            claim_group_id=claim_group.claim_group_id,
            fact_claim_id=raw_factual_claim.fact_claim_id,
            original_sentence=claim_group.original_sentence,
            text_feature_text=claim_group.text_feature_text,
            entities=raw_factual_claim.entities,
            relation=raw_factual_claim.relation,
            constraints=raw_factual_claim.constraints,
            options=options,
            retrieval_progress_callback=emit_retrieval_progress,
        )
        return current_job_index, checked_claim

    results_by_order = {}

    with ThreadPoolExecutor(max_workers=MAX_FACT_CHECK_WORKERS) as executor:
        futures = []
        for job in claim_jobs:
            futures.append(executor.submit(run_fact_check_job, job))

        for future in as_completed(futures):
            current_job_index, checked_claim = future.result()
            results_by_order[current_job_index] = checked_claim

            with progress_lock:
                llm_done_count += 1
                completed_count = llm_done_count

            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": PROGRESS_STAGE_LLM_EVIDENCE_PROGRESS,
                        "status": "running",
                        "message": f"LLM evidence judgement finished {completed_count} of {total_fact_claims} factual claim(s).",
                        "claim_group_id": checked_claim.claim_group_id,
                        "fact_claim_id": checked_claim.fact_claim_id,
                        "completed_llm_evidence_count": completed_count,
                        "fact_check_claim_count": total_fact_claims,
                    }
                )

    factual_claims = []
    for current_job_index in range(len(claim_jobs)):
        factual_claims.append(results_by_order[current_job_index])

    fact_checking = EachFactChecking(
        status=derive_fact_check_status(factual_claims),
        factual_claims=factual_claims,
    )
    build_fact_checking_summary(fact_checking)
    return fact_checking
