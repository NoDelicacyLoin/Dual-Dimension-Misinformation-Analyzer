import os
import threading
from dataclasses import dataclass

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from decision_utils import (
    aggregate_truth_score_from_source_judgments,
    apply_source_judgments_to_evidence,
    calculate_decision_confidence,
    count_borderline_candidates,
    map_truth_score_to_verdict,
    summarize_selected_evidence,
)
from gemini_agent import (
    generate_comprehensive_verdict,
    prepare_claim_for_fact_checking,
)
from nli_filter import filter_top_evidence
from search import fetch_oversampled_evidence

app = FastAPI(title="Fact-Checking API")
NEUTRAL_SCORE = 0.5
VERDICT_BOUNDARIES = [0.25, 0.45, 0.65, 0.85]
BOUNDARY_MARGIN = 0.1
NEAR_NEUTRAL_WINDOW = 0.2
LARGE_STABILIZATION_DELTA = 0.2
SMALL_STABILIZATION_DELTA = 0.1
DEFAULT_RETRIEVAL_RESULTS = 8
OVERSAMPLING_EXTRA_RESULTS = 4
MIN_RESULTS_BEFORE_RETRY = 2


class AnalysisOptions(BaseModel):
    use_query_rewrite: bool = True
    relevance_threshold: float = 0.1
    use_oversampling_retry: bool = True
    use_selective_stabilization: bool = True
    top_k: int = 3
    use_all_eligible_evidence: bool = False
    retrieval_results: int = DEFAULT_RETRIEVAL_RESULTS


class ClaimRequest(BaseModel):
    claim: str
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class EvidenceSource(BaseModel):
    url: str
    content: str
    ai_analysis: str = ""
    evidence_quality: str = ""
    source_role: str = ""
    source_strength: float = 0.0
    source_specificity: float = 0.0


class ClaimMetadata(BaseModel):
    decision_confidence: str = ""
    stabilization_used: bool = False
    stabilization_delta: float = 0.0
    stabilization_result: str = "not_triggered"
    evidence_sufficiency: str = ""
    evidence_quality: str = ""
    retrieval_strategy_used: str = ""
    retrieval_query_used: str = ""
    search_raw_evidence_count: int = 0
    original_raw_evidence_count: int = 0
    selected_evidence_count: int = 0
    fallback_used: bool = False
    excluded_borderline_candidate_count: int = 0


class ClaimResponse(BaseModel):
    status: str
    original_claim: str
    final_claim: str
    decision_stage: str
    failure_reason: str
    truth_score: float
    verdict: str
    explanation: str
    sources: list[EvidenceSource]
    metadata: ClaimMetadata = Field(default_factory=ClaimMetadata)


@dataclass
class RetrievalResult:
    raw_evidence: list[dict]
    final_claim: str
    strategy_used: str
    search_raw_count: int
    original_raw_count: int = 0
    fallback_used: bool = False
    error_type: str = ""
    error_message: str = ""


@dataclass
class SelectionResult:
    selected_evidence: list[dict]
    filter_debug: dict
    final_claim: str
    strategy_used: str
    search_raw_count: int
    original_raw_count: int
    fallback_used: bool


@dataclass
class StabilizationResult:
    truth_score: float
    explanation: str
    used: bool
    delta: float
    result: str


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
    if dominant_reason == "below_usability_floor_no_anchor":
        return (
            "all_evidence_missing_claim_anchor",
            "Retrieved evidence was found, but it did not mention the core entities, numbers, or claim anchors closely enough to support a verdict.",
        )
    if dominant_reason == "below_usability_floor_weak_anchor":
        return (
            "all_evidence_below_usability_floor",
            "Retrieved evidence was related to the topic, but the claim anchor match was still too weak to support a verdict.",
        )

    return (
        "no_selected_evidence",
        "The system could not retain any evidence for final judging.",
    )


@dataclass
class NormalizedAnalysisOptions:
    use_query_rewrite: bool
    relevance_threshold: float
    use_oversampling_retry: bool
    use_selective_stabilization: bool
    top_k: int
    use_all_eligible_evidence: bool
    retrieval_results: int


def normalize_analysis_options(raw_options: AnalysisOptions) -> NormalizedAnalysisOptions:
    relevance_threshold = min(max(raw_options.relevance_threshold, 0.0), 1.0)
    top_k = min(max(raw_options.top_k, 1), 10)
    retrieval_results = min(max(raw_options.retrieval_results, 1), 20)

    return NormalizedAnalysisOptions(
        use_query_rewrite=bool(raw_options.use_query_rewrite),
        relevance_threshold=relevance_threshold,
        use_oversampling_retry=bool(raw_options.use_oversampling_retry),
        use_selective_stabilization=bool(raw_options.use_selective_stabilization),
        top_k=top_k,
        use_all_eligible_evidence=bool(raw_options.use_all_eligible_evidence),
        retrieval_results=retrieval_results,
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
            strategy_used="",
            search_raw_count=len(first_pass.evidence),
            original_raw_count=0,
            fallback_used=False,
            error_type=first_pass.error_type,
            error_message=first_pass.error_message,
        )

    oversampled_result_count = retrieval_results + OVERSAMPLING_EXTRA_RESULTS
    second_pass = fetch_oversampled_evidence(final_claim, max_results=oversampled_result_count)
    if second_pass.error_type or len(second_pass.evidence) <= len(first_pass.evidence):
        return RetrievalResult(
            raw_evidence=first_pass.evidence,
            final_claim=final_claim,
            strategy_used="",
            search_raw_count=len(first_pass.evidence),
            original_raw_count=0,
            fallback_used=False,
            error_type=first_pass.error_type,
            error_message=first_pass.error_message,
        )

    print("[App] Retrieval oversampling retry improved raw evidence count.")
    return RetrievalResult(
        raw_evidence=second_pass.evidence,
        final_claim=final_claim,
        strategy_used="",
        search_raw_count=len(second_pass.evidence),
        original_raw_count=0,
        fallback_used=False,
        error_type=second_pass.error_type,
        error_message=second_pass.error_message,
    )


def build_response(user_claim: str, final_claim: str = "", **updates) -> ClaimResponse:
    metadata_updates = updates.pop("metadata", {})
    metadata = ClaimMetadata(**metadata_updates)

    response_data = {
        "status": "success",
        "original_claim": user_claim,
        "final_claim": final_claim,
        "decision_stage": "",
        "failure_reason": "",
        "truth_score": 0.5,
        "verdict": "Neutral",
        "explanation": "",
        "sources": [],
        "metadata": metadata,
    }
    response_data.update(updates)
    return ClaimResponse(**response_data)


def compare_evidence_sets(candidate_evidence: list[dict], current_evidence: list[dict]) -> bool:
    quality_rank = {"weak": 0, "mixed": 1, "strong": 2}
    candidate_summary = summarize_selected_evidence(candidate_evidence)
    current_summary = summarize_selected_evidence(current_evidence)

    candidate_average_usability = (
        sum(evidence_item.get("usability_score", 0.0) for evidence_item in candidate_evidence) / len(candidate_evidence)
        if candidate_evidence else 0.0
    )
    current_average_usability = (
        sum(evidence_item.get("usability_score", 0.0) for evidence_item in current_evidence) / len(current_evidence)
        if current_evidence else 0.0
    )
    candidate_average_source_quality = (
        sum(evidence_item.get("source_quality_score", 0.60) for evidence_item in candidate_evidence) / len(candidate_evidence)
        if candidate_evidence else 0.0
    )
    current_average_source_quality = (
        sum(evidence_item.get("source_quality_score", 0.60) for evidence_item in current_evidence) / len(current_evidence)
        if current_evidence else 0.0
    )

    candidate_rank = (
        quality_rank[candidate_summary.quality],
        candidate_average_source_quality,
        len(candidate_evidence),
        candidate_average_usability,
    )
    current_rank = (
        quality_rank[current_summary.quality],
        current_average_source_quality,
        len(current_evidence),
        current_average_usability,
    )

    return candidate_rank > current_rank


def score_retrieval_attempt(raw_evidence: list[dict]) -> tuple[float, int]:
    if not raw_evidence:
        return (0.0, 0)

    average_source_quality = sum(
        evidence_item.get("source_quality_score", 0.65)
        for evidence_item in raw_evidence
    ) / len(raw_evidence)

    return (
        average_source_quality,
        len(raw_evidence),
    )


def should_trigger_selective_stabilization(
    truth_score: float,
    decision_confidence: str,
    selected_evidence_count: int,
    evidence_quality: str,
    use_selective_stabilization: bool,
) -> bool:
    if not use_selective_stabilization:
        return False

    if decision_confidence == "low":
        return True

    if decision_confidence != "medium":
        return False

    distance_from_neutral = abs(truth_score - NEUTRAL_SCORE)
    near_verdict_boundary = any(
        abs(truth_score - boundary) < BOUNDARY_MARGIN
        for boundary in VERDICT_BOUNDARIES
    )

    if evidence_quality == "mixed" and selected_evidence_count <= 1:
        return True
    if evidence_quality == "mixed" and distance_from_neutral < NEAR_NEUTRAL_WINDOW:
        return True
    if near_verdict_boundary:
        return True

    return False


def stabilize_result(
    claim_for_verdict: str,
    selected_evidence: list[dict],
    first_truth_score: float,
    first_explanation: str,
    decision_confidence: str,
    evidence_quality: str,
    use_selective_stabilization: bool = True,
) -> StabilizationResult:
    if not should_trigger_selective_stabilization(
        truth_score=first_truth_score,
        decision_confidence=decision_confidence,
        selected_evidence_count=len(selected_evidence),
        evidence_quality=evidence_quality,
        use_selective_stabilization=use_selective_stabilization,
    ):
        return StabilizationResult(first_truth_score, first_explanation, False, 0.0, "not_triggered")

    second_report = generate_comprehensive_verdict(claim_for_verdict, selected_evidence)
    second_truth_score = aggregate_truth_score_from_source_judgments(
        selected_evidence=selected_evidence,
        source_judgments=second_report.get("source_judgments", []),
    )
    second_explanation = second_report.get("explanation", first_explanation)
    stabilization_delta = abs(first_truth_score - second_truth_score)

    crossed_neutral_boundary = (
        (first_truth_score < NEUTRAL_SCORE < second_truth_score) or
        (second_truth_score < NEUTRAL_SCORE < first_truth_score)
    )

    if crossed_neutral_boundary or stabilization_delta >= LARGE_STABILIZATION_DELTA:
        explanation = (
            f"{first_explanation} The result was re-checked because the case was not fully stable. "
            "The second scoring pass pointed in a meaningfully different direction, so the final score was reset to a neutral value."
        ).strip()
        return StabilizationResult(NEUTRAL_SCORE, explanation, True, stabilization_delta, "reset_to_neutral")

    stabilized_truth_score = (first_truth_score + second_truth_score) / 2

    if stabilization_delta < SMALL_STABILIZATION_DELTA:
        return StabilizationResult(stabilized_truth_score, first_explanation, True, stabilization_delta, "confirmed")

    explanation = f"{first_explanation} The result was re-checked because the case was borderline.".strip()
    if second_explanation and second_explanation != first_explanation:
        explanation = (
            f"{explanation} A second scoring pass produced a meaningfully different score, "
            "so the final score was stabilized toward the middle."
        )

    return StabilizationResult(stabilized_truth_score, explanation, True, stabilization_delta, "soft_adjusted")


def retrieve_evidence(
    user_claim: str,
    final_claim: str,
    retrieval_results: int = DEFAULT_RETRIEVAL_RESULTS,
    use_oversampling_retry: bool = True,
) -> RetrievalResult:
    strategy_used = "final_claim_only"
    retrieval_attempt = run_retrieval_with_oversampling_retry(
        final_claim,
        retrieval_results=retrieval_results,
        use_oversampling_retry=use_oversampling_retry,
    )
    raw_evidence = retrieval_attempt.raw_evidence
    active_final_claim = final_claim
    original_raw_count = 0
    fallback_used = False

    search_failed = bool(retrieval_attempt.error_type)
    current_attempt_score = score_retrieval_attempt(retrieval_attempt.raw_evidence)
    search_too_weak = retrieval_attempt.search_raw_count < 2 or current_attempt_score[0] < 0.72

    if (search_failed or search_too_weak) and final_claim != user_claim:
        original_attempt = run_retrieval_with_oversampling_retry(
            user_claim,
            retrieval_results=retrieval_results,
            use_oversampling_retry=use_oversampling_retry,
        )
        original_raw_count = original_attempt.search_raw_count

        original_attempt_score = score_retrieval_attempt(original_attempt.raw_evidence)

        if (
            not original_attempt.error_type
            and original_attempt_score > current_attempt_score
        ):
            raw_evidence = original_attempt.raw_evidence
            active_final_claim = user_claim
            retrieval_attempt = original_attempt
            strategy_used = "final_claim_then_original_fallback"
            fallback_used = True
            print("[App] Retrieval fallback used original claim.")

    if strategy_used == "final_claim_only" and use_oversampling_retry and len(retrieval_attempt.raw_evidence) > retrieval_results:
        strategy_used = "final_claim_with_oversampling_retry"

    return RetrievalResult(
        raw_evidence=raw_evidence,
        final_claim=active_final_claim,
        strategy_used=strategy_used,
        search_raw_count=retrieval_attempt.search_raw_count,
        original_raw_count=original_raw_count,
        fallback_used=fallback_used,
        error_type=retrieval_attempt.error_type,
        error_message=retrieval_attempt.error_message,
    )


def choose_evidence(
    user_claim: str,
    final_claim: str,
    raw_evidence: list[dict],
    relevance_threshold: float,
    top_k: int,
    use_all_eligible_evidence: bool,
    retrieval_results: int,
    use_oversampling_retry: bool,
) -> SelectionResult:
    selected_evidence, filter_debug = filter_top_evidence(
        final_claim,
        raw_evidence,
        relevance_threshold=relevance_threshold,
        top_k=top_k,
        use_all_eligible_evidence=use_all_eligible_evidence,
        return_debug_info=True,
    )

    evidence_summary = summarize_selected_evidence(selected_evidence)
    should_check_original_claim = (
        final_claim != user_claim
        and (
            not selected_evidence
            or (
                not use_all_eligible_evidence
                and len(selected_evidence) < top_k
                and evidence_summary.quality != "strong"
            )
        )
    )

    if not should_check_original_claim:
        return SelectionResult(
            selected_evidence=selected_evidence,
            filter_debug=filter_debug,
            final_claim=final_claim,
            strategy_used="",
            search_raw_count=len(raw_evidence),
            original_raw_count=0,
            fallback_used=False,
        )

    original_attempt = run_retrieval_with_oversampling_retry(
        user_claim,
        retrieval_results=retrieval_results,
        use_oversampling_retry=use_oversampling_retry,
    )

    if original_attempt.error_type:
        return SelectionResult(
            selected_evidence=selected_evidence,
            filter_debug=filter_debug,
            final_claim=final_claim,
            strategy_used="",
            search_raw_count=len(raw_evidence),
            original_raw_count=original_attempt.search_raw_count,
            fallback_used=False,
        )

    fallback_selected_evidence, fallback_filter_debug = filter_top_evidence(
        user_claim,
        original_attempt.raw_evidence,
        relevance_threshold=relevance_threshold,
        top_k=top_k,
        use_all_eligible_evidence=use_all_eligible_evidence,
        return_debug_info=True,
    )

    if not fallback_selected_evidence:
        return SelectionResult(
            selected_evidence=selected_evidence,
            filter_debug=filter_debug,
            final_claim=final_claim,
            strategy_used="",
            search_raw_count=len(raw_evidence),
            original_raw_count=original_attempt.search_raw_count,
            fallback_used=False,
        )

    if selected_evidence and not compare_evidence_sets(fallback_selected_evidence, selected_evidence):
        return SelectionResult(
            selected_evidence=selected_evidence,
            filter_debug=filter_debug,
            final_claim=final_claim,
            strategy_used="",
            search_raw_count=len(raw_evidence),
            original_raw_count=original_attempt.search_raw_count,
            fallback_used=False,
        )

    print("[App] Evidence comparison favored original claim fallback.")
    return SelectionResult(
        selected_evidence=fallback_selected_evidence,
        filter_debug=fallback_filter_debug,
        final_claim=user_claim,
        strategy_used="final_claim_then_original_evidence_fallback",
        search_raw_count=original_attempt.search_raw_count,
        original_raw_count=original_attempt.search_raw_count,
        fallback_used=True,
    )

@app.post("/analyze", response_model=ClaimResponse)
def analyze_claim(request: ClaimRequest):
    user_claim = request.claim.strip()
    options = normalize_analysis_options(request.options)

    if not user_claim:
        return build_response(
            "",
            decision_stage="claim_validation",
            failure_reason="empty_claim",
            explanation="The input claim is empty.",
        )

    claim_preparation = prepare_claim_for_fact_checking(
        user_claim,
        use_query_rewrite=options.use_query_rewrite,
    )

    if not claim_preparation.is_valid_claim:
        return build_response(
            user_claim,
            "",
            decision_stage="claim_validation",
            failure_reason="invalid_claim",
            explanation="The input does not look like a factual claim, so the system did not run fact-checking.",
        )

    final_claim = claim_preparation.final_claim

    retrieval_result = retrieve_evidence(
        user_claim,
        final_claim,
        retrieval_results=options.retrieval_results,
        use_oversampling_retry=options.use_oversampling_retry,
    )
    final_claim = retrieval_result.final_claim
    raw_evidence = retrieval_result.raw_evidence
    metadata_fields = {
        "retrieval_strategy_used": retrieval_result.strategy_used,
        "retrieval_query_used": retrieval_result.final_claim,
        "search_raw_evidence_count": retrieval_result.search_raw_count,
        "original_raw_evidence_count": retrieval_result.original_raw_count,
        "fallback_used": retrieval_result.fallback_used,
    }

    if retrieval_result.error_type:
        error_message = retrieval_result.error_message or "Search failed."
        failure_reason = "search_api_error"
        status = "system_error"

        if retrieval_result.error_type == "no_results":
            failure_reason = "no_search_results"
            status = "success"

        return build_response(
            user_claim,
            final_claim,
            status=status,
            decision_stage="retrieval",
            failure_reason=failure_reason,
            explanation=error_message,
            metadata=metadata_fields,
        )

    selection_result = choose_evidence(
        user_claim=user_claim,
        final_claim=final_claim,
        raw_evidence=raw_evidence,
        relevance_threshold=options.relevance_threshold,
        top_k=options.top_k,
        use_all_eligible_evidence=options.use_all_eligible_evidence,
        retrieval_results=options.retrieval_results,
        use_oversampling_retry=options.use_oversampling_retry,
    )
    final_claim = selection_result.final_claim
    selected_evidence = selection_result.selected_evidence
    filter_debug = selection_result.filter_debug

    retrieval_strategy_used = retrieval_result.strategy_used
    retrieval_query_used = retrieval_result.final_claim
    search_raw_evidence_count = retrieval_result.search_raw_count
    original_raw_evidence_count = retrieval_result.original_raw_count
    fallback_used = retrieval_result.fallback_used

    if selection_result.original_raw_count > original_raw_evidence_count:
        original_raw_evidence_count = selection_result.original_raw_count
    if selection_result.fallback_used:
        retrieval_strategy_used = selection_result.strategy_used
        retrieval_query_used = final_claim
        search_raw_evidence_count = selection_result.search_raw_count
        fallback_used = True

    selected_evidence_count = len(selected_evidence)
    excluded_borderline_candidate_count = count_borderline_candidates(filter_debug)

    evidence_summary = summarize_selected_evidence(selected_evidence)
    evidence_sufficiency = evidence_summary.sufficiency
    evidence_quality = evidence_summary.quality

    if not selected_evidence:
        failure_reason, explanation = explain_empty_selection(raw_evidence, filter_debug)

        return build_response(
            user_claim,
            final_claim,
            decision_stage="evidence_filter",
            failure_reason=failure_reason,
            explanation=explanation,
            metadata={
                "decision_confidence": "low",
                "evidence_sufficiency": evidence_sufficiency,
                "evidence_quality": evidence_quality,
                "retrieval_strategy_used": retrieval_strategy_used,
                "retrieval_query_used": retrieval_query_used,
                "search_raw_evidence_count": search_raw_evidence_count,
                "original_raw_evidence_count": original_raw_evidence_count,
                "selected_evidence_count": selected_evidence_count,
                "fallback_used": fallback_used,
                "excluded_borderline_candidate_count": excluded_borderline_candidate_count,
            },
        )

    verdict_report = generate_comprehensive_verdict(final_claim, selected_evidence)
    source_judgments = verdict_report.get("source_judgments", [])
    apply_source_judgments_to_evidence(selected_evidence, source_judgments)

    first_explanation = verdict_report.get("explanation", "No explanation was generated.")
    first_truth_score = aggregate_truth_score_from_source_judgments(
        selected_evidence=selected_evidence,
        source_judgments=source_judgments,
    )

    decision_confidence = calculate_decision_confidence(
        decision_stage="completed",
        truth_score=first_truth_score,
        selected_evidence_count=len(selected_evidence),
        evidence_sufficiency=evidence_sufficiency,
        evidence_quality=evidence_quality,
    )

    stabilization = stabilize_result(
        claim_for_verdict=final_claim,
        selected_evidence=selected_evidence,
        first_truth_score=first_truth_score,
        first_explanation=first_explanation,
        decision_confidence=decision_confidence,
        evidence_quality=evidence_quality,
        use_selective_stabilization=options.use_selective_stabilization,
    )

    return build_response(
        user_claim,
        final_claim,
        decision_stage="completed",
        truth_score=stabilization.truth_score,
        verdict=map_truth_score_to_verdict(stabilization.truth_score),
        explanation=stabilization.explanation,
        sources=selected_evidence,
        metadata={
            "decision_confidence": decision_confidence,
            "stabilization_used": stabilization.used,
            "stabilization_delta": stabilization.delta,
            "stabilization_result": stabilization.result,
            "evidence_sufficiency": evidence_sufficiency,
            "evidence_quality": evidence_quality,
            "retrieval_strategy_used": retrieval_strategy_used,
            "retrieval_query_used": retrieval_query_used,
            "search_raw_evidence_count": search_raw_evidence_count,
            "original_raw_evidence_count": original_raw_evidence_count,
            "selected_evidence_count": selected_evidence_count,
            "fallback_used": fallback_used,
            "excluded_borderline_candidate_count": excluded_borderline_candidate_count,
        },
    )


def start_server():
    print("[Server] FastAPI is starting on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    from pyngrok import ngrok

    ngrok.kill()

    ngrok_token = os.environ.get("NGROK_TOKEN")
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)

    public_url = ngrok.connect(8000).public_url
    print(f"API is live at: {public_url}/docs")

    server_thread = threading.Thread(target=start_server)
    server_thread.start()
