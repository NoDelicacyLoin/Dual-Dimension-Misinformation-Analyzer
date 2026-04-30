import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from api_contract import (
    AnalyzeResponse,
    AtomizedClaimGroup,
    EachFactChecking,
    EachFactualClaim,
    EachFactualClaimMetadata,
    TextPatternResult,
)
from shared_constants import (
    PROGRESS_STAGE_ANALYSIS,
    PROGRESS_STAGE_ATOMIZER,
    PROGRESS_STAGE_BERT_PROGRESS,
    PROGRESS_STAGE_FACT_CHECKING,
    PROGRESS_STAGE_LLM_EVIDENCE_PROGRESS,
    PROGRESS_STAGE_TAVILY_NLI_PROGRESS,
    PROGRESS_STAGE_TEXT_PATTERN,
    PROGRESS_STAGE_TOKEN_OCCLUSION_PROGRESS,
)


class UserFacingAnalysisError(RuntimeError):
    pass


def emit_progress(callback, event, progress_events, progress_lock=None):
    if progress_lock is None:
        progress_events.append(event)
    else:
        with progress_lock:
            progress_events.append(event)

    if callback is not None:
        callback(event)


def build_overall_risk(
    text_pattern_results: list[TextPatternResult],
    fact_checking: EachFactChecking,
) -> tuple[float | None, str, str]:
    risk_scores = []
    confidence_levels = []

    for result in text_pattern_results:
        if result.status != "success":
            continue
        risk_scores.append(float(result.prediction.risk_score))
        if result.prediction.confidence_level:
            confidence_levels.append(result.prediction.confidence_level)

    fact_risk_scores = []
    for factual_claim in fact_checking.factual_claims:
        if factual_claim.truth_score is None:
            continue
        fact_risk_scores.append(1.0 - float(factual_claim.truth_score))
        if factual_claim.decision_confidence:
            confidence_levels.append(factual_claim.decision_confidence)

    confidence_rank = {"low": 1, "medium": 2, "high": 3}

    combined_scores = risk_scores + fact_risk_scores
    overall_risk_score = (sum(combined_scores) / len(combined_scores)) if combined_scores else None

    if overall_risk_score is None:
        overall_risk_level = ""
    elif overall_risk_score >= 0.66:
        overall_risk_level = "high_risk"
    elif overall_risk_score >= 0.33:
        overall_risk_level = "medium_risk"
    else:
        overall_risk_level = "low_risk"

    overall_risk_confidence = (
        min(confidence_levels, key=lambda item: confidence_rank.get(item, 1)) if confidence_levels else ""
    )

    return overall_risk_score, overall_risk_level, overall_risk_confidence


def analyze_text_pattern_groups(
    claim_groups: list[AtomizedClaimGroup],
    progress_callback=None,
) -> list[TextPatternResult]:
    try:
        from text_pattern.text_risk_service import analyze_text_pattern
    except Exception as error:
        return [
            TextPatternResult(
                claim_group_id=claim_group.claim_group_id,
                original_sentence=claim_group.original_sentence,
                text_feature_text=claim_group.text_feature_text,
                atomization_applied=claim_group.atomization_applied,
                status="error",
                message=f"Text-pattern analysis failed: {error}",
            )
            for claim_group in claim_groups
        ]

    text_pattern_results = []
    total_count = len(claim_groups)
    bert_done_count = 0

    for claim_group in claim_groups:
        bert_progress_sent = False

        def emit_bert_progress():
            nonlocal bert_done_count, bert_progress_sent
            if bert_progress_sent:
                return
            bert_progress_sent = True
            bert_done_count += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": PROGRESS_STAGE_BERT_PROGRESS,
                        "status": "running",
                        "message": f"BERT classification finished {bert_done_count} of {total_count} text unit(s).",
                        "claim_group_id": claim_group.claim_group_id,
                        "fact_claim_id": 0,
                        "completed_text_feature_unit_count": bert_done_count,
                        "text_feature_unit_count": total_count,
                    }
                )

        try:
            text_pattern_result = analyze_text_pattern(
                claim_group.text_feature_text,
                progress_callback=emit_bert_progress,
            )
        except Exception as error:
            text_pattern_result = TextPatternResult(
                status="error",
                message=f"Text-pattern analysis failed: {error}",
            )
        emit_bert_progress()
        text_pattern_result.claim_group_id = claim_group.claim_group_id
        text_pattern_result.original_sentence = claim_group.original_sentence
        text_pattern_result.text_feature_text = claim_group.text_feature_text
        text_pattern_result.atomization_applied = claim_group.atomization_applied
        text_pattern_results.append(text_pattern_result)
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": PROGRESS_STAGE_TOKEN_OCCLUSION_PROGRESS,
                    "status": "running",
                    "message": f"Token occlusion finished {len(text_pattern_results)} of {total_count} text unit(s).",
                    "claim_group_id": claim_group.claim_group_id,
                    "fact_claim_id": 0,
                    "completed_text_feature_unit_count": len(text_pattern_results),
                    "text_feature_unit_count": total_count,
                }
            )

    return text_pattern_results


def build_fact_checking_branch_error(
    claim_groups: list[AtomizedClaimGroup],
    message: str,
) -> EachFactChecking:
    factual_claims = []

    for claim_group in claim_groups:
        for fact_claim in claim_group.fact_check_claims:
            factual_claims.append(
                EachFactualClaim(
                    claim_group_id=claim_group.claim_group_id,
                    fact_claim_id=fact_claim.fact_claim_id,
                    original_sentence=claim_group.original_sentence,
                    text_feature_text=claim_group.text_feature_text,
                    claim=fact_claim.claim,
                    entities=fact_claim.entities,
                    relation=fact_claim.relation,
                    constraints=fact_claim.constraints,
                    status="system_error",
                    explanation=message,
                    decision_confidence="low",
                    metadata=EachFactualClaimMetadata(),
                )
            )

    return EachFactChecking(
        status="degraded",
        explanation=message,
        factual_claims=factual_claims,
    )


def analyze_fact_checking_groups(
    claim_groups: list[AtomizedClaimGroup],
    raw_options,
    progress_callback=None,
) -> EachFactChecking:
    try:
        from fact_checking.fact_check_service import analyze_fact_check_claims

        return analyze_fact_check_claims(
            claim_groups,
            raw_options,
            progress_callback,
        )
    except Exception as error:
        print(f"[fact_checking] branch failed: {type(error).__name__}: {error}")
        return build_fact_checking_branch_error(
            claim_groups,
            f"Evidence-based fact-checking failed: {error}",
        )


def run_analysis(request, progress_callback=None) -> AnalyzeResponse:
    progress_events = []
    progress_lock = threading.Lock()
    user_text = request.claim.strip()

    if not user_text:
        raise UserFacingAnalysisError("Please enter a factual claim or passage to analyze.")

    from atomizer.atomizer_service import atomize_for_pipeline

    atomized = atomize_for_pipeline(user_text)
    if atomized.status == "atomizer_error":
        raise RuntimeError(atomized.message or "Atomizer failed.")

    atomizer_counts = {
        "candidate_claim_group_count": atomized.candidate_claim_group_count,
        "candidate_fact_claim_count": atomized.candidate_fact_claim_count,
        "selected_claim_group_count": atomized.selected_claim_group_count,
        "selected_fact_claim_count": atomized.selected_fact_claim_count,
        "max_claim_group_count": atomized.max_claim_group_count,
        "claim_selection_reason": atomized.claim_selection_reason,
    }
    emit_progress(
        progress_callback,
        {
            "stage": PROGRESS_STAGE_ATOMIZER,
            "status": "completed",
            "message": f"Atomizer finished with {len(atomized.claim_groups)} claim group(s).",
            "claim_group_id": 0,
            "fact_claim_id": 0,
            "text_feature_unit_count": atomized.summary.text_feature_unit_count,
            "fact_check_claim_count": atomized.summary.fact_check_claim_count,
            "ignored_sentence_count": atomized.summary.ignored_sentence_count,
            **atomizer_counts,
        },
        progress_events,
        progress_lock,
    )

    if atomized.status == "invalid_input" or not atomized.claim_groups:
        raise UserFacingAnalysisError(
            "No checkable factual claim was found..."
        )

    claim_groups = atomized.claim_groups

    text_pattern_results = []
    fact_checking = EachFactChecking(status="failed", factual_claims=[])

    def emit_branch_progress(event):
        emit_progress(progress_callback, event, progress_events, progress_lock)

    with ThreadPoolExecutor(max_workers=2) as executor:
        text_pattern_future = executor.submit(analyze_text_pattern_groups, claim_groups, emit_branch_progress)
        fact_checking_future = executor.submit(
            analyze_fact_checking_groups,
            claim_groups,
            request.options,
            emit_branch_progress,
        )

        for finished_future in as_completed([text_pattern_future, fact_checking_future]):
            if finished_future is text_pattern_future:
                text_pattern_results = finished_future.result()
                emit_progress(
                    progress_callback,
                    {
                        "stage": PROGRESS_STAGE_TEXT_PATTERN,
                        "status": "completed",
                        "message": f"Text-pattern analysis finished for {len(text_pattern_results)} claim group(s).",
                        "claim_group_id": 0,
                        "fact_claim_id": 0,
                        "completed_text_feature_unit_count": len(text_pattern_results),
                        "text_feature_unit_count": atomized.summary.text_feature_unit_count,
                    },
                    progress_events,
                    progress_lock,
                )
            else:
                fact_checking = finished_future.result()
                emit_progress(
                    progress_callback,
                    {
                        "stage": PROGRESS_STAGE_FACT_CHECKING,
                        "status": "completed",
                        "message": f"Fact-checking finished for {len(fact_checking.factual_claims)} factual claim(s).",
                        "claim_group_id": 0,
                        "fact_claim_id": 0,
                        "completed_fact_check_claim_count": len(fact_checking.factual_claims),
                        "fact_check_claim_count": atomized.summary.fact_check_claim_count,
                    },
                    progress_events,
                    progress_lock,
                )

    overall_risk_score, overall_risk_level, overall_risk_confidence = build_overall_risk(
        text_pattern_results,
        fact_checking,
    )

    emit_progress(
        progress_callback,
        {
            "stage": PROGRESS_STAGE_ANALYSIS,
            "status": "completed",
            "message": "Overall analysis finished.",
            "claim_group_id": 0,
            "fact_claim_id": 0,
            "text_feature_unit_count": atomized.summary.text_feature_unit_count,
            "fact_check_claim_count": atomized.summary.fact_check_claim_count,
        },
        progress_events,
        progress_lock,
    )

    return AnalyzeResponse(
        status=fact_checking.status,
        original_text=user_text,
        ignored_sentences=atomized.ignored_sentences,
        text_pattern_results=text_pattern_results,
        fact_checking=fact_checking,
        overall_risk_score=overall_risk_score,
        overall_risk_level=overall_risk_level,
        overall_risk_confidence=overall_risk_confidence,
        progress_events=progress_events,
        ignored_sentence_count=atomized.summary.ignored_sentence_count,
        text_feature_unit_count=atomized.summary.text_feature_unit_count,
        fact_check_claim_count=atomized.summary.fact_check_claim_count,
        **atomizer_counts,
    )


def stream_analysis(request):
    event_queue = queue.Queue()

    def emit_event(event):
        payload = {"event": "progress", "data": event}
        event_queue.put(f"data: {json.dumps(payload)}\n\n")

    def worker():
        try:
            response = run_analysis(request, progress_callback=emit_event)
            payload = {"event": "result", "data": response.model_dump()}
            event_queue.put(f"data: {json.dumps(payload)}\n\n")
        except Exception as error:
            payload = {"event": "error", "data": {"message": str(error)}}
            event_queue.put(f"data: {json.dumps(payload)}\n\n")
        finally:
            event_queue.put(None)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        next_item = event_queue.get()
        if next_item is None:
            break
        yield next_item
