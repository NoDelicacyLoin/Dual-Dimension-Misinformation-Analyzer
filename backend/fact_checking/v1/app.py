import os
import threading
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from pyngrok import ngrok

from search import fetch_oversampled_evidence
from nli_filter import filter_top_evidence
from gemini_agent import optimize_claim_for_search, generate_comprehensive_verdict

app = FastAPI(title="Fact-Checking API")
ENABLE_SELECTIVE_STABILIZATION = True


class ClaimRequest(BaseModel):
    claim: str


class EvidenceSource(BaseModel):
    url: str
    content: str
    ai_analysis: str = ""
    evidence_quality: str = ""
    source_role: str = ""
    source_strength: float = 0.0
    source_specificity: float = 0.0


class ClaimResponse(BaseModel):
    status: str
    original_claim: str
    optimized_claim: str
    decision_stage: str
    failure_reason: str
    truth_score: float
    verdict: str
    explanation: str
    sources: list[EvidenceSource]
    decision_confidence: str = ""
    stabilization_used: bool = False
    stabilization_delta: float = 0.0
    stabilization_result: str = "not_triggered"
    evidence_sufficiency: str = ""
    evidence_quality: str = ""

    retrieval_strategy_used: str = ""
    retrieval_query_used: str = ""
    optimized_raw_evidence_count: int = 0
    original_raw_evidence_count: int = 0
    selected_evidence_count: int = 0
    fallback_used: bool = False

def is_search_error(evidence_list: list[dict]) -> bool:
    if not evidence_list:
        return False
    return evidence_list[0].get("url") == "Error"


def get_evidence_count(evidence_list: list[dict]) -> int:
    if not evidence_list:
        return 0
    if evidence_list[0].get("url") == "Error":
        return 0
    return len(evidence_list)


def set_selective_stabilization_enabled(enabled: bool) -> None:
    """
    Allow notebook experiments to toggle stabilization without changing API logic.
    """
    global ENABLE_SELECTIVE_STABILIZATION
    ENABLE_SELECTIVE_STABILIZATION = bool(enabled)

def map_truth_score_to_verdict(truth_score: float) -> str:
    """
    Map a continuous truth score into a product-facing verdict label.
    """
    if truth_score >= 0.85:
        return "True"
    if truth_score >= 0.65:
        return "Mostly True"
    if truth_score >= 0.45:
        return "Neutral"
    if truth_score >= 0.25:
        return "Mostly False"
    return "False"


def normalize_truth_score(raw_truth_score) -> float:
    """
    Keep the score inside a valid range before mapping.
    """
    try:
        normalized_truth_score = float(raw_truth_score)
    except Exception:
        normalized_truth_score = 0.5

    if normalized_truth_score < 0.0:
        return 0.0
    if normalized_truth_score > 1.0:
        return 1.0
    return normalized_truth_score


def normalize_source_role(raw_role: str) -> str:
    """
    Keep source-level stance labels inside a small product-facing set.
    """
    normalized_role = (raw_role or "").strip().lower()

    if normalized_role in {"supports", "support", "supported"}:
        return "supports"
    if normalized_role in {"contradicts", "contradict", "refutes", "refute"}:
        return "contradicts"
    if normalized_role in {"mixed", "conflicted"}:
        return "mixed"
    return "background"


def get_source_quality_weight(evidence_quality: str) -> float:
    """
    Convert source quality labels into simple aggregation weights.
    """
    if evidence_quality == "strong":
        return 1.0
    if evidence_quality == "usable":
        return 0.75
    return 0.5


def aggregate_truth_score_from_source_judgments(
    selected_evidence: list[dict],
    source_judgments: list[dict]
) -> float:
    """
    Aggregate source-level judgments into a deterministic truth score.
    Start from neutral and let usable support or contradiction move the score.
    """
    support_sum = 0.0
    contradiction_sum = 0.0
    support_count = 0
    contradiction_count = 0
    mixed_or_background_count = 0

    for evidence_index, evidence_item in enumerate(selected_evidence, start=1):
        matching_judgment = None

        for source_judgment in source_judgments:
            if source_judgment.get("source_index") == evidence_index:
                matching_judgment = source_judgment
                break

        if not matching_judgment:
            continue

        source_role = normalize_source_role(matching_judgment.get("stance", "background"))
        source_strength = normalize_truth_score(matching_judgment.get("strength", 0.0))
        source_specificity = normalize_truth_score(matching_judgment.get("specificity", 0.0))
        quality_weight = get_source_quality_weight(evidence_item.get("evidence_quality", "weak"))

        source_weight = source_strength * source_specificity * quality_weight

        if source_role == "supports":
            support_sum += source_weight
            support_count += 1
        elif source_role == "contradicts":
            contradiction_sum += source_weight
            contradiction_count += 1
        elif source_role == "mixed":
            support_sum += source_weight * 0.35
            contradiction_sum += source_weight * 0.35
            mixed_or_background_count += 1
        else:
            mixed_or_background_count += 1

    effective_signal = support_sum + contradiction_sum
    if effective_signal <= 0.05:
        return 0.5

    # Keep one weak or mixed source from pushing the system into a directional label.
    if len(selected_evidence) == 1:
        single_evidence_item = selected_evidence[0]
        single_quality = single_evidence_item.get("evidence_quality", "weak")
        single_role = normalize_source_role(source_judgments[0].get("stance", "background")) if source_judgments else "background"

        if single_quality == "weak" or single_role in {"mixed", "background"}:
            balance_score = (support_sum - contradiction_sum) / effective_signal
            guarded_truth_score = 0.5 + (0.08 * balance_score)
            return normalize_truth_score(guarded_truth_score)

    # Keep clearly conflicted evidence near neutral instead of forcing a directional score.
    if support_count >= 1 and contradiction_count >= 1:
        weaker_side = min(support_sum, contradiction_sum)
        stronger_side = max(support_sum, contradiction_sum)

        if weaker_side >= 0.20 and stronger_side > 0:
            conflict_ratio = weaker_side / stronger_side
            if conflict_ratio >= 0.55:
                balance_score = (support_sum - contradiction_sum) / effective_signal
                conflicted_truth_score = 0.5 + (0.12 * balance_score)
                return normalize_truth_score(conflicted_truth_score)

    balance_score = (support_sum - contradiction_sum) / effective_signal
    coverage_factor = min(effective_signal / 1.8, 1.0)
    aggregated_truth_score = 0.5 + (0.45 * balance_score * coverage_factor)

    return normalize_truth_score(aggregated_truth_score)


def apply_source_judgments_to_evidence(
    selected_evidence: list[dict],
    source_judgments: list[dict]
) -> None:
    """
    Copy source-level AI judgments onto each selected evidence item.
    """
    for evidence_index, evidence_item in enumerate(selected_evidence, start=1):
        matching_judgment = None

        for source_judgment in source_judgments:
            if source_judgment.get("source_index") == evidence_index:
                matching_judgment = source_judgment
                break

        if not matching_judgment:
            evidence_item["ai_analysis"] = "No specific analysis was generated for this source."
            evidence_item["source_role"] = "background"
            evidence_item["source_strength"] = 0.0
            evidence_item["source_specificity"] = 0.0
            continue

        evidence_item["ai_analysis"] = (
            matching_judgment.get("analysis", "").strip() or
            "No specific analysis was generated for this source."
        )
        evidence_item["source_role"] = normalize_source_role(matching_judgment.get("stance", "background"))
        evidence_item["source_strength"] = normalize_truth_score(matching_judgment.get("strength", 0.0))
        evidence_item["source_specificity"] = normalize_truth_score(matching_judgment.get("specificity", 0.0))


def summarize_selected_evidence(selected_evidence: list[dict]) -> tuple[str, str]:
    """
    Turn structured evidence metadata into a product-facing summary.
    """
    if not selected_evidence:
        return "insufficient", "weak"

    strong_count = 0
    usable_count = 0

    for evidence_item in selected_evidence:
        evidence_quality = evidence_item.get("evidence_quality", "weak")
        if evidence_quality == "strong":
            strong_count += 1
            usable_count += 1
        elif evidence_quality == "usable":
            usable_count += 1

    if strong_count >= 2:
        return "sufficient", "strong"

    if strong_count >= 1 and usable_count >= 2:
        return "sufficient", "mixed"

    if strong_count >= 1:
        return "limited", "mixed"

    if usable_count >= 2:
        return "limited", "mixed"

    if usable_count >= 1:
        return "limited", "weak"

    return "insufficient", "weak"


def calculate_decision_confidence(
    decision_stage: str,
    truth_score: float,
    selected_evidence_count: int,
    evidence_sufficiency: str,
    evidence_quality: str
) -> str:
    """
    Convert model output and evidence summary into a product-facing confidence band.
    """
    if decision_stage != "completed":
        return "low"

    if evidence_sufficiency == "insufficient":
        return "low"

    distance_from_neutral = abs(truth_score - 0.5)

    if evidence_sufficiency == "limited":
        if selected_evidence_count <= 1:
            return "low"
        if distance_from_neutral < 0.15:
            return "low"
        return "medium"

    if evidence_sufficiency == "sufficient":
        if evidence_quality == "strong" and selected_evidence_count >= 2 and distance_from_neutral >= 0.25:
            return "high"
        return "medium"

    return "low"


def should_trigger_selective_stabilization(
    decision_stage: str,
    decision_confidence: str,
    truth_score: float,
    selected_evidence_count: int,
    evidence_quality: str
) -> bool:
    """
    Trigger stabilization for clearly low-confidence cases and a narrow set
    of medium-confidence edge cases.
    """
    if not ENABLE_SELECTIVE_STABILIZATION:
        return False

    if decision_stage != "completed":
        return False

    if decision_confidence == "low":
        return True

    distance_from_neutral = abs(truth_score - 0.5)
    near_verdict_boundary = (
        abs(truth_score - 0.25) < 0.08 or
        abs(truth_score - 0.45) < 0.08 or
        abs(truth_score - 0.65) < 0.08 or
        abs(truth_score - 0.85) < 0.08
    )

    if decision_confidence == "medium":
        if evidence_quality == "mixed" and selected_evidence_count <= 1:
            return True
        if evidence_quality == "mixed" and distance_from_neutral < 0.20:
            return True
        if near_verdict_boundary:
            return True

    return False


def run_selective_stabilization(
    claim_for_verdict: str,
    selected_evidence: list[dict],
    first_truth_score: float,
    first_explanation: str,
    decision_stage: str,
    decision_confidence: str,
    evidence_quality: str
) -> tuple[float, str, bool, float, str]:
    """
    Re-run only the final scoring step for selected edge cases.
    Keep retrieval and filtering fixed so stabilization stays interpretable.
    """
    if not should_trigger_selective_stabilization(
        decision_stage=decision_stage,
        decision_confidence=decision_confidence,
        truth_score=first_truth_score,
        selected_evidence_count=len(selected_evidence),
        evidence_quality=evidence_quality
    ):
        return first_truth_score, first_explanation, False, 0.0, "not_triggered"

    second_verdict_report = generate_comprehensive_verdict(claim_for_verdict, selected_evidence)
    second_truth_score = aggregate_truth_score_from_source_judgments(
        selected_evidence=selected_evidence,
        source_judgments=second_verdict_report.get("source_judgments", [])
    )
    second_explanation = second_verdict_report.get("explanation", first_explanation)

    stabilization_delta = abs(first_truth_score - second_truth_score)
    crossed_neutral_boundary = (
        (first_truth_score < 0.5 and second_truth_score > 0.5) or
        (first_truth_score > 0.5 and second_truth_score < 0.5)
    )

    if crossed_neutral_boundary or stabilization_delta >= 0.20:
        stabilized_truth_score = 0.5
        stabilized_explanation = (
            f"{first_explanation} The result was re-checked because the case was not fully stable. "
            "The second scoring pass pointed in a meaningfully different direction, so the final score was reset to a neutral value."
        ).strip()
        return stabilized_truth_score, stabilized_explanation, True, stabilization_delta, "reset_to_neutral"

    if stabilization_delta < 0.12:
        stabilized_truth_score = (first_truth_score + second_truth_score) / 2
        return stabilized_truth_score, first_explanation, True, stabilization_delta, "confirmed"

    stabilized_truth_score = (first_truth_score + second_truth_score) / 2
    stabilized_explanation = (
        f"{first_explanation} The result was re-checked because the case was borderline."
    ).strip()
    if second_explanation and second_explanation != first_explanation:
        stabilized_explanation = (
            f"{stabilized_explanation} A second scoring pass produced a meaningfully different score, "
            "so the final score was stabilized toward the middle."
        )

    return stabilized_truth_score, stabilized_explanation, True, stabilization_delta, "soft_adjusted"


@app.post("/analyze", response_model=ClaimResponse)
def analyze_claim(request: ClaimRequest):
    user_claim = request.claim.strip()

    if not user_claim:
        return ClaimResponse(
            status="success",
            original_claim="",
            optimized_claim="",
            decision_stage="claim_validation",
            failure_reason="empty_claim",
            truth_score=0.5,
            verdict="Neutral",
            explanation="The input claim is empty.",
            sources=[],
            decision_confidence="low",
            evidence_sufficiency="insufficient",
            evidence_quality="weak",
            stabilization_result="not_triggered",
            retrieval_strategy_used="none",
            retrieval_query_used="",
            optimized_raw_evidence_count=0,
            original_raw_evidence_count=0,
            selected_evidence_count=0,
            fallback_used=False
        )

    optimized_claim = optimize_claim_for_search(user_claim)

    if optimized_claim == "INVALID_CLAIM":
        return ClaimResponse(
            status="success",
            original_claim=user_claim,
            optimized_claim=optimized_claim,
            decision_stage="claim_validation",
            failure_reason="invalid_claim",
            truth_score=0.5,
            verdict="Neutral",
            explanation="The input does not look like a factual claim, so the system did not run fact-checking.",
            sources=[],
            decision_confidence="low",
            evidence_sufficiency="insufficient",
            evidence_quality="weak",
            stabilization_result="not_triggered",
            retrieval_strategy_used="none",
            retrieval_query_used="",
            optimized_raw_evidence_count=0,
            original_raw_evidence_count=0,
            selected_evidence_count=0,
            fallback_used=False
        )

    retrieval_strategy_used = "optimized_only"
    retrieval_query_used = optimized_claim
    fallback_used = False

    optimized_raw_evidence = fetch_oversampled_evidence(optimized_claim, max_results=8)
    optimized_raw_evidence_count = get_evidence_count(optimized_raw_evidence)

    original_raw_evidence = []
    original_raw_evidence_count = 0

    raw_evidence = optimized_raw_evidence
    claim_for_filtering = optimized_claim

    optimized_search_failed = is_search_error(optimized_raw_evidence)
    optimized_search_too_weak = optimized_raw_evidence_count < 2

    if (optimized_search_failed or optimized_search_too_weak) and optimized_claim != user_claim:
        original_raw_evidence = fetch_oversampled_evidence(user_claim, max_results=8)
        original_raw_evidence_count = get_evidence_count(original_raw_evidence)

        if not is_search_error(original_raw_evidence) and original_raw_evidence_count > optimized_raw_evidence_count:
            raw_evidence = original_raw_evidence
            claim_for_filtering = user_claim
            retrieval_query_used = user_claim
            retrieval_strategy_used = "optimized_then_original_fallback"
            fallback_used = True
            print("[App] Retrieval fallback used original claim.")

    if is_search_error(raw_evidence):
        error_message = raw_evidence[0].get("content", "Search failed.")

        if "No reliable evidence found" in error_message:
            return ClaimResponse(
                status="success",
                original_claim=user_claim,
                optimized_claim=optimized_claim,
                decision_stage="retrieval",
                failure_reason="no_search_results",
                truth_score=0.5,
                verdict="Neutral",
                explanation=error_message,
                sources=[],
                decision_confidence="low",
                evidence_sufficiency="insufficient",
                evidence_quality="weak",
                stabilization_result="not_triggered",
                retrieval_strategy_used=retrieval_strategy_used,
                retrieval_query_used=retrieval_query_used,
                optimized_raw_evidence_count=optimized_raw_evidence_count,
                original_raw_evidence_count=original_raw_evidence_count,
                selected_evidence_count=0,
                fallback_used=fallback_used
            )

        return ClaimResponse(
            status="system_error",
            original_claim=user_claim,
            optimized_claim=optimized_claim,
            decision_stage="retrieval",
            failure_reason="search_api_error",
            truth_score=0.5,
            verdict="Neutral",
            explanation=error_message,
            sources=[],
            decision_confidence="low",
            evidence_sufficiency="insufficient",
            evidence_quality="weak",
            stabilization_result="not_triggered",
            retrieval_strategy_used=retrieval_strategy_used,
            retrieval_query_used=retrieval_query_used,
            optimized_raw_evidence_count=optimized_raw_evidence_count,
            original_raw_evidence_count=original_raw_evidence_count,
            selected_evidence_count=0,
            fallback_used=fallback_used
        )

    selected_evidence = filter_top_evidence(
        claim_for_filtering,
        raw_evidence,
        top_k=3
    )

    if not selected_evidence and claim_for_filtering != user_claim:
        if not original_raw_evidence:
            original_raw_evidence = fetch_oversampled_evidence(user_claim, max_results=8)
            original_raw_evidence_count = get_evidence_count(original_raw_evidence)

        if not is_search_error(original_raw_evidence):
            fallback_selected_evidence = filter_top_evidence(
                user_claim,
                original_raw_evidence,
                top_k=3
            )

            if fallback_selected_evidence:
                selected_evidence = fallback_selected_evidence
                raw_evidence = original_raw_evidence
                claim_for_filtering = user_claim
                retrieval_query_used = user_claim
                retrieval_strategy_used = "optimized_then_original_filter_fallback"
                fallback_used = True
                print("[App] Evidence-filter fallback used original claim.")

    selected_evidence_count = len(selected_evidence)
    evidence_sufficiency, evidence_quality = summarize_selected_evidence(selected_evidence)

    if not selected_evidence or evidence_sufficiency == "insufficient":
        failure_reason = "all_evidence_filtered_by_nli"

        if get_evidence_count(raw_evidence) > 0:
            failure_reason = "too_few_relevant_evidence"

        return ClaimResponse(
            status="success",
            original_claim=user_claim,
            optimized_claim=optimized_claim,
            decision_stage="evidence_filter",
            failure_reason=failure_reason,
            truth_score=0.5,
            verdict="Neutral",
            explanation="The system could not find enough relevant evidence for this claim.",
            sources=[],
            decision_confidence="low",
            evidence_sufficiency="insufficient",
            evidence_quality="weak",
            stabilization_result="not_triggered",
            retrieval_strategy_used=retrieval_strategy_used,
            retrieval_query_used=retrieval_query_used,
            optimized_raw_evidence_count=optimized_raw_evidence_count,
            original_raw_evidence_count=original_raw_evidence_count,
            selected_evidence_count=0,
            fallback_used=fallback_used
        )

    verdict_report = generate_comprehensive_verdict(claim_for_filtering, selected_evidence)
    source_judgments = verdict_report.get("source_judgments", [])
    apply_source_judgments_to_evidence(selected_evidence, source_judgments)
    final_explanation = verdict_report.get("explanation", "No explanation was generated.")
    final_truth_score = aggregate_truth_score_from_source_judgments(
        selected_evidence=selected_evidence,
        source_judgments=source_judgments
    )

    decision_confidence = calculate_decision_confidence(
        decision_stage="completed",
        truth_score=final_truth_score,
        selected_evidence_count=selected_evidence_count,
        evidence_sufficiency=evidence_sufficiency,
        evidence_quality=evidence_quality
    )
    final_truth_score, final_explanation, stabilization_used, stabilization_delta, stabilization_result = run_selective_stabilization(
        claim_for_verdict=claim_for_filtering,
        selected_evidence=selected_evidence,
        first_truth_score=final_truth_score,
        first_explanation=final_explanation,
        decision_stage="completed",
        decision_confidence=decision_confidence,
        evidence_quality=evidence_quality
    )
    final_verdict = map_truth_score_to_verdict(final_truth_score)

    return ClaimResponse(
        status="success",
        original_claim=user_claim,
        optimized_claim=optimized_claim,
        decision_stage="completed",
        failure_reason="",
        truth_score=final_truth_score,
        verdict=final_verdict,
        explanation=final_explanation,
        sources=selected_evidence,
        decision_confidence=decision_confidence,
        stabilization_used=stabilization_used,
        stabilization_delta=stabilization_delta,
        stabilization_result=stabilization_result,
        evidence_sufficiency=evidence_sufficiency,
        evidence_quality=evidence_quality,
        retrieval_strategy_used=retrieval_strategy_used,
        retrieval_query_used=retrieval_query_used,
        optimized_raw_evidence_count=optimized_raw_evidence_count,
        original_raw_evidence_count=original_raw_evidence_count,
        selected_evidence_count=selected_evidence_count,
        fallback_used=fallback_used
    )


def start_server():
    print("[Server] FastAPI is starting on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    ngrok.kill()

    ngrok_token = os.environ.get("ngrok_TOKEN")
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)

    public_url = ngrok.connect(8000).public_url
    print(f"API is live at: {public_url}/docs")

    server_thread = threading.Thread(target=start_server)
    server_thread.start()
