from dataclasses import dataclass

NEUTRAL_SCORE = 0.5
MINIMUM_SIGNAL = 0.1
MAX_SCORE_SHIFT = 0.4
SINGLE_SOURCE_SHIFT = 0.2
VERY_STRONG_SINGLE_SOURCE_SHIFT = 0.3
CONFLICT_SOFTENING_SHIFT = 0.1
HIGH_SOURCE_SCORE = 0.9
CONFLICT_DOMINANCE_RATIO = 0.5
HIGH_CONFIDENCE_DISTANCE = 0.3
LIMITED_EVIDENCE_DISTANCE = 0.2
MIXED_SIGNAL_SHARE = 0.50

BORDERLINE_FILTER_REASONS = {
    "below_usability_floor_no_anchor",
    "below_usability_floor_weak_anchor",
}
@dataclass
class EvidenceSummary:
    sufficiency: str
    quality: str


def count_borderline_candidates(filter_debug_info: dict) -> int:
    """
    Count candidates that looked close enough to review but were still excluded.
    """
    if not filter_debug_info:
        return 0

    return sum(
        1
        for scored_item in filter_debug_info.get("scored_evidence", [])
        if not scored_item.get("passed_threshold")
        and scored_item.get("filter_reason") in BORDERLINE_FILTER_REASONS
    )


def map_truth_score_to_verdict(truth_score: float) -> str:
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
    try:
        normalized_truth_score = float(raw_truth_score)
    except Exception:
        normalized_truth_score = NEUTRAL_SCORE

    if normalized_truth_score < 0.0:
        return 0.0
    if normalized_truth_score > 1.0:
        return 1.0
    return normalized_truth_score


def normalize_source_role(raw_role: str) -> str:
    normalized_role = (raw_role or "").strip().lower()

    if normalized_role in {"supports", "support", "supported"}:
        return "supports"
    if normalized_role in {"contradicts", "contradict", "refutes", "refute"}:
        return "contradicts"
    if normalized_role in {"mixed", "conflicted"}:
        return "mixed"
    return "background"


def get_source_quality_weight(evidence_quality: str) -> float:
    if evidence_quality == "strong":
        return 1.0
    if evidence_quality == "usable":
        return 0.75
    return 0.50


def get_source_judgment_by_index(source_judgments: list[dict], source_index: int) -> dict | None:
    for source_judgment in source_judgments:
        if source_judgment.get("source_index") == source_index:
            return source_judgment
    return None


def get_source_weight(evidence_item: dict, source_judgment: dict) -> float:
    source_strength = normalize_truth_score(source_judgment.get("strength", 0.0))
    source_specificity = normalize_truth_score(source_judgment.get("specificity", 0.0))
    quality_weight = get_source_quality_weight(evidence_item.get("evidence_quality", "weak"))
    return ((source_strength + source_specificity) / 2) * quality_weight


def collect_source_signal_totals(selected_evidence: list[dict], source_judgments: list[dict]) -> dict:
    support_sum = 0.0
    contradiction_sum = 0.0
    support_count = 0
    contradiction_count = 0

    for evidence_index, evidence_item in enumerate(selected_evidence, start=1):
        matching_judgment = get_source_judgment_by_index(source_judgments, evidence_index)
        if not matching_judgment:
            continue

        source_role = normalize_source_role(matching_judgment.get("stance", "background"))
        source_weight = get_source_weight(evidence_item, matching_judgment)

        if source_role == "supports":
            support_sum += source_weight
            support_count += 1
        elif source_role == "contradicts":
            contradiction_sum += source_weight
            contradiction_count += 1
        elif source_role == "mixed":
            support_sum += source_weight * MIXED_SIGNAL_SHARE
            contradiction_sum += source_weight * MIXED_SIGNAL_SHARE

    return {
        "support_sum": support_sum,
        "contradiction_sum": contradiction_sum,
        "support_count": support_count,
        "contradiction_count": contradiction_count,
    }


def score_single_source_case(selected_evidence: list[dict], source_judgments: list[dict], balance_score: float) -> float | None:
    if len(selected_evidence) != 1:
        return None

    single_evidence = selected_evidence[0]
    matching_judgment = source_judgments[0] if source_judgments else {}
    single_role = normalize_source_role(matching_judgment.get("stance", "background"))
    single_quality = single_evidence.get("evidence_quality", "weak")
    single_strength = normalize_truth_score(matching_judgment.get("strength", 0.0))
    single_specificity = normalize_truth_score(matching_judgment.get("specificity", 0.0))

    if single_role in {"mixed", "background"} or single_quality == "weak":
        return normalize_truth_score(NEUTRAL_SCORE + ((SINGLE_SOURCE_SHIFT / 2) * balance_score))

    allowed_shift = SINGLE_SOURCE_SHIFT
    if (
        single_quality == "strong"
        and single_strength >= HIGH_SOURCE_SCORE
        and single_specificity >= HIGH_SOURCE_SCORE
    ):
        allowed_shift = VERY_STRONG_SINGLE_SOURCE_SHIFT

    return normalize_truth_score(NEUTRAL_SCORE + (allowed_shift * balance_score))


def score_conflicted_case(signal_totals: dict, balance_score: float) -> float | None:
    support_sum = signal_totals["support_sum"]
    contradiction_sum = signal_totals["contradiction_sum"]

    if support_sum <= 0 or contradiction_sum <= 0:
        return None

    weaker_side = min(support_sum, contradiction_sum)
    stronger_side = max(support_sum, contradiction_sum)
    if stronger_side == 0:
        return None

    dominance_ratio = weaker_side / stronger_side
    if dominance_ratio < CONFLICT_DOMINANCE_RATIO:
        return None

    return normalize_truth_score(NEUTRAL_SCORE + (CONFLICT_SOFTENING_SHIFT * balance_score))


def score_default_case(balance_score: float, effective_signal: float, evidence_count: int) -> float:
    if evidence_count <= 0:
        return NEUTRAL_SCORE

    coverage = min(effective_signal / evidence_count, 1.0)
    return normalize_truth_score(NEUTRAL_SCORE + (MAX_SCORE_SHIFT * balance_score * coverage))


def aggregate_truth_score_from_source_judgments(
    selected_evidence: list[dict],
    source_judgments: list[dict]
) -> float:
    """
    Start from neutral and move only as far as the source-level support or
    contradiction signal really justifies.
    """
    signal_totals = collect_source_signal_totals(selected_evidence, source_judgments)
    support_sum = signal_totals["support_sum"]
    contradiction_sum = signal_totals["contradiction_sum"]

    effective_signal = support_sum + contradiction_sum
    if effective_signal <= MINIMUM_SIGNAL:
        return NEUTRAL_SCORE

    balance_score = (support_sum - contradiction_sum) / effective_signal

    single_source_score = score_single_source_case(selected_evidence, source_judgments, balance_score)
    if single_source_score is not None:
        return single_source_score

    conflicted_score = score_conflicted_case(signal_totals, balance_score)
    if conflicted_score is not None:
        return conflicted_score

    return score_default_case(balance_score, effective_signal, len(selected_evidence))


def apply_source_judgments_to_evidence(
    selected_evidence: list[dict],
    source_judgments: list[dict]
) -> None:
    for evidence_index, evidence_item in enumerate(selected_evidence, start=1):
        matching_judgment = get_source_judgment_by_index(source_judgments, evidence_index)

        if not matching_judgment:
            evidence_item["ai_analysis"] = "No specific analysis was generated for this source."
            evidence_item["source_role"] = "background"
            evidence_item["source_strength"] = 0.0
            evidence_item["source_specificity"] = 0.0
            continue

        evidence_item["ai_analysis"] = (
            matching_judgment.get("analysis", "").strip()
            or "No specific analysis was generated for this source."
        )
        evidence_item["source_role"] = normalize_source_role(matching_judgment.get("stance", "background"))
        evidence_item["source_strength"] = normalize_truth_score(matching_judgment.get("strength", 0.0))
        evidence_item["source_specificity"] = normalize_truth_score(matching_judgment.get("specificity", 0.0))


def summarize_selected_evidence(selected_evidence: list[dict]) -> EvidenceSummary:
    if not selected_evidence:
        return EvidenceSummary("insufficient", "weak")

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
        return EvidenceSummary("sufficient", "strong")
    if strong_count >= 1 and usable_count >= 2:
        return EvidenceSummary("sufficient", "mixed")
    if usable_count >= 2:
        return EvidenceSummary("sufficient", "mixed")
    if strong_count >= 1:
        return EvidenceSummary("limited", "mixed")
    if usable_count >= 1:
        return EvidenceSummary("limited", "weak")

    return EvidenceSummary("insufficient", "weak")


def calculate_decision_confidence(
    decision_stage: str,
    truth_score: float,
    selected_evidence_count: int,
    evidence_sufficiency: str,
    evidence_quality: str
) -> str:
    if decision_stage != "completed":
        return "low"

    distance_from_neutral = abs(truth_score - NEUTRAL_SCORE)

    if selected_evidence_count <= 0:
        return "low"

    if evidence_sufficiency == "insufficient":
        if evidence_quality == "strong" and distance_from_neutral >= HIGH_CONFIDENCE_DISTANCE:
            return "medium"
        return "low"

    if evidence_sufficiency == "limited":
        if selected_evidence_count <= 1:
            return "low"
        if distance_from_neutral < LIMITED_EVIDENCE_DISTANCE:
            return "low"
        return "medium"

    if evidence_quality == "strong" and selected_evidence_count >= 2 and distance_from_neutral >= HIGH_CONFIDENCE_DISTANCE:
        return "high"

    return "medium"
