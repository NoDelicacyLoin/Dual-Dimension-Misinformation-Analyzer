import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

tokenizer = None
model = None

MIN_KEEP_SCORE = 0.22
USABLE_EVIDENCE_SCORE = 0.48
STRONG_EVIDENCE_SCORE = 0.74
MIN_CLAIM_MATCH_SCORE = 0.10
MIN_MULTI_ANCHOR_SCORE = 0.50
ANCHOR_MISSING_SCORE = 0.80

SHORT_EVIDENCE_LENGTH = 900
PASSAGE_JOIN_LENGTH = 700
MAX_PASSAGE_LENGTH = 1200
MAX_ANCHORS = 5

KEEP_SCORE_BASE = 0.12
KEEP_SCORE_THRESHOLD_WEIGHT = 0.60
CONFLICT_NLI_SCORE = 0.45
STRONG_NLI_SCORE = 0.65
STRONG_ANCHOR_SCORE = 0.65
USABLE_NLI_SCORE = 0.55
USABLE_CLAIM_MATCH_SCORE = 0.20
USABLE_ANCHOR_SCORE = 0.45

PASSAGE_MATCH_WEIGHT = 0.55
PASSAGE_ANCHOR_WEIGHT = 0.45

FINAL_NLI_WEIGHT = 0.45
FINAL_MATCH_WEIGHT = 0.20
FINAL_ANCHOR_WEIGHT = 0.20
FINAL_NUMBER_WEIGHT = 0.15

PRIORITY_NLI_WEIGHT = 0.40
PRIORITY_MATCH_WEIGHT = 0.20
PRIORITY_ANCHOR_WEIGHT = 0.20
PRIORITY_NUMBER_WEIGHT = 0.10
PRIORITY_SOURCE_WEIGHT = 0.10

COMPARISON_PATTERNS = [
    (r"\bbefore\s+(\d+(?:\.\d+)?)", "lt"),
    (r"\bafter\s+(\d+(?:\.\d+)?)", "gt"),
    (r"\bunder\s+(\d+(?:\.\d+)?)", "lt"),
    (r"\bbelow\s+(\d+(?:\.\d+)?)", "lt"),
    (r"\bover\s+(\d+(?:\.\d+)?)", "gt"),
    (r"\babove\s+(\d+(?:\.\d+)?)", "gt"),
    (r"\bless than\s+(\d+(?:\.\d+)?)", "lt"),
    (r"\bmore than\s+(\d+(?:\.\d+)?)", "gt"),
    (r"\bat least\s+(\d+(?:\.\d+)?)", "gte"),
    (r"\bat most\s+(\d+(?:\.\d+)?)", "lte"),
]


@dataclass
class NliScoreResult:
    contradiction_prob: float
    entailment_prob: float
    neutral_prob: float
    relevance_score: float


@dataclass
class ClaimMatchResult:
    claim_match_score: float
    anchor_score: float
    number_score: float
    number_status: str
    overlapping_terms: list[str]
    anchors: list[str]
    missing_anchors: list[str]


@dataclass
class FilterDecision:
    keep: bool
    reason: str
    final_match_score: float
    evidence_quality: str
    claim_match_score: float
    number_score: float
    number_status: str
    nli_score: float
    debug: dict


def load_nli_model():
    global tokenizer, model

    if tokenizer is not None and model is not None:
        return tokenizer, model

    print(f"[NLI Filter] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print("[NLI Filter] Model loaded.")
    return tokenizer, model


def extract_word_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    cleaned_tokens = []

    for token in tokens:
        cleaned_token = token.strip("'")
        if not cleaned_token:
            continue
        if len(cleaned_token) == 1 and not cleaned_token.isdigit():
            continue
        cleaned_tokens.append(cleaned_token)

    return cleaned_tokens


def normalize_text_for_match(text: str) -> str:
    return " ".join(extract_word_tokens(text))


def extract_claim_anchors(text: str) -> list[str]:
    anchors = []

    for match in re.finditer(r'"([^"]{2,80})"', text):
        anchor = match.group(1).strip()
        if anchor:
            anchors.append(anchor)

    title_pattern = r"\b(?:[A-Z][A-Za-z0-9'-]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9'-]+|[A-Z]{2,}))*"
    for match in re.finditer(title_pattern, text):
        anchor = match.group(0).strip()
        anchor_tokens = extract_word_tokens(anchor)
        if anchor_tokens:
            anchors.append(anchor)

    unique_anchors = []
    seen = set()
    for anchor in anchors:
        key = " ".join(extract_word_tokens(anchor))
        if key and key not in seen:
            unique_anchors.append(anchor)
            seen.add(key)

    return unique_anchors[:MAX_ANCHORS]


def extract_numbers(text: str) -> list[float]:
    numbers = []

    for raw_number in re.findall(r"\d+(?:\.\d+)?", text.lower()):
        try:
            numbers.append(float(raw_number))
        except ValueError:
            continue

    return numbers


def extract_comparison_rules(text: str) -> list[tuple[str, float]]:
    rules = []
    lower_text = text.lower()

    for pattern, operator in COMPARISON_PATTERNS:
        for match in re.finditer(pattern, lower_text):
            try:
                threshold = float(match.group(1))
            except ValueError:
                continue
            rules.append((operator, threshold))

    return rules


def score_anchor_match(claim_text: str, evidence_text: str) -> tuple[float, list[str], list[str]]:
    normalized_evidence = normalize_text_for_match(evidence_text)
    anchors = extract_claim_anchors(claim_text)

    anchor_scores = []
    missing_anchors = []

    for anchor in anchors:
        anchor_tokens = extract_word_tokens(anchor)
        if not anchor_tokens:
            continue

        normalized_anchor = normalize_text_for_match(anchor)
        anchor_match = 1.0 if normalized_anchor in normalized_evidence else 0.0
        anchor_scores.append(anchor_match)
        if anchor_match < ANCHOR_MISSING_SCORE:
            missing_anchors.append(anchor)

    anchor_score = sum(anchor_scores) / len(anchor_scores) if anchor_scores else 1.0
    return anchor_score, anchors, missing_anchors


def token_weight(token: str) -> float:
    if any(char.isdigit() for char in token):
        return 1.0
    if len(token) >= 8:
        return 1.0
    if len(token) >= 6:
        return 0.85
    if len(token) >= 4:
        return 0.70
    if len(token) >= 2:
        return 0.50
    return 0.0


def score_claim_match(claim_text: str, evidence_text: str) -> ClaimMatchResult:
    claim_tokens = set(extract_word_tokens(claim_text))
    evidence_tokens = set(extract_word_tokens(evidence_text))
    overlapping_terms = sorted(claim_tokens.intersection(evidence_tokens))
    anchor_score, anchors, missing_anchors = score_anchor_match(claim_text, evidence_text)

    if not claim_tokens:
        claim_match_score = 0.0
    else:
        total_weight = sum(token_weight(token) for token in claim_tokens)
        overlap_weight = sum(token_weight(token) for token in overlapping_terms)
        claim_match_score = overlap_weight / total_weight if total_weight else 0.0

    claim_numbers = extract_numbers(claim_text)
    evidence_numbers = extract_numbers(evidence_text)
    comparison_rules = extract_comparison_rules(claim_text)

    if not claim_numbers:
        number_score = 0.5
        number_status = "not_needed"
    elif not evidence_numbers:
        number_score = 0.4
        number_status = "missing"
    elif comparison_rules:
        comparison_matched = False

        for operator, threshold in comparison_rules:
            for evidence_number in evidence_numbers:
                if operator == "lt" and evidence_number < threshold:
                    comparison_matched = True
                elif operator == "gt" and evidence_number > threshold:
                    comparison_matched = True
                elif operator == "lte" and evidence_number <= threshold:
                    comparison_matched = True
                elif operator == "gte" and evidence_number >= threshold:
                    comparison_matched = True

        if comparison_matched:
            number_score = 0.9
            number_status = "comparison_compatible"
        else:
            number_score = 0.2
            number_status = "comparison_conflict"
    else:
        rounded_claim_numbers = {round(number, 6) for number in claim_numbers}
        rounded_evidence_numbers = {round(number, 6) for number in evidence_numbers}

        if rounded_claim_numbers.intersection(rounded_evidence_numbers):
            number_score = 1.0
            number_status = "exact_match"
        else:
            number_score = 0.2
            number_status = "conflict"

    return ClaimMatchResult(
        claim_match_score=claim_match_score,
        anchor_score=anchor_score,
        number_score=number_score,
        number_status=number_status,
        overlapping_terms=overlapping_terms[:8],
        anchors=anchors,
        missing_anchors=missing_anchors,
    )


def split_evidence_sentences(text: str) -> list[str]:
    clean_text = re.sub(r"\s+", " ", text).strip()
    if not clean_text:
        return []
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", clean_text) if item.strip()]


def select_best_evidence_passage(claim_text: str, evidence_text: str) -> str:
    clean_text = re.sub(r"\s+", " ", evidence_text).strip()
    if len(clean_text) <= SHORT_EVIDENCE_LENGTH:
        return clean_text

    sentences = split_evidence_sentences(clean_text)
    if not sentences:
        return clean_text[:MAX_PASSAGE_LENGTH]

    candidates = []
    for index, sentence in enumerate(sentences):
        chunk = sentence
        if index + 1 < len(sentences) and len(chunk) < PASSAGE_JOIN_LENGTH:
            chunk = f"{chunk} {sentences[index + 1]}"
        candidates.append(chunk[:MAX_PASSAGE_LENGTH])

    best_passage = candidates[0]
    best_score = -1.0

    for candidate in candidates:
        claim_match = score_claim_match(claim_text, candidate)
        passage_score = (
            (PASSAGE_MATCH_WEIGHT * claim_match.claim_match_score)
            + (PASSAGE_ANCHOR_WEIGHT * claim_match.anchor_score)
        )

        if passage_score > best_score:
            best_score = passage_score
            best_passage = candidate

    return best_passage


def score_nli_relevance(claim_text: str, evidence_text: str) -> NliScoreResult:
    active_tokenizer, active_model = load_nli_model()

    encoded_inputs = active_tokenizer(
        evidence_text,
        claim_text,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        model_outputs = active_model(**encoded_inputs)
        probabilities = F.softmax(model_outputs.logits, dim=1)[0]

    contradiction_prob = probabilities[0].item()
    entailment_prob = probabilities[1].item()
    neutral_prob = probabilities[2].item()
    relevance_score = contradiction_prob + entailment_prob

    return NliScoreResult(
        contradiction_prob=contradiction_prob,
        entailment_prob=entailment_prob,
        neutral_prob=neutral_prob,
        relevance_score=relevance_score,
    )


def decide_keep(
    claim_text: str,
    evidence_text: str,
    source_quality_score: float,
    relevance_threshold: float,
) -> FilterDecision:
    nli_result = score_nli_relevance(claim_text, evidence_text)
    claim_match = score_claim_match(claim_text, evidence_text)

    final_match_score = (
        (FINAL_NLI_WEIGHT * nli_result.relevance_score)
        + (FINAL_MATCH_WEIGHT * claim_match.claim_match_score)
        + (FINAL_ANCHOR_WEIGHT * claim_match.anchor_score)
        + (FINAL_NUMBER_WEIGHT * claim_match.number_score)
    )

    keep_score = max(MIN_KEEP_SCORE, KEEP_SCORE_BASE + (KEEP_SCORE_THRESHOLD_WEIGHT * relevance_threshold))
    has_strong_number_signal = claim_match.number_status in {"exact_match", "comparison_compatible"}
    has_multiple_anchors = len(claim_match.anchors) >= 2

    if (
        has_multiple_anchors
        and claim_match.anchor_score < MIN_MULTI_ANCHOR_SCORE
        and not has_strong_number_signal
    ):
        keep = False
        reason = "missing_claim_anchor"
        evidence_quality = "weak"
    elif claim_match.claim_match_score < MIN_CLAIM_MATCH_SCORE and not has_strong_number_signal:
        keep = False
        reason = "low_claim_match"
        evidence_quality = "weak"
    elif final_match_score < keep_score:
        if nli_result.relevance_score < relevance_threshold and claim_match.claim_match_score < 0.15:
            reason = "below_relevance_threshold"
        elif (
            claim_match.number_status in {"conflict", "comparison_conflict"}
            and nli_result.relevance_score < CONFLICT_NLI_SCORE
        ):
            reason = "conflicting_claim_details"
        else:
            reason = "low_claim_match"
        keep = False
        evidence_quality = "weak"
    else:
        keep = True
        if (
            final_match_score >= STRONG_EVIDENCE_SCORE
            and nli_result.relevance_score >= STRONG_NLI_SCORE
            and claim_match.anchor_score >= STRONG_ANCHOR_SCORE
        ):
            evidence_quality = "strong"
        elif (
            final_match_score >= USABLE_EVIDENCE_SCORE
            and claim_match.anchor_score >= USABLE_ANCHOR_SCORE
        ) or (
            nli_result.relevance_score >= USABLE_NLI_SCORE
            and claim_match.claim_match_score >= USABLE_CLAIM_MATCH_SCORE
            and claim_match.anchor_score >= USABLE_ANCHOR_SCORE
        ):
            evidence_quality = "usable"
        else:
            evidence_quality = "weak"
        reason = "passed"

    selection_priority = (
        (PRIORITY_NLI_WEIGHT * nli_result.relevance_score)
        + (PRIORITY_MATCH_WEIGHT * claim_match.claim_match_score)
        + (PRIORITY_ANCHOR_WEIGHT * claim_match.anchor_score)
        + (PRIORITY_NUMBER_WEIGHT * claim_match.number_score)
        + (PRIORITY_SOURCE_WEIGHT * source_quality_score)
    )

    debug = {
        "claim_match_score": claim_match.claim_match_score,
        "number_score": claim_match.number_score,
        "number_status": claim_match.number_status,
        "overlapping_terms": claim_match.overlapping_terms,
        "claim_anchor_score": claim_match.anchor_score,
        "claim_anchors": claim_match.anchors,
        "missing_claim_anchors": claim_match.missing_anchors,
        "contradiction_prob": nli_result.contradiction_prob,
        "entailment_prob": nli_result.entailment_prob,
        "neutral_prob": nli_result.neutral_prob,
        "relevance_score": nli_result.relevance_score,
        "final_match_score": final_match_score,
        "selection_priority": selection_priority,
    }

    return FilterDecision(
        keep=keep,
        reason=reason,
        final_match_score=final_match_score,
        evidence_quality=evidence_quality,
        claim_match_score=claim_match.claim_match_score,
        number_score=claim_match.number_score,
        number_status=claim_match.number_status,
        nli_score=nli_result.relevance_score,
        debug=debug,
    )


def build_empty_debug_info(
    user_claim: str,
    relevance_threshold: float,
    top_k: int,
    use_all_eligible_evidence: bool,
) -> dict:
    return {
        "claim_used": user_claim,
        "threshold_used": relevance_threshold,
        "evidence_scored_count": 0,
        "evidence_above_threshold_count": 0,
        "top_k_used": top_k,
        "use_all_eligible_evidence": use_all_eligible_evidence,
        "scored_evidence": [],
    }


def build_debug_info(
    user_claim: str,
    relevance_threshold: float,
    top_k: int,
    use_all_eligible_evidence: bool,
    all_scored_evidence: list[dict],
    kept_evidence: list[dict],
) -> dict:
    return {
        "claim_used": user_claim,
        "threshold_used": relevance_threshold,
        "evidence_scored_count": len(all_scored_evidence),
        "evidence_above_threshold_count": len(kept_evidence),
        "top_k_used": top_k,
        "use_all_eligible_evidence": use_all_eligible_evidence,
        "scored_evidence": all_scored_evidence,
    }


def filter_top_evidence(
    user_claim: str,
    oversampled_evidence: list[dict],
    relevance_threshold: float = 0.2,
    top_k: int = 3,
    use_all_eligible_evidence: bool = False,
    return_debug_info: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    if not oversampled_evidence:
        if return_debug_info:
            return [], build_empty_debug_info(
                user_claim,
                relevance_threshold,
                top_k,
                use_all_eligible_evidence,
            )
        return []

    kept_evidence = []
    all_scored_evidence = []

    for evidence_item in oversampled_evidence:
        raw_evidence_text = evidence_item.get("content", "").strip()
        if not raw_evidence_text:
            continue
        evidence_text = select_best_evidence_passage(user_claim, raw_evidence_text)

        source_quality = evidence_item.get("source_quality", "general_web")
        source_quality_score = evidence_item.get("source_quality_score", 0.60)
        filter_decision = decide_keep(
            claim_text=user_claim,
            evidence_text=evidence_text,
            source_quality_score=source_quality_score,
            relevance_threshold=relevance_threshold,
        )

        all_scored_evidence.append(
            {
                "url": evidence_item.get("url", ""),
                "content_preview": evidence_text[:200],
                "source_quality": source_quality,
                "source_quality_score": source_quality_score,
                "passed_threshold": filter_decision.keep,
                "filter_reason": filter_decision.reason,
                "evidence_quality": filter_decision.evidence_quality,
                **filter_decision.debug,
            }
        )

        if filter_decision.keep:
            kept_evidence.append(
                {
                    "url": evidence_item.get("url", ""),
                    "content": evidence_text,
                    "evidence_quality": filter_decision.evidence_quality,
                    "source_quality": source_quality,
                    "source_quality_score": source_quality_score,
                    "relevance_score": filter_decision.nli_score,
                    "claim_match_score": filter_decision.claim_match_score,
                    "number_score": filter_decision.number_score,
                    "final_match_score": filter_decision.final_match_score,
                    "selection_priority": filter_decision.debug["selection_priority"],
                }
            )

    if not kept_evidence:
        reason_counts = {}
        for scored_item in all_scored_evidence:
            reason = scored_item.get("filter_reason", "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        dominant_reason = max(reason_counts, key=reason_counts.get, default="unknown")
        if dominant_reason == "below_relevance_threshold":
            print("[NLI Filter] Warning: Retrieved evidence looked off-topic for the claim.")
        elif dominant_reason == "conflicting_claim_details":
            print("[NLI Filter] Warning: Evidence discussed the topic, but key claim details conflicted.")
        elif dominant_reason == "missing_claim_anchor":
            print("[NLI Filter] Warning: Evidence missed the main entity or title in the claim.")
        elif dominant_reason == "low_claim_match":
            print(
                "[NLI Filter] Warning: Evidence was somewhat related, "
                "but still not close enough to the claim."
            )
        else:
            print("[NLI Filter] Warning: No evidence survived filtering.")

        if return_debug_info:
            return [], build_debug_info(
                user_claim,
                relevance_threshold,
                top_k,
                use_all_eligible_evidence,
                all_scored_evidence,
                kept_evidence,
            )
        return []

    kept_evidence.sort(
        key=lambda evidence_item: (
            evidence_item["selection_priority"],
            evidence_item["relevance_score"],
            evidence_item["final_match_score"],
            evidence_item.get("source_quality_score", 0.60),
        ),
        reverse=True,
    )
    all_scored_evidence.sort(
        key=lambda evidence_item: evidence_item.get("selection_priority", 0.0),
        reverse=True,
    )

    selected_evidence = kept_evidence if use_all_eligible_evidence else kept_evidence[:top_k]

    final_evidence = []
    for evidence_item in selected_evidence:
        final_evidence.append(
            {
                "url": evidence_item["url"],
                "content": evidence_item["content"],
                "evidence_quality": evidence_item["evidence_quality"],
                "source_quality": evidence_item.get("source_quality", "general_web"),
                "source_quality_score": evidence_item.get("source_quality_score", 0.60),
            }
        )

    if return_debug_info:
        return final_evidence, build_debug_info(
            user_claim,
            relevance_threshold,
            top_k,
            use_all_eligible_evidence,
            all_scored_evidence,
            kept_evidence,
        )

    return final_evidence
