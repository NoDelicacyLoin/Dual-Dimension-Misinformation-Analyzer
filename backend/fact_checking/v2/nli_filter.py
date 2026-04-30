import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

print(f"[NLI Filter] Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

print("[NLI Filter] Model loaded.")

STOPWORDS = {
    "a", "an", "the",
    "and", "or", "but",
    "to", "of", "in", "on", "at", "for", "from", "by", "with", "as",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "have", "has", "had",
    "it", "its", "he", "she", "they", "them", "his", "her", "their",
    "you", "your", "we", "our", "i", "me", "my",
    "who", "that", "this", "these", "those",
}

MIN_USABILITY_FLOOR = 0.08
USABLE_EVIDENCE_CUTOFF = 0.30
STRONG_EVIDENCE_CUTOFF = 0.55
STRONG_TERM_COVERAGE = 0.45


@dataclass
class UsabilityResult:
    score: float
    debug: dict


@dataclass
class FilterDecision:
    keep: bool
    reason: str
    usability_score: float
    evidence_quality: str
    usability_debug: dict


def normalize_text_tokens(text: str) -> list[str]:
    text = text.lower()
    raw_tokens = re.findall(r"[a-z0-9']+", text)
    normalized_tokens = []

    for token in raw_tokens:
        cleaned_token = token.strip("'")
        if len(cleaned_token) < 3:
            continue
        if cleaned_token in STOPWORDS:
            continue
        normalized_tokens.append(cleaned_token)

    return normalized_tokens


def build_claim_profile(user_claim: str) -> dict:
    claim_tokens = normalize_text_tokens(user_claim)
    number_tokens = [token for token in claim_tokens if any(char.isdigit() for char in token)]
    claim_terms = [token for token in claim_tokens if token not in number_tokens]

    claim_phrases = []
    for phrase_size in (3, 2):
        for start_index in range(len(claim_terms) - phrase_size + 1):
            claim_phrases.append(" ".join(claim_terms[start_index:start_index + phrase_size]))

    return {
        "number_tokens": number_tokens,
        "claim_terms": claim_terms,
        "claim_phrases": claim_phrases,
    }


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
    scored_evidence: list[dict],
) -> dict:
    return {
        "claim_used": user_claim,
        "threshold_used": relevance_threshold,
        "evidence_scored_count": len(all_scored_evidence),
        "evidence_above_threshold_count": len(scored_evidence),
        "top_k_used": top_k,
        "use_all_eligible_evidence": use_all_eligible_evidence,
        "scored_evidence": all_scored_evidence,
    }


def score_decision_usability(claim_profile: dict, evidence_text: str) -> UsabilityResult:
    evidence_token_set = set(normalize_text_tokens(evidence_text))
    lower_evidence_text = evidence_text.lower()
    claim_term_set = set(claim_profile["claim_terms"])
    number_token_set = set(claim_profile["number_tokens"])
    evidence_number_tokens = {token for token in evidence_token_set if any(char.isdigit() for char in token)}
    claim_phrase_list = claim_profile["claim_phrases"]

    overlapping_claim_terms = claim_term_set.intersection(evidence_token_set)
    overlapping_number_tokens = number_token_set.intersection(evidence_token_set)
    overlapping_claim_phrases = [
        phrase for phrase in claim_phrase_list
        if phrase in lower_evidence_text
    ]

    term_coverage = 0.0
    if claim_term_set:
        term_coverage = len(overlapping_claim_terms) / len(claim_term_set)

    phrase_coverage = 0.0
    if claim_phrase_list:
        phrase_coverage = len(overlapping_claim_phrases) / len(claim_phrase_list)

    weighted_parts = []
    if claim_term_set:
        weighted_parts.append((term_coverage, 0.55))
    if claim_phrase_list:
        weighted_parts.append((phrase_coverage, 0.25))
    if number_token_set:
        number_coverage = len(overlapping_number_tokens) / len(number_token_set)
        weighted_parts.append((number_coverage, 0.20))
    else:
        number_coverage = None

    if weighted_parts:
        total_weight = sum(weight for _, weight in weighted_parts)
        usability_score = sum(score * weight for score, weight in weighted_parts) / total_weight
    else:
        usability_score = 0.0

    has_conflicting_numbers = (
        bool(number_token_set)
        and bool(evidence_number_tokens)
        and not overlapping_number_tokens
    )
    if has_conflicting_numbers:
        usability_score *= 0.5

    usability_debug = {
        "overlapping_anchor_terms_count": len(overlapping_claim_terms),
        "overlapping_number_tokens_count": len(overlapping_number_tokens),
        "overlapping_anchor_terms": sorted(overlapping_claim_terms)[:6],
        "overlapping_anchor_phrases_count": len(overlapping_claim_phrases),
        "overlapping_anchor_phrases": overlapping_claim_phrases[:4],
        "term_coverage": term_coverage,
        "phrase_coverage": phrase_coverage,
        "anchor_strength_score": usability_score,
        "number_coverage": number_coverage,
        "has_conflicting_numbers": has_conflicting_numbers,
        "usability_score": usability_score,
    }

    return UsabilityResult(usability_score, usability_debug)


def should_keep_evidence(
    claim_profile: dict,
    evidence_text: str,
    contradiction_prob: float,
    entailment_prob: float,
    relevance_score: float,
    relevance_threshold: float,
) -> FilterDecision:
    usability_result = score_decision_usability(claim_profile, evidence_text)
    usability_score = usability_result.score
    usability_debug = usability_result.debug

    if relevance_score <= relevance_threshold:
        return FilterDecision(False, "below_relevance_threshold", usability_score, "weak", usability_debug)

    if usability_score < MIN_USABILITY_FLOOR:
        has_anchor_match = (
            usability_debug["overlapping_anchor_terms_count"] > 0
            or usability_debug["overlapping_number_tokens_count"] > 0
        )
        filter_reason = "below_usability_floor_weak_anchor" if has_anchor_match else "below_usability_floor_no_anchor"
        return FilterDecision(False, filter_reason, usability_score, "weak", usability_debug)

    if usability_score >= STRONG_EVIDENCE_CUTOFF and usability_debug["term_coverage"] >= STRONG_TERM_COVERAGE:
        evidence_quality = "strong"
    elif usability_score >= USABLE_EVIDENCE_CUTOFF:
        evidence_quality = "usable"
    else:
        evidence_quality = "weak"

    return FilterDecision(True, "passed", usability_score, evidence_quality, usability_debug)


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

    scored_evidence = []
    all_scored_evidence = []
    claim_profile = build_claim_profile(user_claim)

    with torch.no_grad():
        for each_evidence in oversampled_evidence:
            evidence_text = each_evidence.get("content", "").strip()
            if not evidence_text:
                continue

            encoded_inputs = tokenizer(
                evidence_text,
                user_claim,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            model_outputs = model(**encoded_inputs)
            probabilities = F.softmax(model_outputs.logits, dim=1)[0]

            contradiction_prob = probabilities[0].item()
            entailment_prob = probabilities[1].item()
            neutral_prob = probabilities[2].item()
            relevance_score = contradiction_prob + entailment_prob

            filter_decision = should_keep_evidence(
                claim_profile=claim_profile,
                evidence_text=evidence_text,
                contradiction_prob=contradiction_prob,
                entailment_prob=entailment_prob,
                relevance_score=relevance_score,
                relevance_threshold=relevance_threshold,
            )

            all_scored_evidence.append({
                "url": each_evidence.get("url", ""),
                "content_preview": evidence_text[:200],
                "relevance_score": relevance_score,
                "contradiction_prob": contradiction_prob,
                "entailment_prob": entailment_prob,
                "neutral_prob": neutral_prob,
                "usability_score": filter_decision.usability_score,
                "source_quality": each_evidence.get("source_quality", "general_web"),
                "source_quality_score": each_evidence.get("source_quality_score", 0.60),
                "evidence_quality": filter_decision.evidence_quality,
                "passed_threshold": filter_decision.keep,
                "filter_reason": filter_decision.reason,
                "usability_debug": filter_decision.usability_debug,
            })

            if filter_decision.keep:
                scored_evidence.append({
                    "url": each_evidence.get("url"),
                    "content": evidence_text,
                    "relevance_score": relevance_score,
                    "usability_score": filter_decision.usability_score,
                    "source_quality": each_evidence.get("source_quality", "general_web"),
                    "source_quality_score": each_evidence.get("source_quality_score", 0.60),
                    "evidence_quality": filter_decision.evidence_quality,
                })

    if not scored_evidence:
        filter_reason_counts: dict[str, int] = {}
        for scored_item in all_scored_evidence:
            filter_reason = scored_item.get("filter_reason", "unknown")
            filter_reason_counts[filter_reason] = filter_reason_counts.get(filter_reason, 0) + 1

        dominant_reason = max(filter_reason_counts, key=filter_reason_counts.get, default="unknown")

        if dominant_reason == "below_relevance_threshold":
            print("[NLI Filter] Warning: No evidence passed the relevance threshold.")
        elif dominant_reason == "below_usability_floor_no_anchor":
            print("[NLI Filter] Warning: No evidence matched the claim anchors strongly enough.")
        elif dominant_reason == "below_usability_floor_weak_anchor":
            print("[NLI Filter] Warning: Evidence was topic-related, but claim anchor match remained too weak.")
        else:
            print("[NLI Filter] Warning: No evidence survived filtering.")

        if return_debug_info:
            return [], build_debug_info(
                user_claim,
                relevance_threshold,
                top_k,
                use_all_eligible_evidence,
                all_scored_evidence,
                scored_evidence,
            )
        return []

    scored_evidence.sort(
        key=lambda evidence_item: (
            evidence_item["usability_score"],
            evidence_item.get("source_quality_score", 0.60),
            evidence_item["relevance_score"],
        ),
        reverse=True,
    )
    all_scored_evidence.sort(key=lambda evidence_item: evidence_item["relevance_score"], reverse=True)

    selected_evidence = scored_evidence if use_all_eligible_evidence else scored_evidence[:top_k]

    final_evidence = [
        {
            "url": filtered_evidence["url"],
            "content": filtered_evidence["content"],
            "evidence_quality": filtered_evidence["evidence_quality"],
            "usability_score": filtered_evidence["usability_score"],
            "source_quality": filtered_evidence.get("source_quality", "general_web"),
            "source_quality_score": filtered_evidence.get("source_quality_score", 0.60),
        }
        for filtered_evidence in selected_evidence
    ]

    if return_debug_info:
        return final_evidence, build_debug_info(
            user_claim,
            relevance_threshold,
            top_k,
            use_all_eligible_evidence,
            all_scored_evidence,
            scored_evidence,
        )

    return final_evidence
