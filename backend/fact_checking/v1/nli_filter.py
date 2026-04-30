import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from search import fetch_oversampled_evidence
import re

# 1. 全局加载模型
MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

print(f"[NLI Filter] Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

print("[NLI Filter] Model loaded.")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "am",
    "to", "of", "in", "on", "at", "for", "from", "by", "with", "about", "into",
    "after", "before", "over", "under", "between", "through", "during", "without",
    "as", "it", "its", "he", "she", "they", "them", "his", "her", "their", "you",
    "your", "we", "our", "i", "me", "my", "do", "does", "did", "done", "have",
    "has", "had", "will", "would", "should", "could", "can", "may", "might",
    "not", "no", "yes", "just", "very", "more", "most", "some", "any", "all",
    "people", "says", "said"
}

NEUTRAL_REJECTION_CUTOFF = 0.85
MIN_SUPPORT_SIGNAL = 0.08
USABLE_EVIDENCE_CUTOFF = 0.38
STRONG_EVIDENCE_CUTOFF = 0.75
STRONG_KEY_TERM_COVERAGE = 0.50


def normalize_text_tokens(text: str) -> list[str]:
    """
    Break text into simple lowercase tokens for lightweight overlap checks.
    """
    text = text.lower()
    raw_tokens = re.findall(r"[a-z0-9']+", text)
    normalized_tokens = []

    for token in raw_tokens:
        cleaned_token = token.strip("'")
        if len(cleaned_token) < 4:
            continue
        if cleaned_token in STOPWORDS:
            continue
        normalized_tokens.append(cleaned_token)

    return normalized_tokens


def build_claim_profile(user_claim: str) -> dict:
    """
    Build a lightweight claim profile for evidence usability checks.
    """
    claim_tokens = normalize_text_tokens(user_claim)
    number_tokens = [token for token in claim_tokens if any(char.isdigit() for char in token)]
    key_terms = [token for token in claim_tokens if token not in number_tokens]

    return {
        "all_tokens": claim_tokens,
        "number_tokens": number_tokens,
        "key_terms": key_terms
    }


def score_decision_usability(claim_profile: dict, evidence_text: str) -> tuple[float, dict]:
    """
    Estimate whether an evidence item is usable for final verification,
    not just loosely related to the same topic.
    """
    evidence_tokens = normalize_text_tokens(evidence_text)
    evidence_token_set = set(evidence_tokens)

    key_term_set = set(claim_profile["key_terms"])
    number_token_set = set(claim_profile["number_tokens"])

    overlapping_key_terms = key_term_set.intersection(evidence_token_set)
    overlapping_number_tokens = number_token_set.intersection(evidence_token_set)

    total_key_term_count = len(key_term_set)
    if total_key_term_count > 0:
        key_term_coverage = len(overlapping_key_terms) / total_key_term_count
    else:
        key_term_coverage = 0.0

    capped_key_term_count = min(len(overlapping_key_terms), 4)
    key_term_strength = capped_key_term_count / 4

    if number_token_set:
        number_score = len(overlapping_number_tokens) / len(number_token_set)
    else:
        number_score = 0.5

    usability_score = (0.5 * key_term_strength) + (0.3 * key_term_coverage) + (0.2 * number_score)

    usability_debug = {
        "overlapping_key_terms_count": len(overlapping_key_terms),
        "overlapping_number_tokens_count": len(overlapping_number_tokens),
        "overlapping_key_terms": sorted(list(overlapping_key_terms))[:6],
        "key_term_coverage": key_term_coverage,
        "usability_score": usability_score
    }

    return usability_score, usability_debug


def should_keep_evidence(
    claim_profile: dict,
    evidence_text: str,
    contradiction_prob: float,
    entailment_prob: float,
    neutral_prob: float,
    relevance_score: float,
    relevance_threshold: float
) -> tuple[bool, str, float, str, dict]:
    """
    Decide whether an evidence item is relevant enough and usable enough
    for the final verdict stage.
    """
    usability_score, usability_debug = score_decision_usability(claim_profile, evidence_text)

    if relevance_score <= relevance_threshold:
        return False, "below_relevance_threshold", usability_score, "weak", usability_debug

    # Let a few more borderline passages survive when they are otherwise useful.
    if neutral_prob >= NEUTRAL_REJECTION_CUTOFF:
        return False, "too_neutral", usability_score, "weak", usability_debug

    strongest_support = max(contradiction_prob, entailment_prob)
    if strongest_support < MIN_SUPPORT_SIGNAL:
        return False, "too_weak_individually", usability_score, "weak", usability_debug

    overlapping_key_terms_count = usability_debug["overlapping_key_terms_count"]
    overlapping_number_tokens_count = usability_debug["overlapping_number_tokens_count"]

    if overlapping_key_terms_count == 0 and overlapping_number_tokens_count == 0:
        return False, "no_claim_anchor", usability_score, "weak", usability_debug

    key_term_coverage = usability_debug["key_term_coverage"]

    if usability_score >= STRONG_EVIDENCE_CUTOFF and key_term_coverage >= STRONG_KEY_TERM_COVERAGE:
        evidence_quality = "strong"
    elif usability_score >= USABLE_EVIDENCE_CUTOFF:
        evidence_quality = "usable"
    else:
        evidence_quality = "weak"

    if evidence_quality == "weak":
        return False, "not_decision_usable", usability_score, evidence_quality, usability_debug

    return True, "passed", usability_score, evidence_quality, usability_debug


def filter_top_evidence(
    user_claim: str,
    oversampled_evidence: list[dict],
    relevance_threshold: float = 0.2,
    top_k: int = 3,
    return_debug_info: bool = False
) -> list[dict] | tuple[list[dict], dict]:
    """
    Read the retrieved evidence items one by one.
    Compute how relevant each evidence item is to the user's claim.
    Keep only the top-k evidence items above the relevance threshold.

    If return_debug_info=True, also return detailed scoring information
    for notebook analysis and threshold debugging.
    """

    if not oversampled_evidence:
        if return_debug_info:
            debug_info = {
                "claim_used": user_claim,
                "threshold_used": relevance_threshold,
                "evidence_scored_count": 0,
                "evidence_above_threshold_count": 0,
                "top_k_used": top_k,
                "scored_evidence": []
            }
            return [], debug_info
        return []

    if oversampled_evidence[0].get("url") == "Error":
        if return_debug_info:
            debug_info = {
                "claim_used": user_claim,
                "threshold_used": relevance_threshold,
                "evidence_scored_count": 0,
                "evidence_above_threshold_count": 0,
                "top_k_used": top_k,
                "scored_evidence": []
            }
            return [], debug_info
        return []

    scored_evidence = []
    all_scored_evidence = []
    claim_profile = build_claim_profile(user_claim)

    with torch.no_grad():

        for each_evidence in oversampled_evidence:
            evidence_text = each_evidence.get("content", "").strip()
            if not evidence_text:
                continue

            evidence_url = each_evidence.get("url", "")

            encoded_inputs = tokenizer(
                evidence_text, user_claim, truncation=True, max_length=512, return_tensors="pt"
            )

            model_outputs = model(**encoded_inputs)
            probabilities = F.softmax(model_outputs.logits, dim=1)[0]

            contradiction_prob = probabilities[0].item()
            entailment_prob = probabilities[1].item()
            neutral_prob = probabilities[2].item()

            relevance_score = contradiction_prob + entailment_prob
            passed_filter, filter_reason, usability_score, evidence_quality, usability_debug = should_keep_evidence(
                claim_profile=claim_profile,
                evidence_text=evidence_text,
                contradiction_prob=contradiction_prob,
                entailment_prob=entailment_prob,
                neutral_prob=neutral_prob,
                relevance_score=relevance_score,
                relevance_threshold=relevance_threshold
            )

            all_scored_evidence.append({
                "url": evidence_url,
                "content_preview": evidence_text[:200],
                "relevance_score": relevance_score,
                "contradiction_prob": contradiction_prob,
                "entailment_prob": entailment_prob,
                "neutral_prob": neutral_prob,
                "usability_score": usability_score,
                "evidence_quality": evidence_quality,
                "passed_threshold": passed_filter,
                "filter_reason": filter_reason,
                "usability_debug": usability_debug
            })

            if passed_filter:
                scored_evidence.append({
                    "url": each_evidence.get("url"),
                    "content": evidence_text,
                    "relevance_score": relevance_score,
                    "usability_score": usability_score,
                    "evidence_quality": evidence_quality
                })

    if not scored_evidence:
        print("[NLI Filter] Warning: All evidence was filtered out as irrelevant.")

        if return_debug_info:
            debug_info = {
                "claim_used": user_claim,
                "threshold_used": relevance_threshold,
                "evidence_scored_count": len(all_scored_evidence),
                "evidence_above_threshold_count": 0,
                "top_k_used": top_k,
                "scored_evidence": all_scored_evidence
            }
            return [], debug_info

        return []

    # 按相关度从高到低排序
    scored_evidence.sort(
        key=lambda evidence_item: (
            evidence_item["usability_score"],
            evidence_item["relevance_score"]
        ),
        reverse=True
    )
    all_scored_evidence.sort(key=lambda evidence_item: evidence_item["relevance_score"], reverse=True)

    final_evidence = []
    top_scored_evidence = scored_evidence[:top_k]

    for filtered_evidence in top_scored_evidence:
        final_evidence.append({
            "url": filtered_evidence["url"],
            "content": filtered_evidence["content"],
            "evidence_quality": filtered_evidence["evidence_quality"],
            "usability_score": filtered_evidence["usability_score"]
        })

    if return_debug_info:
        debug_info = {
            "claim_used": user_claim,
            "threshold_used": relevance_threshold,
            "evidence_scored_count": len(all_scored_evidence),
            "evidence_above_threshold_count": len(scored_evidence),
            "top_k_used": top_k,
            "scored_evidence": all_scored_evidence
        }
        return final_evidence, debug_info

    return final_evidence


# --- 本地测试区块 ---
if __name__ == "__main__":
    threshold = 0.8
    top_k = 8
    test_claims = [
        "Drinking warm lemon water every morning completely detoxifies the liver.",
        "Albert Einstein failed his high school math classes.",
        "Sharks do not get cancer."
    ]

    for each_claim in test_claims:
        print("\n" + "=" * 80)
        print(f"\n[Test Claim]: {each_claim}")
        test_oversampled = fetch_oversampled_evidence(each_claim, max_results=8)

        print("=" * 40)
        print("Test at threshold 0.90:")
        results_loose, debug_loose = filter_top_evidence(
            each_claim,
            test_oversampled,
            relevance_threshold=0.90,
            top_k=8,
            return_debug_info=True
        )
        print(f"Passed count: {len(results_loose)}")
        print("\n--- Filtered Results ---")
        for r in results_loose:
            print(f"URL: {r['url']}\nContent: {r['content']}\n")

        print("\n--- Debug Summary ---")
        print(f"Evidence scored: {debug_loose['evidence_scored_count']}")
        print(f"Evidence above threshold: {debug_loose['evidence_above_threshold_count']}")

        print("=" * 40)
        print("Test at threshold 0.50:")
        results_strict, debug_strict = filter_top_evidence(
            each_claim,
            test_oversampled,
            relevance_threshold=0.50,
            top_k=8,
            return_debug_info=True
        )
        print(f"Passed count: {len(results_strict)}")
        print("\n--- Filtered Results ---")
        for r in results_strict:
            print(f"URL: {r['url']}\nContent: {r['content']}\n")

        print("\n--- Debug Summary ---")
        print(f"Evidence scored: {debug_strict['evidence_scored_count']}")
        print(f"Evidence above threshold: {debug_strict['evidence_above_threshold_count']}")

        print("=" * 80 + "\n")
