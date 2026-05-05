import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

tokenizer = None
model = None

SHORT_EVIDENCE_LENGTH = 900
PASSAGE_JOIN_LENGTH = 700
MAX_PASSAGE_LENGTH = 1200
MAX_ANCHORS = 5
DEFAULT_SOURCE_QUALITY_SCORE = 0.60


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
    tokens = []

    for token in re.findall(r"[a-z0-9']+", text.lower()):
        cleaned_token = token.strip("'")
        if not cleaned_token:
            continue
        if len(cleaned_token) == 1 and not cleaned_token.isdigit():
            continue
        tokens.append(cleaned_token)

    return tokens


def normalize_text_for_match(text: str) -> str:
    return " ".join(extract_word_tokens(text))


def extract_claim_anchors(claim_text: str) -> list[str]:
    anchors = []

    for match in re.finditer(r'"([^"]{2,80})"', claim_text):
        anchors.append(match.group(1).strip())

    title_pattern = r"\b(?:[A-Z][A-Za-z0-9'-]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9'-]+|[A-Z]{2,}))*"
    for match in re.finditer(title_pattern, claim_text):
        anchors.append(match.group(0).strip())

    unique_anchors = []
    seen = set()
    for anchor in anchors:
        anchor_key = normalize_text_for_match(anchor)
        if anchor_key and anchor_key not in seen:
            unique_anchors.append(anchor)
            seen.add(anchor_key)

    return unique_anchors[:MAX_ANCHORS]


def extract_numbers(text: str) -> list[float]:
    numbers = []

    for raw_number in re.findall(r"\d+(?:\.\d+)?", text.lower()):
        try:
            numbers.append(float(raw_number))
        except ValueError:
            continue

    return numbers


def score_number_match(claim_text: str, evidence_text: str) -> tuple[float, str]:
    claim_numbers = extract_numbers(claim_text)
    evidence_numbers = extract_numbers(evidence_text)

    if not claim_numbers:
        return 0.5, "not_needed"
    if not evidence_numbers:
        return 0.4, "missing"

    rounded_claim_numbers = {round(number, 6) for number in claim_numbers}
    rounded_evidence_numbers = {round(number, 6) for number in evidence_numbers}

    if rounded_claim_numbers.intersection(rounded_evidence_numbers):
        return 1.0, "exact_match"
    return 0.6, "different_numbers"


def score_claim_match(claim_text: str, evidence_text: str) -> dict:
    claim_tokens = set(extract_word_tokens(claim_text))
    evidence_tokens = set(extract_word_tokens(evidence_text))
    overlapping_terms = sorted(claim_tokens.intersection(evidence_tokens))

    anchors = extract_claim_anchors(claim_text)
    normalized_evidence = normalize_text_for_match(evidence_text)
    missing_anchors = []
    matched_anchor_count = 0

    for anchor in anchors:
        normalized_anchor = normalize_text_for_match(anchor)
        if normalized_anchor and normalized_anchor in normalized_evidence:
            matched_anchor_count += 1
        else:
            missing_anchors.append(anchor)

    anchor_score = matched_anchor_count / len(anchors) if anchors else 1.0
    number_score, number_status = score_number_match(claim_text, evidence_text)

    if not claim_tokens:
        token_match_score = 0.0
    else:
        token_match_score = len(overlapping_terms) / len(claim_tokens)

    return {
        "claim_match_score": token_match_score,
        "anchor_score": anchor_score,
        "number_score": number_score,
        "number_status": number_status,
        "overlapping_terms": overlapping_terms[:8],
        "anchors": anchors,
        "missing_anchors": missing_anchors,
    }


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

    best_passage = sentences[0][:MAX_PASSAGE_LENGTH]
    best_score = -1.0

    for index, sentence in enumerate(sentences):
        passage = sentence
        if index + 1 < len(sentences) and len(passage) < PASSAGE_JOIN_LENGTH:
            passage = f"{passage} {sentences[index + 1]}"
        passage = passage[:MAX_PASSAGE_LENGTH]

        claim_match = score_claim_match(claim_text, passage)
        passage_score = (
            (0.55 * claim_match["claim_match_score"])
            + (0.45 * claim_match["anchor_score"])
        )

        if passage_score > best_score:
            best_score = passage_score
            best_passage = passage

    return best_passage


def score_nli_relevance(claim_text: str, evidence_text: str) -> dict:
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

    return {
        "contradiction_prob": contradiction_prob,
        "entailment_prob": entailment_prob,
        "neutral_prob": neutral_prob,
        "relevance_score": contradiction_prob + entailment_prob,
    }


def score_evidence(
    claim_text: str,
    evidence_text: str,
    source_quality_score: float,
    relevance_threshold: float,
) -> dict:
    nli_score = score_nli_relevance(claim_text, evidence_text)
    claim_match = score_claim_match(claim_text, evidence_text)

    nli_relevance = nli_score["relevance_score"]
    token_match = claim_match["claim_match_score"]
    anchor_match = claim_match["anchor_score"]
    number_match = claim_match["number_score"]

    final_match_score = (
        (0.45 * nli_relevance)
        + (0.20 * token_match)
        + (0.20 * anchor_match)
        + (0.15 * number_match)
    )
    selection_priority = (
        (0.40 * nli_relevance)
        + (0.20 * token_match)
        + (0.20 * anchor_match)
        + (0.10 * number_match)
        + (0.10 * source_quality_score)
    )

    keep_score = max(0.22, 0.12 + (0.60 * relevance_threshold))
    number_matches_claim = claim_match["number_status"] == "exact_match"
    has_multiple_claim_anchors = len(claim_match["anchors"]) >= 2

    keep_evidence = True
    filter_reason = "passed"

    if has_multiple_claim_anchors and anchor_match < 0.50 and not number_matches_claim:
        keep_evidence = False
        filter_reason = "missing_claim_anchor"
    elif token_match < 0.10 and not number_matches_claim:
        keep_evidence = False
        filter_reason = "low_claim_match"
    elif final_match_score < keep_score:
        keep_evidence = False
        if nli_relevance < relevance_threshold and token_match < 0.15:
            filter_reason = "below_relevance_threshold"
        else:
            filter_reason = "low_claim_match"

    if not keep_evidence:
        evidence_quality = "weak"
    elif final_match_score >= 0.74 and nli_relevance >= 0.65 and anchor_match >= 0.65:
        evidence_quality = "strong"
    elif final_match_score >= 0.48 and anchor_match >= 0.45:
        evidence_quality = "usable"
    else:
        evidence_quality = "weak"

    return {
        "keep": keep_evidence,
        "filter_reason": filter_reason,
        "evidence_quality": evidence_quality,
        "final_match_score": final_match_score,
        "selection_priority": selection_priority,
        **nli_score,
        **claim_match,
    }


def build_debug_info(
    user_claim: str,
    relevance_threshold: float,
    top_k: int,
    use_all_eligible_evidence: bool,
    scored_evidence: list[dict],
    kept_evidence: list[dict],
) -> dict:
    return {
        "claim_used": user_claim,
        "threshold_used": relevance_threshold,
        "evidence_scored_count": len(scored_evidence),
        "evidence_above_threshold_count": len(kept_evidence),
        "top_k_used": top_k,
        "use_all_eligible_evidence": use_all_eligible_evidence,
        "scored_evidence": scored_evidence,
    }


def filter_top_evidence(
    user_claim: str,
    oversampled_evidence: list[dict],
    relevance_threshold: float = 0.2,
    top_k: int = 3,
    use_all_eligible_evidence: bool = False,
    return_debug_info: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    scored_evidence = []
    kept_evidence = []

    for evidence_item in oversampled_evidence:
        raw_evidence_text = evidence_item.get("content", "").strip()
        if not raw_evidence_text:
            continue

        evidence_text = select_best_evidence_passage(user_claim, raw_evidence_text)
        evidence_url = evidence_item.get("url", "")
        source_quality = evidence_item.get("source_quality", "general_web")
        source_quality_score = evidence_item.get("source_quality_score", DEFAULT_SOURCE_QUALITY_SCORE)
        evidence_score = score_evidence(
            claim_text=user_claim,
            evidence_text=evidence_text,
            source_quality_score=source_quality_score,
            relevance_threshold=relevance_threshold,
        )

        scored_evidence.append(
            {
                "url": evidence_url,
                "content_preview": evidence_text[:200],
                "source_quality": source_quality,
                "source_quality_score": source_quality_score,
                "passed_threshold": evidence_score["keep"],
                "filter_reason": evidence_score["filter_reason"],
                "evidence_quality": evidence_score["evidence_quality"],
                "claim_match_score": evidence_score["claim_match_score"],
                "number_score": evidence_score["number_score"],
                "number_status": evidence_score["number_status"],
                "overlapping_terms": evidence_score["overlapping_terms"],
                "claim_anchor_score": evidence_score["anchor_score"],
                "claim_anchors": evidence_score["anchors"],
                "missing_claim_anchors": evidence_score["missing_anchors"],
                "contradiction_prob": evidence_score["contradiction_prob"],
                "entailment_prob": evidence_score["entailment_prob"],
                "neutral_prob": evidence_score["neutral_prob"],
                "relevance_score": evidence_score["relevance_score"],
                "final_match_score": evidence_score["final_match_score"],
                "selection_priority": evidence_score["selection_priority"],
            }
        )

        if evidence_score["keep"]:
            kept_evidence.append(
                {
                    "url": evidence_url,
                    "content": evidence_text,
                    "evidence_quality": evidence_score["evidence_quality"],
                    "source_quality": source_quality,
                    "source_quality_score": source_quality_score,
                    "relevance_score": evidence_score["relevance_score"],
                    "claim_match_score": evidence_score["claim_match_score"],
                    "number_score": evidence_score["number_score"],
                    "final_match_score": evidence_score["final_match_score"],
                    "selection_priority": evidence_score["selection_priority"],
                }
            )

    if not kept_evidence:
        debug_info = build_debug_info(
            user_claim=user_claim,
            relevance_threshold=relevance_threshold,
            top_k=top_k,
            use_all_eligible_evidence=use_all_eligible_evidence,
            scored_evidence=scored_evidence,
            kept_evidence=kept_evidence,
        )
        return ([], debug_info) if return_debug_info else []

    kept_evidence.sort(
        key=lambda evidence_item: (
            evidence_item["selection_priority"],
            evidence_item["relevance_score"],
            evidence_item["final_match_score"],
            evidence_item.get("source_quality_score", DEFAULT_SOURCE_QUALITY_SCORE),
        ),
        reverse=True,
    )
    scored_evidence.sort(
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
                "source_quality_score": evidence_item.get("source_quality_score", DEFAULT_SOURCE_QUALITY_SCORE),
            }
        )

    debug_info = build_debug_info(
        user_claim=user_claim,
        relevance_threshold=relevance_threshold,
        top_k=top_k,
        use_all_eligible_evidence=use_all_eligible_evidence,
        scored_evidence=scored_evidence,
        kept_evidence=kept_evidence,
    )
    return (final_evidence, debug_info) if return_debug_info else final_evidence
