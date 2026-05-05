import json
import re

from api_contract import AtomizedClaimGroup, AtomizedFactClaim, AtomizerOutput, AtomizerSummary
from atomizer.atomizer_gemini import generate_atomizer_json, is_gemini_available
from atomizer.prompts import build_claim_group_ranking_prompt


MAX_SENTENCES_FOR_ATOMIZER = 12
MAX_FACT_CLAIMS_PER_GROUP = 5
LONG_TEXT_BATCH_SIZE = 8
LONG_TEXT_CONTEXT_SENTENCES = 2
MAX_CLAIM_GROUPS_FOR_OUTPUT = 5
GENERIC_REFERENCE_NOUNS = {
    "mission",
    "launch",
    "program",
    "project",
    "bill",
    "law",
    "policy",
    "claim",
    "report",
    "plan",
    "campaign",
    "event",
}
GENERIC_REFERENCE_MODIFIERS = {
    "planned",
    "scheduled",
    "proposed",
    "next",
    "current",
    "same",
}


def normalize_input_text(raw_text: str) -> str:
    compact_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    compact_text = re.sub(r"[ \t]+", " ", compact_text)
    compact_text = re.sub(r"(?<=[a-z0-9][.!?])(?=[A-Z])", " ", compact_text)
    compact_text = re.sub(r"(?<=\d)(?=[A-Z])", " ", compact_text)
    compact_text = re.sub(r"\n{3,}", "\n\n", compact_text)
    return compact_text.strip()


def split_into_sentences(text: str, limit: int | None = MAX_SENTENCES_FOR_ATOMIZER) -> list[str]:
    protected_text = re.sub(r"\b([A-Z])\.(?=\s+[A-Z])", r"\1<dot>", text)
    protected_text = re.sub(r'([.!?]["\'])\s+(?=[A-Z])', r"\1<split>", protected_text)
    sentence_items = [
        sentence.replace("<dot>", ".").replace("<split>", " ").strip()
        for sentence in re.split(r"(?<=[.!?])\s+|<split>", protected_text)
        if sentence.strip()
    ]
    if limit is None:
        return sentence_items
    return sentence_items[:limit]


def trim_json_fence(response_text: str) -> str:
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?%?", text))


def clean_string_list(raw_items) -> list[str]:
    if not isinstance(raw_items, list):
        return []

    clean_items = []
    for raw_item in raw_items:
        item = re.sub(r"\s+", " ", str(raw_item)).strip()
        if item and item not in clean_items:
            clean_items.append(item)
    return clean_items[:5]


def has_unresolved_generic_reference(text: str) -> bool:
    match = re.match(r"^(The|This|That)\s+([A-Za-z]+)(?:\s+([A-Za-z]+))?\b", text)
    if not match:
        return False

    first_word = match.group(2).lower()
    second_word = (match.group(3) or "").lower()
    if first_word in GENERIC_REFERENCE_NOUNS:
        return True
    return first_word in GENERIC_REFERENCE_MODIFIERS and second_word in GENERIC_REFERENCE_NOUNS


def has_unresolved_reference(text: str) -> bool:
    lower_text = text.lower()
    starts_with_reference = re.search(r"^(he|she|they|it|this|that)\b", lower_text)
    object_reference = re.search(
        r"\b(said|claimed|reported|stated|thought|believed|posted|wrote)\s+(it|this|that)\b",
        lower_text,
    )
    return (
        starts_with_reference is not None
        or object_reference is not None
        or has_unresolved_generic_reference(text)
    )


def make_fact_claim(
    fact_index: int,
    claim_text: str,
    entities: list[str] | None = None,
    relation: str = "",
    constraints: list[str] | None = None,
) -> AtomizedFactClaim:
    return AtomizedFactClaim(
        fact_claim_id=fact_index,
        claim=re.sub(r"\s+", " ", claim_text).strip(),
        entities=clean_string_list(entities or []),
        relation=re.sub(r"\s+", " ", relation).strip(),
        constraints=clean_string_list(constraints or []),
    )


def add_fact_claim(
    fact_claims: list[AtomizedFactClaim],
    seen_claims: set[str],
    raw_claim_text,
    entities: list[str] | None = None,
    relation: str = "",
    constraints: list[str] | None = None,
    sentence_numbers: set[str] | None = None,
) -> None:
    if not isinstance(raw_claim_text, str):
        return

    claim_text = re.sub(r"\s+", " ", raw_claim_text).strip()
    if len(claim_text) < 8:
        return
    if sentence_numbers is not None:
        if has_unresolved_reference(claim_text):
            return
        if not extract_numbers(claim_text).issubset(sentence_numbers):
            return

    claim_key = normalize_text(claim_text)
    if not claim_key or claim_key in seen_claims:
        return

    fact_claims.append(
        make_fact_claim(
            len(fact_claims) + 1,
            claim_text,
            entities=entities,
            relation=relation,
            constraints=constraints,
        )
    )
    seen_claims.add(claim_key)


def make_claim_group(
    claim_group_id: int,
    original_sentence: str,
    fact_claims: list[AtomizedFactClaim],
) -> AtomizedClaimGroup:
    return AtomizedClaimGroup(
        claim_group_id=claim_group_id,
        original_sentence=original_sentence,
        text_feature_text=original_sentence,
        atomization_applied=len(fact_claims) > 1,
        fact_check_claims=fact_claims,
    )


def build_summary(ignored_sentences: list[str], claim_groups: list[AtomizedClaimGroup]) -> AtomizerSummary:
    return AtomizerSummary(
        ignored_sentence_count=len(ignored_sentences),
        text_feature_unit_count=len(claim_groups),
        fact_check_claim_count=sum(len(group.fact_check_claims) for group in claim_groups),
    )


def make_atomizer_output(
    original_text: str,
    ignored_sentences: list[str],
    claim_groups: list[AtomizedClaimGroup],
) -> AtomizerOutput:
    return AtomizerOutput(
        status="success" if claim_groups else "invalid_input",
        original_text=original_text,
        ignored_sentences=ignored_sentences,
        claim_groups=claim_groups,
        summary=build_summary(ignored_sentences, claim_groups),
    )


def make_atomizer_error_output(original_text: str, message: str) -> AtomizerOutput:
    return AtomizerOutput(
        status="atomizer_error",
        original_text=original_text,
        ignored_sentences=[],
        claim_groups=[],
        summary=AtomizerSummary(),
        message=message,
    )


def make_invalid_input_output(original_text: str) -> AtomizerOutput:
    return AtomizerOutput(
        status="invalid_input",
        original_text=original_text,
        ignored_sentences=[],
        claim_groups=[],
        summary=AtomizerSummary(),
        candidate_claim_group_count=0,
        candidate_fact_claim_count=0,
        selected_claim_group_count=0,
        selected_fact_claim_count=0,
        max_claim_group_count=MAX_CLAIM_GROUPS_FOR_OUTPUT,
        claim_selection_reason="",
    )


def renumber_claim_groups(claim_groups: list[AtomizedClaimGroup]) -> list[AtomizedClaimGroup]:
    renumbered_groups = []
    for group_index, group in enumerate(claim_groups, start=1):
        fact_claims = []
        for fact_index, fact_claim in enumerate(group.fact_check_claims, start=1):
            fact_claims.append(
                make_fact_claim(
                    fact_index,
                    fact_claim.claim,
                    entities=fact_claim.entities,
                    relation=fact_claim.relation,
                    constraints=fact_claim.constraints,
                )
            )
        renumbered_groups.append(make_claim_group(group_index, group.original_sentence, fact_claims))
    return renumbered_groups


def get_sentence_context(sentence_items: list[str], sentence_text: str) -> str:
    sentence_index = 0
    normalized_sentence = normalize_text(sentence_text)

    for index, item in enumerate(sentence_items):
        if normalize_text(item) == normalized_sentence:
            sentence_index = index
            break

    start_index = max(0, sentence_index - 2)
    context_items = sentence_items[start_index:sentence_index + 1]
    return " ".join(context_items)


def get_matched_sentence(sentence_text, sentence_lookup: dict[str, str], target_sentence_keys: set[str]) -> str:
    if not isinstance(sentence_text, str):
        return ""

    matched_sentence = sentence_lookup.get(normalize_text(sentence_text))
    if not matched_sentence:
        return ""
    if normalize_text(matched_sentence) not in target_sentence_keys:
        return ""
    return matched_sentence


def add_ignored_sentence(
    sentence_text,
    sentence_lookup: dict[str, str],
    target_sentence_keys: set[str],
    ignored_sentences: list[str],
    assigned_sentences: set[str],
) -> None:
    matched_sentence = get_matched_sentence(sentence_text, sentence_lookup, target_sentence_keys)
    if not matched_sentence or matched_sentence in assigned_sentences:
        return

    ignored_sentences.append(matched_sentence)
    assigned_sentences.add(matched_sentence)


def get_fact_claims(raw_fact_claims: list, sentence_context: str, seen_claims: set[str]) -> list[AtomizedFactClaim]:
    fact_claims: list[AtomizedFactClaim] = []
    sentence_numbers = extract_numbers(sentence_context)

    for raw_fact_claim in raw_fact_claims[:MAX_FACT_CLAIMS_PER_GROUP]:
        if not isinstance(raw_fact_claim, dict):
            continue

        add_fact_claim(
            fact_claims,
            seen_claims,
            raw_fact_claim.get("claim", ""),
            entities=raw_fact_claim.get("entities", []),
            relation=str(raw_fact_claim.get("relation") or ""),
            constraints=raw_fact_claim.get("constraints", []),
            sentence_numbers=sentence_numbers,
        )

    return fact_claims


def validate_llm_output( original_text: str,  sentence_items: list[str],
    raw_payload: dict, target_sentences: list[str] | None = None) -> AtomizerOutput:
    
    sentence_lookup = {normalize_text(sentence): sentence for sentence in sentence_items}
    target_sentence_keys = {normalize_text(sentence) for sentence in target_sentences or sentence_items}
    ignored_sentences: list[str] = []
    claim_groups: list[AtomizedClaimGroup] = []
    assigned_sentences: set[str] = set()
    seen_claims: set[str] = set()

    raw_ignored = raw_payload.get("ignored_sentences", [])
    if isinstance(raw_ignored, list):
        for sentence_text in raw_ignored:
            add_ignored_sentence(
                sentence_text,
                sentence_lookup,
                target_sentence_keys,
                ignored_sentences,
                assigned_sentences,
            )

    raw_groups = raw_payload.get("claim_groups", [])
    if not isinstance(raw_groups, list):
        return make_atomizer_error_output(original_text, "Atomizer returned an invalid claim group structure.")

    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            continue

        original_sentence = raw_group.get("original_sentence")
        text_feature_text = raw_group.get("text_feature_text")
        raw_fact_claims = raw_group.get("fact_check_claims")

        if not isinstance(original_sentence, str):
            continue
        if text_feature_text != original_sentence or not isinstance(raw_fact_claims, list):
            continue

        matched_sentence = get_matched_sentence(original_sentence, sentence_lookup, target_sentence_keys)
        if not matched_sentence or matched_sentence in assigned_sentences:
            continue

        sentence_context = get_sentence_context(sentence_items, matched_sentence)
        valid_fact_claims = get_fact_claims(raw_fact_claims, sentence_context, seen_claims)

        if not valid_fact_claims:
            ignored_sentences.append(matched_sentence)
            assigned_sentences.add(matched_sentence)
            continue

        claim_groups.append(make_claim_group(len(claim_groups) + 1, matched_sentence, valid_fact_claims))
        assigned_sentences.add(matched_sentence)

    for sentence_text in sentence_items:
        add_ignored_sentence(
            sentence_text,
            sentence_lookup,
            target_sentence_keys,
            ignored_sentences,
            assigned_sentences,
        )

    return make_atomizer_output(original_text, ignored_sentences, claim_groups)


def parse_selected_claim_group_ids(raw_payload: dict, claim_groups: list[AtomizedClaimGroup], top_k: int) -> list[int]:
    valid_ids = {group.claim_group_id for group in claim_groups}
    selected_ids = []
    raw_ids = raw_payload.get("selected_claim_group_ids", [])

    if not isinstance(raw_ids, list):
        return selected_ids

    for raw_id in raw_ids:
        try:
            claim_group_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if claim_group_id not in valid_ids:
            continue
        if claim_group_id in selected_ids:
            continue
        selected_ids.append(claim_group_id)
        if len(selected_ids) >= top_k:
            break

    return selected_ids


def select_top_claim_groups(
    original_text: str,
    claim_groups: list[AtomizedClaimGroup],
    top_k: int,
) -> tuple[list[AtomizedClaimGroup], str]:
    if len(claim_groups) <= top_k:
        return renumber_claim_groups(claim_groups), ""

    if not is_gemini_available():
        raise RuntimeError("claim ranking is unavailable because Gemini is not configured.")

    prompt = build_claim_group_ranking_prompt(original_text, claim_groups, top_k)
    response_text = generate_atomizer_json(prompt)
    if not response_text:
        raise RuntimeError("claim ranking returned no response.")

    raw_payload = json.loads(trim_json_fence(response_text))
    if not isinstance(raw_payload, dict):
        raise RuntimeError("claim ranking returned an invalid response.")

    selected_ids = parse_selected_claim_group_ids(raw_payload, claim_groups, top_k)
    if not selected_ids:
        raise RuntimeError("claim ranking did not select any valid claim groups.")

    selection_reason = str(raw_payload.get("selection_reason") or "").strip()

    selected_id_set = set(selected_ids)
    selected_groups = [
        group
        for group in claim_groups
        if group.claim_group_id in selected_id_set
    ]
    selected_groups.sort(key=lambda group: selected_ids.index(group.claim_group_id))
    return renumber_claim_groups(selected_groups), selection_reason


def finalize_atomizer_output(
    original_text: str,
    ignored_sentences: list[str],
    claim_groups: list[AtomizedClaimGroup],
    max_claim_groups: int = MAX_CLAIM_GROUPS_FOR_OUTPUT,
) -> AtomizerOutput:
    candidate_claim_group_count = len(claim_groups)
    candidate_fact_claim_count = sum(len(group.fact_check_claims) for group in claim_groups)

    try:
        selected_claim_groups, selection_reason = select_top_claim_groups(
            original_text,
            claim_groups,
            max_claim_groups,
        )
    except Exception as error:
        return make_atomizer_error_output(original_text, f"Atomizer claim selection failed: {error}")

    output = make_atomizer_output(original_text, ignored_sentences, selected_claim_groups)
    output.candidate_claim_group_count = candidate_claim_group_count
    output.candidate_fact_claim_count = candidate_fact_claim_count
    output.selected_claim_group_count = len(selected_claim_groups)
    output.selected_fact_claim_count = output.summary.fact_check_claim_count
    output.max_claim_group_count = max_claim_groups
    output.claim_selection_reason = selection_reason
    return output
