import json

from api_contract import AtomizedClaimGroup, AtomizerOutput
from atomizer.atomizer_gemini import generate_atomizer_json, is_gemini_available
from atomizer.atomizer_utils import (
    LONG_TEXT_BATCH_SIZE,
    LONG_TEXT_CONTEXT_SENTENCES,
    MAX_CLAIM_GROUPS_FOR_OUTPUT,
    MAX_SENTENCES_FOR_ATOMIZER,
    add_fact_claim,
    finalize_atomizer_output,
    make_atomizer_error_output,
    make_claim_group,
    make_invalid_input_output,
    normalize_input_text,
    split_into_sentences,
    trim_json_fence,
    validate_llm_output,
)
from atomizer.prompts import build_atomizer_prompt


def add_claim_groups_from_batch(
    batch_output: AtomizerOutput,
    ignored_sentences: list[str],
    claim_groups: list[AtomizedClaimGroup],
    seen_claims: set[str],
) -> None:
    ignored_sentences.extend(batch_output.ignored_sentences)

    for raw_group in batch_output.claim_groups:
        fact_claims = []
        for raw_fact_claim in raw_group.fact_check_claims:
            add_fact_claim(
                fact_claims,
                seen_claims,
                raw_fact_claim.claim,
                entities=raw_fact_claim.entities,
                relation=raw_fact_claim.relation,
                constraints=raw_fact_claim.constraints,
            )

        if fact_claims:
            claim_groups.append(
                make_claim_group(
                    len(claim_groups) + 1,
                    raw_group.original_sentence,
                    fact_claims,
                )
            )


def atomize_sentence_batch(
    batch_text: str,
    sentence_items: list[str],
    context_sentences: list[str] | None = None,
    target_sentences: list[str] | None = None,
) -> AtomizerOutput:
    if not is_gemini_available():
        return make_atomizer_error_output(batch_text, "Atomizer is unavailable because Gemini is not configured.")

    prompt = build_atomizer_prompt(
        batch_text,
        sentence_items,
        context_sentences=context_sentences,
        target_sentences=target_sentences,
    )
    try:
        response_text = generate_atomizer_json(prompt)
        if not response_text:
            return make_atomizer_error_output(batch_text, "Atomizer did not return a response.")

        llm_payload = json.loads(trim_json_fence(response_text))
        if not isinstance(llm_payload, dict):
            return make_atomizer_error_output(batch_text, "Atomizer returned an invalid response.")

        return validate_llm_output(
            batch_text,
            sentence_items,
            llm_payload,
            target_sentences=target_sentences,
        )
    except Exception as error:
        return make_atomizer_error_output(batch_text, f"Atomizer failed: {error}")


def atomize_long_text(
    raw_text: str,
    batch_size: int = LONG_TEXT_BATCH_SIZE,
    context_sentence_count: int = LONG_TEXT_CONTEXT_SENTENCES,
    max_claim_groups: int = MAX_CLAIM_GROUPS_FOR_OUTPUT,
) -> AtomizerOutput:
    normalized_text = normalize_input_text(raw_text)
    if not normalized_text:
        return make_invalid_input_output(raw_text)

    sentence_items = split_into_sentences(normalized_text, limit=None)
    if not sentence_items:
        return make_invalid_input_output(normalized_text)

    ignored_sentences = []
    claim_groups: list[AtomizedClaimGroup] = []
    seen_claims = set()

    for start_index in range(0, len(sentence_items), batch_size):
        batch_sentences = sentence_items[start_index:start_index + batch_size]
        context_start_index = max(0, start_index - context_sentence_count)
        context_sentences = sentence_items[context_start_index:start_index]
        prompt_sentences = context_sentences + batch_sentences
        batch_text = " ".join(prompt_sentences)
        batch_output = atomize_sentence_batch(
            batch_text,
            prompt_sentences,
            context_sentences=context_sentences,
            target_sentences=batch_sentences,
        )
        if batch_output.status == "atomizer_error":
            return make_atomizer_error_output(normalized_text, batch_output.message or "Atomizer failed.")

        add_claim_groups_from_batch(batch_output, ignored_sentences, claim_groups, seen_claims)

    return finalize_atomizer_output(normalized_text, ignored_sentences, claim_groups, max_claim_groups)


def atomize_for_pipeline(raw_text: str) -> AtomizerOutput:
    normalized_text = normalize_input_text(raw_text)
    if not normalized_text:
        return make_invalid_input_output(raw_text)

    sentence_items = split_into_sentences(normalized_text, limit=None)
    if not sentence_items:
        return make_invalid_input_output(normalized_text)

    if len(sentence_items) > MAX_SENTENCES_FOR_ATOMIZER:
        return atomize_long_text(normalized_text)

    atomizer_output = atomize_sentence_batch(normalized_text, sentence_items)
    if atomizer_output.status == "atomizer_error":
        return atomizer_output

    return finalize_atomizer_output(
        normalized_text,
        atomizer_output.ignored_sentences,
        atomizer_output.claim_groups,
    )
