import json
import re
from difflib import SequenceMatcher

from atomizer.atomizer_gemini import generate_atomizer_json, is_gemini_available


MAX_SENTENCES_FOR_ATOMIZER = 12
MAX_FACT_CLAIMS_PER_GROUP = 5
LONG_TEXT_BATCH_SIZE = 8
LONG_TEXT_CONTEXT_SENTENCES = 2


def normalize_input_text(raw_text: str) -> str:
    compact_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    compact_text = re.sub(r"[ \t]+", " ", compact_text)
    compact_text = re.sub(r"(?<=[a-z0-9][.!?])(?=[A-Z])", " ", compact_text)
    compact_text = re.sub(r"(?<=\d)(?=[A-Z])", " ", compact_text)
    compact_text = re.sub(r"\n{3,}", "\n\n", compact_text)
    return compact_text.strip()


def split_into_sentences(text: str, limit: int | None = MAX_SENTENCES_FOR_ATOMIZER) -> list[str]:
    protected_text = re.sub(r"\b([A-Z])\.(?=\s+[A-Z])", r"\1<dot>", text)
    sentence_items = [
        sentence.replace("<dot>", ".").strip()
        for sentence in re.split(r"(?<=[.!?])\s+", protected_text)
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


def extract_word_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower()))


def has_unresolved_reference(text: str) -> bool:
    lower_text = text.lower()
    starts_with_reference = re.search(r"^(he|she|they|it|this|that)\b", lower_text)
    object_reference = re.search(
        r"\b(said|claimed|reported|stated|thought|believed|posted|wrote)\s+(it|this|that)\b",
        lower_text,
    )
    return starts_with_reference is not None or object_reference is not None


def make_fact_claim(fact_index: int, claim_text: str) -> dict:
    return {
        "fact_claim_id": fact_index,
        "claim": re.sub(r"\s+", " ", claim_text).strip(),
    }


def make_claim_group(claim_group_id: int, original_sentence: str, fact_claims: list[dict]) -> dict:
    return {
        "claim_group_id": claim_group_id,
        "original_sentence": original_sentence,
        "text_feature_text": original_sentence,
        "atomization_applied": len(fact_claims) > 1,
        "fact_check_claims": fact_claims,
    }


def build_summary(ignored_sentences: list[str], claim_groups: list[dict]) -> dict[str, int]:
    return {
        "ignored_sentence_count": len(ignored_sentences),
        "text_feature_unit_count": len(claim_groups),
        "fact_check_claim_count": sum(len(group["fact_check_claims"]) for group in claim_groups),
    }


def make_atomizer_output(original_text: str, ignored_sentences: list[str], claim_groups: list[dict]) -> dict:
    return {
        "status": "success" if claim_groups else "invalid_input",
        "original_text": original_text,
        "ignored_sentences": ignored_sentences,
        "claim_groups": claim_groups,
        "summary": build_summary(ignored_sentences, claim_groups),
    }


def build_fallback_output(original_text: str, sentence_items: list[str]) -> dict:
    claim_groups: list[dict] = []
    for sentence_text in sentence_items:
        claim_group_id = len(claim_groups) + 1
        fact_claim = make_fact_claim(1, sentence_text)
        claim_groups.append(make_claim_group(claim_group_id, sentence_text, [fact_claim]))

    return make_atomizer_output(original_text, [], claim_groups)


def build_atomizer_prompt(
    original_text: str,
    sentence_items: list[str],
    context_note: str = "",
    context_sentences: list[str] | None = None,
    target_sentences: list[str] | None = None,
    include_context_note: bool = False,
) -> str:
    escaped_original_text = json.dumps(original_text)
    sentence_block = "\n".join(f"{index}. {sentence_text}" for index, sentence_text in enumerate(sentence_items, start=1))
    context_block = "\n".join(
        f"{index}. {sentence_text}"
        for index, sentence_text in enumerate(context_sentences or [], start=1)
    )
    target_block = "\n".join(
        f"{index}. {sentence_text}"
        for index, sentence_text in enumerate(target_sentences or sentence_items, start=1)
    )
    context_note_rule = ""
    context_note_schema = ""
    if include_context_note:
        context_note_rule = """
- Use the previous context note and context sentences only to resolve local references such as he, she, they, it, this, and that.
- Do not output claim_groups for context sentences. Output claim_groups only for target sentences.
- Return next_context_note as one short sentence describing the current subject, object, and topic that may help the next batch resolve references.
- Keep next_context_note factual and brief. Do not include verdicts, opinions, or new information not implied by the target sentences.
""".strip()
        context_note_schema = ',\n  "next_context_note": "Brief context note for the next batch."'

    return f"""
You are an atomizer for a dual-branch misinformation analysis system.

Your job:
1. Review the input sentence by sentence.
2. Ignore non-factual, subjective, vague, greeting-like, reaction-only, or non-checkable sentences.
3. For each factual sentence, create exactly one claim_group.
4. Keep text_feature_text exactly equal to the original sentence.
5. Extract one or more standalone fact_check_claims only when they are genuinely factual and checkable.

Hard rules:
- Return valid JSON only.
- Do not rewrite, summarize, clean, or rephrase the original sentences.
- original_sentence must be copied exactly from the input.
- text_feature_text must be copied exactly from the original sentence.
- Ignore personal opinion, vague rumor, greeting-like, reaction-only, or non-checkable sentences.
- Do not ignore checkable reporting, attribution, quotation, or belief sentences.
- Preserve reporting, attribution, and belief meaning. For example, "People once thought X" must stay about what people once thought, not become only "X".
- A fact_check_claim must be standalone enough for web search and evidence checking.
- If a sentence uses he, she, they, it, this, or that, resolve the reference in the fact_check_claim only when the nearby previous context clearly identifies the subject or object.
- If the subject or object reference is unclear, ignore that sentence instead of outputting an incomplete claim.
- You may perform minimal structural decomposition only when necessary to split a conjunction into standalone claims.
- Minimal structural decomposition means preserving the same facts while only adjusting function words or inflection required for grammar.
- Do not add new facts, entities, times, numbers, causes, locations, qualifiers, or explanations.
- Do not remove negation, comparison, quantity, or time meaning.
- Avoid duplicate or near-duplicate fact_check_claims.
- If two candidate claims have the same main subject, same relation, and same important conditions, keep only one natural version.
- Do not merge claims when negation, time, quantity, comparison, modality, attribution, or degree changes the meaning.
- If a sentence has one factual claim, keep atomization_applied as false and keep one fact_check_claim.
- If a sentence has multiple factual claims, keep atomization_applied as true.
- If there are no factual claim groups at all, set status to "invalid_input".
{context_note_rule}

Examples:
- Input context: Donald Trump praised the tax bill. He said it would help workers.
  Fact claims: Donald Trump praised the tax bill. / Donald Trump said the tax bill would help workers.
- Input: Donald Trump said, "The stock market had its best year."
  Fact claim: Donald Trump said the stock market had its best year.
- Input: He said it would happen soon.
  No fact claim because the subject and object are unclear.
- Input: The Earth orbits the Sun. Earth orbits around the Sun.
  Keep only one claim because the subject, relation, and condition are the same.
- Input: The project is almost finished. The project is finished.
  Keep both claims because "almost finished" and "finished" do not mean the same thing.

Return exactly this JSON structure:
{{
  "status": "success",
  "original_text": {escaped_original_text},
  "ignored_sentences": [],
  "claim_groups": [
    {{
      "claim_group_id": 1,
      "original_sentence": "Exact original sentence.",
      "text_feature_text": "Exact original sentence.",
      "atomization_applied": true,
      "fact_check_claims": [
        {{
          "fact_claim_id": 1,
          "claim": "Standalone factual claim."
        }}
      ]
    }}
  ],
  "summary": {{
    "ignored_sentence_count": 0,
    "text_feature_unit_count": 1,
    "fact_check_claim_count": 1
  }}{context_note_schema}
}}

Input text:
{original_text}

Input sentences:
{sentence_block}

Previous context note:
{context_note or "None"}

Context sentences:
{context_block or "None"}

Target sentences:
{target_block}
""".strip()


def is_grounded_fact_claim(claim_text: str, source_sentence: str) -> bool:
    cleaned_claim = re.sub(r"\s+", " ", claim_text).strip()
    cleaned_source = re.sub(r"\s+", " ", source_sentence).strip()

    if len(cleaned_claim) < 8:
        return False
    if has_unresolved_reference(cleaned_claim):
        return False

    claim_numbers = extract_numbers(cleaned_claim)
    source_numbers = extract_numbers(cleaned_source)
    if not claim_numbers.issubset(source_numbers):
        return False

    if normalize_text(cleaned_claim) in normalize_text(cleaned_source):
        return True

    claim_tokens = extract_word_tokens(cleaned_claim)
    source_tokens = extract_word_tokens(cleaned_source)
    if not claim_tokens or not source_tokens:
        return False

    token_overlap_ratio = len(claim_tokens & source_tokens) / len(claim_tokens)
    similarity_ratio = SequenceMatcher(None, normalize_text(cleaned_claim), normalize_text(cleaned_source)).ratio()

    return token_overlap_ratio >= 0.65 and similarity_ratio >= 0.35


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


def validate_llm_output(original_text: str, sentence_items: list[str], raw_payload: dict) -> dict:
    fallback_output = build_fallback_output(original_text, sentence_items)
    sentence_lookup = {normalize_text(sentence): sentence for sentence in sentence_items}
    ignored_sentences: list[str] = []
    claim_groups: list[dict] = []
    assigned_sentences: set[str] = set()
    seen_claims: set[str] = set()

    raw_ignored = raw_payload.get("ignored_sentences", [])
    if isinstance(raw_ignored, list):
        for sentence_text in raw_ignored:
            if not isinstance(sentence_text, str):
                continue
            matched_sentence = sentence_lookup.get(normalize_text(sentence_text))
            if not matched_sentence or matched_sentence in assigned_sentences:
                continue
            ignored_sentences.append(matched_sentence)
            assigned_sentences.add(matched_sentence)

    raw_groups = raw_payload.get("claim_groups", [])
    if not isinstance(raw_groups, list):
        return fallback_output

    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            continue

        original_sentence = raw_group.get("original_sentence")
        text_feature_text = raw_group.get("text_feature_text")
        raw_fact_claims = raw_group.get("fact_check_claims")

        if not isinstance(original_sentence, str):
            continue
        if not isinstance(text_feature_text, str):
            continue
        if text_feature_text != original_sentence:
            continue
        if not isinstance(raw_fact_claims, list):
            continue

        matched_sentence = sentence_lookup.get(normalize_text(original_sentence))
        if not matched_sentence:
            continue
        if matched_sentence in assigned_sentences:
            continue

        valid_fact_claims: list[dict] = []
        sentence_context = get_sentence_context(sentence_items, matched_sentence)
        for raw_fact_claim in raw_fact_claims[:MAX_FACT_CLAIMS_PER_GROUP]:
            if not isinstance(raw_fact_claim, dict):
                continue

            claim_text = raw_fact_claim.get("claim", "")
            if not isinstance(claim_text, str):
                continue
            if not is_grounded_fact_claim(claim_text, sentence_context):
                continue
            claim_key = normalize_text(claim_text)
            if claim_key in seen_claims:
                continue

            fact_claim = make_fact_claim(len(valid_fact_claims) + 1, claim_text)
            valid_fact_claims.append(fact_claim)
            seen_claims.add(claim_key)

        if not valid_fact_claims:
            ignored_sentences.append(matched_sentence)
            assigned_sentences.add(matched_sentence)
            continue

        claim_group_id = len(claim_groups) + 1
        final_fact_claims = []
        for index, fact_claim in enumerate(valid_fact_claims, start=1):
            final_fact_claims.append(make_fact_claim(index, fact_claim["claim"]))
        claim_groups.append(make_claim_group(claim_group_id, matched_sentence, final_fact_claims))
        assigned_sentences.add(matched_sentence)

    for sentence_text in sentence_items:
        if sentence_text in assigned_sentences:
            continue
        if sentence_text in ignored_sentences:
            continue
        ignored_sentences.append(sentence_text)

    return make_atomizer_output(original_text, ignored_sentences, claim_groups)


def keep_target_sentence_groups(atomizer_output: dict, target_sentences: list[str]) -> dict:
    target_sentence_keys = {normalize_text(sentence) for sentence in target_sentences}
    ignored_sentences = [
        sentence
        for sentence in atomizer_output.get("ignored_sentences", [])
        if normalize_text(sentence) in target_sentence_keys
    ]
    claim_groups = [
        group
        for group in atomizer_output.get("claim_groups", [])
        if normalize_text(group.get("original_sentence", "")) in target_sentence_keys
    ]
    return make_atomizer_output(atomizer_output.get("original_text", ""), ignored_sentences, claim_groups)


def atomize_sentence_batch(
    batch_text: str,
    sentence_items: list[str],
    context_note: str = "",
    context_sentences: list[str] | None = None,
    target_sentences: list[str] | None = None,
    include_context_note: bool = False,
) -> dict:
    fallback_output = build_fallback_output(batch_text, sentence_items)
    if not is_gemini_available():
        return fallback_output

    prompt = build_atomizer_prompt(
        batch_text,
        sentence_items,
        context_note=context_note,
        context_sentences=context_sentences,
        target_sentences=target_sentences,
        include_context_note=include_context_note,
    )
    try:
        response_text = generate_atomizer_json(prompt)
        if not response_text:
            return fallback_output

        llm_payload = json.loads(trim_json_fence(response_text))
        if not isinstance(llm_payload, dict):
            return fallback_output

        output = validate_llm_output(batch_text, sentence_items, llm_payload)
        if target_sentences is not None:
            output = keep_target_sentence_groups(output, target_sentences)
        output["next_context_note"] = str(llm_payload.get("next_context_note") or "").strip()
        return output
    except Exception as error:
        print(f"[Atomizer] Gemini atomization failed. Falling back to sentence-based mode: {error}")
        return fallback_output


def atomize_text(raw_text: str) -> dict:
    normalized_text = normalize_input_text(raw_text)
    if not normalized_text:
        return {
            "status": "invalid_input",
            "original_text": raw_text,
            "ignored_sentences": [],
            "claim_groups": [],
            "summary": {
                "ignored_sentence_count": 0,
                "text_feature_unit_count": 0,
                "fact_check_claim_count": 0,
            },
        }

    sentence_items = split_into_sentences(normalized_text)
    if not sentence_items:
        return {
            "status": "invalid_input",
            "original_text": normalized_text,
            "ignored_sentences": [],
            "claim_groups": [],
            "summary": {
                "ignored_sentence_count": 0,
                "text_feature_unit_count": 0,
                "fact_check_claim_count": 0,
            },
        }

    return atomize_sentence_batch(normalized_text, sentence_items)


def atomize_long_text(
    raw_text: str,
    batch_size: int = LONG_TEXT_BATCH_SIZE,
    context_sentence_count: int = LONG_TEXT_CONTEXT_SENTENCES,
) -> dict:
    normalized_text = normalize_input_text(raw_text)
    if not normalized_text:
        return {
            "status": "invalid_input",
            "original_text": raw_text,
            "ignored_sentences": [],
            "claim_groups": [],
            "summary": {
                "ignored_sentence_count": 0,
                "text_feature_unit_count": 0,
                "fact_check_claim_count": 0,
            },
        }

    sentence_items = split_into_sentences(normalized_text, limit=None)
    if not sentence_items:
        return {
            "status": "invalid_input",
            "original_text": normalized_text,
            "ignored_sentences": [],
            "claim_groups": [],
            "summary": {
                "ignored_sentence_count": 0,
                "text_feature_unit_count": 0,
                "fact_check_claim_count": 0,
            },
        }

    ignored_sentences = []
    claim_groups = []
    seen_claims = set()
    context_note = ""

    for start_index in range(0, len(sentence_items), batch_size):
        batch_sentences = sentence_items[start_index:start_index + batch_size]
        context_start_index = max(0, start_index - context_sentence_count)
        context_sentences = sentence_items[context_start_index:start_index]
        prompt_sentences = context_sentences + batch_sentences
        batch_text = " ".join(prompt_sentences)
        batch_output = atomize_sentence_batch(
            batch_text,
            prompt_sentences,
            context_note=context_note,
            context_sentences=context_sentences,
            target_sentences=batch_sentences,
            include_context_note=True,
        )
        next_context_note = batch_output.get("next_context_note", "")
        if next_context_note:
            context_note = next_context_note

        ignored_sentences.extend(batch_output.get("ignored_sentences", []))

        for raw_group in batch_output.get("claim_groups", []):
            fact_claims = []
            for raw_fact_claim in raw_group.get("fact_check_claims", []):
                claim_text = raw_fact_claim.get("claim", "")
                claim_key = normalize_text(claim_text)
                if not claim_key or claim_key in seen_claims:
                    continue
                fact_claims.append(make_fact_claim(len(fact_claims) + 1, claim_text))
                seen_claims.add(claim_key)

            if not fact_claims:
                continue

            claim_groups.append(
                make_claim_group(
                    len(claim_groups) + 1,
                    raw_group.get("original_sentence", ""),
                    fact_claims,
                )
            )

    return make_atomizer_output(normalized_text, ignored_sentences, claim_groups)
