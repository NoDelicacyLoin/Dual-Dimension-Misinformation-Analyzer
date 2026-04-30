import json
import os
import time
from dataclasses import dataclass
import re

from google import genai
from google.genai import types

api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    client = genai.Client(api_key=api_key)
else:
    print("[gemini_agent.py] Warning: GEMINI_API_KEY environment variable not found.")
    client = None

MODEL_ID = "gemini-2.5-flash-lite"
MAX_GEMINI_RETRIES = 3
RETRYABLE_GEMINI_MARKERS = [
    "503",
    "UNAVAILABLE",
    "high demand",
    "rate limit",
    "RESOURCE_EXHAUSTED"
]


@dataclass
class ClaimPreparationResult:
    is_valid_claim: bool
    final_claim: str


def is_retryable_gemini_error(error: Exception) -> bool:
    """
    Detect temporary Gemini failures that are worth retrying.
    """
    error_text = str(error)
    return any(marker in error_text for marker in RETRYABLE_GEMINI_MARKERS)


def generate_content_with_retry(contents: str, response_json: bool = False):
    """
    Retry a few times when Gemini returns a temporary availability error.
    """
    if not client:
        return None

    request_config = None
    if response_json:
        request_config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )

    for attempt_index in range(MAX_GEMINI_RETRIES):
        try:
            return client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=request_config
            )
        except Exception as error:
            should_retry = is_retryable_gemini_error(error)
            is_last_attempt = attempt_index == MAX_GEMINI_RETRIES - 1

            if not should_retry or is_last_attempt:
                raise

            retry_delay_seconds = 2 ** attempt_index
            print(
                f"[Gemini Agent] Temporary API issue on attempt {attempt_index + 1}. "
                f"Retrying in {retry_delay_seconds}s..."
            )
            time.sleep(retry_delay_seconds)


def prepare_claim_for_fact_checking(raw_claim: str, use_query_rewrite: bool = True) -> ClaimPreparationResult:
    """
    Check whether the input is a fact-checkable claim and optionally rewrite it
    into a search-friendly form.
    """
    if not client:
        return ClaimPreparationResult(True, raw_claim)

    rewrite_instruction = (
        "If the claim is already clear and searchable, return it unchanged."
        if use_query_rewrite else
        "Do not rewrite the claim. If it is valid, return it unchanged."
    )

    prompt = f"""
You are preparing a user claim for a fact-checking retrieval system.

Return exactly one of these outputs:
- INVALID_CLAIM
- the final claim text to use for fact-checking

Use INVALID_CLAIM when the input is:
- empty
- only an opinion, joke, greeting, or vague reaction
- a request rather than a claim
- too incomplete to fact-check as a factual statement

Use the final claim text when the input is a factual statement that can in principle be checked.

Rules for the final claim text:
- Preserve the original meaning exactly.
- Do not negate, correct, verify, or fact-check the claim.
- Do not introduce new facts, explanations, assumptions, or background information.
- Do not make the claim more specific or more general than the original.
- Do not remove meaning-bearing words such as negation, tense, aspect, comparison, quantity, or time-related words.
- Keep the main entities, relation, and key claim wording.
- Prefer natural English phrasing over keyword fragments.
- Prefer keeping a factual claim as a declarative sentence.
- Do not rewrite a statement as a question unless the original input is already phrased as a question.
- Do not add descriptive fillers such as appositions, category labels, or explanatory phrases.
- Do not shorten a claim just to make it look more like a search query.
- {rewrite_instruction}

Good examples:
Input: Albert Einstein failed math in school
Output: Albert Einstein failed math in school

Input: China has the largest population in the world
Output: China has the largest population in the world

Input: so like people say coffee actually dehydrates you
Output: Coffee dehydrates you

Input: i heard drinking lemon water detoxifies the liver
Output: Drinking lemon water detoxifies the liver

Bad rewrite examples:
Input: China has the largest population in the world
Bad Output: China largest population world

Input: COVID vaccines cause infertility
Bad Output: Do COVID vaccines cause infertility

Input: China has the largest population in the world
Bad Output: China is a country with the largest population in the world

User input: "{raw_claim}"

Output:
""".strip()

    try:
        response = generate_content_with_retry(prompt, response_json=False)
        final_claim = (response.text or "").strip()

        if not final_claim:
            return ClaimPreparationResult(True, raw_claim)

        if final_claim == "INVALID_CLAIM":
            return ClaimPreparationResult(False, "")

        raw_claim_lower = raw_claim.lower()
        final_claim_lower = final_claim.lower()

        raw_has_not = " not " in f" {raw_claim_lower} "
        final_has_not = " not " in f" {final_claim_lower} "

        if raw_has_not != final_has_not:
            print("[Gemini Agent] Rewrite changed negation pattern. Using original claim instead.")
            return ClaimPreparationResult(True, raw_claim)

        raw_numbers = set(re.findall(r"\d+(?:\.\d+)?", raw_claim_lower))
        final_numbers = set(re.findall(r"\d+(?:\.\d+)?", final_claim_lower))
        if raw_numbers != final_numbers:
            print("[Gemini Agent] Rewrite changed number pattern. Using original claim instead.")
            return ClaimPreparationResult(True, raw_claim)

        return ClaimPreparationResult(True, final_claim)

    except Exception as error:
        print(f"[Gemini Agent] Claim preparation failed: {error}")
        return ClaimPreparationResult(True, raw_claim)


def generate_comprehensive_verdict(claim: str, selected_evidence: list[dict]) -> dict:
    """
    Use the filtered evidence to produce source-level judgments in JSON.
    The backend will aggregate these judgments into the final truth score.
    """
    if not client:
        return {
            "explanation": "Gemini API key is missing.",
            "source_judgments": []
        }

    evidence_lines = []
    for evidence_index, evidence_item in enumerate(selected_evidence, start=1):
        evidence_text = evidence_item.get("content", "").strip()
        evidence_lines.append(f"Evidence {evidence_index}: {evidence_text}")

    evidence_block = "\n".join(evidence_lines) if evidence_lines else "No relevant evidence was found."

    prompt = f"""
You are a careful fact-checking assistant.

Your task is to interpret each evidence item separately,
using only the evidence provided below.

Do not use outside knowledge.
Do not assume missing facts.
Do not strengthen weak evidence.
Do not output a final verdict label such as True or False.
Do not decide the final truth score for the system.
The backend will aggregate your source-level judgments later.

Evidence handling rules:
- Judge the quality of each evidence item separately.
- Ignore evidence that is mostly page chrome, navigation text, headlines without substance, or vague commentary.
- Do not treat indirect background context as decisive proof.
- If an evidence item does not mention the key entity, event, number, place, or policy in the claim, treat it as weak or irrelevant.
- Use one stance for each source:
  - supports
  - contradicts
  - mixed
  - background
- The stance must always be judged relative to the original claim above.
- Use supports only when the source makes the original claim more likely to be true.
- Use contradicts only when the source makes the original claim more likely to be false.
- Use mixed when the source contains both helpful and harmful signals.
- Use background when the source is related context but does not directly verify the claim.
- strength means how strong the source signal is.
- specificity means how directly the source addresses the exact claim.
- Keep strength and specificity between 0.0 and 1.0.
- Use low values for weak, vague, indirect, or partial evidence.
- If the claim says someone did nothing, and the evidence shows they did something, that stance is contradicts.
- If the claim says something is false, and the evidence says it is true, that stance is contradicts.
- If the claim says something is true, and the evidence says it is true, that stance is supports.
- Do not label a source as supports just because it strongly states a fact. The label depends on whether that fact supports or contradicts the original claim.

Claim:
\"{claim}\"

Evidence:
{evidence_block}

Return valid JSON with this structure:
{{
  "explanation": "Short explanation in 2 to 4 sentences.",
  "source_judgments": [
    {{
      "source_index": 1,
      "stance": "supports",
      "strength": 0.8,
      "specificity": 0.9,
      "analysis": "Short source-level explanation."
    }}
  ]
}}

Important reminder:
- The important output is the source_judgments list.
- Every stance label must be relative to the original claim, not relative to the source alone.
""".strip()

    try:
        response = generate_content_with_retry(prompt, response_json=True)
        response_text = (response.text or "").strip()

        if response_text.startswith("```json"):
            response_text = response_text[7:].strip()
        if response_text.endswith("```"):
            response_text = response_text[:-3].strip()

        verdict_report = json.loads(response_text)

        if "explanation" not in verdict_report:
            verdict_report["explanation"] = "The model did not return a full explanation."
        if "source_judgments" not in verdict_report or not isinstance(verdict_report["source_judgments"], list):
            verdict_report["source_judgments"] = []

        return verdict_report

    except Exception as error:
        print(f"[Gemini Agent] Verdict generation failed: {error}")
        return {
            "explanation": "The AI verdict step failed.",
            "source_judgments": []
        }
