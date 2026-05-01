import json
import os
import re
import time
from dataclasses import dataclass

from api_contract import EachEvidence, EachFactualClaim

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

api_key = os.environ.get("GEMINI_API_KEY")
if api_key and genai is not None:
    client = genai.Client(api_key=api_key)
else:
    if genai is None:
        print("[gemini_agent.py] Warning: google-genai package is not installed.")
    elif not api_key:
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

def is_gemini_available() -> bool:
    return client is not None


@dataclass
class PreparedClaim:
    is_valid_claim: bool
    final_claim: str
    search_query: str
    main_entities: list[str]
    relation: str
    constraints: list[str]


def is_retryable_gemini_error(error: Exception) -> bool:
    """
    Detect temporary Gemini failures that are worth retrying.
    """
    error_text = str(error)
    return any(marker in error_text for marker in RETRYABLE_GEMINI_MARKERS)


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text.lower()))


def clean_string_list(raw_items) -> list[str]:
    if not isinstance(raw_items, list):
        return []

    cleaned_items = []
    for raw_item in raw_items:
        item = str(raw_item).strip()
        if item and item not in cleaned_items:
            cleaned_items.append(item)
    return cleaned_items[:5]


def is_too_empty_for_fact_checking(text: str) -> bool:
    stripped_text = text.strip()
    if not stripped_text:
        return True

    tokens = re.findall(r"[a-zA-Z0-9]+", stripped_text)
    return len(tokens) < 2


def normalize_gemini_stance(raw_stance: str) -> str:
    stance = (raw_stance or "").strip().lower()
    if stance in {"supports", "support", "supported"}:
        return "supports"
    if stance in {"contradicts", "contradict", "refutes", "refute"}:
        return "contradicts"
    if stance in {"mixed", "conflicted"}:
        return "mixed"
    return "background"


def build_empty_verdict_report(explanation: str) -> dict:
    return {
        "explanation": explanation,
        "overall_truth_score": None,
        "source_judgments": [],
    }


def parse_gemini_json(response_text: str) -> dict:
    cleaned_text = (response_text or "").strip()

    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:].strip()
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3].strip()

    try:
        parsed_json = json.loads(cleaned_text)
    except json.JSONDecodeError:
        safe_text = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", cleaned_text)
        parsed_json = json.loads(safe_text)

    return parsed_json if isinstance(parsed_json, dict) else {}


def apply_source_judgments_to_claim(
    factual_claim: EachFactualClaim,
    source_judgments: list[dict],
) -> int:
    applied_count = 0

    for evidence_index, evidence_item in enumerate(factual_claim.evidence, start=1):
        source_judgment = None
        for judgment in source_judgments:
            if judgment.get("source_index") == evidence_index:
                source_judgment = judgment
                break

        if not source_judgment:
            evidence_item.ai_analysis = "No specific analysis was generated for this source."
            evidence_item.stance = "background"
            continue

        evidence_item.ai_analysis = (
            str(source_judgment.get("analysis", "")).strip()
            or "No specific analysis was generated for this source."
        )
        evidence_item.stance = normalize_gemini_stance(str(source_judgment.get("stance", "background")))
        applied_count += 1

    return applied_count


def apply_gemini_verdict_to_factual_claim(claim: str, factual_claim: EachFactualClaim) -> int:
    verdict_report = generate_verdict_report(claim, factual_claim.evidence)
    factual_claim.explanation = verdict_report.get("explanation", "No explanation was generated.")
    factual_claim.metadata.gemini_truth_score = verdict_report.get("overall_truth_score")
    source_judgments = verdict_report.get("source_judgments", []) if isinstance(verdict_report, dict) else []
    return apply_source_judgments_to_claim(factual_claim, source_judgments)


def generate_content_with_retry(contents: str, response_json: bool = False):
    """
    Retry a few times when Gemini returns a temporary availability error.
    """
    if not client:
        return None

    request_config = None
    if response_json and types is not None:
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


def build_claim_preparation_prompt(raw_claim: str) -> str:
    return f"""
You are preparing a user claim for a fact-checking retrieval system.

Return valid JSON only.

Do not decide whether the claim is true, false, plausible, absurd, or likely.
Even if the claim looks obviously false or nonsensical, keep it as a fact-checkable claim
when it states a concrete relationship that evidence could support or refute.

Output fields:
- claim_for_verdict: the claim text the final evidence judgement should use.
- search_query: the query text the retrieval step should search with.
- main_entities: important people, titles, organisations, places, or objects explicitly named in the claim.
- relation: the short relation or predicate being checked, copied from the claim when possible.
- constraints: important limits or conditions explicitly stated in the claim.

Rules for claim_for_verdict:
- Preserve the original meaning exactly.
- Do not negate, correct, verify, or fact-check the claim.
- Do not introduce new facts, explanations, assumptions, or background information.
- Do not make the claim more specific or more general than the original.
- Do not remove meaning-bearing words such as negation, tense, aspect, comparison, quantity, or time-related words.
- Keep the main entities, the relation between them, and key claim wording.
- Keep factual conditions such as time, place, quantity, comparison, negation, exclusivity, and scope.
- Prefer natural English phrasing over keyword fragments.
- Prefer keeping a factual claim as a declarative sentence.
- Do not rewrite a statement as a question unless the original input is already phrased as a question.
- Do not add descriptive fillers such as appositions, category labels, or explanatory phrases.
- Do not shorten a claim just to make it look more like a search query.

Rules for search_query:
- Keep the same core claim, entities, relation, and factual conditions.
- Prefer concise search terms over a full sentence.
- Preserve quoted titles, numbers, negation, and comparison words.
- Preserve relation words and factual constraints when they carry meaning.
- You may add a neutral entity-type hint such as film, album, book, person, organization, event, cast, author, or location when the wording clearly implies it.
- Do not add an answer, correction, or verdict.
- If unsure, use claim_for_verdict as the search_query.
- If the claim is already clear and searchable, keep the verdict claim unchanged. You may still make the search query more search-friendly.

Rules for main_entities, relation, and constraints:
- Only use information present in the claim.
- Do not add background knowledge.
- Do not infer the answer.
- main_entities should preserve exact names or titles when possible.
- relation should be short and natural, such as the verb phrase or relation phrase from the claim.
- constraints should include explicit time, place, number, comparison, negation, exclusivity, or scope words when they matter.

Good examples:
Input: Albert Einstein failed math in school
Output: {{"claim_for_verdict": "Albert Einstein failed math in school", "search_query": "Albert Einstein failed math in school", "main_entities": ["Albert Einstein"], "relation": "failed math", "constraints": ["in school"]}}

Input: China has the largest population in the world
Output: {{"claim_for_verdict": "China has the largest population in the world", "search_query": "China largest population in the world", "main_entities": ["China"], "relation": "has largest population", "constraints": ["in the world"]}}

Input: so like people say coffee actually dehydrates you
Output: {{"claim_for_verdict": "Coffee dehydrates you", "search_query": "coffee dehydrates you", "main_entities": ["Coffee"], "relation": "dehydrates", "constraints": []}}

Input: i heard drinking lemon water detoxifies the liver
Output: {{"claim_for_verdict": "Drinking lemon water detoxifies the liver", "search_query": "lemon water detoxifies liver", "main_entities": ["lemon water", "liver"], "relation": "detoxifies", "constraints": []}}

Bad rewrite examples:
Input: China has the largest population in the world
Bad claim_for_verdict: China largest population world

Input: COVID vaccines cause infertility
Bad claim_for_verdict: Do COVID vaccines cause infertility

Input: China has the largest population in the world
Bad claim_for_verdict: China is a country with the largest population in the world

User input: "{raw_claim}"

Output:
""".strip()


def prepare_claim_for_fact_checking(raw_claim: str, use_query_rewrite: bool = True) -> PreparedClaim:
    """
    Lightly guard against empty input and optionally prepare a search query.
    The verdict claim keeps the user's meaning; the search query may be more
    retrieval-friendly.
    """
    if is_too_empty_for_fact_checking(raw_claim):
        return PreparedClaim(False, "", "", [], "", [])

    if not use_query_rewrite:
        return PreparedClaim(True, raw_claim, raw_claim, [], "", [])

    if not client:
        return PreparedClaim(True, raw_claim, raw_claim, [], "", [])

    prompt = build_claim_preparation_prompt(raw_claim)

    try:
        response = generate_content_with_retry(prompt, response_json=True)
        try:
            preparation = parse_gemini_json(response.text or "")
        except Exception:
            preparation = {}

        if isinstance(preparation, dict):
            final_claim = str(preparation.get("claim_for_verdict") or raw_claim).strip()
            search_query = str(preparation.get("search_query") or final_claim).strip()
            main_entities = clean_string_list(preparation.get("main_entities", []))
            relation = str(preparation.get("relation") or "").strip()
            constraints = clean_string_list(preparation.get("constraints", []))
        else:
            final_claim = raw_claim
            search_query = raw_claim
            main_entities = []
            relation = ""
            constraints = []

        if not final_claim:
            return PreparedClaim(True, raw_claim, raw_claim, [], "", [])

        raw_claim_lower = raw_claim.lower()
        final_claim_lower = final_claim.lower()

        raw_claim_has_not = " not " in f" {raw_claim_lower} "
        final_claim_has_not = " not " in f" {final_claim_lower} "

        if raw_claim_has_not != final_claim_has_not:
            print("[Gemini Agent] Rewrite changed negation pattern. Using original claim instead.")
            return PreparedClaim(True, raw_claim, raw_claim, main_entities, relation, constraints)

        raw_numbers = extract_numbers(raw_claim)
        final_numbers = extract_numbers(final_claim)
        if raw_numbers != final_numbers:
            print("[Gemini Agent] Rewrite changed number pattern. Using original claim instead.")
            return PreparedClaim(True, raw_claim, raw_claim, main_entities, relation, constraints)

        if not search_query:
            search_query = final_claim

        search_query_lower = search_query.lower()
        search_query_has_not = " not " in f" {search_query_lower} "
        search_numbers = extract_numbers(search_query)
        if (
            raw_claim_has_not != search_query_has_not
            or raw_numbers != search_numbers
        ):
            search_query = final_claim

        return PreparedClaim(True, final_claim, search_query, main_entities, relation, constraints)

    except Exception as error:
        print(f"[Gemini Agent] Claim preparation failed: {error}")
        return PreparedClaim(True, raw_claim, raw_claim, [], "", [])


def build_verdict_prompt(claim: str, evidence_block: str) -> str:
    return f"""
You are a careful fact-checking assistant.

Your task is to interpret each evidence item separately,
using only the evidence provided below.

Do not use outside knowledge.
Do not assume missing facts.
Do not strengthen weak evidence.
Do not output a final verdict label such as True or False.
Do not decide the final verdict for the system.
The backend will aggregate your source-level judgments later.

Evidence handling rules:
- Judge the quality of each evidence item separately.
- Ignore evidence that is mostly page chrome, navigation text, headlines without substance, or vague commentary.
- Do not treat indirect background context as decisive proof.
- First identify the claim's main entities, relation, and factual conditions.
- Factual conditions include time, place, number, comparison, negation, exclusivity, and scope.
- Judge whether the evidence addresses the same relation under the same important conditions.
- Supports requires the same relation, not just the same entities or time period.
- If the evidence states the opposite or an incompatible relation, use contradicts.
- If an evidence item only mentions one entity from the claim but not the relation being checked, treat it as background.
- If an evidence item discusses the right topic but misses an important condition, treat it as weak, mixed, or background depending on whether it still bears on the claim.
- If an evidence item does not mention the key entity, event, number, place, relation, or policy in the claim, treat it as weak or irrelevant.
- Use one stance for each source:
  - supports
  - contradicts
  - mixed
  - background
- The stance must always be judged relative to the original claim above.
- Use supports only when the source makes the original claim more likely to be true.
- Use contradicts only when the source makes the original claim more likely to be false.
- Use mixed when the source contains both helpful and harmful signals.
- Use mixed only when the source genuinely contains meaningful signal in both directions.
- If the source mostly leans one way, use supports or contradicts with lower strength instead of mixed.
- Use background only when the source is merely topical context and does not meaningfully bear on whether the claim is true or false.
- If a source partially supports or partially contradicts the claim, prefer supports, contradicts, or mixed with low strength instead of background.
- Do not use background for a source that directly discusses the same statement, quote, statistic, policy, event, or factual comparison as the claim, even if the signal is weak.
- If a source directly evaluates whether a statement is accurate, misleading, false, true, exaggerated, or unsupported, background is almost never appropriate.
- If a source directly mentions the same core claim subject and clearly talks about the same statement, quote, statistic, policy, or event, prefer supports, contradicts, or mixed with low strength instead of background.
- For claims about two entities, evidence should normally mention both entities or clearly discuss the relation between them.
- For scope, negation, and comparison claims, make sure the evidence addresses the same constraint before marking it as supports or contradicts.
- For membership, ownership, location, inclusion, exclusion, and part-of claims, check whether the evidence affirms the same relation or the opposite relation.
- Use background only for evidence that is mostly side context, generic news, unrelated biography, directory text, page shell, or very indirect topical mention.
- Do not label a source as supports just because it strongly states a fact. The label depends on whether that fact supports or contradicts the original claim.

Claim:
\"{claim}\"

Evidence:
{evidence_block}

Return valid JSON with this structure:
{{
  "explanation": "Short explanation in 2 to 4 sentences.",
  "overall_truth_score": 0.72,
  "source_judgments": [
    {{
      "source_index": 1,
      "stance": "supports",
      "analysis": "Short source-level explanation."
    }}
  ]
}}

Important reminder:
- The important output is the source_judgments list.
- Every stance label must be relative to the original claim, not relative to the source alone.
- Return one source_judgment for every evidence item.
- overall_truth_score must be a number from 0.0 to 1.0 based only on the selected evidence.
- overall_truth_score is only a light calibration signal. It is not the final system verdict.
""".strip()


def generate_verdict_report(claim: str, selected_evidence: list[EachEvidence]) -> dict:
    """
    Use the filtered evidence to produce source-level judgments in JSON.
    The backend will aggregate these judgments into the final truth score.
    """
    if not client:
        return build_empty_verdict_report("Gemini API key is missing.")

    evidence_lines = []
    for evidence_index, evidence_item in enumerate(selected_evidence, start=1):
        evidence_text = evidence_item.content.strip()
        evidence_lines.append(f"Evidence {evidence_index}: {evidence_text}")

    evidence_block = "\n".join(evidence_lines) if evidence_lines else "No relevant evidence was found."

    prompt = build_verdict_prompt(claim, evidence_block)

    try:
        response = generate_content_with_retry(prompt, response_json=True)
        verdict_report = parse_gemini_json(response.text or "")

        if "explanation" not in verdict_report:
            verdict_report["explanation"] = "The model did not return a full explanation."
        try:
            gemini_truth_score = float(verdict_report.get("overall_truth_score"))
        except Exception:
            gemini_truth_score = None
        if gemini_truth_score is not None:
            if gemini_truth_score < 0.0:
                gemini_truth_score = 0.0
            elif gemini_truth_score > 1.0:
                gemini_truth_score = 1.0
        verdict_report["overall_truth_score"] = gemini_truth_score

        if "source_judgments" not in verdict_report or not isinstance(verdict_report["source_judgments"], list):
            verdict_report["source_judgments"] = []

        fixed_judgments = []
        seen_indices = set()

        for raw_judgment in verdict_report["source_judgments"]:
            if not isinstance(raw_judgment, dict):
                continue

            try:
                source_index = int(raw_judgment.get("source_index", 0))
            except Exception:
                continue

            if source_index < 1 or source_index > len(selected_evidence):
                continue
            if source_index in seen_indices:
                continue

            fixed_judgments.append(
                {
                    "source_index": source_index,
                    "stance": str(raw_judgment.get("stance", "background")).strip().lower(),
                    "analysis": str(raw_judgment.get("analysis", "")).strip(),
                }
            )
            seen_indices.add(source_index)

        verdict_report["source_judgments"] = fixed_judgments

        return verdict_report

    except Exception as error:
        print(f"[Gemini Agent] Verdict generation failed: {error}")
        return build_empty_verdict_report("The AI verdict step failed.")
