import json

from api_contract import AtomizedClaimGroup


def build_atomizer_prompt(
    original_text: str,
    sentence_items: list[str],
    context_sentences: list[str] | None = None,
    target_sentences: list[str] | None = None,
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
- For ordinary reporting or quotation sentences, fact-check the embedded factual proposition, not the reporting frame.
- Remove weak source frames such as "critics said", "officials said", or "analysts reported" when the checkable fact is inside the clause.
- Preserve attribution only when the act of saying, announcing, believing, posting, writing, or historical thinking is itself the factual point.
- Preserve belief framing when the claim is about what people believed or argued, not about whether the embedded statement is true.
- For example, "Trump said X" should usually become "X".
- For example, "People once thought X" must stay about what people once thought.
- A fact_check_claim must be standalone enough for web search and evidence checking.
- Each fact_check_claim must include entities, relation, and constraints.
- entities should list the main people, organizations, places, objects, titles, or events in the claim.
- relation should be the short relationship or predicate being checked.
- constraints should list important time, place, number, comparison, negation, exclusivity, or scope conditions.
- Use only information present in the original sentence or clearly resolved from nearby context.
- If a sentence uses he, she, they, it, this, or that, resolve the reference in the fact_check_claim only when the nearby previous context clearly identifies the subject or object.
- If the subject or object reference is unclear, ignore that sentence instead of outputting an incomplete claim.
- Generic references such as "the mission", "the launch", "the project", "the bill", or "the policy" must also be resolved from nearby context.
- Do not output vague fact_check_claims that still depend on generic subjects like "the mission" or "the planned launch".
- Descriptor + generic noun phrases such as "the tax bill", "the spending bill", or "the federal policy" must also be resolved if nearby context names a more specific subject.
- Proper-name subjects such as "the Artemis mission" or "the First Amendment" are specific enough.
- When resolving a generic reference, put the resolved subject in the claim and in entities.
- If context names a specific law, amendment, program, policy, project, mission, or event, do not fall back to a descriptor-only subject such as "the tax bill" or "the policy".
- Wrong: "The tax bill passed the Senate." when nearby context says "The Tax Cuts and Jobs Act".
- Correct: "The Tax Cuts and Jobs Act passed the Senate."
- Time, place, comparison, and condition words from the target sentence must be preserved.
- Use context conditions only when the target sentence clearly refers back to them. Do not blindly inherit the most recent year, place, or event from context.
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

Examples:
- Input context: The Artemis program is led by NASA. The mission was delayed after technical problems.
  Fact claims: The Artemis program is led by NASA. / The Artemis mission was delayed after technical problems.
- Input context: The Artemis program is led by NASA. The mission was delayed after technical problems. This pushed the planned launch into 2026.
  Fact claims: The Artemis program is led by NASA. / The Artemis mission was delayed after technical problems. / The Artemis mission's planned launch was pushed into 2026.
- Input context: Donald Trump praised the tax bill. He said it would help workers.
  Fact claims: Donald Trump praised the tax bill. / Donald Trump said the tax bill would help workers.
- Input context: Donald Trump praised the tax bill. The bill passed the Senate.
  Fact claims: Donald Trump praised the tax bill. / The tax bill passed the Senate.
- Input context: The Tax Cuts and Jobs Act changed corporate tax rates. The tax bill passed the Senate.
  Fact claims: The Tax Cuts and Jobs Act changed corporate tax rates. / The Tax Cuts and Jobs Act passed the Senate.
- Input context: The First Amendment protects free speech. The amendment was ratified in 1791.
  Fact claims: The First Amendment protects free speech. / The First Amendment was ratified in 1791.
- Input: Donald Trump said, "The stock market had its best year."
  Fact claim: The stock market had its best year.
- Input context: Donald Trump praised the tax bill. Critics said the bill mainly helped large corporations.
  Fact claims: Donald Trump praised the tax bill. / The tax bill mainly helped large corporations.
- Input: Officials announced the law in 2024.
  Fact claim: Officials announced the law in 2024.
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
          "claim": "Standalone factual claim.",
          "entities": ["Main entity"],
          "relation": "short relation",
          "constraints": ["important condition"]
        }}
      ]
    }}
  ],
  "summary": {{
    "ignored_sentence_count": 0,
    "text_feature_unit_count": 1,
    "fact_check_claim_count": 1
  }}
}}

Input text:
{original_text}

Input sentences:
{sentence_block}

Context sentences:
{context_block or "None"}

Target sentences:
{target_block}
""".strip()


def build_claim_group_ranking_prompt(original_text: str, claim_groups: list[AtomizedClaimGroup], top_k: int) -> str:
    group_lines = []
    for group in claim_groups:
        fact_claim_lines = []
        for fact_claim in group.fact_check_claims:
            details = []
            if fact_claim.entities:
                details.append("entities: " + ", ".join(fact_claim.entities))
            if fact_claim.relation:
                details.append("relation: " + fact_claim.relation)
            if fact_claim.constraints:
                details.append("constraints: " + ", ".join(fact_claim.constraints))
            detail_text = f" ({'; '.join(details)})" if details else ""
            fact_claim_lines.append(f"- {fact_claim.claim}{detail_text}")

        group_lines.append(
            "\n".join(
                [
                    f"Group {group.claim_group_id}",
                    f"Original sentence: {group.original_sentence}",
                    "Fact-checking claims:",
                    "\n".join(fact_claim_lines),
                ]
            )
        )

    group_block = "\n\n".join(group_lines)
    escaped_original_text = json.dumps(original_text)

    return f"""
You are selecting the most important factual units for a long-text misinformation analysis system.

The atomizer has already extracted factual claim groups. Your job is only to choose the top {top_k} claim groups that best represent the user's input.

Selection rules:
- Rank by importance to the user's original text, not by whether a claim looks true or false.
- Do not prefer a claim because it seems surprising, controversial, suspicious, or likely false.
- Prefer claims that carry the main factual message of the text.
- Prefer claims that other sentences depend on.
- Prefer specific checkable factual claims over background details.
- Judge retrieval readiness from the fact-checking claims, not from the original sentence.
- The original sentence may contain pronouns such as he, it, this, or they, as long as the fact-checking claims have resolved them clearly.
- Select only groups whose fact-checking claims are standalone enough for open-web retrieval.
- Do not select groups whose fact-checking claims still depend on vague references such as "a related law", "a new policy", "the government", "this", or "that" unless the needed topic, entity, or condition is included in the fact-checking claim.
- If an important group cannot be made specific enough from the provided fact-checking claims, skip it rather than selecting a vague claim.
- Keep enough variety to cover the main factual topics.
- Do not select duplicate or near-duplicate groups if another selected group already covers the same fact.
- Use only the group ids provided below.

Return valid JSON only:
{{
  "selected_claim_group_ids": [1, 2, 3],
  "selection_reason": "Brief reason for the selection. Do not mention group ids."
}}

Original text:
{escaped_original_text}

Claim groups:
{group_block}
""".strip()
