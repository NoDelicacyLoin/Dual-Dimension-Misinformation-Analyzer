# Dual-Dimension Backend Pipeline Specification

This document describes the intended backend pipeline for the Dual-Dimension Misinformation Analyzer.
It is written as an implementation contract: another developer or AI agent should be able to rebuild the backend from this document without adding duplicate logic, hidden rewrite steps, or experiment residue.

## Core Product Idea

The product analyzes user text through two dimensions:

1. **Text-pattern risk**: whether the wording pattern looks risky or misinformation-like.
2. **Evidence-based fact-checking**: whether external evidence supports or contradicts factual claims.

The two branches share the same atomized input, but they do different jobs.

- The text-pattern branch checks wording, not truth.
- The fact-checking branch checks factual support, not rhetorical style.
- The atomizer prepares clean units for both branches.
- The aggregate layer combines branch outputs into a final response.

## Public API

### `POST /analyze`

Runs the full analysis and returns one complete JSON response.

Request body:

```json
{
  "claim": "User input text.",
  "options": {
    "use_query_rewrite": false,
    "relevance_threshold": 0.1,
    "top_k": 3,
    "use_all_eligible_evidence": false,
    "retrieval_results": 10
  }
}
```

### `POST /analyze/stream`

Runs the same analysis, but returns Server-Sent Events.

Event shapes:

```json
{"event": "progress", "data": {"stage": "atomizer_finished", "...": "..."}}
{"event": "result", "data": {"status": "success", "...": "..."}}
{"event": "error", "data": {"message": "Error message."}}
```

The final `result` event must contain the same schema as `POST /analyze`.

## Public Response Contract

Successful or degraded completed analyses return `AnalyzeResponse`.
User-facing input errors, such as empty input or no checkable factual claim, stop the pipeline and return an API error instead of a partial analysis response.

```json
{
  "status": "success",
  "original_text": "Original user input.",
  "ignored_sentences": [],
  "text_pattern_results": [],
  "fact_checking": {
    "status": "success",
    "truth_score": 0.82,
    "verdict": "Mostly True",
    "explanation": "Aggregated mean truth score over 2 factual claim(s).",
    "factual_claims": []
  },
  "overall_risk_score": 0.42,
  "overall_risk_level": "medium_risk",
  "overall_risk_confidence": "medium",
  "progress_events": [],
  "ignored_sentence_count": 0,
  "text_feature_unit_count": 1,
  "fact_check_claim_count": 1,
  "candidate_claim_group_count": 1,
  "candidate_fact_claim_count": 1,
  "selected_claim_group_count": 1,
  "selected_fact_claim_count": 1,
  "max_claim_group_count": 0,
  "claim_selection_reason": "",
  "message": ""
}
```

### Top-Level Status

`AnalyzeResponse.status` describes completed backend runs:

- `success`: all factual claims completed successfully.
- `partial_success`: at least one factual claim succeeded, but not all.
- `degraded`: an external component was unavailable or failed. Claim-level `system_error` maps to this top-level meaning.
- `no_evidence`: no usable evidence was found.
- `failed`: no branch produced a useful result.

## Pipeline Overview

The full backend pipeline is:

```text
user input
-> atomizer
-> text-pattern branch + fact-checking branch
-> aggregate result
-> optional streaming progress
```

After atomization, the text-pattern branch and fact-checking branch should run in parallel.
Inside the fact-checking branch, factual claims may run with a small worker pool.
Current limit: `MAX_FACT_CHECK_WORKERS = 2`.
The final `factual_claims` list must keep the original atomizer order, even if claim jobs finish out of order.

## Step 1: User Input

Input comes from `ClaimRequest.claim`.

Rules:

- Trim leading and trailing whitespace.
- If empty, stop early with a user-facing error. The frontend normally blocks this before the request is sent.
- Do not do dataset-specific preprocessing here.
- Do not perform claim rewriting here.

Empty input API error:

```json
{
  "detail": "Please enter a factual claim or passage to analyze."
}
```

## Step 2: Atomizer

The atomizer owns claim preparation.

It receives raw user text and returns factual/text units for the downstream branches.
It is the only stage that should resolve pronouns, split multi-fact sentences, and prepare standalone checkable claims.

### Atomizer Responsibilities

The atomizer should:

- normalize spacing and simple sentence-boundary issues;
- split input into sentence-like units;
- ignore non-factual, subjective, vague, greeting-like, reaction-only, or non-checkable sentences;
- keep each text-pattern unit close to the original sentence;
- extract standalone factual claims for fact-checking;
- resolve clear local pronouns such as `he`, `she`, `they`, `it`, `this`, and `that`;
- resolve clear generic local references such as `the mission`, `the launch`, `the project`, `the bill`, and `the policy`;
- resolve descriptor + generic noun references such as `the tax bill` if nearby context names a more specific subject;
- let the atomizer LLM perform the reference resolution; backend validation should reject unresolved generic references rather than inventing subjects;
- preserve important time, place, number, negation, comparison, attribution, and scope conditions;
- return `entities`, `relation`, and `constraints` for each factual claim;
- deduplicate exact or near-exact repeated fact-checking claims;
- select at most `MAX_CLAIM_GROUPS_FOR_OUTPUT` claim groups before downstream analysis.

The atomizer should not:

- decide whether a claim is true or false;
- add background knowledge;
- create a claim if the reference is unclear;
- turn a vague claim into a specific one without textual support;
- generate retrieval queries;
- perform evidence search.

### Short Text Path

If the normalized text has up to `MAX_SENTENCES_FOR_ATOMIZER` sentences, use the short atomizer path:

```text
normalize input
-> split sentences
-> call Gemini atomizer if available
-> validate output
-> keep at most MAX_CLAIM_GROUPS_FOR_OUTPUT claim groups
-> stop with an atomizer error if Gemini is unavailable or output validation fails
```

Atomizer failure is terminal.
If the atomizer fails, the backend must not run text-pattern analysis or fact-checking.
The API should return an error response, and the streaming endpoint should emit an `error` event.
If the atomizer returns `invalid_input` because no checkable factual claim was found, stop with a user-facing error rather than running either analysis branch.

### Long Text Path

If the normalized text has more than `MAX_SENTENCES_FOR_ATOMIZER` sentences, use the long-text path:

```text
normalize input
-> split all sentences
-> process target sentences in batches
-> include a small previous-context window
-> merge batch outputs
-> deduplicate claims
-> rank claim groups globally
-> keep at most MAX_CLAIM_GROUPS_FOR_OUTPUT claim groups
```

If any long-text atomizer batch fails, stop the full analysis with an atomizer error.
If claim selection is needed and fails, stop the full analysis with an atomizer error.

Current atomizer constants:

```text
MAX_SENTENCES_FOR_ATOMIZER = 12
LONG_TEXT_BATCH_SIZE = 8
LONG_TEXT_CONTEXT_SENTENCES = 2
MAX_CLAIM_GROUPS_FOR_OUTPUT = 5
```

### Atomizer Output Shape

The atomizer returns an `AtomizerOutput` model from `api_contract.py`.
Cross-file pipeline data should use contract models rather than loose dicts.

Serialized shape:

```json
{
  "status": "success",
  "original_text": "Normalized original text.",
  "ignored_sentences": [],
  "claim_groups": [],
  "summary": {
    "ignored_sentence_count": 0,
    "text_feature_unit_count": 1,
    "fact_check_claim_count": 1
  },
  "candidate_claim_group_count": 1,
  "candidate_fact_claim_count": 1,
  "selected_claim_group_count": 1,
  "selected_fact_claim_count": 1,
  "max_claim_group_count": 5,
  "claim_selection_reason": ""
}
```

Both short-text and long-text paths should return candidate/selected count fields.

### Claim Group Shape

Each claim group represents one text unit for the text-pattern branch and one or more checkable claims for fact-checking.

```json
{
  "claim_group_id": 1,
  "original_sentence": "Exact sentence from the user text.",
  "text_feature_text": "Exact sentence from the user text.",
  "atomization_applied": false,
  "fact_check_claims": [
    {
      "fact_claim_id": 1,
      "claim": "Standalone checkable factual claim.",
      "entities": ["Main entity"],
      "relation": "short relation",
      "constraints": ["time/place/number/scope condition"]
    }
  ]
}
```

Field meanings:

- `claim_group_id`: integer id, unique within one response.
- `original_sentence`: original text unit shown to users.
- `text_feature_text`: text sent to the text-pattern branch. It should normally equal `original_sentence`.
- `atomization_applied`: `true` when the sentence produced more than one fact-checking claim.
- `fact_check_claims`: standalone claims sent to fact-checking.

### Factual Claim Fields From Atomizer

Each `fact_check_claim` must include:

- `fact_claim_id`: integer id within its claim group.
- `claim`: standalone checkable claim.
- `entities`: key people, organizations, places, objects, titles, or events.
- `relation`: short predicate/relation being checked.
- `constraints`: important time, place, number, negation, comparison, exclusivity, or scope conditions.

These fields are part of the claim itself. They are not retrieval metadata.

## Step 3: Text-Pattern Branch

The text-pattern branch receives `claim_group["text_feature_text"]`.

It returns one `TextPatternResult` per claim group.

```json
{
  "claim_group_id": 1,
  "original_sentence": "Original sentence.",
  "text_feature_text": "Text sent to model.",
  "atomization_applied": false,
  "status": "success",
  "prediction": {
    "risk_level": "low_risk",
    "risk_score": 0.39,
    "confidence_level": "medium",
    "low_risk_probability": 0.52,
    "medium_risk_probability": 0.17,
    "high_risk_probability": 0.30
  },
  "influential_words": [],
  "technical_details": {},
  "message": ""
}
```

Text-pattern rules:

- It should not fact-check.
- It should not read evidence.
- It should not alter the factual claim.
- It should return a stable schema even on error.

On error:

```json
{
  "status": "error",
  "message": "Error message."
}
```

The orchestrator fills `claim_group_id`, `original_sentence`, `text_feature_text`, and `atomization_applied` after the model returns.

## Step 4: Fact-Checking Branch

The fact-checking branch receives the atomizer's `fact_check_claims`.

It must not re-atomize the input.
It must not re-derive `entities`, `relation`, or `constraints`.
It must not call Gemini to rewrite the main claim before searching.

The main path is:

```text
checkable claim + atomizer constraints
-> direct search query
-> Tavily retrieval
-> NLI filtering
-> Gemini evidence judgment
-> backend score aggregation
```

For multiple factual claims, run at most `MAX_FACT_CHECK_WORKERS = 2` claim pipelines at the same time.
Each individual claim still follows the same main path.

### Fact-Checking Input

Each fact-checking run receives:

```json
{
  "claim_group_id": 1,
  "fact_claim_id": 1,
  "original_sentence": "Original sentence.",
  "text_feature_text": "Text-pattern text.",
  "claim": "Standalone checkable factual claim.",
  "entities": [],
  "relation": "",
  "constraints": []
}
```

This becomes `EachFactualClaim`.

### EachFactualClaim Output

```json
{
  "claim_group_id": 1,
  "fact_claim_id": 1,
  "original_sentence": "Original sentence.",
  "text_feature_text": "Text-pattern text.",
  "claim": "Standalone checkable factual claim.",
  "entities": [],
  "relation": "",
  "constraints": [],
  "status": "success",
  "truth_score": 0.9,
  "verdict": "True",
  "explanation": "Short evidence-based explanation.",
  "decision_confidence": "high",
  "evidence_sufficiency": "sufficient",
  "evidence": [],
  "metadata": {}
}
```

### Fact-Checking Metadata

`metadata` is only for retrieval/runtime information:

```json
{
  "retrieval_query_used": "search query actually used",
  "retrieval_queries_tried": ["query 1", "query 2"],
  "fallback_used": false,
  "search_raw_evidence_count": 10,
  "selected_evidence_count": 3,
  "gemini_truth_score": 0.86
}
```

Do not store `entities`, `relation`, or `constraints` in metadata.

### Fact-Checking Status Values

Claim-level status values:

- `invalid_request`: claim is empty or too short to check.
- `no_evidence`: retrieval and filtering returned no usable selected evidence after any allowed fallback.
- `system_error`: search or external service failed.
- `degraded`: evidence existed, but Gemini judgment failed or was unavailable.
- `success`: evidence was judged and a truth score/verdict was produced. Weak evidence is represented by `evidence_sufficiency` and `decision_confidence`, not by a separate status.

Important distinction:

- `no_evidence` means the retrieval/filter path could not provide selected evidence, so no evidence-based verdict should be produced.
- `evidence_sufficiency = "insufficient"` or `"low"` means evidence reached Gemini and can still produce a verdict, but the frontend must display the weak evidence basis using `decision_confidence` and `evidence_sufficiency`.

### Search

Fact-checking should build one direct fact-check target from the atomizer output:

```text
fact_check_target = checkable_claim + relevant atomizer constraints
```

The default `build_search_queries()` should return only this direct fact-check target.
It should append constraints that are not already present in the claim text.
The same fact-check target should be used for search, NLI evidence filtering, and Gemini evidence judgment.

Do not build extra query variants in the main path.
Do not create entity-only searches in the main path.
Do not split the claim into separate subject/object searches.

### Rewrite Fallback

`options.use_query_rewrite` means **rewrite fallback for empty selected evidence**, not normal query preparation.

Fallback can run only when:

- the user enabled `use_query_rewrite`, and
- the first retrieval/filter pass produced `selected_evidence_count == 0`, and
- the search did not fail with a real system error.

Fallback should ask Gemini for one alternative `search_query`.
It must not change the checkable claim or atomizer constraints used for judgment.

Fallback output:

```json
{
  "is_valid_claim": true,
  "search_query": "fallback search query"
}
```

### Retrieval Result

Internal retrieval shape:

```json
{
  "raw_evidence": [],
  "search_query": "query used",
  "search_raw_count": 8,
  "error_type": "",
  "error_message": ""
}
```

Retrieval makes one Tavily request per search query.
The default request asks Tavily for 10 results.
If retrieval or filtering cannot produce selected evidence, the claim returns `no_evidence` unless the optional rewrite fallback is enabled.

### Evidence Selection

The NLI filter receives:

```text
fact_check_target
raw_evidence
relevance_threshold
top_k
use_all_eligible_evidence
```

It returns:

```json
{
  "selected_evidence": [],
  "filter_debug": {},
  "search_raw_count": 8
}
```

Each selected evidence item must fit `EachEvidence`:

```json
{
  "stance": "",
  "evidence_quality": "strong",
  "url": "https://source.example",
  "content": "Evidence text snippet.",
  "ai_analysis": ""
}
```

Before Gemini judgment:

- `evidence_quality` comes from retrieval/NLI filtering.
- `stance` may be empty or provisional.
- `ai_analysis` is empty.

After Gemini judgment:

- `stance` must be one of `supports`, `contradicts`, `mixed`, or `background`.
- `ai_analysis` explains the source-level judgment.

### Gemini Evidence Judgment

Gemini receives:

```text
fact_check_target
selected evidence content
```

Gemini must judge only the selected evidence.
It must not use outside knowledge.
It must not decide the final backend verdict.

Expected Gemini JSON:

```json
{
  "explanation": "Short explanation in 2 to 4 sentences.",
  "overall_truth_score": 0.72,
  "source_judgments": [
    {
      "source_index": 1,
      "stance": "supports",
      "analysis": "Source-level explanation."
    }
  ]
}
```

Backend handling:

- copy `explanation` to `EachFactualClaim.explanation`;
- store `overall_truth_score` in `metadata.gemini_truth_score`;
- apply each source judgment to the matching evidence item;
- if a source judgment is missing, mark that source as `background` with default analysis;
- if Gemini is unavailable, set status `degraded`.

### Evidence Sufficiency

Evidence sufficiency is decided after Gemini stance judgment.

Evidence scoring:

```text
strong evidence = 2
usable evidence = 1
weak/background/other = 0
```

Sufficiency:

```text
all selected evidence is background -> low
score >= 3 -> sufficient
score >= 2 -> limited
else -> insufficient
```

Decision-usable sources are selected sources where:

- `stance` is not `background`, and
- `evidence_quality` is `strong` or `usable`.

If evidence sufficiency is `insufficient` or `low`, the claim can still have `status = "success"` when a truth score and verdict were produced.
The frontend should use `verdict`, `decision_confidence`, and `evidence_sufficiency` together.

### Truth Score Aggregation

Backend stance score:

```text
strong support = +2
usable support = +1
strong contradiction = -2
usable contradiction = -1
```

If there is no directional score:

```text
backend_truth_score = 0.5
```

Otherwise:

```text
balance_score = (support_score - contradiction_score) / (support_score + contradiction_score)
backend_truth_score = 0.5 + (0.4 * balance_score)
```

Gemini truth score is a light calibration signal:

```text
final_truth_score = 0.85 * backend_truth_score + 0.15 * gemini_truth_score
```

Clamp final score to `[0.0, 1.0]`.
Round to 4 decimals.

### Verdict Mapping

Claim and branch verdicts use the same truth-score boundaries:

```text
truth_score is None -> verdict = None
truth_score >= 0.85 -> True
truth_score >= 0.65 -> Mostly True
truth_score >= 0.45 -> Neutral
truth_score >= 0.25 -> Mostly False
else -> False
```

### Decision Confidence

Decision confidence starts from usable directional source count:

```text
0 or 1 usable source -> low
2 usable sources -> medium
3+ usable sources -> high
```

Then reduce confidence if the score is close to neutral:

```text
abs(truth_score - 0.5) < 0.15
```

If near neutral:

- `high` becomes `medium`;
- `medium` or `low` becomes `low`.

### Fact-Checking Branch Summary

After all factual claims are processed, build `EachFactChecking`:

```json
{
  "status": "success",
  "truth_score": 0.82,
  "verdict": "Mostly True",
  "explanation": "Aggregated mean truth score over 2 successful factual claim(s).",
  "factual_claims": []
}
```

Branch truth score:

- if claims with numeric truth scores exist, average the available truth scores;
- else `truth_score = null`.

Branch status:

```text
all claims success -> success
some claims success -> partial_success
any degraded or system_error -> degraded
any no_evidence -> no_evidence
else -> failed
```

## Step 5: Aggregate Result

The orchestrator combines:

- atomizer counts;
- text-pattern results;
- fact-checking summary;
- overall status;
- overall risk score;
- progress events.

### Overall Status

Overall status is based mainly on fact-checking claim outcomes:

```text
all factual claims success -> success
some factual claims success -> partial_success
any degraded or system_error -> degraded
any no_evidence -> no_evidence
else -> failed
```

If atomizer returns invalid input, stop early with a user-facing error and do not run either branch.

### Overall Risk Score

Overall risk combines:

1. text-pattern risk scores from successful text-pattern results;
2. factual risk scores from fact-checking claims that produced numeric truth scores.

Factual risk:

```text
factual_risk = 1.0 - truth_score
```

Overall score:

```text
overall_risk_score = mean(text_pattern_risk_scores + factual_risk_scores)
```

Only branch outputs with numeric scores contribute to the numeric risk score.
`no_evidence` does not automatically count as factual risk.
Weak evidence can still contribute when it produced a numeric truth score/verdict, with confidence and sufficiency shown separately.

Overall risk level:

```text
score >= 0.66 -> high_risk
score >= 0.33 -> medium_risk
else -> low_risk
```

Overall risk confidence:

- collect confidence labels from successful text-pattern and fact-checking outputs;
- return the weakest confidence using this rank:

```text
low < medium < high
```

## Step 6: Streaming Progress

Streaming is optional but should reflect the same pipeline.

Progress events are stored in `AnalyzeResponse.progress_events`.
When streaming, the same progress events are also emitted as SSE `progress` events.

### Progress Event Base Fields

Every progress event should include:

```json
{
  "stage": "stage_name",
  "status": "running_or_completed",
  "message": "Human-readable message.",
  "claim_group_id": 0,
  "fact_claim_id": 0
}
```

### Stage Names

Use these exact stage names:

```text
atomizer_finished
bert_progress
token_occlusion_progress
text_pattern_finished
tavily_nli_progress
llm_evidence_progress
fact_checking_finished
analysis_finished
```

### Atomizer Finished Event

```json
{
  "stage": "atomizer_finished",
  "status": "completed",
  "message": "Atomizer finished with 2 claim group(s).",
  "claim_group_id": 0,
  "fact_claim_id": 0,
  "text_feature_unit_count": 2,
  "fact_check_claim_count": 3,
  "ignored_sentence_count": 1,
  "candidate_claim_group_count": 8,
  "candidate_fact_claim_count": 12,
  "selected_claim_group_count": 5,
  "selected_fact_claim_count": 7,
  "max_claim_group_count": 5,
  "claim_selection_reason": "Brief reason."
}
```

### Text-Pattern Progress Events

BERT classification:

```json
{
  "stage": "bert_progress",
  "status": "running",
  "completed_text_feature_unit_count": 1,
  "text_feature_unit_count": 2
}
```

Token occlusion:

```json
{
  "stage": "token_occlusion_progress",
  "status": "running",
  "completed_text_feature_unit_count": 1,
  "text_feature_unit_count": 2
}
```

Text-pattern branch finished:

```json
{
  "stage": "text_pattern_finished",
  "status": "completed",
  "completed_text_feature_unit_count": 2,
  "text_feature_unit_count": 2
}
```

### Fact-Checking Progress Events

Tavily + NLI:

```json
{
  "stage": "tavily_nli_progress",
  "status": "running",
  "completed_tavily_nli_count": 1,
  "fact_check_claim_count": 3
}
```

LLM evidence judgment:

```json
{
  "stage": "llm_evidence_progress",
  "status": "running",
  "completed_llm_evidence_count": 1,
  "fact_check_claim_count": 3
}
```

Fact-checking branch finished:

```json
{
  "stage": "fact_checking_finished",
  "status": "completed",
  "completed_fact_check_claim_count": 3,
  "fact_check_claim_count": 3
}
```

Overall analysis finished:

```json
{
  "stage": "analysis_finished",
  "status": "completed",
  "text_feature_unit_count": 2,
  "fact_check_claim_count": 3
}
```

## Frontend Display Boundary

The backend must return enough information for the frontend to distinguish:

1. **User Input Text**: the full user input.
2. **Original Sentence / Text Unit**: the sentence-like unit selected by atomizer.
3. **Checkable Claim**: the standalone factual claim sent to fact-checking.
4. **Search Query**: technical retrieval metadata, not the factual claim.

Rules:

- Always show `original_text` as `User Input Text`.
- Use claim group headers for `original_sentence`.
- Show `EachFactualClaim.claim` as `Checkable Claim`.
- Do not label search queries as claims.
- Do not show `N/A` as if it were a real verdict; use no-final-verdict language for `no_evidence`, and show verdicts together with `decision_confidence` and `evidence_sufficiency` when evidence is insufficient.

## Ownership Boundaries

These boundaries are important for keeping the backend simple.

### Atomizer Owns

- sentence splitting;
- factual/non-factual filtering;
- pronoun/context resolution;
- splitting multi-fact sentences;
- checkable claim preparation;
- `entities`, `relation`, `constraints`;
- final claim group selection up to `MAX_CLAIM_GROUPS_FOR_OUTPUT`.

### Text-Pattern Branch Owns

- wording-pattern risk;
- risk probabilities;
- influential words;
- text-pattern confidence.

### Fact-Checking Branch Owns

- retrieval;
- NLI evidence filtering;
- optional empty-search rewrite fallback;
- Gemini source-level evidence judgment;
- truth score;
- verdict;
- evidence sufficiency;
- decision confidence.

### Aggregate Layer Owns

- top-level status;
- overall risk score;
- overall risk level;
- overall risk confidence;
- response assembly;
- progress event storage.

## Anti-Redundancy Rules

Do not add code that violates these rules:

1. Do not prepare or rewrite the main claim in fact-checking. The atomizer already prepares checkable claims.
2. Do not store claim fields inside metadata. Metadata is only runtime/retrieval information.
3. Do not create multiple search planners. Main search is the checkable claim plus atomizer constraints; fallback is one optional Gemini query.
4. Do not split entity searches in the main path.
5. Do not re-rank long-text claims outside the atomizer.
6. Do not let the text-pattern branch influence evidence selection.
7. Do not let fact-checking alter text-pattern input.
8. Do not create dataset-specific rules for LIAR, FEVER, or ISOT in production code.
9. Do not use notebook/test code as product logic.
10. Do not introduce helper layers unless they remove real duplication.

## Minimal Backend File Shape

A clean backend can be organized like this:

```text
api_contract.py
shared_constants.py
app.py
analysis_orchestrator.py
atomizer/
  atomizer_service.py
  atomizer_gemini.py
text_pattern/
  text_risk_service.py
  predict_text_risk_local.py
fact_checking/
  fact_check_service.py
  retrieval_service.py
  search.py
  nli_filter.py
  gemini_agent.py
  decision_utils.py
  recovery.py
```

Expected file responsibilities:

- `api_contract.py`: request, response, and cross-stage pipeline models.
- `shared_constants.py`: shared stage names and scoring constants.
- `app.py`: HTTP routes and static frontend serving.
- `analysis_orchestrator.py`: pipeline order, parallel branches, aggregation, streaming.
- `atomizer_service.py`: atomization and long-text selection.
- `text_risk_service.py`: adapt local model output into `TextPatternResult`.
- `fact_check_service.py`: claim-level fact-checking flow and branch summary.
- `retrieval_service.py`: Tavily retrieval and NLI selection orchestration.
- `search.py`: raw Tavily API access and page cleanup.
- `nli_filter.py`: evidence scoring and selection.
- `gemini_agent.py`: Gemini evidence judgment and optional fallback search query.
- `decision_utils.py`: score, verdict, sufficiency, confidence.
- `recovery.py`: narrow fallback utilities.

## Current Product Limitations To Preserve Honestly

The backend should not pretend to solve these fully yet:

- open-web retrieval may fail even when a claim is checkable;
- long articles may need stronger article-level summarization later;
- top-K claim selection may omit some factual claims;
- evidence sufficiency can be insufficient even when one useful source exists;
- `no_evidence` is not the same as false;
- weak or insufficient evidence is not the same as false;
- noisy copied articles may need preprocessing outside the current scope.

These limitations should be reflected in output wording rather than hidden with extra fallback logic.
