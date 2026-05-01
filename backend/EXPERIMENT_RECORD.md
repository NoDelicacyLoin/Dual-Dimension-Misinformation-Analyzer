# Backend Experiment Record

Date: 2026-04-30

This file records the cleanup and validation work done while moving the backend toward a final project version. It is intended as report material, not as pipeline source of truth. The implementation contract remains `PIPELINE_SPEC.md`.

## 1. Cleanup Goal

The main goal was to reduce patch-like logic and make the pipeline easier to explain:

- keep atomization, text-pattern risk, fact-checking, and aggregation as separate responsibilities;
- remove fallback paths that were not part of the current product contract;
- keep cross-file data in contract models instead of loose dictionaries;
- prefer simple, direct control flow over extra helper layers;
- avoid hardcoded fixes for individual examples.

The guiding rule was: if the spec does not describe a behavior, remove it; if the spec describes it but it still looks unnecessary, consider removing it.

## 2. Pipeline Behavior Confirmed

Current pipeline shape:

```text
user input
-> atomizer
-> text-pattern branch and fact-checking branch in parallel
-> aggregate result
-> frontend display
```

Important behavior decisions:

- Atomizer failure is terminal. Text-pattern analysis and fact-checking do not run after atomizer failure.
- Empty input and no-checkable-claim cases are user-facing errors.
- Text-pattern and fact-checking branches run in parallel after atomization.
- The fact-checking branch processes factual claims with a small thread pool, currently 2 workers.
- Optional query rewrite is only an empty-selected-evidence fallback.
- If rewrite fallback is disabled or still returns no selected evidence, the claim ends as `no_evidence`.
- Gemini evidence judgment is run once per claim on the selected evidence. The previous second-pass stabilization path was removed.

## 3. Atomizer Context Resolution

Original issue:

- Claims such as `The mission was delayed after technical problems.` could remain too generic after atomization.
- The desired output was a standalone fact-checkable claim such as `The Artemis mission was delayed after technical problems.`

Design decision:

- Gemini should perform local reference resolution in the atomizer prompt.
- Backend validation should reject unresolved generic references.
- Backend validation should not invent missing subjects from context, because that creates hardcoded and patch-like behavior.

Prompt rules were strengthened for:

- local pronouns such as `he`, `she`, `they`, `it`, `this`, and `that`;
- generic references such as `the mission`, `the launch`, `the project`, `the bill`, and `the policy`;
- descriptor plus generic noun phrases such as `the tax bill`, when nearby context names a more specific subject;
- proper-name subjects such as `the Artemis mission` and `the First Amendment`, which are specific enough.

Backend guard behavior:

```text
The bill passed the Senate.                       -> reject
The planned launch was pushed into 2026.          -> reject
The tax bill passed the Senate.                   -> pass
The Tax Cuts and Jobs Act passed the Senate.      -> pass
The Artemis mission was delayed after problems.   -> pass
The First Amendment was ratified in 1791.         -> pass
```

## 4. Atomizer Gemini Tests

### Artemis Context Test

Input:

```text
I saw this claim in a short online post. The Artemis program is led by NASA and aims to return humans to the Moon. The mission was delayed after technical problems. This pushed the planned launch into 2026.
```

Observed atomizer output:

```text
status: success
ignored: I saw this claim in a short online post.
groups: 3

The Artemis program is led by NASA.
The Artemis program aims to return humans to the Moon.
The Artemis mission was delayed after technical problems.
The Artemis mission's planned launch was pushed into 2026.
```

Result:

- The atomizer resolved `The mission` to `The Artemis mission`.
- The atomizer resolved `This pushed the planned launch` to `The Artemis mission's planned launch`.
- Time constraint `2026` was preserved for search and fact-checking.

### Public Policy and Amendment Context Test

Input:

```text
A city council approved a new public health policy in 2025. The policy required schools to report vaccination rates. The First Amendment protects free speech. The amendment was ratified in 1791. The Tax Cuts and Jobs Act changed corporate tax rates. The tax bill passed the Senate.
```

Observed selected claims included:

```text
The public health policy required schools to report vaccination rates.
The First Amendment was ratified in 1791.
```

Result:

- `The policy` was resolved to `The public health policy`.
- `The amendment` was resolved to `The First Amendment`.
- A valid tax-bill claim could be omitted when the top-5 claim-group limit is reached. This is expected behavior under the current frontend limit.

### Tax Cuts and Jobs Act Test

Input:

```text
The Tax Cuts and Jobs Act changed corporate tax rates. The tax bill passed the Senate. The bill was signed into law later that year.
```

Earlier output before prompt strengthening:

```text
The tax bill passed the Senate.
The tax bill was signed into law later that year.
```

Output after prompt strengthening:

```text
The Tax Cuts and Jobs Act changed corporate tax rates.
The Tax Cuts and Jobs Act passed the Senate.
The Tax Cuts and Jobs Act was signed into law later that year.
```

Result:

- Gemini now resolves the descriptor phrase `the tax bill` when the nearby context names the specific act.
- The backend did not add this subject by code.

## 5. Long-Text Atomizer Test

Long-text atomizer behavior:

- splits long input into batches;
- includes previous sentence context;
- merges candidate claim groups;
- ranks globally;
- keeps at most 5 claim groups for the frontend.

Observed long-text output after cleanup:

```text
status: success
ignored_sentence_count: 5
text_feature_unit_count: 5
fact_check_claim_count: 5
candidate_claim_group_count: 12
candidate_fact_claim_count: 13
selected_claim_group_count: 5
selected_fact_claim_count: 5
```

Selected claim groups:

```text
Donald Trump praised the tax bill.
The tax bill would help workers and small businesses.
The tax bill passed the Senate.
The tax bill mainly helped large corporations.
The corporate tax cuts would increase the federal deficit.
```

Ignored examples:

```text
The project was almost finished at the time.
Later, officials said the project was finished.
It gave agencies new powers.
Some supporters called the bill historic.
I think the whole debate was ridiculous.
```

Result:

- The atomizer selected 5 groups as required by the frontend contract.
- Vague or unresolved references were filtered out.
- Some factual but lower-priority claims may be omitted because the product intentionally caps output at 5 groups.

## 6. Retrieval and Fallback Cleanup

Previous behavior:

- Retrieval could use an oversampling-style retry path.
- This added extra search latency and made the main fact-checking path harder to explain.

Current behavior:

- Default `retrieval_results` is 10.
- The main retrieval path performs one Tavily search per search query.
- The main search query is the checkable claim plus atomizer constraints.
- Optional rewrite fallback is separate and only runs when selected evidence count is 0 and the user enables rewrite fallback.

Reason:

- The simpler version is easier to explain.
- Searching 10 results on the first attempt is clearer than searching fewer and then rerunning hidden fallback logic.

## 7. Frontend Display Checks

User-facing display changes validated through local browser screenshots:

- `Original Text` was renamed to `User Input Text`.
- The user input block is shown even on no-checkable-claim error pages.
- `Decision Confidence` was shortened to `Confidence`.
- Claim-level explanation text is hidden from the frontend.
- Matrix explanation is generated from risk level plus fact-checking verdict.
- Summary counts were shortened to reduce wrapping:

```text
Detected: X factual - Filtered: Y non-factual
Checked: X factual - Rewrite: On/Off
```

## 8. Stabilization Cleanup

Previous behavior:

- Selective stabilization could run Gemini evidence judgment a second time on the same selected evidence.
- It was enabled by default through the frontend and backend request options.
- This added latency and made borderline verdicts harder to explain.

Current behavior:

- The stabilization option was removed from the public request contract.
- The frontend no longer sends `use_selective_stabilization`.
- The active fact-checking branch no longer calls `stabilize_result()`.
- A successful claim now follows one direct path:

```text
selected evidence
-> Gemini evidence judgment
-> evidence summary
-> truth score aggregation
-> decision confidence
-> verdict
```

Reason:

- It removes a hidden second LLM call.
- It keeps the explanation close to the evidence actually shown to the user.
- It reduces latency without changing Tavily retrieval, NLI filtering, or the main Gemini evidence prompt.

## 9. Fact-Checking Internal Parallelism

Previous behavior:

- After atomization, the text-pattern branch and fact-checking branch ran in parallel.
- Inside the fact-checking branch, factual claims still ran one after another.
- For multi-claim inputs, claim 2 waited for claim 1 to finish Tavily retrieval, NLI filtering, and Gemini evidence judgment.

Current behavior:

- The fact-checking branch uses `ThreadPoolExecutor(max_workers=2)`.
- Each worker still calls the same single-claim function.
- Results are stored by job order and returned in the original atomizer order.
- Progress counters are protected with a small `threading.Lock`.

Reason:

- Tavily and Gemini are network-bound, so threads can reduce waiting time for multiple factual claims.
- A limit of 2 workers avoids firing all possible claims at external APIs at once.
- CUDA was not introduced because the main delay is external API waiting, and the local NLI model was left on its existing CPU path.

### Full Pipeline Smoke Test

Input:

```text
I saw this claim in a short online post. The Artemis program is led by NASA and aims to return humans to the Moon. The mission was delayed after technical problems. This pushed the planned launch into 2026.
```

First run after adding fact-checking worker threads:

- Atomizer succeeded.
- Text-pattern branch completed.
- Fact-checking branch failed before retrieval because both branches could touch `transformers` during first import.
- Error: `AutoModelForSequenceClassification` could not be imported from the top-level `transformers` module.

Fix:

- Changed active text-pattern and NLI imports to import `AutoModelForSequenceClassification` and `AutoTokenizer` from the `transformers.models.auto` modules directly.
- Added a small lock around first NLI model load so two fact-check workers do not try to initialize the same model at the same time.

Second full pipeline run:

```text
elapsed_seconds: 58.18
status: success
selected_claim_groups: 3
selected_fact_claims: 4
fact_checking_status: success
fact_checking_verdict: Mostly True
fact_checking_truth_score: 0.754575
```

Observed factual claim results:

```text
The Artemis program is led by NASA.
-> True, truth_score 0.915, sufficient evidence

The Artemis program aims to return humans to the Moon.
-> True, truth_score 0.915, sufficient evidence

The Artemis mission was delayed after technical problems.
-> Mostly True, truth_score 0.6583, sufficient evidence

The Artemis mission's planned launch was pushed into 2026.
-> Neutral, truth_score 0.53, insufficient evidence
```

Interpretation:

- The 2-worker fact-checking branch ran successfully and returned results in original atomizer order.
- The last claim is a useful quality-warning case: retrieval and evidence judgment may be sensitive to how the launch claim is phrased and to current open-web results.

## 10. Timing Observations

Observed local timing was variable because Gemini and search calls depend on network and external service latency.

Approximate atomizer timings from local tests:

```text
Artemis short context test:        about 12 seconds
Mixed context short test:          about 23 seconds
Tax Cuts and Jobs Act test:        about 13 seconds
Long-text atomizer test, earlier:  about 75 seconds
Long-text atomizer test, later:    about 37 seconds
```

Full pipeline timing for the Artemis example:

```text
Displayed summary:
Detected: 4 factual · Filtered: 1 non-factual
Checked: 4 factual · Rewrite: Off

MAX_FACT_CHECK_WORKERS = 1: about 101 seconds
MAX_FACT_CHECK_WORKERS = 2: about 43 seconds
```

Interpretation:

- Long text is slower because it requires multiple Gemini batch calls plus global ranking.
- The largest remaining speed risk is external API latency, especially Tavily and Gemini response time.
- The worker thread optimization is only inside the fact-checking branch, because fact-checking is the slow branch.
- The text-pattern branch is already relatively fast, so adding threads there would add complexity without much practical benefit.
- The 2-worker version is faster because separate factual claims can wait for Tavily and Gemini at the same time instead of one after another.

## 11. External Evaluation Note

The user reported that a FEVER-style test still achieved about 70 percent success after the cleanup. This suggests the simplification did not obviously damage the main fact-checking behavior, but it should be treated as a project sanity check rather than a formal benchmark unless the test setup is documented separately.

## 12. Current Known Limitations

- Fact-checking branch uses only 2 worker threads to reduce rate-limit risk.
- Long-text atomization can still be slow for many sentences.
- Top-5 claim group selection can omit valid factual claims.
- If Gemini fails to resolve a local reference, backend validation may reject the claim rather than trying a code-level repair.

## 13. Next Optimization Candidates

Most useful next checks:

- measure fact-checking time by claim and by stage;
- compare fact-checking latency with 1 worker versus 2 workers;
- keep retrieval as one main Tavily call unless there is a clear product reason to retry;
- keep prompt changes centralized instead of adding special-case backend patches.
