# Fact-checking Report Notes

This file records useful design notes, testing observations, and report-ready explanations for the fact-checking branch. It should be updated whenever a meaningful implementation decision or experiment finding appears.

## Core Pipeline

The fact-checking branch takes factual claims from the atomizer and checks them with a retrieval-augmented pipeline:

1. Receive factual claims from the atomizer.
2. Rewrite or normalize the claim for search when useful.
3. Retrieve candidate evidence from web search.
4. Filter and rank evidence with NLI-style relevance and claim matching.
5. Send selected evidence to Gemini for evidence-based judgement.
6. Aggregate claim-level results into the branch-level fact-checking output.
7. Return schema-shaped results to the orchestrator and frontend.

The branch is designed to keep retrieval, evidence selection, language-model judgement, and final decision logic separate. This makes failures easier to diagnose: a bad result can usually be traced to retrieval, filtering, Gemini interpretation, or final aggregation.

## Important Status Meanings

The branch separates `no_evidence` from `insufficient_evidence`.

`no_evidence` means no selected evidence survived the retrieval and filtering stage. In this case the claim should not be sent to Gemini, because there is no evidence for Gemini to judge. The system may attempt fallback or stabilization, but if evidence is still unavailable, the result should say that no evidence was found.

`insufficient_evidence` means evidence exists and has been judged, but the evidence is not strong enough or numerous enough to support a confident final factual conclusion. This is different from `no_evidence`: the system has evidence, but the evidence is thin.

It is acceptable for a claim to have a directional verdict while also having `evidence_sufficiency = insufficient`. For example, one strong or usable source may be enough to suggest a likely verdict, but still not enough to mark the evidence base as fully sufficient. In report language, this means the system can make a low-confidence evidence-based judgement while honestly flagging that the evidence base is limited.

## Evidence Sufficiency

Evidence sufficiency is intentionally simple. It should depend mainly on the number and quality of usable evidence items, especially `strong` evidence. It should not depend directly on whether an evidence item's stance is `background`.

This distinction matters because `background` is a stance judgement, not an evidence-quality judgement. A background source can still be useful for context, but it should not automatically decide sufficiency. The sufficiency rule should remain simple and explainable: how much usable evidence does the system actually have?

## LIAR Testing Observations

The current LIAR tests are mainly used for stability and pipeline diagnosis, not for high accuracy. LIAR is difficult because many labels reflect political context, rhetorical framing, missing context, or partial truth rather than only literal factual support.

### LIAR Stability Batch: Seed 42

A 24-claim LIAR batch was run with `STABILITY_RANDOM_SEED = 42` and `ROWS_PER_LABEL = 4`, sampled across `true`, `mostly-true`, `half-true`, `barely-true`, `false`, and `pants-fire`.

The stability audit found `0` flow-level issues:

- No claim with empty selected evidence was incorrectly sent through as a normal judged verdict.
- Successful claim results had complete schema fields, including verdict and truth score.
- `no_evidence` and `insufficient_evidence` were kept separate.
- No system-level runtime errors appeared.

This supports the conclusion that the fact-checking branch is stable enough to move beyond LIAR smoke testing. The remaining disagreements with LIAR labels should mainly be treated as diagnostic examples, not immediate pipeline failures.

The batch also showed that LIAR remains difficult as an accuracy benchmark. Several claims were judged as `True` or `Neutral` even when LIAR labels were `false`, `barely-true`, or `pants-fire`. These cases are useful for studying retrieval quality, contextual framing, and the difference between literal evidence support and political fact-checking labels.

#### LIAR Batch Metrics

These metrics were calculated by mapping LIAR labels to the system verdict scale:

| LIAR label | Mapped verdict |
| --- | --- |
| `true` | `True` |
| `mostly-true` | `Mostly True` |
| `half-true` | `Neutral` |
| `barely-true` | `Mostly False` |
| `false` | `False` |
| `pants-fire` | `False` |

| Metric | Value |
| --- | ---: |
| Total claims | 24 |
| Judged claims | 14 |
| No-verdict claims (`insufficient_evidence` or `no_evidence`) | 10 |
| Verdict coverage | 58.3% |
| Flow-level stability issues | 0 |
| Strict accuracy, counting no-verdict as incorrect | 20.8% |
| Strict accuracy, judged claims only | 35.7% |
| Relaxed accuracy, counting no-verdict as incorrect | 33.3% |
| Relaxed accuracy, judged claims only | 57.1% |
| Strict Macro-F1, counting no-verdict as incorrect | 0.235 |
| Strict Macro-F1, judged claims only | 0.280 |
| Mean raw evidence count | 6.46 |
| Mean selected evidence count | 2.33 |
| Mean runtime per claim | 8.52 seconds |

The LIAR accuracy should not be read as final fact-checking performance. This batch was designed as a stability and diagnostic test. LIAR labels often include context, rhetorical framing, and partial-truth reasoning, while the current system mainly judges whether retrieved evidence supports or contradicts the literal claim. A more suitable performance benchmark should be run on FEVER after the LIAR stability stage.

| Claim status | Count |
| --- | ---: |
| `success` | 14 |
| `insufficient_evidence` | 7 |
| `no_evidence` | 3 |

#### Relaxed LIAR Matching

Strict exact matching is harsh for LIAR because the dataset label space does not match the system verdict space. LIAR has labels such as `pants-fire` and `barely-true`, while the system uses a simpler five-level verdict scale. A relaxed matching rule can therefore be useful as a secondary diagnostic metric.

The relaxed rule treats nearby labels as acceptable:

| LIAR label | Accepted system verdicts |
| --- | --- |
| `true` | `True`, `Mostly True` |
| `mostly-true` | `True`, `Mostly True`, `Neutral` |
| `half-true` | `Mostly True`, `Neutral`, `Mostly False` |
| `barely-true` | `Neutral`, `Mostly False`, `False` |
| `false` | `Mostly False`, `False` |
| `pants-fire` | `Mostly False`, `False` |

Macro-F1 is kept as a strict metric because relaxed matching is a many-to-many acceptance rule rather than a standard single-label classification setup. For the relaxed view, accuracy and per-label relaxed match rate are more interpretable.

| LIAR label | Total | Judged | Strict correct | Relaxed correct | Strict accuracy | Relaxed accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `barely-true` | 4 | 2 | 0 | 0 | 0.00 | 0.00 |
| `false` | 4 | 1 | 0 | 1 | 0.00 | 0.25 |
| `half-true` | 4 | 4 | 2 | 2 | 0.50 | 0.50 |
| `mostly-true` | 4 | 2 | 0 | 2 | 0.00 | 0.50 |
| `pants-fire` | 4 | 3 | 1 | 1 | 0.25 | 0.25 |
| `true` | 4 | 2 | 2 | 2 | 0.50 | 0.50 |

The seed-42 run preserves the stability conclusion while giving a more informative judged-only relaxed score. The result should still be described as a claim-only LIAR diagnostic result, not as final fact-checking performance.

#### Why LIAR Batch Accuracy Is Limited

Several factors can suppress LIAR batch accuracy even when the fact-checking pipeline is behaving correctly.

First, LIAR and the system do not use the same label space. LIAR has six labels, including `pants-fire` and `barely-true`, while the system uses a simpler five-level verdict scale. Exact matching is therefore an imperfect evaluation method. Relaxed matching should be reported alongside strict matching.

Second, the current LIAR notebook uses a claim-only setup. It sends only the statement text into the backend, without adding LIAR metadata such as speaker, venue, political party, state, or statement context. This is a valid setup because it reflects what the product receives from a normal user input, but it is not the easiest setup for LIAR. Many LIAR labels depend on context, time, speaker, or rhetorical framing. A future context-enhanced LIAR experiment could add speaker and context to the search query or prompt, and should be reported separately from the claim-only experiment.

Third, the current truth-score aggregation is deliberately simple. If most usable evidence supports a claim, the score can move strongly toward `True`. This makes the system stable and explainable, but it also means fine-grained LIAR labels such as `mostly-true`, `half-true`, and `barely-true` are harder to match exactly. This is especially visible when LIAR marks a claim as partially true because of missing context, while retrieved evidence supports the literal wording of the claim.

Fourth, the system can abstain with `no_evidence` or `insufficient_evidence`. This is useful product behavior because the system should not force a verdict when the evidence base is weak. However, if all abstentions are counted as incorrect, all-claims accuracy will be lower. For this reason, reports should include both all-claims accuracy and judged-only accuracy.

Fifth, retrieval is not optimized specifically for LIAR. Adding benchmark-specific hints such as `PolitiFact` to every query could improve LIAR scores, but that would make the evaluation less representative of a general user-facing misinformation analyzer. It is better to keep the main LIAR run claim-only, then optionally run a separate context-enhanced or benchmark-assisted experiment if needed.

The current notebook is therefore usable as a claim-only LIAR stability and diagnostic test. It is suitable for checking whether the backend behaves consistently, whether evidence handling is clean, and whether the system can produce schema-valid outputs. It should not be presented as the final or fairest measure of fact-checking accuracy.

## FEVER Testing Plan

After the LIAR stability stage, FEVER should be used as the next batch benchmark because its labels are closer to evidence-based fact checking:

- `SUPPORTS`: the claim is supported by evidence.
- `REFUTES`: the claim is contradicted by evidence.
- `NOT ENOUGH INFO`: the evidence is missing or insufficient.

The first FEVER notebook uses a 10-claim balanced sample from `dataset/FEVER/paper_test.csv`:

| FEVER label | Sample count |
| --- | ---: |
| `SUPPORTS` | 4 |
| `REFUTES` | 3 |
| `NOT ENOUGH INFO` | 3 |

System verdicts are mapped to FEVER labels as follows:

| Backend result | FEVER prediction |
| --- | --- |
| `True`, `Mostly True` | `SUPPORTS` |
| `False`, `Mostly False` | `REFUTES` |
| `Neutral` | `NOT ENOUGH INFO` |
| `no_evidence`, `insufficient_evidence` | `NOT ENOUGH INFO` |

This mapping is more natural than the LIAR mapping because FEVER's `NOT ENOUGH INFO` class matches the product behavior of refusing a strong verdict when evidence is missing or thin. The first FEVER run should record accuracy, Macro-F1, status distribution, confusion matrix, and selected-evidence examples.

### FEVER Batch: 10 Claims

The first FEVER batch test used 10 claims from `dataset/FEVER/paper_test.csv`, sampled with `FEVER_RANDOM_SEED = 42`:

| FEVER label | Sample count |
| --- | ---: |
| `SUPPORTS` | 4 |
| `REFUTES` | 3 |
| `NOT ENOUGH INFO` | 3 |

Overall metrics:

| Metric | Value |
| --- | ---: |
| Total claims | 10 |
| Accuracy | 70.0% |
| Macro-F1 | 0.669 |
| Flow-level stability issues | 0 |
| Mean raw evidence count | 7.70 |
| Mean selected evidence count | 2.90 |
| Mean runtime per claim | 10.20 seconds |

This result is from the rerun after simplifying claim preparation and fixing the notebook audit rule for `invalid_request`.

Status distribution:

| Claim status | Count |
| --- | ---: |
| `success` | 8 |
| `insufficient_evidence` | 2 |

Per-class metrics:

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `SUPPORTS` | 0.750 | 0.750 | 0.750 |
| `REFUTES` | 0.750 | 1.000 | 0.857 |
| `NOT ENOUGH INFO` | 0.500 | 0.333 | 0.400 |

Confusion matrix:

| Gold label | Predicted `NOT ENOUGH INFO` | Predicted `REFUTES` | Predicted `SUPPORTS` |
| --- | ---: | ---: | ---: |
| `NOT ENOUGH INFO` | 1 | 1 | 1 |
| `REFUTES` | 0 | 3 | 0 |
| `SUPPORTS` | 1 | 0 | 3 |

The result is substantially more meaningful than LIAR for this project because FEVER's label space is closer to the system's evidence-based decision structure. The main weakness in this small batch is the `NOT ENOUGH INFO` class: the system sometimes gives a directional verdict when FEVER labels the claim as unverifiable, because web retrieval can find real-world evidence even when FEVER's closed evidence set marks the claim as not enough information.

Important examples from this batch:

- `Meteora is the sophomore album of Linkin Park.` was correctly predicted as `SUPPORTS`.
- `Prince Charles and Lady Diana were married in July 1981.` was correctly predicted as `SUPPORTS`.
- `Samsung entered the construction and shipbuilding industries in the mid-1950s.` was correctly predicted as `REFUTES`.
- `Fraud is accidental deception.` was correctly predicted as `REFUTES`.
- `Benzodiazepine was globally the most prescribed dance move in 1977.` was gold `REFUTES` and was correctly predicted as `REFUTES` after the claim-preparation change. This confirms that absurd but concrete claims should not be rejected before retrieval.
- `Corsica is adjacent to Haute-Corse.` was gold `NOT ENOUGH INFO` but predicted `SUPPORTS`; this remains a real model-side stance error. Evidence that Haute-Corse is part of Corsica should not support the claim that Corsica is adjacent to Haute-Corse. The prompt rule helped document the intended reasoning, but the rerun shows this issue is not fully solved.
- `Uganda was ruled by the French.` was gold `NOT ENOUGH INFO` but predicted `REFUTES`; open-web evidence showed Uganda was a British protectorate, so the system contradicted the claim even though FEVER treats it as not enough information.
- `Ang Lee is a writer.` was gold `SUPPORTS` but predicted `NOT ENOUGH INFO`; selected evidence focused on filmmaker/director context rather than direct writing evidence. Under the current responsibility split, this is acceptable: retrieval and filtering did not surface direct support, so the final insufficient-evidence result is a reasonable abstention rather than a clear decision bug.

This first FEVER result suggests that the fact-checking branch is working as an evidence-based verifier, but future FEVER experiments should distinguish between open-web verification and FEVER's original closed-evidence setting.

### FEVER Batch: 30 Claims

A larger FEVER batch was run with `FEVER_30_RANDOM_SEED = 43`, using a balanced 30-claim sample:

| FEVER label | Sample count |
| --- | ---: |
| `SUPPORTS` | 10 |
| `REFUTES` | 10 |
| `NOT ENOUGH INFO` | 10 |

Overall metrics:

| Metric | Value |
| --- | ---: |
| Total claims | 30 |
| Accuracy | 56.7% |
| Macro-F1 | 0.549 |
| Flow-level stability issues | 0 |
| Mean raw evidence count | 7.83 |
| Mean selected evidence count | 2.77 |
| Mean runtime per claim | 7.33 seconds |

Per-class metrics:

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `SUPPORTS` | 0.562 | 0.900 | 0.692 |
| `REFUTES` | 0.800 | 0.400 | 0.533 |
| `NOT ENOUGH INFO` | 0.444 | 0.400 | 0.421 |

Status distribution:

| FEVER label | `success` | `insufficient_evidence` |
| --- | ---: | ---: |
| `SUPPORTS` | 9 | 1 |
| `REFUTES` | 6 | 4 |
| `NOT ENOUGH INFO` | 6 | 4 |

Confusion matrix:

| Gold label | Predicted `NOT ENOUGH INFO` | Predicted `REFUTES` | Predicted `SUPPORTS` |
| --- | ---: | ---: | ---: |
| `NOT ENOUGH INFO` | 4 | 1 | 5 |
| `REFUTES` | 4 | 4 | 2 |
| `SUPPORTS` | 1 | 0 | 9 |

This larger batch confirms that the pipeline is stable, but it also shows a clear prediction bias toward `SUPPORTS`. The system correctly identifies many supported claims, but it often predicts `SUPPORTS` for `REFUTES` and `NOT ENOUGH INFO` cases. This likely comes from open-web retrieval finding related true facts and Gemini treating them as direct support, even when the benchmark label depends on a more specific contradiction or the absence of FEVER evidence.

The `REFUTES` class has high precision but low recall in this batch. When the system predicts `REFUTES`, it is usually right, but several gold refutations are still marked `NOT ENOUGH INFO`, meaning the selected evidence did not provide enough direct contradiction. This suggests future work should focus on contradiction-sensitive retrieval and stance judgement, not only on retrieving more evidence.

The `NOT ENOUGH INFO` class remains difficult because the system uses open-web retrieval. Some NEI claims are predicted as `SUPPORTS` when the web provides plausible evidence for a broader or related statement. This should be discussed as an evaluation mismatch between FEVER's closed-evidence setup and the product's open-web fact-checking setting.

Wrong-case audit from this rerun:

| Error pattern | Count |
| --- | ---: |
| Gold `NOT ENOUGH INFO`, predicted `SUPPORTS` | 5 |
| Gold `NOT ENOUGH INFO`, predicted `REFUTES` | 1 |
| Gold `REFUTES`, predicted `NOT ENOUGH INFO` | 4 |
| Gold `REFUTES`, predicted `SUPPORTS` | 2 |
| Gold `SUPPORTS`, predicted `NOT ENOUGH INFO` | 1 |

The wrong-case evidence table showed that most `NOT ENOUGH INFO -> SUPPORTS` errors came from selected evidence judged as `supports`, often because open-web snippets contained real related facts. Most `REFUTES -> NOT ENOUGH INFO` errors came from evidence judged as background, indicating that retrieval/filtering did not surface a clean contradiction for those claims.

#### REFUTES -> NOT ENOUGH INFO Audit

The audit isolated four gold `REFUTES` cases that became `NOT ENOUGH INFO`:

| Claim | Status | Selected evidence pattern |
| --- | --- | --- |
| `The Quiet only stars Hillary Clinton.` | `insufficient_evidence` | Selected evidence was about Hillary Clinton, not the film `The Quiet` or its cast. |
| `Carey Hayes is illiterate.` | `insufficient_evidence` | Selected evidence was mostly unrelated `Hayes` or video material and did not establish literacy or biography. |
| `Andrew Moray led an uprising against occupation in 1397.` | `insufficient_evidence` | Only one weak/background historical source was selected. |
| `The New England Patriots failed to reach seven Super Bowls.` | `insufficient_evidence` | Evidence mentioned Patriots/Super Bowl context but did not cleanly settle the exact comparison. |

The raw retrieval audit for `The Quiet only stars Hillary Clinton.` showed that the search results were about Hillary Clinton rather than the film `The Quiet`. The filter selected the best available sources, but the candidate pool did not include the key cast evidence needed to refute the claim. This makes the failure mainly a retrieval/query disambiguation problem, not a Gemini stance problem.

This points to a future improvement: retrieval should be more entity-aware. Ambiguous claims may need search queries that preserve or infer the intended entity type, such as film title, person, organization, event, or work. This should be implemented as a general retrieval improvement, not by hardcoding FEVER examples.

#### Entity-Aware Search Query Adjustment

After auditing `The Quiet only stars Hillary Clinton.`, the backend was adjusted to separate the claim used for final judgement from the query used for retrieval.

Previously, claim preparation produced one `final_claim`, and that same text was used for both search and evidence judgement. This was too limited for ambiguous claims. In the `The Quiet` example, the claim itself was already unchanged, but search still focused on Hillary Clinton because she is the stronger web entity. The problem was therefore not a bad rewrite of the claim; it was that retrieval needed a more entity-aware query.

The current design keeps two fields:

| Field | Purpose |
| --- | --- |
| `claim_for_verdict` / `final_claim` | The meaning-preserving claim used for evidence filtering, Gemini judgement, stabilization, and scoring. |
| `search_query` | A retrieval-oriented query that may add neutral entity-type hints such as `film`, `cast`, `album`, `book`, `person`, or `organization` when the wording clearly implies them. |

For example, the verdict claim can remain:

`The Quiet only stars Hillary Clinton.`

while the search query can become:

`"The Quiet" film cast Hillary Clinton`

This is not benchmark hardcoding. It is a general retrieval improvement: the system still judges the original claim, but it searches with a query that is less likely to be dominated by the wrong entity. The implementation deliberately stays simple. Gemini is only asked to produce a JSON object with the verdict claim and search query; the backend keeps existing safety checks so negation, numbers, and meaning-bearing words are not changed in the verdict claim.

The FEVER notebook's raw retrieval audit cell was also updated so it prints both `claim_for_verdict` and `search_query`, then reproduces the backend behavior: retrieval uses `search_query`, while evidence filtering uses the verdict claim.

One follow-up issue was found after this adjustment. The older fallback logic assumed that the same text was used for both retrieval and filtering. After splitting `search_query` from the verdict claim, that assumption was no longer always true. A query such as `"The Quiet" film cast Hillary Clinton` may retrieve one evidence pool, while the original claim remains the target for filtering and final judgement.

The fallback behavior was therefore clarified:

1. Search first with the original atomic claim, using the normal retrieval result count.
2. Filter and rank retrieved evidence against the verdict claim.
3. If selected evidence is missing or weak, generate a rewritten relation-aware search query.
4. Search again with the rewritten query.
5. Filter the fallback evidence against the same verdict claim.
6. Keep whichever selection is stronger.

The system does not split the initial search budget evenly across multiple queries. This keeps the main path simple and avoids reducing the quality of the primary search. The rewritten query is only used as a fallback search path when the first evidence pool does not survive filtering well enough.

#### Why Entity-Split Search Was Not Used

During this adjustment, an entity-split search strategy was considered. For example, for the claim:

`The Quiet only stars Hillary Clinton.`

one possible approach would be to split the claim into entities and search `The Quiet` and `Hillary Clinton` separately.

This direction was also tested experimentally, and the batch accuracy dropped noticeably rather than improving. This made the approach unattractive both conceptually and empirically.

This approach was rejected because the system needs relation retrieval, not separate entity retrieval. Searching `Hillary Clinton` alone would likely return many high-quality but irrelevant results about Hillary Clinton, which was already the original failure pattern. Those results can make retrieval look successful while still failing to answer the actual relation in the claim: whether Hillary Clinton is the only star of the film `The Quiet`.

Searching `The Quiet` alone is also not a complete solution. It may retrieve a film page or cast list, but the retrieved text can contain many unrelated details. If the useful cast information appears later in a long page, it may be truncated before reaching the evidence filter and Gemini judgement. In that case, the relevant fact exists on the page, but the evidence item still does not expose it to the backend.

For these reasons, the entity-split option was set aside. The current design keeps the original claim for judgement and uses a relation-aware search query instead, such as:

`"The Quiet" film cast Hillary Clinton`

This query keeps the important entities together with the relation being checked. It is a general retrieval improvement rather than a benchmark-specific rule. If this query fails, future fallback should still remain relation-oriented, such as searching for the film cast, rather than searching the strongest entity by itself.

One implementation detail is worth noting. The main search path remains the original claim. Rewrite is treated as a fallback rescue step rather than the default first search. This means that even when the user-facing rewrite option is disabled, the backend can still use a rewritten query as a fallback after the original search fails or returns weak selected evidence. In report language, the rewrite step is no longer the primary search strategy; it is a stabilization-style retrieval fallback.

This also keeps the search budget simple. The backend does not split the first eight search results across the original claim and the rewritten query. It searches the original claim first. Only if the selected evidence is empty, weak, or too thin does it generate a rewritten relation-aware query and run a second retrieval pass. The two evidence pools are not blindly merged. Instead, each pool is filtered against the same verdict claim, and the stronger selected-evidence set is kept.

### Stabilization Trigger Fix

The stabilization rerun is intended to re-check borderline or low-confidence successful verdicts. A review found that confidence had been moved after stabilization so the final confidence would match the final score. That part was correct, but it also meant the stabilization trigger no longer had a useful confidence value to read.

The order was adjusted so the branch now calculates a provisional confidence immediately after the first truth-score aggregation. Stabilization then uses that provisional confidence to decide whether to rerun Gemini on the selected evidence. After stabilization, confidence is calculated again so the final output still describes the final score.

Current stabilization trigger:

| Condition | Stabilization rerun? |
| --- | --- |
| `use_selective_stabilization = False` | No |
| Provisional confidence is `low` | Yes |
| Provisional confidence is `medium` and score is near neutral | Yes |
| Provisional confidence is `medium` and score is near a verdict boundary | Yes |
| Provisional confidence is `high` | No |

This keeps stabilization selective. It is not a general second pass for every claim; it is only a second pass for cases where the first successful judgement looks unstable or borderline.

### FEVER 30 Reruns After Fallback-Only Rewrite

After changing rewrite into a fallback retrieval step, the same balanced FEVER 30 sample was rerun. There were two useful observations.

The first rerun showed an apparent improvement, but logs also showed that some claims caused three search calls. This came from overlapping fallback behavior: the service first searched the original claim, then tried a rewrite fallback, while `retrieve_evidence()` could still fall back internally from the rewritten query to the original claim. The fallback rewrite call was then adjusted to disable that internal original-claim fallback. This keeps retry behavior easier to explain: the service owns the original-claim search and the rewrite fallback, while retrieval only performs the requested search pass.

The notebook was rerun after that cleanup. The cleaner run is more useful as the current reference result.

Overall metrics:

| Metric | Earlier FEVER 30 | First fallback-only run | Clean fallback-only run |
| --- | ---: | ---: | ---: |
| Total claims | 30 | 30 | 30 |
| Accuracy | 56.7% | 63.3% | 56.7% |
| Macro-F1 | 0.549 | 0.624 | 0.547 |
| Flow-level stability issues | 0 | 0 | 0 |
| Mean raw evidence count | 7.83 | 7.83 | 7.80 |
| Mean selected evidence count | 2.77 | 2.87 | 2.97 |
| Mean runtime per claim | 7.33 seconds | 10.21 seconds | 8.93 seconds |

Clean-run per-class metrics:

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `SUPPORTS` | 0.533 | 0.800 | 0.640 |
| `REFUTES` | 0.750 | 0.300 | 0.429 |
| `NOT ENOUGH INFO` | 0.545 | 0.600 | 0.571 |

Clean-run confusion matrix:

| Gold label | Predicted `NOT ENOUGH INFO` | Predicted `REFUTES` | Predicted `SUPPORTS` |
| --- | ---: | ---: | ---: |
| `NOT ENOUGH INFO` | 6 | 1 | 3 |
| `REFUTES` | 3 | 3 | 4 |
| `SUPPORTS` | 2 | 0 | 8 |

The clean run shows that fallback-only rewrite is not a clear accuracy win on this 30-claim sample. It preserved stability and improved `NOT ENOUGH INFO` recall, but `REFUTES` recall fell to `0.300`. The fallback mechanism only actually helped trigger an alternate query in one case, and that case still ended as `insufficient_evidence`. This suggests that the fallback condition is currently too conservative to help cases where pre-Gemini evidence selection looks strong but Gemini later judges all sources as background.

The remaining `REFUTES -> NOT ENOUGH INFO` cases were:

| Claim | Status | Evidence pattern |
| --- | --- | --- |
| `The Quiet only stars Hillary Clinton.` | `insufficient_evidence` | Selected evidence still focused on Hillary Clinton rather than the film cast. |
| `Carey Hayes is illiterate.` | `insufficient_evidence` | Selected evidence was about unrelated or weak `Hayes` material. |
| `Andrew Moray led an uprising against occupation in 1397.` | `insufficient_evidence` | Selected evidence remained historical background without a clean contradiction. |

The `The Quiet` case revealed an important limitation of the fallback trigger. The pre-Gemini filter selected three evidence items and rated them as strong or usable, so the rewrite fallback did not trigger. Gemini then judged all selected evidence as `background`, producing `insufficient_evidence`. This means the failure is no longer just "no selected evidence"; it is a case where NLI-style filtering overestimates evidence quality, and the later Gemini stance step correctly rejects the selected evidence as background.

This led to a post-Gemini background fallback. If Gemini judges all selected evidence as background, and no rewrite fallback has already been used, the backend now attempts one relation-aware rewrite retrieval before returning `insufficient_evidence`. This keeps the current distinction between `no_evidence` and `insufficient_evidence`, while giving clear retrieval failures one more chance after Gemini exposes that the selected evidence was only topical background.

The motivating example is:

`The Quiet only stars Hillary Clinton.`

The pre-Gemini filter selected Hillary Clinton sources and rated them as strong or usable, so the normal rewrite fallback did not trigger. Gemini correctly judged those sources as background because they did not discuss the film cast. The post-Gemini fallback is designed for exactly this pattern: selected evidence exists, but the evidence is merely topical and cannot support a verdict.

The fallback layers were also simplified. `fact_check_service` now owns retrieval fallback decisions. `retrieval_service` no longer performs its own original-claim fallback; it only searches the query it is given, with optional oversampling retry when too few results are returned. This keeps the responsibilities clearer:

| Layer | Responsibility |
| --- | --- |
| `retrieval_service` | Run the requested search query and basic oversampling retry. |
| `fact_check_service` | Decide whether to try original claim search, rewrite fallback, or post-Gemini fallback. |
| `stabilization` | Rerun Gemini judgement only for unstable successful verdicts. |

### FEVER-Driven Backend Adjustments

The FEVER 10-claim batch led to two backend adjustments.

First, fact-checking claim preparation was simplified. The fact-checking branch should not perform heavy claim-validity judgement because factual claim filtering mainly belongs to the atomizer. The fact-checking branch now only applies a very light local guard against empty or extremely short inputs, then uses Gemini only for conservative query rewriting. It should not reject a claim because it looks absurd, implausible, or obviously false. This is important for benchmark claims such as `Benzodiazepine was globally the most prescribed dance move in 1977.`, which are nonsensical but still fact-checkable and refutable.

Second, the Gemini evidence-stance prompt was clarified with two general rules:

- For identity, occupation, authorship, membership, or category claims, evidence that states a matching role or category should count as direct signal rather than background.
- For spatial relation claims, evidence that something is inside, part of, within, or contained by something else should not be treated as support for adjacency, next-to, or bordering claims.

These are general evidence-interpretation rules rather than benchmark-specific hardcoding. They address the `Ang Lee is a writer.` and `Corsica is adjacent to Haute-Corse.` examples without adding case-specific logic.

After rerunning the FEVER 10-claim notebook, the claim-preparation change worked: the Benzodiazepine example moved from `invalid_request` to a correct `REFUTES` prediction. However, the Corsica example remained wrong, and the Ang Lee example remained an abstention because direct writing evidence was not retrieved. This is useful evidence that not every error should be handled by adding more prompt rules. Some issues may require better retrieval snippets, relation-aware scoring, or accepting that open-web verification differs from FEVER's closed evidence setup.

### Prompt Design Notes

Prompt changes should not become an endless list of one-off fixes. For this project, prompts should stay close to a small set of stable principles:

- Define the task boundary clearly: Gemini judges selected evidence against the claim, while the backend performs aggregation and final scoring.
- Keep output schemas strict and simple so the backend can parse them reliably.
- Add rules only when they describe a reusable reasoning distinction, not when they merely patch one sample.
- Prefer general semantic rules over case-specific examples. For example, distinguishing `part of` from `adjacent to` is a reusable spatial-relation rule; mentioning only Corsica and Haute-Corse would be hardcoding.
- Avoid asking Gemini to do work that belongs elsewhere. The atomizer should decide whether text contains factual claims; the fact-checking branch should only perform light guarding and evidence-based checking.
- If the prompt keeps growing because of repeated failures, consider moving that behavior into code, tests, or a separate model step instead of continuing to add more prompt text.

In practice, the prompt should act like a concise instruction sheet for evidence interpretation, not a collection of benchmark answers.

### Row 62: Clinton Wedding Claim

The claim about Bill and Hillary Clinton attending Donald Trump's last wedding behaved well. Retrieval found relevant evidence, filtering selected useful sources, and the final verdict matched the direction of the dataset label. This is an example of the pipeline working cleanly end to end.

### Row 466: Small Business Tax Claim

The claim about small businesses being taxed at higher rates than corporations produced `insufficient_evidence`.

The raw evidence was mostly general tax material, such as corporate tax pages, small business tax guides, pass-through business discussions, and broad policy analysis. Some sources were related, but they did not directly settle the exact claim. The filter selected the best available items, but those items were still weak or only partly relevant.

This looks mainly like a retrieval problem, not a Gemini problem. The system did not retrieve enough direct evidence for the specific claim, so the later stages had limited material to work with. The final `insufficient_evidence` result is reasonable.

### Row 745: Minimum Wage Claim

The claim about the minimum wage rising by `$2.35` in two years and `31 percent` behaved well as a diagnostic case.

Retrieval found a directly relevant PolitiFact-style source. Filtering kept the direct source and rejected generic minimum wage pages. Gemini then had enough targeted evidence to make a directional judgement. Even if the evidence sufficiency is marked as insufficient because only one strong source was selected, the pipeline behavior is still sensible: the system gives a directional verdict while acknowledging that the evidence base is thin.

### Row 763: Quran / Christians and Jews Claim

This case shows an important limitation of LIAR-style evaluation.

The claim says, roughly, that the Quran tells readers not to take Christians and Jews as friends. The selected evidence includes sources that directly quote or discuss the relevant verse, plus sources that explain historical or interpretive context.

At the literal evidence level, the system can reasonably decide that the claim is supported: there is text that appears to match the claim's wording. However, LIAR or PolitiFact may label this kind of claim as `half-true` because the political statement leaves out important context, translation nuance, historical setting, or interpretive limitations.

This is a task-semantics issue rather than a simple pipeline failure. The current fact-checking branch mostly asks:

> Does the selected evidence support or contradict the literal factual claim?

LIAR labels often ask a broader question:

> Is the public claim accurate, complete, and non-misleading in context?

This means the branch can be stable and internally reasonable while still disagreeing with LIAR on context-heavy claims. To model this better later, the system may need an explicit context or misleadingness dimension, rather than forcing every case into literal support versus contradiction.

## Current Engineering Judgement

The fact-checking branch is now close to a complete first production-style version of the core mechanism:

- It has a clear retrieval-to-verdict pipeline.
- It separates no-evidence cases from insufficient-evidence cases.
- It avoids sending empty evidence into Gemini.
- It returns a complete branch summary from the fact-checking service itself.
- It calculates confidence after stabilization so confidence matches the final score.
- It produces schema-shaped outputs that can be tested directly.

The remaining work is less about inventing the wheel and more about validating and tuning it:

- Improve retrieval for hard political claims where the first search results are too generic.
- Test on FEVER after LIAR stability is acceptable.
- Decide whether contextual misleadingness should be a separate dimension.
- Review how `mixed` or context-heavy evidence should affect the final truth score.
- Keep notebook tests focused on diagnosing retrieval, filtering, Gemini judgement, and aggregation separately.

## Work Log

This section records the development reasoning behind the current fact-checking branch. It is intended as report material, not just implementation notes.

### Backend Flow Cleanup

Initial review found that the fact-checking branch had several scattered responsibilities. The service returned factual-claim details, while the orchestrator later filled in branch-level `truth_score`, `verdict`, and `explanation`. This made direct fact-checking tests produce schema-shaped but semantically incomplete outputs.

The branch summary was moved into the fact-checking service itself. This made `analyze_fact_check_claims()` responsible for returning a complete `EachFactChecking` object, whether it is called directly from notebooks or through the orchestrator.

Another issue was confidence calculation. Confidence was originally calculated before stabilization, meaning the confidence could describe the pre-stabilized score rather than the final score. The order was changed so stabilization happens before confidence and verdict mapping.

### Evidence Status Semantics

The branch originally blurred cases where no evidence was selected with cases where evidence existed but was weak. This was cleaned into two separate statuses:

- `no_evidence`: no selected evidence survived retrieval/filtering, so Gemini should not be called.
- `insufficient_evidence`: selected evidence exists and can be inspected, but it is not strong enough for a confident final verdict.

This distinction became important in notebooks because it made failure modes easier to diagnose. If selected evidence is empty, the problem is upstream of Gemini. If evidence exists but sufficiency is low, the issue may be evidence quality, source selection, or Gemini stance judgement.

### LIAR Stability Stage

LIAR was used first as a stability and diagnostic test, not as the main performance benchmark. The dataset is difficult for this system because LIAR labels often include political context, partial truth, speaker context, and rhetorical framing.

The LIAR notebook produced stable backend behavior: schema outputs were complete, `no_evidence` and `insufficient_evidence` were separated, and empty evidence was not sent into Gemini. The seed-42 LIAR batch reached `57.1%` relaxed judged-only accuracy, but this was treated as a diagnostic result rather than final fact-checking performance.

The LIAR stage also clarified report methodology: strict matching and relaxed matching should both be shown because the system's verdict scale does not exactly match LIAR's six-label scale.

### FEVER Stage

FEVER was introduced after LIAR because it is closer to evidence-based fact checking. FEVER's labels map more naturally to the backend:

- `SUPPORTS` maps from `True` and `Mostly True`.
- `REFUTES` maps from `False` and `Mostly False`.
- `NOT ENOUGH INFO` maps from `Neutral`, `no_evidence`, and `insufficient_evidence`.

The first 10-claim FEVER run reached `60.0%` accuracy and `0.583` Macro-F1. It revealed one important backend issue: the claim-preparation step rejected an absurd but concrete claim as `invalid_request`.

The claim-preparation logic was then simplified. Fact-checking now only applies a light local guard against empty or extremely short inputs. It does not reject a claim because the claim looks absurd, implausible, or obviously false. This keeps the fact-checking branch focused on evidence checking and leaves factual-claim detection mainly to the atomizer.

After rerunning the same FEVER batch, the result improved to `70.0%` accuracy and `0.669` Macro-F1 with `0` stability issues. The previously rejected Benzodiazepine claim was correctly processed and predicted as `REFUTES`.

A larger 30-claim FEVER batch was then run with seed 43. After a full runtime restart and rerun, it achieved `56.7%` accuracy and `0.549` Macro-F1 with `0` stability issues. This confirmed backend stability on a larger sample, while exposing a prediction bias toward `SUPPORTS` and weaker recall on `REFUTES`.

### Remaining FEVER Findings

Two notable issues remain after the rerun.

The Corsica example is still predicted as `SUPPORTS` even though the evidence mainly says Haute-Corse is part of Corsica. This shows a relation-understanding problem: part-whole evidence should not support an adjacency claim.

The Ang Lee example is still predicted as `NOT ENOUGH INFO`. Evidence retrieved for Ang Lee focuses on filmmaker/director identity rather than direct writing evidence. Under the current pipeline responsibility split, this is acceptable: retrieval/filtering did not surface direct support, so the final insufficient-evidence result is a reasonable abstention.

The Uganda example is different. The system predicts `REFUTES` because open-web evidence says Uganda was a British protectorate, not ruled by the French. FEVER marks it `NOT ENOUGH INFO`, but this likely reflects FEVER's closed-evidence setting rather than a wrong open-web judgement.

### Prompt Iteration Principle

The FEVER rerun showed that prompt changes can help but should not become the only solution. The project should only add prompt rules when they express general reasoning principles. Case-specific fixes should be avoided. If a failure keeps recurring, it may need code-level handling, better retrieval, or a clearer evaluation distinction rather than more prompt text.

### FEVER 30 Rerun After Fallback Cleanup

After centralising fallback handling in the fact-checking service, the FEVER 30 notebook was rerun with seed 43.

| Metric | Value |
|---|---:|
| Claims | 30 |
| Accuracy | 46.67% |
| Macro-F1 | 0.4565 |
| Stability issues | 0 |
| Mean raw evidence count | 7.7333 |
| Mean selected evidence count | 2.9333 |
| Mean runtime | 10.3647 seconds |
| Fallback used | 2 / 30 |

Per-class results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| SUPPORTS | 0.467 | 0.700 | 0.560 |
| REFUTES | 0.750 | 0.300 | 0.429 |
| NOT ENOUGH INFO | 0.364 | 0.400 | 0.381 |

The run still had no schema or stability failures, so the backend flow remained structurally stable. The lower score mainly reflects prediction quality, not broken output structure.

The main error pattern was:

- `NOT ENOUGH INFO` often became `SUPPORTS` when open-web evidence contained plausible supporting background.
- `REFUTES` often became `NOT ENOUGH INFO` when retrieval did not find direct contradiction.
- Some `SUPPORTS` cases became `NOT ENOUGH INFO` when retrieval found related entity pages but missed the specific relation needed by the claim.

Two fallback cases appeared in this run:

- `The Fly is a film from the United States.`
- `Jimi Hendrix was trained for surgical operations.`

This shows that fallback is now being recorded in the notebook, but it is not over-active. Only 2 out of 30 claims used it.

The two fallback cases were:

| Row index | FEVER id | Gold label | Final status | Final verdict | Fallback search query |
|---:|---:|---|---|---|---|
| 1197 | 182042 | SUPPORTS | insufficient_evidence | None | `"The Fly" film United States` |
| 2128 | 90130 | NOT ENOUGH INFO | insufficient_evidence | None | `Jimi Hendrix surgical training` |

Both cases ended as `insufficient_evidence`, so fallback did not artificially force a verdict. This is a useful stability point: rewrite fallback can change the search query, but the later Gemini/evidence-sufficiency stage still controls whether the selected evidence is strong enough for a final decision.

The `The Quiet only stars Hillary Clinton.` case remains useful as a retrieval/filtering diagnostic. The raw search was pulled toward Hillary Clinton pages rather than the film cast relation. The filter then selected three high-quality-looking sources, but Gemini judged them all as background. The final result was `insufficient_evidence`, which is semantically reasonable given the evidence actually shown to Gemini, but it still misses the FEVER `REFUTES` label.

This case shows a remaining limitation in fallback triggering. Before Gemini, the system sees three selected evidence items and therefore does not treat retrieval as weak. After Gemini, the evidence is revealed to be background-only. This supports the later decision to add a post-Gemini background fallback: if all selected evidence becomes background, the service should get one more chance to search with the rewritten query before returning `insufficient_evidence`.

### Relation and Condition Aware Evidence Selection

The next backend adjustment shifted the main focus away from fallback and toward evidence quality. The key problem was that the system was often selecting pages that were topically related to one entity in the claim, but did not actually cover the relation or condition that made the claim checkable.

For example, `The Quiet only stars Hillary Clinton.` was not mainly a fallback problem. The first search returned Hillary Clinton pages. Those pages were related to one named entity, but they did not cover the film title or the cast/star relation. Gemini correctly treated them as background, but by that point the selected evidence was already poor.

The filter was changed conservatively. It does not use a hand-written relation or stopword dictionary. Instead, it keeps two simple checks:

- whether evidence covers explicit anchors in the claim, such as quoted titles, people, organisations, or capitalised names;
- whether numeric or comparison details are compatible, using the existing number-checking logic.

Long retrieved pages are now reduced to the most claim-relevant passage before evidence scoring. This means the downstream NLI model and Gemini see a shorter passage that is more likely to contain the actual evidence, rather than a whole page of entity background.

The intended behaviour is:

- evidence about only Hillary Clinton should not be strong evidence for a claim about who stars in `The Quiet`;
- evidence listing the cast of `The Quiet` should be easier for the NLI model and Gemini to use because the passage extraction step can surface the cast-related passage;
- evidence missing a key time, place, number, negation, comparison, or scope condition should be handled mainly by the NLI model and Gemini prompt, not by brittle hand-written word lists.

The Gemini prompt was also updated to use the same product logic. It now explicitly asks the model to judge whether evidence addresses the same entity relation under the same important conditions, rather than treating general topical relevance as enough.

This is a broader project direction: the fact-checking branch should not simply retrieve related web pages. It should retrieve and select evidence that can actually decide the claim.

Post-Gemini all-background cases were also kept conservative. If Gemini judges all selected evidence as background, the service first tries rewrite fallback. If no usable fallback evidence exists, it reruns the Gemini judgment once on the same selected evidence. This is not intended to raise accuracy by force; it is a small guard against one unstable all-background response while preserving the rule that evidence sufficiency is decided after evidence is actually judged.

After review, the filter implementation was deliberately simplified again. A draft version used hand-written stopword, condition, and related-term lists, but this was rejected because it looked like benchmark-specific patching and made the code less natural. The current filter keeps only simple anchor coverage, existing number/comparison checks, NLI relevance, and passage extraction. Relation and condition understanding is mainly handled by the NLI model and Gemini prompt rather than brittle word lists.

### FEVER 30 After NLI Filter Cleanup

After simplifying `nli_filter.py` and rerunning the FEVER 30 batch with seed 43, the result improved compared with the previous 30-claim run.

| Metric | Value |
|---|---:|
| Claims | 30 |
| Accuracy | 60.00% |
| Macro-F1 | 0.5751 |
| Stability issues | 0 |
| Mean raw evidence count | 7.8333 |
| Mean selected evidence count | 2.9667 |
| Mean runtime | 11.7533 seconds |
| Fallback used | 9 / 30 |

Per-class results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| SUPPORTS | 0.562 | 0.900 | 0.692 |
| REFUTES | 1.000 | 0.300 | 0.462 |
| NOT ENOUGH INFO | 0.545 | 0.600 | 0.571 |

This run reduced the number of incorrect predictions from 16 to 12 while keeping stability issues at zero. The main remaining weakness is still low `REFUTES` recall: the system is careful when it predicts `REFUTES`, but it misses many refuting cases and maps them to `SUPPORTS` or `NOT ENOUGH INFO`.

The `The Quiet only stars Hillary Clinton.` case still fails. The selected evidence remains Hillary Clinton-related background rather than film-cast evidence. The filter now rejects some weaker Hillary Clinton pages as missing claim anchors, but NLI still gives very high relevance to several Hillary Clinton pages, so they can still enter top-k. This confirms that the remaining limitation is primarily retrieval/evidence selection, not the final aggregation logic.

Fallback was used more often in this run. This helped some cases but should not become the main mechanism. The core direction remains improving first-pass evidence quality rather than making fallback more aggressive.

The follow-up audit cell compared the original query with the final batch query. For `The Quiet only stars Hillary Clinton.`, the final query was `The Quiet stars Hillary Clinton`, which dropped the scope word `only` and still retrieved Hillary Clinton background pages rather than film-cast evidence. Two cleanup changes followed:

- search query validation now preserves the same meaning-bearing words as the original claim, so fallback rewrite cannot silently drop words such as `only`;
- anchor matching now checks explicit anchor phrases instead of loose token overlap, so a page containing the ordinary word `quiet` no longer partially matches the title `The Quiet`.

### Claim Frame Retrieval Direction

Open-web fact-checking systems commonly separate atomic claim extraction, evidence retrieval, and claim verification. More recent retrieval-augmented systems also use structured query generation rather than relying on a single raw claim search. The backend now follows this direction in a simple way.

Claim preparation now returns a small claim frame:

- `claim_for_verdict`
- `search_query`
- `main_entities`
- `relation`
- `constraints`

These fields must come from the claim itself. They are not allowed to add background knowledge or infer the answer.

The fact-checking service now tries a small set of search queries for each claim:

- the original atomic claim;
- Gemini's retrieval-friendly search query;
- a frame-based query built from entities, relation, and constraints.

The service then chooses the evidence selection with the strongest selected evidence. This moves the main retrieval logic away from a single brittle search string while keeping the code readable and avoiding domain-specific rules. The notebook records `retrieval_queries_tried`, `claim_entities`, `claim_relation`, and `claim_constraints` so failures can be audited directly.

### FEVER Small Batch After Claim Frame Retrieval

After adding claim-frame retrieval, a smaller FEVER notebook was created for quick regression testing before rerunning the full 30-claim batch. The small batch uses seed 43 and samples 3 claims from each FEVER class.

| Metric | Value |
|---|---:|
| Claims | 9 |
| Accuracy | 77.78% |
| Macro-F1 | 0.7746 |
| Stability issues | 0 |
| Mean raw evidence count | 7.7778 |
| Mean selected evidence count | 2.7778 |
| Mean runtime | 19.6000 seconds |

Per-class results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| SUPPORTS | 0.750 | 1.000 | 0.857 |
| REFUTES | 1.000 | 0.667 | 0.800 |
| NOT ENOUGH INFO | 0.667 | 0.667 | 0.667 |

This run is not large enough to be treated as a final benchmark result, but it is useful as a regression check. The schema was correct, all stability checks passed, and relation/condition examples such as bonobo population, Andrew Moray's opposition to Edward I, Hollow Man 2, and Uganda under British rule were handled correctly.

Two failures remain informative:

| Claim | FEVER label | Backend result | Main issue |
|---|---|---|---|
| `Carey Hayes is illiterate.` | REFUTES | NOT ENOUGH INFO | Search results mention similar names or generic illiteracy, but not enough usable evidence about Carey Hayes himself. |
| `The Catcher in the Rye examines themes such as innocence and connection.` | NOT ENOUGH INFO | SUPPORTS | Open-web evidence supports the theme claim, so this is likely a benchmark/open-web evidence mismatch rather than a clear backend failure. |

An earlier run of the same small batch produced a `degraded` result for the Carey Hayes claim because Gemini returned JSON containing an invalid backslash escape. The backend now parses Gemini JSON through a shared lightweight parser that strips code fences and retries after escaping unsafe bare backslashes. After this fix, the same claim becomes normal `insufficient_evidence`, which is the expected behavior when retrieval finds weak topical evidence but not usable evidence about the claim subject.

The Carey Hayes case is also a useful limitation example. It is not an obvious backend flow bug: the system searched the claim, found weak topical or name-similar material, and correctly avoided forcing a verdict. Returning `insufficient_evidence` is acceptable for a fact-checking product when the selected evidence does not directly establish or refute the claim.

To turn this case into a confident `REFUTES` result, the system would need a stronger retrieval planner. Instead of searching mainly for the literal claim phrase, it would need to infer a better evidence-seeking route, such as checking Carey Hayes's public biography, profession, writing credits, interviews, education, or other reliable profile sources. This is a next-stage retrieval-intelligence problem rather than a stability problem in the current backend flow.

### FEVER 30 After Claim Frame Retrieval

After the small-batch regression looked stable, the same runtime was used to run a 30-claim balanced FEVER batch. This used the current claim-frame retrieval backend and the same seed-based sample setup.

| Metric | Value |
|---|---:|
| Claims | 30 |
| Accuracy | 66.67% |
| Macro-F1 | 0.6572 |
| Stability issues | 0 |
| Mean raw evidence count | 7.9000 |
| Mean selected evidence count | 2.9667 |
| Mean runtime | 27.6163 seconds |
| Fallback used | 9 / 30 |

Per-class results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| SUPPORTS | 0.667 | 0.800 | 0.727 |
| REFUTES | 0.800 | 0.800 | 0.800 |
| NOT ENOUGH INFO | 0.500 | 0.400 | 0.444 |

Confusion matrix:

| FEVER label | Predicted NEI | Predicted REFUTES | Predicted SUPPORTS |
|---|---:|---:|---:|
| NOT ENOUGH INFO | 4 | 2 | 4 |
| REFUTES | 2 | 8 | 0 |
| SUPPORTS | 2 | 0 | 8 |

This is an improvement over the previous FEVER 30 run after NLI cleanup, where accuracy was 60.00% and macro-F1 was 0.5751. The main improvement is that `REFUTES` recall increased from 0.300 to 0.800 while stability issues stayed at zero.

The remaining errors are mostly explainable:

- Some FEVER `NOT ENOUGH INFO` cases become `SUPPORTS` or `REFUTES` because open-web evidence provides direct support or contradiction. Examples include `Corsica is adjacent to Haute-Corse.`, `Uganda was ruled by the French.`, and `Ted Cruz is a North American.` This is partly a mismatch between FEVER's closed evidence setting and open-web retrieval.
- Some true or false claims still become `insufficient_evidence` when retrieval does not find evidence that directly covers the claim relation. Examples include `Stephen King wrote 7 novels under a pen name.`, `Ang Lee is a writer.`, `Benzodiazepine was globally the most prescribed dance move in 1977.`, and `Bonobos live south of the Ganges River.`
- The `Ang Lee is a writer.` case is now better behaved than earlier runs: it no longer confidently contradicts the claim from filmmaker/director evidence. It returns `insufficient_evidence`, which is conservative when selected evidence does not directly establish the writer relation.

Overall, this FEVER 30 run supports treating the current fact-checking backend as a stable stage. The remaining ceiling is mainly retrieval intelligence: deciding what kind of evidence should be searched for when the literal claim query does not find direct evidence.

### Strict FEVER vs Open-Web-Adjusted Reading

The strict FEVER result should still be reported because it is the benchmark score under the dataset labels:

| Evaluation view | Accuracy | Macro-F1 | Notes |
|---|---:|---:|---|
| Strict FEVER | 66.67% | 0.6572 | Uses the original FEVER labels directly. |
| Open-web-adjusted | 86.67% | about 0.83 | Treats FEVER `NOT ENOUGH INFO` cases as acceptable when open-web retrieval found direct support or contradiction. |
| Mismatch-excluded | 83.33% | about 0.815 | Removes the open-web/FEVER mismatch cases from the denominator. |

The adjusted view is not meant to replace the strict benchmark score. It is included because this project is an open-web fact-checking product, while FEVER labels are based on a closed evidence setting. A FEVER claim can be labelled `NOT ENOUGH INFO` even when the wider web contains evidence that supports or refutes it.

In this run, six incorrect strict-FEVER predictions are better interpreted as open-web/closed-label mismatch cases:

| Claim | FEVER label | Backend prediction | Why it is a mismatch |
|---|---|---|---|
| `Manchester United has had two home fields.` | NOT ENOUGH INFO | SUPPORTS | Open-web evidence can list Manchester United's home grounds. |
| `John Goodman starred in a music video.` | NOT ENOUGH INFO | SUPPORTS | Open-web evidence can directly discuss Goodman appearing in a music video. |
| `Uganda was ruled by the French.` | NOT ENOUGH INFO | REFUTES | Open-web evidence indicates Uganda was under British rule, which contradicts the claim. |
| `Ted Cruz is a North American.` | NOT ENOUGH INFO | SUPPORTS | Public biographical/geographical evidence supports this broad claim. |
| `Corsica is adjacent to Haute-Corse.` | NOT ENOUGH INFO | SUPPORTS | Open-web geographical evidence supports the adjacency relation. |
| `Andrew Moray led an educational program for farmers in 1297.` | NOT ENOUGH INFO | REFUTES | Open-web historical evidence about Andrew Moray in 1297 points to military/political activity, not an educational farming program. |

For example, `Uganda was ruled by the French.` is counted wrong under strict FEVER because the gold label is `NOT ENOUGH INFO`. However, an open-web fact-checking system can reasonably return `REFUTES` if retrieved evidence shows Uganda was a British protectorate rather than ruled by France. This is a product-evaluation mismatch rather than a normal backend error.

### Fact-Checking Backend Cleanup

After the FEVER small batch and FEVER 30 runs, the fact-checking backend was treated as a stable stage and cleaned toward a more final implementation style. This cleanup did not add new fact-checking behavior. It focused on reducing experiment residue and keeping the code easier to explain.

The main cleanup points were:

- the truth-score-to-verdict boundary rules were centralized in `decision_utils.py` so the branch summary and claim-level verdicts use the same rule;
- repeated fallback judgment-reset code in `fact_check_service.py` was consolidated into a single simple path;
- an obsolete `INVALID_CLAIM` branch was removed from Gemini claim preparation because the current prompt no longer emits that value;
- deprecated compatibility and experiment output files under `fact_checking` were removed so the entrypoint is clearly `fact_check_service.py`.

This keeps the current fact-checking branch focused on the main product flow: prepare claim, retrieve evidence, select evidence, judge evidence, aggregate verdict, and return a schema-stable result.

The scoring step was also made slightly more natural for product output. Gemini now returns an `overall_truth_score` based on the selected evidence, and the backend stores it as `metadata.gemini_truth_score`. The final `truth_score` is still mainly controlled by the backend's source-level stance aggregation, but the Gemini score is mixed in with a small weight. This avoids every confident result landing on the same fixed values such as `0.9000` or `0.1000`, while keeping the final decision rule transparent.

The NLI filtering file was then cleaned more aggressively because it still looked like an experimental tuning file. The public function stayed the same: `filter_top_evidence()` still receives a claim and retrieved evidence, and returns selected evidence plus optional debug information. Internally, the code was reorganized into a more linear flow:

- normalize text and extract claim anchors;
- score number compatibility without using claim-specific comparison rules;
- choose the best passage from long retrieved pages;
- score NLI relevance;
- combine these signals into a keep/reject decision;
- sort retained evidence by selection priority.

This keeps the core behavior understandable without introducing a heavier retrieval framework. Earlier comparison-pattern rules were removed during cleanup because they made the filter look like a collection of hardcoded patches. The current version keeps only a minimal number signal and leaves finer relation judgment to the evidence judgment step.

### Conservative Verdict Policy vs Higher Strict-FEVER Accuracy

The FEVER small-batch run after the relation-aware Gemini prompt showed an important product trade-off. The claim `Bonaire was excluded from the Netherlands Antilles until 2010.` was corrected after the prompt emphasized relation checking: evidence stating that Bonaire was `part of` the Netherlands Antilles until 2010 should contradict, not support, a claim saying it was `excluded from` the Netherlands Antilles until 2010.

The same run also produced a useful borderline example: `The New England Patriots failed to reach seven Super Bowls.` The system retrieved evidence where one selected source contradicted the claim, while the remaining selected sources were background. A more aggressive decision policy could classify this as `REFUTES` based on the one directional source. That would improve strict FEVER accuracy for this case.

The current backend intentionally does not make every one-source directional signal into a confident final verdict. It still considers evidence sufficiency, source quality, decision confidence, and selective stabilization. In this case, the final score was stabilized to `Neutral` because the evidence set was not strongly consistent across selected sources.

This is a deliberate product choice. The system could achieve slightly better strict benchmark accuracy by trusting any single usable support/refute signal more aggressively. However, for an open-web fact-checking product, a conservative verdict is easier to justify: users should be able to distinguish between "there is one useful directional source" and "the evidence set is strong enough for a confident verdict." Evidence sufficiency remains visible in the output so users can judge whether the verdict is well supported.

### Backend Aggregate Cleanup

A later backend review found that the product-level aggregate result was still mostly a text-pattern aggregate. `overall_risk_score`, `overall_risk_level`, and `overall_risk_confidence` were being calculated from the text-pattern branch only, while the fact-checking branch was returned separately.

This was inconsistent with the intended dual-dimension pipeline. If the text-pattern model sees low rhetorical risk but evidence strongly refutes the claim, the product should not present the overall risk as low simply because the wording looks calm.

The aggregate logic was therefore adjusted in a simple way:

- successful text-pattern results contribute their normal risk score;
- successful fact-checking claims contribute factual risk as `1 - truth_score`;
- insufficient or no-evidence fact-checking results do not force factual risk, because lack of evidence is not the same as falsehood;
- the overall risk level is derived from the combined score using low, medium, and high bands.

This keeps the aggregate simple and reportable while making it match the product idea more closely: text-pattern risk and evidence-based factual risk are both dimensions of the final output.

## Report-Friendly Summary

The fact-checking component uses retrieval-augmented evidence checking rather than relying on a language model alone. It first searches for external evidence, filters candidate sources, asks a language model to judge the claim against selected evidence, and then aggregates the result into a structured verdict. The system distinguishes between cases where no evidence is available and cases where evidence exists but is not sufficient. This distinction improves interpretability because users can see whether the limitation comes from retrieval failure or weak evidence.

Initial LIAR testing suggests that the core pipeline is functioning, but LIAR is not a pure factual entailment benchmark. Some LIAR labels depend on missing context or rhetorical framing. Therefore, disagreement with LIAR does not always indicate a broken fact-checking pipeline. It may indicate that the system is checking literal factual support while LIAR is also judging contextual misleadingness.
