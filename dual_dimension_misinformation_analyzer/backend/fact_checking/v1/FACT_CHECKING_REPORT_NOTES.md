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

## Report-Friendly Summary

The fact-checking component uses retrieval-augmented evidence checking rather than relying on a language model alone. It first searches for external evidence, filters candidate sources, asks a language model to judge the claim against selected evidence, and then aggregates the result into a structured verdict. The system distinguishes between cases where no evidence is available and cases where evidence exists but is not sufficient. This distinction improves interpretability because users can see whether the limitation comes from retrieval failure or weak evidence.

Initial LIAR testing suggests that the core pipeline is functioning, but LIAR is not a pure factual entailment benchmark. Some LIAR labels depend on missing context or rhetorical framing. Therefore, disagreement with LIAR does not always indicate a broken fact-checking pipeline. It may indicate that the system is checking literal factual support while LIAR is also judging contextual misleadingness.
