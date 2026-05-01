# Atomizer Report Notes

This file records the design path of the atomizer, especially decisions that may be useful later in the report. The point is not only to describe the final design, but to show what we tried, what failed, and why the current direction was chosen.

## From Sentence Splitting To Claim Preparation

At first, we treated the atomizer mainly as a sentence splitter: user input would be split into sentence-like units, and each unit would then be passed to the text-pattern and fact-checking branches.

This worked for short single-claim input, but it was not enough for real articles. In long text, many sentences contain pronouns, quotations, attribution, repeated claims, or background statements. A sentence like `He said it would help workers and small businesses.` is not a useful fact-checking query on its own, because `he` and `it` are unresolved. Sending this directly into retrieval makes the downstream fact-checking branch weaker even if the fact-checking logic itself is correct.

So the role of the atomizer was changed:

1. It still produces text units for the text-pattern branch.
2. It also prepares standalone factual claims for fact-checking.
3. It should resolve clear local references when the nearby context is enough.
4. It should ignore subjective, vague, or non-checkable sentences.

This moved part of the responsibility upstream. The fact-checking branch should not have to guess who `he` is or what `it` means. Its job is to retrieve evidence and judge a clear claim.

## Pronoun Resolution Experiment

We first considered keeping every claim as close as possible to the original sentence. That avoided changing the user's wording, but it created weak retrieval queries.

Example:

```text
Donald Trump praised the tax bill during a speech in Ohio.
He said it would help workers and small businesses.
```

If the second sentence is sent as `He said it would help workers and small businesses.`, the claim is not standalone. The search branch has no reliable subject or object.

After adding nearby context, the atomizer can output:

```text
Donald Trump said the tax bill would help workers and small businesses.
```

This is better for retrieval because it keeps the factual relationship clear: subject, relation, and object are all present. This confirmed that the atomizer should resolve clear local references such as `he`, `they`, `it`, and `this`, as long as the context is unambiguous.

The limitation is that context resolution can still be wrong. If the wrong subject is carried forward, the error can spread into later claims. For that reason, the rule is conservative: if the reference cannot be clearly resolved, the atomizer should ignore the claim rather than create a misleading one.

## Long-Text Batch Design

We then tried to use the short-text atomizer on a long article. This was not a valid long-text test, because the short path only processes the first group of sentences. It could tell us whether the prompt worked locally, but it could not tell us whether the system handled a full article.

The long-text design was changed to:

1. split the full article into sentences;
2. process target sentences in batches;
3. pass a small number of previous sentences as context;
4. maintain a rolling context note between batches;
5. output claims only for the current target batch;
6. merge the batch outputs;
7. deduplicate repeated claims.

The current test uses target batches of 8 sentences with 2 previous sentences as context. The context sentences are not counted as new target sentences. This keeps the pipeline simple while reducing the chance that a pronoun at the start of a batch loses its referent.

We also considered using one separate LLM call to read the whole article and create an article-level entity state. That could help with long-range references, but it adds another layer of LLM logic and makes errors harder to trace. Instead, the current design asks Gemini to return a small `next_context_note` as part of the same batch call. This avoids an extra model call while still carrying useful context forward.

## Controlled Long-Text Test

We created a controlled long-text example instead of relying only on ISOT articles. This was important because ISOT can be noisy, copied from web pages, and not always ideal for testing atomizer logic. A controlled example lets us test specific cases:

- pronoun resolution;
- quotation handling;
- subjective sentence filtering;
- near-duplicate claims;
- conditions such as timing and degree;
- batch-boundary context.

The controlled example includes:

```text
Donald Trump praised the tax bill during a speech in Ohio.
He said it would help workers and small businesses.
...
Trump said, "The stock market had its best year."
...
The tax bill passed the Senate.
The Senate passed the tax bill.
```

Current output shows clear improvement:

- `He said it would help workers...` becomes `Donald Trump said the tax bill would help workers and small businesses.`
- `This helped...` becomes `The passage of the tax bill helped the president claim a major political victory.`
- the quotation becomes `The stock market had its best year.`
- `I think the whole debate was ridiculous.` is ignored.
- `almost finished` and `finished` are kept as different claims.

This suggests that the atomizer is now moving in the right direction for long text.

## Reporting And Attribution

We discussed whether a sentence like `Critics said the bill mainly helped large corporations.` should be checked as:

```text
Critics said the bill mainly helped large corporations.
```

or as:

```text
The bill mainly helped large corporations.
```

The current product direction is that the fact-checking branch should usually check the embedded factual proposition. The frontend can still show the original sentence, so users can see the reporting context. This means the atomizer should usually remove ordinary reporting frames such as `X said that...` when the main factual content is inside the quote or reported clause.

However, attribution should be preserved when the attribution itself is the factual claim. For example, `In 2024, officials announced the policy.` is partly about the announcement event, so removing the attribution may change the meaning.

Current tests show this is not fully solved yet. Some outputs still preserve reporting frames:

```text
Critics said the bill mainly helped large corporations.
Several economists said the corporate tax cuts would increase the federal deficit.
Officials said the project was finished.
```

This is not a downstream fact-checking bug. It is an atomizer boundary issue: the system has not always converted reported factual content into the cleanest checkable claim.

The prompt was then tightened around this boundary. The new instruction is not to list many reporting verbs as hard rules, but to express the product logic directly: ordinary reporting and quotation should expose the embedded factual proposition, while announcement, belief, posting, writing, and historical-thinking events should keep attribution when that event is what the sentence is actually claiming.

This is still a prompt-level solution, not a separate rule engine. That keeps the code simple, but it means local tests must continue checking examples where attribution should be removed and examples where it should remain.

## Deduplication Experiment

We expected simple duplicate claims to be removed. Exact duplicates are easy, but semantic duplicates are harder.

Example:

```text
The tax bill passed the Senate.
The Senate passed the tax bill.
```

These mean the same thing, but the current deduplication may keep both because the surface wording is different.

The better long-term idea is to deduplicate by subject, relation, and conditions:

- subject: `the tax bill`;
- relation: `passed`;
- object: `the Senate`;
- conditions: time, place, degree, modality, negation.

But this should stay simple. We do not want a heavy NLP layer just for deduplication. For now, exact deduplication is safe, and semantic deduplication can be added only if repeated claims become a major product problem.

One important warning: not all similar claims are duplicates. `The project was almost finished.` and `The project was finished.` must be kept separate because `almost` changes the meaning.

## Conditions And Time Context

We also identified a condition-propagation risk. Long articles often shift between different times, places, or legal contexts.

For example, an article may discuss a 2026 event, then mention a law passed in 2024, then use `it` in the next sentence. The atomizer should not blindly carry the most recent year or the article's main year into every later claim.

The current rule should be:

1. the target sentence's own conditions have priority;
2. context conditions can be reused only when the target sentence clearly refers back to them;
3. if the condition is ambiguous, the atomizer should keep the claim conservative or ignore it.

This matters because fact-checking relies heavily on conditions. A claim can become false or misleading if the wrong year, location, comparison, or degree is attached.

This is a next test target. The current controlled example tests pronouns and duplicates, but it does not yet test condition inheritance deeply enough.

The local test set now includes a condition-shift example:

```text
In 2026, the government announced a new policy.
A related law was passed in 2024.
It gave agencies new powers.
```

The expected behavior is that the atomizer resolves `It` to the related law, while not blindly attaching `2026` to the later law claim.

## ISOT Testing Decision

We initially tried using ISOT `Fake.csv` as a long-text atomizer test. That was useful as a stress test, but it was not the best first benchmark.

Problems found:

- copied article text can be messy;
- some examples contain web artifacts or missing punctuation;
- fake-news articles often contain many background claims, quotations, and reactions;
- checking every factual sentence makes the result page too long and dilutes the overall verdict.

Because of this, ISOT Fake articles should not be the first source for atomizer correctness. The better order is:

1. controlled long-text examples;
2. cleaner real articles, such as ISOT True articles if needed;
3. noisy Fake articles only as stress tests.

Dirty article cleaning is outside the current project scope. The product should not be judged mainly on whether it can repair badly copied text.

## Core Claims And Aggregation

The long-text test also showed a product-level issue: extracting every factual sentence is not always desirable.

A long article may contain many factual statements, but not all of them are equally important. Some are central claims; others are background, attribution, or context. If the system fact-checks all of them equally, three problems appear:

1. the backend becomes slow;
2. the result page becomes too long;
3. the overall verdict can be diluted by background facts.

This is not only an atomizer issue. It connects to aggregation. The future pipeline may need a simple claim-priority layer:

- core claims should influence the overall verdict more;
- background claims should still be shown, but should not dominate the final result;
- very low-value claims may be ignored before expensive retrieval.

This should be handled carefully. We do not want the atomizer to become a complicated article-understanding system, but long-text support needs some distinction between central and background facts.

We considered putting an importance label directly on each claim during batch atomization. That was rejected because each batch only sees part of the article. A claim may look important inside one batch but be background in the full article.

The chosen approach is a second, whole-text selection step after atomization:

1. atomize the full article into candidate claim groups;
2. give the candidate groups back to Gemini with the original text;
3. ask it to select the top 5 claim groups by importance to the user's input;
4. keep only those selected groups for the normal backend pipeline.

This keeps the main pipeline simple. The selected groups still go through both branches: text-pattern and fact-checking. The ranking step does not only limit fact-checking, because that would change the shape of the product. For long text, the product should explain that only the most important factual units are checked.

The ranking prompt also explicitly says not to rank by whether a claim looks true, false, surprising, suspicious, or controversial. The intended ranking basis is centrality to the user's text, factual coverage, and usefulness as a representative checkable unit.

This is still not perfect. LLM ranking can have its own bias, but it is cleaner than a hardcoded rule list and more global than per-batch importance scoring.

After this change, the controlled long-text atomizer test produced the intended shape:

```text
Candidate claim groups: 16
Selected claim groups: 5
Max claim groups: 5
```

The selected groups covered the main tax-bill storyline rather than every background sentence. This confirmed that whole-text top-K selection is a useful direction for long text.

However, the frontend end-to-end test then exposed a new limitation. The selected top-K groups still pass through the normal pipeline, but some selected claims can be too vague for open-web fact-checking:

```text
In 2026, the government announced a new policy.
A related law was passed in 2024.
```

These claims are grammatically factual, but not grounded enough for reliable web retrieval. `the government`, `a new policy`, and `a related law` are too underspecified unless the surrounding topic is carried into the checkable claim. The fact-checking branch may then retrieve any 2026 policy or any 2024 law and incorrectly treat it as support.

This means top-K selection solves the volume problem, but not the grounding problem. Long-text atomization must also prefer or produce claims with enough entity/topic context for retrieval.

We clarified that grounding should be judged on the `fact_check_claim`, not on the original sentence. An original sentence may contain a pronoun:

```text
He said it would help workers and small businesses.
```

This can still be selected if the atomizer has produced a standalone fact-checking claim:

```text
Donald Trump said the tax bill would help workers and small businesses.
```

The group should be skipped only when the fact-checking claim itself remains vague or context-dependent, such as:

```text
A related law was passed in 2024.
```

unless the claim includes enough topic or entity context to make `related law` searchable and judgeable.

The frontend screenshots also showed that the result page becomes confusing for long text when:

- the claim header shows the original sentence, but fact-checking uses a different resolved checkable claim;
- a sentence with multiple fact-checking claims is displayed as one large claim card;
- the label `Rewrite Claim` suggests query rewriting, even though the text is actually the checkable claim prepared by the atomizer;
- the summary does not clearly explain that many candidate factual units were found but only the top 5 were selected.

This confirmed that long-text support is not only an atomizer problem. It also needs frontend wording and aggregation/reporting changes.

The backend response was then extended to report both candidate and selected counts. This supports clearer frontend wording:

```text
16 Factual Claim(s) Detected · 2 Non-Factual Claim(s) Filtered
5 Factual Claim(s) Checked · Query Rewrite: Enabled
```

This makes long-text top-K selection visible to the user instead of making it look like the system only found five claims.

The context-resolution test also changed how we think about query rewrite. Earlier, the fact-checking branch used Gemini preparation to rewrite a raw claim into a better search query. After the atomizer started producing context-complete checkable claims, this extra rewrite step became less central.

For example:

```text
Original sentence:
It aims to return humans to the Moon.

Checkable claim:
The Artemis program aims to return humans to the Moon.
```

In this case, the retrieval branch no longer needs another LLM call to understand what `It` means. The atomizer has already supplied the subject, relation, and condition needed for search.

This led to a simplification:

1. keep Gemini in `gemini_agent.py` for evidence judgement;
2. remove claim-preparation rewrite from the main fact-checking path;
3. search the atomizer's checkable claim directly;
4. keep Gemini query rewrite only as an optional fallback when the first search returns no raw evidence.

The frontend option was changed from a default query-rewrite switch into an optional rewrite fallback for empty search. This reduces normal Gemini calls and makes the pipeline easier to explain: atomizer prepares the claim, retrieval searches it, Gemini judges selected evidence.

The next redundancy was that the atomizer was already resolving subject, relation, and conditions in natural language, but fact-checking was still receiving only the final claim string. This meant the structured claim fields such as `entities`, `relation`, and `constraints` stayed empty unless the old Gemini rewrite/preparation path ran.

That was cleaned up by extending each atomizer `fact_check_claim`:

```json
{
  "fact_claim_id": 1,
  "claim": "The Artemis program aims to return humans to the Moon.",
  "entities": ["The Artemis program", "humans", "the Moon"],
  "relation": "aims to return",
  "constraints": []
}
```

Fact-checking now receives these fields directly as top-level `EachFactualClaim` fields. It does not re-derive them. `metadata` is reserved for retrieval/runtime details such as the query used, fallback status, raw evidence count, and Gemini calibration score. Search still uses the checkable claim string for now; the structured fields can later support debugging or non-LLM search variants.

## Current Boundary

The current backend boundary is:

1. Atomizer prepares standalone factual claims.
2. Text-pattern branch checks wording risk for each text unit.
3. Fact-checking branch retrieves and judges evidence for each factual claim.
4. Recovery improves search when retrieval is weak.
5. Aggregation combines branch outputs into a final verdict.

This means search-query generation should remain in fact-checking recovery, not in the atomizer. The atomizer can make a claim standalone, but it should not become the retrieval planner.

Frontend wording should also match this boundary. `Rewrite Claim` was replaced with `Checkable Claim`, because the displayed text is the factual claim prepared for evidence checking, not just a query rewrite.

## Current Remaining Issues

The current atomizer is better than the first version, but several issues remain:

- reporting frames are not always removed when the embedded proposition should be checked;
- semantic duplicates are not always merged;
- long text still produces too many claims;
- condition inheritance needs more testing;
- article-level core-claim selection is not yet implemented;
- frontend and aggregation need to reflect that long-text outputs may contain both core and background claims.

These are not small frontend display issues. They are pipeline-level design issues that should guide the next round of atomizer and aggregation work.
