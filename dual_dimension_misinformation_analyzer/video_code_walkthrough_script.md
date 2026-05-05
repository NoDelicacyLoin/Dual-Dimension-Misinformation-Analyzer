# COMP3000 Video Script: Short Recording Version

Target length: about 4:45 to 5:00.

Important: only read the narration. `Screen cues` are recording notes, not spoken lines.

## Timing Plan

| Time | Focus |
|---|---|
| 0:00-0:20 | Project aim |
| 0:20-1:00 | Overall pipeline |
| 1:00-2:00 | Product demo |
| 2:00-2:40 | Atomizer |
| 2:40-3:20 | Text-pattern branch |
| 3:25-4:25 | Fact-checking branch |
| 4:25-4:45 | Evaluation and limitations |

## 0:00-0:20 Project Aim

Narration:

Hello, this is Zijian Yang, and this is my Dual-Dimension Misinformation Analyzer.

The aim is to analyse claims from two angles: language-pattern risk and evidence-based factual support. This matters because wording risk and factual truth are different.

Screen cues:

1. Product home page with title visible.

## 0:20-1:00 Overall Pipeline

Narration:

The backend is built with FastAPI. The frontend calls `/analyze/stream`, so progress can be streamed back while the analysis is running.

The main pipeline is in `run_analysis`. The input first goes to the atomizer. Then the system runs two branches in parallel: text-pattern analysis and evidence-based fact-checking.

At the end, the system aggregates the two outputs into the final result.

Screen cues:

1. `backend/app.py`: show `/analyze/stream`.
2. `analysis_orchestrator.py`: show `run_analysis`.
3. Show `atomize_for_pipeline`.
4. Show `ThreadPoolExecutor(max_workers=2)`.
5. Briefly show `build_overall_risk`.

## 1:00-2:00 Product Demo

Demo input:

`I saw this claim in a short online post. The Artemis program is led by NASA and aims to return humans to the Moon. The mission was delayed after technical problems. This pushed the planned launch into 2026.`

Narration:

Now I enter a longer input and start the analysis.

The loading page shows the backend stages and progress circles.

On the result page, the left side shows text-pattern risk, confidence, probabilities, and influential words. The right side shows evidence, stance, quality, truth score, and verdict.

Here, "the mission" becomes the standalone checkable claim "the Artemis mission".

Screen cues:

1. Product input page with Artemis text.
2. Click Start Analysis.
3. Hold on loading page.
4. Show result summary.
5. Show text-pattern panel.
6. Show fact-checking panel and Checkable Claim.

## 2:00-2:40 Atomizer

Narration:

The atomizer prepares the input before both branches run. First, it splits the passage into sentence-level units.

Then it extracts standalone factual claims and records `entities`, `relation`, and `constraints`. These fields help preserve the subject, the relationship, and important conditions such as time, place, number, or negation.

For longer text, it also uses a small context window, so references like "the mission" can be resolved when the previous sentence makes the meaning clear.

Screen cues:

1. `atomizer_service.py`: show `atomize_for_pipeline`.
2. Show `split_into_sentences`.
3. `prompts.py`: show `entities`, `relation`, `constraints`.
4. `atomizer_utils.py`: show `LONG_TEXT_CONTEXT_SENTENCES`.
5. Show `validate_llm_output`.

## 2:40-3:20 Text-Pattern Branch

Narration:

The text-pattern branch uses a LIAR-trained BERT model. The original six labels are mapped into three risk levels: low, medium, and high.

It also calculates uncertainty using entropy, so the result can include a confidence level.

For explainability, I use token occlusion. The system masks tokens one by one and measures how much the prediction confidence changes. This branch analyses wording risk, not factual truth.

Screen cues:

1. `predict_text_risk_local.py`: show `MODEL_SETTINGS`.
2. Show `risk_label_groups`.
3. Show `get_uncertainty` and `get_confidence_level`.
4. Show `get_token_importance`.
5. Show `risk_score`.

## 3:25-4:25 Fact-Checking Branch

Narration:

The fact-checking branch receives structured claims from the atomizer. For each claim, it builds a search query and retrieves evidence with Tavily.

Raw search results are not passed directly to the LLM. They are filtered with NLI relevance, token overlap, anchor matching, and number matching, so loosely related pages are removed.

NLI checks semantic relevance, while anchor and number matching preserve names, dates, and quantities.

Gemini then gives source-level judgements: supporting, contradicting, mixed, or background.

The backend calculates the truth score from evidence stance and quality, so the final verdict is controlled by backend logic.

Screen cues:

1. `fact_check_service.py`: show `run_fact_check_for_checkable_claim`.
2. Show `build_search_queries` and `search_for_evidence`.
3. `retrieval_service.py`: show `choose_evidence`.
4. `nli_filter.py`: show `score_evidence`.
5. `gemini_agent.py`: show `build_verdict_prompt` stance labels.
6. `decision_utils.py`: show `aggregate_truth_score` and `set_verdict_from_truth_score`.

## 4:25-4:45 Evaluation And Limitations

Narration:

Evaluation used LIAR and FEVER. Text-pattern achieved 48.46 percent accuracy and 0.4743 macro-F1.

Strict FEVER scored 56 percent accuracy and 0.523 macro-F1; the secondary open-web adjusted audit reached 82 percent and 0.7484 macro-F1.

Limitations are claim-based retrieval on implicit or multi-hop claims, Tavily coverage, filtering errors, Gemini stance errors, and claim-level text-pattern analysis.

Screen cues:

1. Show LIAR metrics.
2. Show FEVER metrics.
3. End on product result page.
