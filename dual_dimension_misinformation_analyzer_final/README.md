# Dual-Dimension Misinformation Analyzer

This project is a COMP3000 final-year project that analyzes user text through two complementary dimensions:

1. **Text-pattern risk**: whether the wording pattern resembles misinformation-style claims.
2. **Evidence-based fact-checking**: whether external evidence supports or contradicts extracted factual claims.

The system is designed to avoid treating a language-model response as a final truth judgement. Instead, it atomizes user input into checkable factual claims, runs a local text-pattern classifier, retrieves web evidence, filters sources, asks Gemini for source-level evidence judgements, and then uses backend scoring logic to produce the final verdict.

## Main Features

- FastAPI backend with `/analyze` and `/analyze/stream` endpoints.
- Static frontend served by the backend.
- Atomizer that splits long passages into sentence-level units and extracts standalone factual claims.
- Claim metadata extraction: `entities`, `relation`, and `constraints`.
- Local text-pattern risk branch using a six-label LIAR-style classifier mapped into three product risk levels.
- Token occlusion output for influential words.
- Tavily-based retrieval for external evidence.
- NLI, token, anchor, and number matching for evidence filtering.
- Gemini source-level stance judgement: `supports`, `contradicts`, `mixed`, or `background`.
- Backend truth-score aggregation, verdict mapping, evidence sufficiency, and decision confidence.
- Server-Sent Events progress updates for the frontend loading pipeline.

## Pipeline

```text
user input
-> atomizer
-> text-pattern branch + fact-checking branch
-> aggregate risk and verdict
-> frontend result display
```

The atomizer prepares shared structured input for both branches. The text-pattern branch checks wording risk, not factual truth. The fact-checking branch checks evidence support, not rhetorical style.

## Project Structure

```text
backend/
  app.py                         FastAPI app and static frontend serving
  api_contract.py                Pydantic request/response models
  analysis_orchestrator.py       Pipeline control, parallel branch execution, aggregation
  shared_constants.py            Shared progress stage names and scoring constants
  atomizer/                      Gemini-based claim atomization and validation
  text_pattern/                  Local text-risk classifier integration
  fact_checking/                 Retrieval, evidence filtering, Gemini judgement, scoring

frontend/
  index.html                     User interface
  script.js                      Streaming request handling and result rendering
  style.css                      Frontend styling
  assets/                        Static demo/example assets

dataset/
  LIAR/                          LIAR dataset split used for text-pattern work
  FEVER/                         FEVER sample used for fact-checking evaluation

test/
  *.ipynb                        Development and evaluation notebooks
```

## Requirements

- Python 3.10 or later
- A local text-pattern model folder at:

```text
backend/text_pattern/model_copy/final_six_label_to_3risk_hf/
```

- Environment variables:

```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
$env:TAVILY_API_KEY="your-tavily-api-key"
```

The Gemini key is required for atomization and source-level evidence judgement. The Tavily key is required for web evidence retrieval.

## Installation

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The text-pattern branch expects the local Hugging Face model files to be present in `backend/text_pattern/model_copy/final_six_label_to_3risk_hf/`. Large model weights are not suitable for normal GitHub upload and should be supplied separately or handled with Git LFS.

## Running Locally

Run the backend from the `backend` directory:

```powershell
cd backend
uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The backend serves the frontend and exposes:

- `GET /`: frontend page
- `POST /analyze`: full JSON response
- `POST /analyze/stream`: streamed progress events plus final result

## Evaluation Summary

The text-pattern branch was evaluated on LIAR-style labels mapped into three risk levels:

- Accuracy: **48.46%**
- Macro-F1: **0.4743**

The fact-checking branch was evaluated with a FEVER-based test set:

- Strict FEVER-style accuracy: **56.0%**
- Strict macro-F1: **0.523**
- Secondary open-web adjusted audit accuracy: **82.0%**
- Secondary open-web adjusted macro-F1: **0.7484**

The adjusted open-web score is a diagnostic view for open-web evidence mismatch. It does not replace the stricter FEVER-style score.

## Limitations

- Retrieval is claim-based and can fail on implicit, relational, or multi-hop claims.
- The fact-checking branch does not reconstruct full evidence timelines.
- Results depend on Tavily search coverage and source availability.
- Evidence filtering can discard useful sources or keep loosely related sources.
- Gemini may assign an incorrect evidence stance.
- The text-pattern branch is claim-level rather than document-level.
- A text-pattern risk score should be treated as a risk signal, not as a truth detector.

## Development History

The Git history includes earlier development snapshots:

- `archive/fake-news-detector-v1`: early fake-news detector and fact-checking prototype.
- `archive/dual-dimension-dev`: intermediate dual-dimension development version with pipeline notes and v1/v2/v3 fact-checking iterations.
- final submission version: cleaned pipeline with atomizer, text-pattern branch, fact-checking branch, streaming frontend, and evaluation materials.

These historical versions show how the project moved from a simpler fake-news detector toward the final dual-dimension misinformation analysis pipeline.
