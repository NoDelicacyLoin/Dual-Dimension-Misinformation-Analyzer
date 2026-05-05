# Dual-Dimension Misinformation Analyzer

This is the final COMP3000 project repository. The `master` branch contains the submitted application directly at the repository root.

## Repository Layout

```text
backend/    FastAPI backend, atomizer, text-pattern branch, fact-checking branch
dataset/    LIAR, FEVER, and ISOT data used for development and evaluation
frontend/   Static frontend served by the backend
test/       Development and evaluation notebooks
```

Archived development material is kept on archive branches instead of the `master` branch:

- `archive/previous-work`
- `archive/dual-dimension-dev`
- `archive/fake-news-detector-v1`

## System Summary

The system analyzes misinformation risk through two dimensions:

1. **Text-pattern risk**: a local classifier estimates whether wording resembles misinformation-style claims.
2. **Evidence-based fact-checking**: factual claims are extracted, searched with Tavily, filtered with NLI and lexical matching, judged at source level with Gemini, and scored by backend logic.

```text
user input
-> atomizer
-> text-pattern branch + fact-checking branch
-> aggregate risk and verdict
-> frontend result display
```

## Model Files

Large model files are not committed to GitHub. The final project expects the local text-pattern model at:

```text
backend/text_pattern/model_copy/final_six_label_to_3risk_hf/
```

If the project is demonstrated on the original development machine, the model can stay in that local folder. If it must be reproduced on another machine, the model should be supplied separately through Git LFS, Hugging Face Hub, cloud storage, or the university submission system.

## API Keys

Runtime API keys are required for the evidence-based branch:

```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
$env:TAVILY_API_KEY="your-tavily-api-key"
```

No API keys are included in this repository.
