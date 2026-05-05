# COMP3000 Dual-Dimension Misinformation Analyzer Repository

This repository contains the final COMP3000 project implementation together with archived development work.

## Repository Layout

```text
dual_dimension_misinformation_analyzer_final/
  Final cleaned implementation used for the submitted system and video walkthrough.

dual_dimension_misinformation_analyzer/
  Earlier integrated development version, including intermediate backend/frontend iterations,
  notebooks, experiments, and evaluation materials.

archived/previous work/
  Earlier research, Colab notebooks, LIAR-related experiments, result tables, and figures.
  Large model weights and zip archives are intentionally excluded.
```

## Final System

The final system is in `dual_dimension_misinformation_analyzer_final`.

It analyzes misinformation risk through two dimensions:

1. **Text-pattern risk**: a local classifier estimates whether the wording pattern resembles misinformation-style claims.
2. **Evidence-based fact-checking**: factual claims are extracted, searched with Tavily, filtered with NLI and lexical matching, judged at source level with Gemini, and scored by backend logic.

The final pipeline is:

```text
user input
-> atomizer
-> text-pattern branch + fact-checking branch
-> aggregate risk and verdict
-> frontend result display
```

See `dual_dimension_misinformation_analyzer_final/README.md` for installation and running instructions.

## Large Model Files

Large model files are not committed to this repository. This avoids GitHub's normal repository file-size limit and keeps the repository cloneable.

The final project expects the text-pattern model to be available locally at:

```text
dual_dimension_misinformation_analyzer_final/backend/text_pattern/model_copy/final_six_label_to_3risk_hf/
```

If the project is demonstrated on the original development machine, the model can stay in that local folder. If the project must be reproduced on another machine, the model should be supplied separately, for example through a university submission upload, Git LFS, Hugging Face Hub, Google Drive, or OneDrive.

## API Keys

The evidence-based branch requires API keys at runtime:

```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
$env:TAVILY_API_KEY="your-tavily-api-key"
```

No API keys are included in this repository.

## Development History

The archive folders are included to show the project iteration process:

- early fake-news and LIAR experiments;
- model training and prompt/fine-tuning exploration;
- progressive fact-checking branch versions;
- frontend and backend integration work;
- final consolidation into the dual-dimension analyzer.

Model checkpoints, generated caches, and compressed archive exports are excluded. Notebooks, datasets, result tables, source code, and figures are retained where practical.
