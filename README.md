# Dual Dimension Misinformation Analyzer Development Archive

This branch preserves the intermediate development version before the final submission version was consolidated.

## Included

- `backend/atomizer` and `backend/atomizer/v1`: atomization pipeline iterations.
- `backend/fact_checking/v1`, `v2`, and `v3`: successive fact-checking pipeline iterations.
- `backend/text_pattern`: local text-risk prediction integration, excluding copied model weights.
- `frontend` and `frontend/v2`: frontend iterations.
- `dataset` and `test`: datasets and local/Colab test notebooks used during development.

## Excluded

Generated Python caches, logs, copied model weights, and large binary model files are intentionally excluded.

The final cleaned implementation is maintained on the repository `master` branch.
