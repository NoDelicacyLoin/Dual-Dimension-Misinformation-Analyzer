"""
Deprecated compatibility module.

The product-facing API now lives in backend/app.py.
The fact-checking branch entrypoint now lives in fact_checking/fact_check_service.py.
"""

from fact_checking.fact_check_service import (
    analyze_fact_check_claims,
    normalize_analysis_options,
    run_fact_check_for_atomic_claim,
)

__all__ = [
    "analyze_fact_check_claims",
    "normalize_analysis_options",
    "run_fact_check_for_atomic_claim",
]
