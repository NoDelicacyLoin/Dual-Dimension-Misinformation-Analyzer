from typing import Any

from pydantic import BaseModel, Field

from shared_constants import DEFAULT_RETRIEVAL_RESULTS


class AnalysisOptions(BaseModel):
    use_query_rewrite: bool = False
    relevance_threshold: float = 0.1
    top_k: int = 3
    use_all_eligible_evidence: bool = False
    retrieval_results: int = DEFAULT_RETRIEVAL_RESULTS


class ClaimRequest(BaseModel):
    claim: str
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class AtomizedFactClaim(BaseModel):
    fact_claim_id: int
    claim: str
    entities: list[str] = Field(default_factory=list)
    relation: str = ""
    constraints: list[str] = Field(default_factory=list)


class AtomizedClaimGroup(BaseModel):
    claim_group_id: int
    original_sentence: str
    text_feature_text: str
    atomization_applied: bool = False
    fact_check_claims: list[AtomizedFactClaim] = Field(default_factory=list)


class AtomizerSummary(BaseModel):
    ignored_sentence_count: int = 0
    text_feature_unit_count: int = 0
    fact_check_claim_count: int = 0


class AtomizerOutput(BaseModel):
    status: str
    original_text: str = ""
    ignored_sentences: list[str] = Field(default_factory=list)
    claim_groups: list[AtomizedClaimGroup] = Field(default_factory=list)
    summary: AtomizerSummary = Field(default_factory=AtomizerSummary)
    candidate_claim_group_count: int = 0
    candidate_fact_claim_count: int = 0
    selected_claim_group_count: int = 0
    selected_fact_claim_count: int = 0
    max_claim_group_count: int = 0
    claim_selection_reason: str = ""
    message: str = ""


class EachEvidence(BaseModel):
    stance: str = ""
    evidence_quality: str = ""
    url: str = ""
    content: str = ""
    ai_analysis: str = ""


class TextPatternPrediction(BaseModel):
    risk_level: str = ""
    risk_score: float = 0.0
    confidence_level: str = ""
    low_risk_probability: float = 0.0
    medium_risk_probability: float = 0.0
    high_risk_probability: float = 0.0


class TextPatternResult(BaseModel):
    claim_group_id: int = 0
    original_sentence: str = ""
    text_feature_text: str = ""
    atomization_applied: bool = False
    status: str = "error"
    prediction: TextPatternPrediction = Field(default_factory=TextPatternPrediction)
    influential_words: list[dict[str, Any]] = Field(default_factory=list)
    technical_details: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class EachFactualClaimMetadata(BaseModel):
    retrieval_query_used: str = ""
    retrieval_queries_tried: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    search_raw_evidence_count: int = 0
    selected_evidence_count: int = 0
    gemini_truth_score: float | None = None


class EachFactualClaim(BaseModel):
    claim_group_id: int
    fact_claim_id: int
    original_sentence: str
    text_feature_text: str
    claim: str
    entities: list[str] = Field(default_factory=list)
    relation: str = ""
    constraints: list[str] = Field(default_factory=list)
    status: str
    truth_score: float | None = None
    verdict: str | None = None
    explanation: str = ""
    decision_confidence: str = ""
    evidence_sufficiency: str = ""
    evidence: list[EachEvidence] = Field(default_factory=list)
    metadata: EachFactualClaimMetadata = Field(default_factory=EachFactualClaimMetadata)


class EachFactChecking(BaseModel):
    status: str = ""
    truth_score: float | None = None
    verdict: str | None = None
    explanation: str = ""
    factual_claims: list[EachFactualClaim] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    status: str
    original_text: str
    ignored_sentences: list[str] = Field(default_factory=list)
    text_pattern_results: list[TextPatternResult] = Field(default_factory=list)
    fact_checking: EachFactChecking = Field(default_factory=EachFactChecking)
    overall_risk_score: float | None = None
    overall_risk_level: str = ""
    overall_risk_confidence: str = ""
    progress_events: list[dict[str, Any]] = Field(default_factory=list)
    ignored_sentence_count: int = 0
    text_feature_unit_count: int = 0
    fact_check_claim_count: int = 0
    candidate_claim_group_count: int = 0
    candidate_fact_claim_count: int = 0
    selected_claim_group_count: int = 0
    selected_fact_claim_count: int = 0
    max_claim_group_count: int = 0
    claim_selection_reason: str = ""
    message: str = ""
