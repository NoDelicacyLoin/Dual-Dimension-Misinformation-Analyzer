from api_contract import TextPatternPrediction, TextPatternResult

from text_pattern.predict_text_risk_local import model_output as analyze_text_risk


def analyze_text_pattern(text: str, progress_callback=None) -> TextPatternResult:
    try:
        raw_result = analyze_text_risk(text, progress_callback=progress_callback)
    except Exception as error:
        return TextPatternResult(
            status="error",
            message=str(error),
        )

    if not isinstance(raw_result, dict):
        return TextPatternResult(
            status="error",
            message="Text-pattern module returned an unexpected result type.",
        )

    raw_prediction = raw_result.get("prediction", {})
    if not isinstance(raw_prediction, dict):
        raw_prediction = {}

    probabilities = raw_prediction.get("probabilities", {})
    if not isinstance(probabilities, dict):
        probabilities = {}

    return TextPatternResult(
        status=raw_result.get("status", "error"),
        prediction=TextPatternPrediction(
            risk_level=raw_prediction.get("risk_level", ""),
            risk_score=float(raw_prediction.get("risk_score", 0.0) or 0.0),
            confidence_level=raw_prediction.get("confidence_level", ""),
            low_risk_probability=float(probabilities.get("low_risk", 0.0) or 0.0),
            medium_risk_probability=float(probabilities.get("medium_risk", 0.0) or 0.0),
            high_risk_probability=float(probabilities.get("high_risk", 0.0) or 0.0),
        ),
        influential_words=raw_result.get("influential_words", []),
        technical_details=raw_result.get("technical_details", {}),
        message=raw_result.get("message", ""),
    )
