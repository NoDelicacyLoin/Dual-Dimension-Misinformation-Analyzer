import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# Example output:
# {
#   "status": "success",
#   "claim": "The Great Wall of China is visible from the Moon with the naked eye.",
#   "analysis_type": "text_feature_analysis",
#   "prediction": {
#     "risk_level": "low_risk",
#     "risk_score": 0.3739,
#     "confidence_level": "low",
#     "probabilities": {
#       "low_risk": 0.5656,
#       "medium_risk": 0.1209,
#       "high_risk": 0.3134
#     }
#   },
#   "influential_words": [
#     {
#       "text": "visible",
#       "rank": 1,
#       "confidence_drop": 0.3708
#     },
#     {
#       "text": "naked",
#       "rank": 2,
#       "confidence_drop": 0.3092
#     },
#     {
#       "text": "from",
#       "rank": 3,
#       "confidence_drop": 0.2787
#     },
#     {
#       "text": "is",
#       "rank": 4,
#       "confidence_drop": 0.2771
#     },
#     {
#       "text": "moon",
#       "rank": 5,
#       "confidence_drop": 0.2725
#     },
#     {
#       "text": "great",
#       "rank": 6,
#       "confidence_drop": 0.2631
#     },
#     {
#       "text": "the",
#       "rank": 7,
#       "confidence_drop": 0.2517
#     },
#     {
#       "text": "the",
#       "rank": 8,
#       "confidence_drop": 0.2378
#     }
#   ],
#   "technical_details": {
#     "predicted_class_probability": 0.5656,
#     "uncertainty": {
#       "margin": 0.2522,
#       "entropy": 0.9414,
#       "normalized_entropy": 0.8569
#     },
#     "confidence_rule": {
#       "high": "normalized_entropy < 0.65",
#       "medium": "0.65 <= normalized_entropy < 0.85",
#       "low": "normalized_entropy >= 0.85"
#     },
#     "token_importance_method": "token_occlusion",
#     "token_importance_target": "low_risk",
#     "max_sequence_length": 128,
#     "word_count": 14,
#     "token_count": 17
#   }
# }



PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_SETTINGS = {
    "model_dir": PROJECT_ROOT
    / "backend"
    / "fake_news_text_features"
    / "model_training"
    / "model_copy"
    / "final_six_label_to_3risk_hf",
    "max_sequence_length": 128,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "risk_labels": ["low_risk", "medium_risk", "high_risk"],
    "risk_label_groups": {
        "low_risk": [4, 5],
        "medium_risk": [2, 3],
        "high_risk": [0, 1],
    },
    "high_confidence_entropy_cutoff": 0.65,
    "low_confidence_entropy_cutoff": 0.85,
}

TOKENIZER = None
MODEL = None


def load_model():
    global TOKENIZER, MODEL

    if MODEL is not None:
        return TOKENIZER, MODEL

    model_dir = MODEL_SETTINGS["model_dir"]
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
    MODEL = AutoModelForSequenceClassification.from_pretrained(model_dir)
    MODEL.to(MODEL_SETTINGS["device"])
    MODEL.eval()

    return TOKENIZER, MODEL


def get_model_inputs(text):
    tokenizer, _ = load_model()

    model_inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=False,
        max_length=MODEL_SETTINGS["max_sequence_length"],
    )

    return {
        input_name: input_tensor.to(MODEL_SETTINGS["device"])
        for input_name, input_tensor in model_inputs.items()
    }


def get_risk_probabilities(model_inputs):
    _, model = load_model()

    with torch.no_grad():
        output = model(**model_inputs)
        six_label_probabilities = torch.softmax(output.logits, dim=-1)

    risk_probabilities = []

    for risk_label in MODEL_SETTINGS["risk_labels"]:
        six_label_ids = MODEL_SETTINGS["risk_label_groups"][risk_label]
        risk_probability = six_label_probabilities[0, six_label_ids].sum()
        risk_probabilities.append(risk_probability)

    return torch.stack(risk_probabilities)


def get_uncertainty(risk_probabilities):
    sorted_probabilities = torch.sort(risk_probabilities, descending=True).values
    margin = float(sorted_probabilities[0] - sorted_probabilities[1])
    entropy = float(
        -(risk_probabilities * torch.log(risk_probabilities.clamp_min(1e-12))).sum()
    )
    normalized_entropy = entropy / math.log(len(MODEL_SETTINGS["risk_labels"]))

    return {
        "margin": round(margin, 4),
        "entropy": round(entropy, 4),
        "normalized_entropy": round(normalized_entropy, 4),
    }


def get_confidence_level(normalized_entropy):
    if normalized_entropy >= MODEL_SETTINGS["low_confidence_entropy_cutoff"]:
        return "low"
    if normalized_entropy >= MODEL_SETTINGS["high_confidence_entropy_cutoff"]:
        return "medium"
    return "high"


def get_token_importance(model_inputs, predicted_risk_id, predicted_risk_label, original_probability):
    tokenizer, _ = load_model()

    if tokenizer.mask_token_id is None:
        return {
            "method": "token_occlusion",
            "target_label": predicted_risk_label,
            "tokens": [],
        }

    input_ids = model_inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    special_token_ids = set(tokenizer.all_special_ids)

    important_tokens = []

    for token_index, token_id in enumerate(input_ids[0].tolist()):
        if token_id in special_token_ids:
            continue

        masked_inputs = {}
        for input_name, input_tensor in model_inputs.items():
            masked_inputs[input_name] = input_tensor.clone()

        masked_inputs["input_ids"][0, token_index] = tokenizer.mask_token_id

        masked_risk_probabilities = get_risk_probabilities(masked_inputs)
        masked_confidence = float(masked_risk_probabilities[predicted_risk_id])
        confidence_drop = original_probability - masked_confidence

        if confidence_drop <= 0:
            continue

        token = tokens[token_index]
        important_tokens.append(
            {
                "text": token[2:] if token.startswith("##") else token,
                "confidence_drop": round(confidence_drop, 4),
            }
        )

    important_tokens = sorted(
        important_tokens,
        key=lambda token_data: token_data["confidence_drop"],
        reverse=True,
    )

    influential_words = []
    for token_rank, token_data in enumerate(important_tokens[:8], start=1):
        influential_words.append(
            {
                "text": token_data["text"],
                "rank": token_rank,
                "confidence_drop": token_data["confidence_drop"],
            }
        )

    return influential_words


def model_output(text):
    claim = str(text).strip()

    if not claim:
        return {
            "status": "error",
            "message": "The input claim is empty.",
        }

    model_inputs = get_model_inputs(claim)
    risk_probabilities = get_risk_probabilities(model_inputs).detach().cpu()

    predicted_risk_id = int(torch.argmax(risk_probabilities).item())
    predicted_risk_label = MODEL_SETTINGS["risk_labels"][predicted_risk_id]
    predicted_class_probability = float(risk_probabilities[predicted_risk_id])
    uncertainty = get_uncertainty(risk_probabilities)
    confidence_level = get_confidence_level(uncertainty["normalized_entropy"])

    probabilities = {}
    for risk_label, risk_probability in zip(
        MODEL_SETTINGS["risk_labels"],
        risk_probabilities.tolist(),
    ):
        probabilities[risk_label] = round(risk_probability, 4)

    risk_score = 0.5 * risk_probabilities[1] + risk_probabilities[2]
    influential_words = get_token_importance(
        model_inputs,
        predicted_risk_id,
        predicted_risk_label,
        predicted_class_probability,
    )

    return {
        "status": "success",
        "claim": claim,
        "analysis_type": "text_feature_analysis",
        "prediction": {
            "risk_level": predicted_risk_label,
            "risk_score": round(float(risk_score), 4),
            "confidence_level": confidence_level,
            "probabilities": probabilities,
        },
        "influential_words": influential_words,
        "technical_details": {
            "predicted_class_probability": round(predicted_class_probability, 4),
            "uncertainty": uncertainty,
            "confidence_rule": {
                "high": "normalized_entropy < 0.65",
                "medium": "0.65 <= normalized_entropy < 0.85",
                "low": "normalized_entropy >= 0.85",
            },
            "token_importance_method": "token_occlusion",
            "token_importance_target": predicted_risk_label,
            "max_sequence_length": MODEL_SETTINGS["max_sequence_length"],
            "word_count": len(claim.split()),
            "token_count": int(model_inputs["attention_mask"].sum().item()),
        },
    }


if __name__ == "__main__":
    text = ["The Great Wall of China is visible from the Moon with the naked eye.", "Humans use only 10 percent of their brains."]
    for i in range(2):
        print(f"Text {i+1} Result:\n")
        result = model_output(text[i])
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print('='*20)
