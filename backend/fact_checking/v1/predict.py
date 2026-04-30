import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# 1. Load the model and tokenizer globally.
# This ensures we only download and load the model into memory ONCE when the server starts.
MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
print(f"Loading NLI model ({MODEL_NAME}). This might take a minute...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")

def calculate_possibility_score(user_claim: str, evidence_list: list[dict]) -> tuple[float, str]:
    """
    Compares the user's claim against the retrieved evidence using an NLI model.
    Returns a tuple containing:
    1. The Truth Score (float between 0.0 and 1.0, where 1.0 is True and 0.0 is Fake).
    2. The final verdict string ("Fake", "True", or "Neutral").
    """
    if not evidence_list or evidence_list[0].get("url") == "Error":
        # 找不到证据时，真实度应该是 0.5 (未经验证/中立)，而不是 0.0 (绝对虚假)
        return 0.5, "Neutral"
    
    text_snippets = [item["content"] for item in evidence_list]
    combined_evidence = " ".join(text_snippets)


    # The NLI model expects two inputs to compare: Premise (the facts) and Hypothesis (the claim).
    # We use truncation=True to ensure we don't crash the model if the evidence is too long.
    encoded_inputs = tokenizer(
        combined_evidence,
        user_claim,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Disable gradient calculation because we are only doing inference, not training.
    # This is a crucial step to save GPU memory and speed up the process.
    with torch.no_grad():
        model_outputs = model(**encoded_inputs)
        # Logits are the raw, unnormalized scores outputted by the final layer of the neural network.
        raw_logits = model_outputs.logits

    # Convert the raw logits into mathematical probabilities (between 0 and 1) using Softmax.
    probabilities = F.softmax(raw_logits, dim=1)[0]

    # For the 'cross-encoder/nli-deberta-v3-base' model, the output labels are mapped as:
    # Index 0: Contradiction (The evidence proves the claim is Fake)
    # Index 1: Entailment (The evidence proves the claim is True)
    # Index 2: Neutral (The evidence is unrelated to the claim)
    fake_probability = probabilities[0].item()
    true_probability = probabilities[1].item()
    neutral_probability = probabilities[2].item()

    # Determine the final verdict based on which probability is the highest.
    if fake_probability > true_probability and fake_probability > neutral_probability:
        verdict = "Fake"
    elif true_probability > fake_probability and true_probability > neutral_probability:
        verdict = "True"
    else:
        verdict = "Neutral"

    # --- 核心修改：计算“真实度评分 (Truthfulness Score)” ---
    # 公式：(真概率 * 100%) + (中立概率 * 50%) + (假概率 * 0%)
    truth_score = (true_probability * 1.0) + (neutral_probability * 0.5) + (fake_probability * 0.0)
    
    # 返回这个综合评分
    return truth_score, verdict

# little test
if __name__ == "__main__":
    test_claim = "Drinking coffee cures all diseases."

    # 将测试数据从纯文本字符串升级为字典结构
    test_evidence = [
        {"url": "http://mock1.com", "content": "Studies show coffee reduces the risk of heart failure by 12%."},
        {"url": "http://mock2.com", "content": "Drinking coffee is not associated with an overall cancer risk reduction."},
        {"url": "http://mock3.com", "content": "There is no medical evidence that coffee cures all diseases."}
    ]

    print("\nRunning inference...")
    score, final_verdict = calculate_possibility_score(test_claim, test_evidence)
    # ... 下面保持不变 ...

    print("\n--- NLI Analysis Results ---")
    print(f"Claim: {test_claim}")
    print(f"Verdict: {final_verdict}")
    # Format the score as a percentage with 2 decimal places
    print(f"Truth Score: {score * 100:.2f}%")