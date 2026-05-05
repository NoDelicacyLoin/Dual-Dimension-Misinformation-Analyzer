import json
import sys
import time

import requests


def main() -> None:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    claim = " ".join(sys.argv[2:]).strip() if len(sys.argv) > 2 else "The Earth orbits the Sun."

    payload = {
        "claim": claim,
        "options": {
            "use_query_rewrite": True,
            "relevance_threshold": 0.35,
            "use_oversampling_retry": True,
            "use_selective_stabilization": True,
            "top_k": 3,
            "use_all_eligible_evidence": False,
            "retrieval_results": 8,
        },
    }

    print("Base URL:", base_url)
    print("Claim:", claim)

    home_response = requests.get(base_url + "/", timeout=20)
    script_response = requests.get(base_url + "/script.js", timeout=20)
    print("home:", home_response.status_code)
    print("script.js:", script_response.status_code)

    start_time = time.time()

    with requests.post(
        base_url + "/analyze/stream",
        json=payload,
        stream=True,
        timeout=(10, 600),
    ) as response:
        print("stream:", response.status_code)

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            event = json.loads(line.removeprefix("data: "))
            elapsed_seconds = round(time.time() - start_time, 1)

            if event["event"] == "progress":
                data = event["data"]
                counts = []
                if "text_feature_unit_count" in data:
                    counts.append("text_units=" + str(data.get("text_feature_unit_count")))
                if "fact_check_claim_count" in data:
                    counts.append("facts=" + str(data.get("fact_check_claim_count")))
                count_text = " | " + ", ".join(counts) if counts else ""
                print(elapsed_seconds, "progress:", data.get("stage"), "-", data.get("message") + count_text)
            elif event["event"] == "result":
                data = event["data"]
                print(elapsed_seconds, "result:", data.get("status"))
                print("overall risk:", data.get("overall_risk_level"), data.get("overall_risk_score"))
                print("fact status:", data.get("fact_checking", {}).get("status"))
                break
            elif event["event"] == "error":
                print(elapsed_seconds, "error:", event["data"])
                break


if __name__ == "__main__":
    main()
