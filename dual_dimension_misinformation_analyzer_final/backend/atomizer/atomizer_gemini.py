import os
import time

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


MODEL_ID = "gemini-2.5-flash-lite"
MAX_GEMINI_RETRIES = 3
RETRYABLE_GEMINI_MARKERS = [
    "503",
    "UNAVAILABLE",
    "high demand",
    "rate limit",
    "RESOURCE_EXHAUSTED",
]

client = None
client_api_key = ""


def get_client():
    global client
    global client_api_key

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or genai is None:
        return None

    if client is None or client_api_key != api_key:
        client = genai.Client(api_key=api_key)
        client_api_key = api_key

    return client


def is_gemini_available() -> bool:
    return get_client() is not None


def generate_atomizer_json(prompt: str) -> str | None:
    gemini_client = get_client()
    if not gemini_client:
        return None

    request_config = None
    if types is not None:
        request_config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )

    for attempt_index in range(MAX_GEMINI_RETRIES):
        try:
            response = gemini_client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=request_config,
            )
            return (response.text or "").strip()
        except Exception as error:
            error_text = str(error)
            should_retry = any(marker in error_text for marker in RETRYABLE_GEMINI_MARKERS)
            is_last_attempt = attempt_index == MAX_GEMINI_RETRIES - 1

            if not should_retry or is_last_attempt:
                raise

            time.sleep(2 ** attempt_index)

    return None
