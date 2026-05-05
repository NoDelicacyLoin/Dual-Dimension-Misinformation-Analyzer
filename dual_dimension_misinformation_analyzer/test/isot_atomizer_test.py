import csv
import json
import os
import sys


DATA_FILE = "Fake.csv"
ROW_INDEX = 0
BATCH_SIZE = 8
INCLUDE_TITLE = False
PREVIEW_LENGTH = 900


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")
ISOT_ROOT = os.path.join(PROJECT_ROOT, "dataset", "ISOT")

if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from atomizer.atomizer_gemini import is_gemini_available
from atomizer.atomizer_service import atomize_long_text, normalize_input_text, split_into_sentences


def read_isot_row() -> dict:
    csv_path = os.path.join(ISOT_ROOT, DATA_FILE)

    with open(csv_path, encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_index, row in enumerate(reader):
            if row_index == ROW_INDEX:
                return row

    raise ValueError(f"Could not find row {ROW_INDEX} in {csv_path}")


def build_article_text(row: dict) -> str:
    title = row.get("title", "").strip()
    text = row.get("text", "").strip()

    if INCLUDE_TITLE and title:
        return f"{title}. {text}"

    return text


def print_article(row: dict, article_text: str, sentence_items: list[str]) -> None:
    print(f"Dataset file: {DATA_FILE}")
    print(f"Row index: {ROW_INDEX}")
    print(f"Title: {row.get('title', '')}")
    print(f"Subject: {row.get('subject', '')}")
    print(f"Date: {row.get('date', '')}")
    print(f"Article input length: {len(article_text)}")
    print(f"Total sentences after cleaning: {len(sentence_items)}")
    print(f"Batch size: {BATCH_SIZE}")

    print("\nArticle preview:")
    print(article_text[:PREVIEW_LENGTH])

    print("\nSentence batches:")
    for start_index in range(0, len(sentence_items), BATCH_SIZE):
        batch_number = (start_index // BATCH_SIZE) + 1
        batch_sentences = sentence_items[start_index:start_index + BATCH_SIZE]
        print(f"\nBatch {batch_number}:")
        for sentence_index, sentence in enumerate(batch_sentences, start=start_index + 1):
            print(f"{sentence_index}. {sentence}")


def print_final_output(output: dict) -> None:
    print("\n" + "=" * 100)
    print("Merged atomizer output")
    print("-" * 100)

    print("\nAtomizer summary:")
    print(json.dumps(output.get("summary", {}), ensure_ascii=False, indent=2))

    ignored_sentences = output.get("ignored_sentences", [])
    print("\nIgnored sentences:")
    if ignored_sentences:
        for sentence in ignored_sentences:
            print(f"- {sentence}")
    else:
        print("- none")

    claim_groups = output.get("claim_groups", [])
    print("\nClaim groups:")
    if not claim_groups:
        print("- none")
        return

    for group in claim_groups:
        print(f"\nGroup {group.get('claim_group_id')}:")
        print(f"Original sentence: {group.get('original_sentence')}")
        print(f"Text-pattern text: {group.get('text_feature_text')}")
        print(f"Atomization applied: {group.get('atomization_applied')}")

        for fact_claim in group.get("fact_check_claims", []):
            print(f"  Fact {fact_claim.get('fact_claim_id')}: {fact_claim.get('claim')}")


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Backend root: {BACKEND_ROOT}")
    print(f"ISOT root: {ISOT_ROOT}")
    print(f"GEMINI_API_KEY: {'available' if os.environ.get('GEMINI_API_KEY') else 'missing'}")
    print(f"Gemini available: {is_gemini_available()}")

    row = read_isot_row()
    article_text = build_article_text(row)
    sentence_items = split_into_sentences(normalize_input_text(article_text), limit=None)

    print_article(row, article_text, sentence_items)
    output = atomize_long_text(article_text, batch_size=BATCH_SIZE)
    print_final_output(output)


if __name__ == "__main__":
    main()
