import json
import os
import sys
from getpass import getpass


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")
TEXT_PATH = os.path.join(TEST_DIR, "long_text_test_example.txt")

if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from atomizer.atomizer_gemini import is_gemini_available
from atomizer.atomizer_service import (
    LONG_TEXT_BATCH_SIZE,
    LONG_TEXT_CONTEXT_SENTENCES,
    MAX_CLAIM_GROUPS_FOR_OUTPUT,
    atomize_long_text,
    normalize_input_text,
    split_into_sentences,
)
from api_contract import AtomizerOutput


BATCH_SIZE = LONG_TEXT_BATCH_SIZE
CONTEXT_SENTENCE_COUNT = LONG_TEXT_CONTEXT_SENTENCES
MAX_CLAIM_GROUPS = MAX_CLAIM_GROUPS_FOR_OUTPUT


def configure_gemini_key() -> None:
    if os.environ.get("GEMINI_API_KEY"):
        return

    api_key = "AIzaSyC5CfBmlrTDBPMqoDMe4OUaW_ODSduG_Lk"
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key


def print_sentences(sentence_items: list[str]) -> None:
    print(f"Total sentences after cleaning: {len(sentence_items)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Context sentence count: {CONTEXT_SENTENCE_COUNT}")

    for start_index in range(0, len(sentence_items), BATCH_SIZE):
        batch_number = (start_index // BATCH_SIZE) + 1
        context_start_index = max(0, start_index - CONTEXT_SENTENCE_COUNT)
        context_sentences = sentence_items[context_start_index:start_index]
        target_sentences = sentence_items[start_index:start_index + BATCH_SIZE]

        print("\n" + "=" * 100)
        print(f"Batch {batch_number}")
        print("-" * 100)

        print("Context sentences:")
        if context_sentences:
            for sentence_index, sentence in enumerate(context_sentences, start=context_start_index + 1):
                print(f"{sentence_index}. {sentence}")
        else:
            print("- none")

        print("\nTarget sentences:")
        for sentence_index, sentence in enumerate(target_sentences, start=start_index + 1):
            print(f"{sentence_index}. {sentence}")


def print_atomizer_output(output: AtomizerOutput) -> None:
    print("\n" + "=" * 100)
    print("Merged atomizer output")
    print("-" * 100)

    print(f"Status: {output.status}")
    if output.message:
        print(f"Message: {output.message}")

    print("\nSummary:")
    print(json.dumps(output.summary.model_dump(), ensure_ascii=False, indent=2))
    print(f"Candidate claim groups: {output.candidate_claim_group_count}")
    print(f"Candidate factual claims: {output.candidate_fact_claim_count}")
    print(f"Selected claim groups: {output.selected_claim_group_count}")
    print(f"Selected factual claims: {output.selected_fact_claim_count}")
    print(f"Max claim groups: {output.max_claim_group_count}")
    if output.claim_selection_reason:
        print(f"Selection reason: {output.claim_selection_reason}")

    print("\nIgnored sentences:")
    if output.ignored_sentences:
        for sentence in output.ignored_sentences:
            print(f"- {sentence}")
    else:
        print("- none")

    print("\nClaim groups:")
    if not output.claim_groups:
        print("- none")
        return

    for group in output.claim_groups:
        print(f"\nGroup {group.claim_group_id}:")
        print(f"Original sentence: {group.original_sentence}")
        print(f"Text-pattern text: {group.text_feature_text}")
        print(f"Atomization applied: {group.atomization_applied}")

        for fact_claim in group.fact_check_claims:
            print(f"  Fact {fact_claim.fact_claim_id}: {fact_claim.claim}")
            print(f"    Entities: {fact_claim.entities}")
            print(f"    Relation: {fact_claim.relation}")
            print(f"    Constraints: {fact_claim.constraints}")


def main() -> None:
    configure_gemini_key()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Backend root: {BACKEND_ROOT}")
    print(f"Text path: {TEXT_PATH}")
    print(f"GEMINI_API_KEY: {'available' if os.environ.get('GEMINI_API_KEY') else 'missing'}")
    print(f"Gemini available: {is_gemini_available()}")

    with open(TEXT_PATH, encoding="utf-8") as text_file:
        article_text = text_file.read().strip()

    sentence_items = split_into_sentences(normalize_input_text(article_text), limit=None)
    print_sentences(sentence_items)

    output = atomize_long_text(
        article_text,
        batch_size=BATCH_SIZE,
        context_sentence_count=CONTEXT_SENTENCE_COUNT,
        max_claim_groups=MAX_CLAIM_GROUPS,
    )
    print_atomizer_output(output)


if __name__ == "__main__":
    main()
