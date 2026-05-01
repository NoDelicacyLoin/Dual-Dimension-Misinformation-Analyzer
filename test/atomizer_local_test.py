import json
import os
import sys


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")

if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from atomizer.atomizer_gemini import is_gemini_available
from atomizer.atomizer_service import atomize_text


TEXT_CASES = [
    (
        "single factual claim",
        "The Great Wall of China is visible from the Moon with the naked eye.",
    ),
    (
        "multiple factual claims in one sentence",
        "The Great Wall of China is visible from the Moon with the naked eye and the Earth orbits the Sun.",
    ),
    (
        "pronoun subject depends on previous sentence",
        "Donald Trump posted a message on Twitter. He said the stock market had its best year.",
    ),
    (
        "pronoun subject and object depend on previous sentence",
        "Donald Trump praised the tax bill. He said it would help workers.",
    ),
    (
        "unclear pronoun should be ignored",
        "He said it would happen soon.",
    ),
    (
        "historical belief should stay attributed",
        "People once thought the Sun orbits around Earth.",
    ),
    (
        "ordinary reported claim should expose inner proposition",
        "A White House official said the policy would lower taxes.",
    ),
    (
        "quoted claim should expose inner proposition",
        'Donald Trump said, "The stock market had its best year."',
    ),
    (
        "announcement event should keep attribution",
        "Officials announced the new policy in 2024.",
    ),
    (
        "time condition should not be blindly inherited",
        "In 2026, the government announced a new policy. A related law was passed in 2024. It gave agencies new powers.",
    ),
    (
        "subjective sentence should be ignored",
        "I think this story is ridiculous. The Earth orbits the Sun.",
    ),
    (
        "this reference depends on previous sentence",
        "The tax bill passed the Senate. This helped the president claim a victory.",
    ),
    (
        "messy copied article text",
        "Donald J. Trump (@realDonaldTrump) December 31, 2017Trump's tweet went down about as well as you'd expect. He couldn't do it. As critics noted, the message was deleted.",
    ),
    (
        "near duplicate claims",
        "The Earth orbits the Sun. Earth orbits around the Sun.",
    ),
    (
        "similar wording but different degree",
        "The project is almost finished. The project is finished.",
    ),
    (
        "short news-style paragraph",
        "Donald Trump praised the tax bill. He said it would help workers. Critics said the bill mainly helped corporations.",
    ),
]


def print_atomizer_output(case_name: str, text: str, output: dict) -> None:
    print("\n" + "=" * 100)
    print(case_name)
    print("-" * 100)
    print("Input:")
    print(text)

    print("\nSummary:")
    print(json.dumps(output.get("summary", {}), ensure_ascii=False, indent=2))

    ignored_sentences = output.get("ignored_sentences", [])
    print("\nIgnored sentences:")
    if ignored_sentences:
        for sentence in ignored_sentences:
            print(f"- {sentence}")
    else:
        print("- none")

    print("\nClaim groups:")
    claim_groups = output.get("claim_groups", [])
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
            print(f"    Entities: {fact_claim.get('entities', [])}")
            print(f"    Relation: {fact_claim.get('relation', '')}")
            print(f"    Constraints: {fact_claim.get('constraints', [])}")


def run_atomizer_cases() -> None:
    for case_name, text in TEXT_CASES:
        output = atomize_text(text)
        print_atomizer_output(case_name, text, output)


if __name__ == "__main__":
    GEMINI_API_KEY = "AIzaSyC5CfBmlrTDBPMqoDMe4OUaW_ODSduG_Lk"

    if GEMINI_API_KEY:
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

    print(f"Backend root: {BACKEND_ROOT}")
    print(f"GEMINI_API_KEY: {'available' if os.environ.get('GEMINI_API_KEY') else 'missing'}")
    print(f"Gemini available: {is_gemini_available()}")

    run_atomizer_cases()
