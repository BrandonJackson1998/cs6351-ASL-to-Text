"""
Gloss-to-English Translator (Phase 3)
Uses Claude API to translate ASL gloss sequences into grammatical English.

ASL uses topic-comment grammar structure and omits articles, so this is
a real translation task, not just concatenation.

Example:
    Input glosses: [STORE, I, GO, YESTERDAY]
    Output: "I went to the store yesterday."
"""

import argparse


def translate_glosses(glosses: list[str]) -> str:
    """Translate an ordered list of ASL glosses to natural English using Claude API."""
    raise NotImplementedError("Phase 3 - implement Claude API gloss translation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate ASL glosses to English")
    parser.add_argument("--glosses", type=str, required=True,
                        help='Comma-separated glosses, e.g. "STORE,I,GO,YESTERDAY"')
    args = parser.parse_args()

    gloss_list = [g.strip() for g in args.glosses.split(",")]
    result = translate_glosses(gloss_list)
    print(f"English: {result}")
