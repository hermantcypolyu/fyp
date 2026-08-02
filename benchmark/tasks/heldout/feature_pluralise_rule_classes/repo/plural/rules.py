"""The generic suffix rules the pluraliser applies."""

from dataclasses import dataclass

VOWELS = "aeiou"


@dataclass(frozen=True)
class SuffixRule:
    """Rewrite a word ending, optionally only after a consonant."""

    suffix: str
    replacement: str
    requires_consonant_before: bool = False

    def matches(self, word: str) -> bool:
        lowered = word.lower()
        if not lowered.endswith(self.suffix):
            return False
        if self.requires_consonant_before:
            stem = lowered[: -len(self.suffix)]
            if not stem or stem[-1] in VOWELS:
                return False
        return True

    def apply(self, word: str) -> str:
        return word[: -len(self.suffix)] + self.replacement


#: Tried in order; the first match wins.
SUFFIX_RULES = [
    SuffixRule("y", "ies", requires_consonant_before=True),
    SuffixRule("s", "ses"),
    SuffixRule("x", "xes"),
    SuffixRule("z", "zes"),
    SuffixRule("ch", "ches"),
    SuffixRule("sh", "shes"),
]
