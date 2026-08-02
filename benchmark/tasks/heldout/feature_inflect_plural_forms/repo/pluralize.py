def pluralize(word: str) -> str:
    if word.endswith("y"):
        return word[:-1] + "ies"
    if word.endswith("s"):
        return word + "es"
    return word + "s"
