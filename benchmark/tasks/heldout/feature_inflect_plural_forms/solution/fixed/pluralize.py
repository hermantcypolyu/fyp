def pluralize(word: str) -> str:
    irregular = {
        "child": "children",
        "person": "people",
        "mouse": "mice",
        "leaf": "leaves",
    }
    if word in irregular:
        return irregular[word]
    if word.endswith("y"):
        return word[:-1] + "ies"
    if word.endswith("s"):
        return word + "es"
    return word + "s"
