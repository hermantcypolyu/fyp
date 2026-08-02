def normalize(text: str) -> str:
    if not text:
        return ""
    words = [piece for piece in text.lower().replace("-", " ").split() if piece]
    return " ".join(words)
