import re


def slugify(value):
    text = str(value).strip().lower()
    if not text:
        raise ValueError("value cannot be empty")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")
