import re

_strip_end_re = re.compile(r"\n\s+$")


def strip_end(src: str) -> str:
    return _strip_end_re.sub("\n", src)
