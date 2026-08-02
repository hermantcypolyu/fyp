import re
from html import _replace_charref  # type: ignore[attr-defined]
from urllib.parse import quote

_charref_re = re.compile(
    r"&(#[0-9]{1,7};"
    r"|#[xX][0-9a-fA-F]+;"
    r"|[^\t\n\f <&#;]{1,32};)"
)


def unescape(s: str) -> str:
    """
    Copy from `html.unescape`, but `_charref` is different. CommonMark
    does not accept entity references without a trailing semicolon
    """
    if "&" not in s:
        return s
    return _charref_re.sub(_replace_charref, s)


def escape(s: str, quote: bool = True) -> str:
    """Escape characters of ``&<>``. If quote=True, ``"`` will be
    converted to ``&quote;``."""
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    if quote:
        s = s.replace('"', "&quot;")
    return s


def escape_url(link: str) -> str:
    """Escape URL for safety."""
    safe = (
        ":/?#@"  # gen-delims - '[]' (rfc3986)
        "!$&()*+,;="  # sub-delims - "'" (rfc3986)
        "%"  # leave already-encoded octets alone
    )
    return quote(unescape(link), safe=safe)
