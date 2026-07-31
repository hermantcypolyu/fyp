"""python-dotenv - read key/value pairs from a .env file.

This is a reduced extract: only the parser is present.
"""

from dotenv.parser import Binding, Original, parse_stream

__all__ = ["Binding", "Original", "parse_stream"]
