import re

_whitespace_re = re.compile(r"\s+")


def basic_clean(text: str) -> str:
    text = text.strip()
    text = _whitespace_re.sub(" ", text)
    return text
