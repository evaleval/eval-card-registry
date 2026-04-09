import re


def normalize(value: str) -> str:
    """Lowercase, strip, collapse whitespace, remove punctuation except - and /.

    Spaces, hyphens, and underscores are all collapsed to a single space so
    that ``lm-evaluation-harness`` and ``lm evaluation harness`` match.
    """
    value = value.lower()
    value = value.strip()
    value = re.sub(r"[^\w\s\-/]", "", value)       # remove punctuation first
    value = re.sub(r"[\s_\-]+", " ", value).strip() # collapse spaces/hyphens/underscores
    return value
