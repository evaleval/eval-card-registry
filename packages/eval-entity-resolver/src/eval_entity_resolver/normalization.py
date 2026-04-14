import re


def normalize(value: str) -> str:
    """Lowercase, strip, collapse all separators (space/_/-/slash) to one space.

    Collapsing ``/`` with the other separators means ``tau-bench-2/airline``
    and ``tau-bench-2_airline`` normalize identically — critical for generalized
    ingestion where the same benchmark may appear with slash, underscore, or
    hyphen separators across configs. False merges across distinct canonical
    IDs are prevented by fuzzy's suffix-stripping being the *only* stem rewrite
    we apply (no generic similarity).

    Dots between digits are converted to spaces first so that version
    numbers like ``4.5`` and ``4-5`` normalize identically (both → ``4 5``).
    """
    value = value.lower()
    value = value.strip()
    # Convert dots between digits to spaces (e.g. "4.5" → "4 5")
    value = re.sub(r"(?<=\d)\.(?=\d)", " ", value)
    value = re.sub(r"[^\w\s\-/]", "", value)         # remove punctuation first
    value = re.sub(r"[\s_\-/]+", " ", value).strip() # collapse separators
    return value
