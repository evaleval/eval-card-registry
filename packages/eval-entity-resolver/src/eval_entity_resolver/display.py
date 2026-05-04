"""Display-name humanization for canonical model slugs.

Single source of truth for converting machine slugs (`gpt-4o-2024-05-13`) into
human-friendly display names (`GPT-4o (2024-05-13)`). Used by refresh scripts
and the seed migration; consumers (frontend, API) should NOT re-humanize but
read `canonical_models.display_name` directly.

Rules in priority order:
1. Strip org prefix (`openai/gpt-5` -> `gpt-5`).
2. Strip and parenthesize a trailing date suffix:
   - `-YYYY-MM-DD` -> ` (YYYY-MM-DD)`
   - `-YYYYMMDD`  -> ` (YYYY-MM-DD)`
   - `-MMDD` (4-digit) -> ` (MMDD)`
3. Per-token formatting:
   - Known acronyms render uppercase (`gpt` -> `GPT`).
   - Mixed-case overrides apply (`moe` -> `MoE`).
   - Param sizes uppercase the unit (`7b` -> `7B`, `a22b` -> `A22B`,
     `8x7b` -> `8x7B`, `30m` -> `30M`).
   - Number+letter version tags preserve case (`4o` -> `4o`).
   - O-series stays lowercase (`o1`, `o3`).
   - Vendor-name overrides (`deepseek` -> `DeepSeek`).
   - Default: capitalize first letter.
4. Glue an acronym token to the next token with a hyphen when the next
   token is a bare version number (digits + optional `.NN` + optional
   single non-size letter): `GPT 5 Mini` -> `GPT-5 Mini`,
   `GPT 4o ...` -> `GPT-4o ...`. Skipped when the next token is a param
   size like `7B`.
"""

from __future__ import annotations

import re

ACRONYMS: frozenset[str] = frozenset(
    {
        "gpt",
        "glm",
        "llm",
        "vl",
        "vlm",
        "qvq",
        "qwq",
        "mt",
        "vit",
        "clip",
        "dit",
        "hf",
        "ocr",
        "tts",
        "asr",
        "moe",
        "mlp",
        "rlhf",
    }
)

# Tokens whose canonical rendering is mixed case rather than ALL CAPS.
CASE_OVERRIDES: dict[str, str] = {
    "moe": "MoE",
    "vit": "ViT",
    "dit": "DiT",
}

# Vendor / family tokens whose canonical rendering doesn't match a simple
# capitalize() — e.g., `deepseek` should display as `DeepSeek`. Keep the
# list short; this is for tokens the auto-rule mangles, not a general
# branding registry.
TOKEN_OVERRIDES: dict[str, str] = {
    "deepseek": "DeepSeek",
    "openai": "OpenAI",
    "stepfun": "StepFun",
    "moonshotai": "MoonshotAI",
    "mistralai": "MistralAI",
}

# Suffixes treated as parameter-count units, NOT version letters. When a
# token like `7b` appears after an acronym, we do NOT hyphen-glue it.
_SIZE_SUFFIXES: frozenset[str] = frozenset({"b", "m", "k"})


def humanize_model_slug(slug: str) -> str:
    """Render a model slug as a human display name.

    Accepts a bare slug (`gpt-4o-2024-05-13`) or a full canonical id
    (`openai/gpt-4o-2024-05-13`); the org prefix is dropped.
    """
    if not slug:
        return ""
    if "/" in slug:
        slug = slug.split("/", 1)[1]

    slug, suffix = _strip_date_suffix(slug)

    tokens = slug.split("-")
    formatted = [_format_token(t) for t in tokens]

    out: list[str] = []
    i = 0
    while i < len(formatted):
        cur_lower = tokens[i].lower()
        if (
            i + 1 < len(formatted)
            and cur_lower in ACRONYMS
            and _is_version_token(tokens[i + 1])
        ):
            out.append(f"{formatted[i]}-{formatted[i + 1]}")
            i += 2
        else:
            out.append(formatted[i])
            i += 1

    return " ".join(out) + suffix


def _strip_date_suffix(slug: str) -> tuple[str, str]:
    """Pop a trailing date or 4-digit code; return (slug_without, ' (suffix)').

    Order matters: more specific patterns first, since a partial match
    against a less-specific pattern would mis-render (e.g. `2025` as a
    bare 4-digit code when it's actually the year half of `2025-08`).
    """
    # Full ISO date: `-YYYY-MM-DD`
    m = re.search(r"-(20\d{2}-\d{2}-\d{2})$", slug)
    if m:
        return slug[: m.start()], f" ({m.group(1)})"
    # Compact date: `-YYYYMMDD`
    m = re.search(r"-(20\d{6})$", slug)
    if m:
        d = m.group(1)
        return slug[: m.start()], f" ({d[:4]}-{d[4:6]}-{d[6:8]})"
    # Year-month: `-YYYY-MM` (e.g. `gpt-5-2025-08`)
    m = re.search(r"-(20\d{2})-(\d{2})$", slug)
    if m:
        return slug[: m.start()], f" ({m.group(1)}-{m.group(2)})"
    # Cohere convention: `-MM-YYYY` (e.g. `command-r-08-2024`).
    # Render as `(YYYY-MM)` for ISO-ordered display.
    m = re.search(r"-(\d{2})-(20\d{2})$", slug)
    if m:
        return slug[: m.start()], f" ({m.group(2)}-{m.group(1)})"
    # Bare 4-digit code: `-NNNN` (e.g. `grok-4-0709`, `kimi-k2-0711`).
    m = re.search(r"-(\d{4})$", slug)
    if m:
        return slug[: m.start()], f" ({m.group(1)})"
    return slug, ""


def _format_token(tok: str) -> str:
    if not tok:
        return tok
    low = tok.lower()
    if low in CASE_OVERRIDES:
        return CASE_OVERRIDES[low]
    if low in ACRONYMS:
        return low.upper()
    if low in TOKEN_OVERRIDES:
        return TOKEN_OVERRIDES[low]
    # Param size: 7b, 70b, 1.5b, 30m
    if re.fullmatch(r"\d+(?:\.\d+)?[bmk]", low):
        return low[:-1] + low[-1].upper()
    # MoE active-expert form: a22b, a3b
    if re.fullmatch(r"a\d+(?:\.\d+)?b", low):
        return "A" + low[1:-1] + "B"
    # MxNb: 8x7b -> 8x7B
    if re.fullmatch(r"\d+x\d+(?:\.\d+)?b", low):
        return low[:-1] + "B"
    # Number followed by a single lowercase letter that's NOT a size suffix:
    # version tags like `4o`, `5o` — keep as-is.
    if re.fullmatch(r"\d+(?:\.\d+)?[a-z]", low) and low[-1] not in _SIZE_SUFFIXES:
        return low
    # O-series: o1, o3, o4
    if re.fullmatch(r"o\d+", low):
        return low
    # Default: capitalize first letter, preserve rest.
    return tok[0].upper() + tok[1:] if tok[0].isalpha() else tok


def _is_version_token(tok: str) -> bool:
    """True if `tok` looks like a version (e.g. `5`, `4.5`, `4o`) and not
    a parameter size (`7b`, `70m`)."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([a-z]?)", tok.lower())
    if not m:
        return False
    return m.group(2) not in _SIZE_SUFFIXES
