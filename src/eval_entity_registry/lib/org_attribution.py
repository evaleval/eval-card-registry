"""Seed-time org attribution for malformed-org canonical ids.

A draft id with no `/` whose leading token is a known developer org couldn't have
its org parsed — the org delimiter is a `-`/`.` instead of `/` — so it lands with
a null org_id, orphaned from its developer (`cohere-march-2024`, `deepseek-coder`).
This pass attributes the org from that leading token. org_id is DECOUPLED from the
id string (the registry already maps `Qwen/…` → org `alibaba`), so attribution is
safe even when the org token also appears in the model NAME (`deepseek-coder` →
org `deepseek`, id kept verbatim).

It does NOT touch an entry that is actually a DUPLICATE of an existing real repo
of the same developer (`deepseek-v2-lite-chat` ≡ `deepseek-ai/DeepSeek-V2-Lite-Chat`,
`nvidia.nemotron-nano-9b-v2` ≡ `nvidia/NVIDIA-Nemotron-Nano-9B-v2`). Those don't
normalize-collide (the real repo's name repeats the org), and org-tagging them in
isolation would collide on the humanized display name — they need a fold into the
real repo, which is a separate concern. Detection: same developer + same
org-token-stripped model name as a more-canonical (slashed) entry.
"""
import re
from collections import defaultdict

_LEAD_RE = re.compile(r"([A-Za-z0-9]+)([-.])(.+)")
_PLACEHOLDER_ORGS = frozenset({"unknown", "none", "na", ""})
_SEP = re.compile(r"[-_.\s]+")


def _dev_of(e: dict, hf_to_dev: dict) -> str | None:
    org = e.get("org_id")
    if org and str(org).lower() not in _PLACEHOLDER_ORGS:
        return str(org).lower()
    cid = str(e.get("id", ""))
    if "/" in cid:
        return hf_to_dev.get(cid.split("/", 1)[0].lower())
    m = _LEAD_RE.match(cid)
    return hf_to_dev.get(m.group(1).lower()) if m else None


def _name_key(cid: str, dev: str | None) -> str:
    """Model name normalized after stripping a leading developer token, so
    `deepseek-ai/DeepSeek-V2-Lite-Chat` and `deepseek-v2-lite-chat` share a key."""
    name = cid.split("/", 1)[1] if "/" in cid else cid
    nl = name.lower()
    if dev:
        for tok in (dev, dev.replace("-", ""), dev.split("-")[0]):
            if tok and nl.startswith(tok):
                nl = nl[len(tok):]
                break
    return _SEP.sub("", nl).strip("-_. ")


def attribute_orgs(entries: list[dict], hf_to_dev: dict[str, str]):
    """For each malformed-org draft id whose leading token is a known org:
    - if it DUPLICATES an existing real repo of that org (same developer + same
      org-token-stripped model name), return it in the merge map {draft -> real
      repo} so the fold collapses it into the real repo;
    - otherwise attribute org_id (standalone draft) in place.

    Returns (entries, merge_map). The merge map carries the cross-spelling
    duplicates the resolver's normalize missed (it's org-blind: `deepseek` vs
    `deepseek-ai`, the org repeated in the model name, letter-dot not split)."""
    seen: dict[tuple, list[str]] = defaultdict(list)
    for e in entries:
        dev = _dev_of(e, hf_to_dev)
        if dev:
            seen[(dev, _name_key(str(e.get("id", "")), dev))].append(str(e.get("id", "")))

    merges: dict[str, str] = {}
    for e in entries:
        cid = str(e.get("id", ""))
        if "/" in cid:
            continue
        org = e.get("org_id")
        if org and str(org).lower() not in _PLACEHOLDER_ORGS:
            continue
        m = _LEAD_RE.match(cid)
        if not m:
            continue
        dev = hf_to_dev.get(m.group(1).lower())
        if not dev:
            continue
        twins = [i for i in seen.get((dev, _name_key(cid, dev)), []) if i != cid and "/" in i]
        if twins:
            merges[cid] = sorted(twins)[0]   # fold the draft into the real repo
        else:
            e["org_id"] = dev                 # standalone draft: just attribute
    return entries, merges
