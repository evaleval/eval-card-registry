#!/usr/bin/env python3
"""For each sweep wrong-fold with no existing canonical, check our OFFLINE HF
authorities for the model's real identity, so minting uses a real HF id (or we
discover it already exists and should repoint instead of mint).

Authorities (all offline):
  - oracle resolved-repo set: every fixed_hf_model_id in hf_model_id_resolution
    (a real HF repo referenced by some EEE submission) — authoritative "on HF".
  - hub_stats.generated.yaml ids (from cfahlgren1/hub-stats).
  - existing canonical ids (flexible normalized-substring match), to catch a
    model already present under a different spelling.

Per case prints: real-id guess, whether an HF repo / hub-stats id / canonical
matches it (with the matched id), so a human can decide mint-with-real-id vs
repoint vs discuss.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"
HUB = ROOT / "seed" / "models" / "sources" / "hub_stats.generated.yaml"
ORACLE = Path("/Users/jchim/projects/evaleval/hf_model_id_resolution.json")

_SEP = re.compile(r"[/_.:\-]+")


def norm(s: str) -> str:
    return _SEP.sub("-", s.lower()).strip("-")


# alias -> (real-identity search term as a normalized HF-body guess, note)
GUESSES = {
    "moonshotai/k2": ("kimi-k2", "Moonshot Kimi K2 (ambiguous: which K2?)"),
    "unknown/lfm-3b": ("lfm-3b", "LiquidAI LFM-3B (orig API-only?)"),
    "unknown/lfm-7b": ("lfm-7b", "LiquidAI LFM-7B?"),
    "unknown/starling-lm-alpha-8x7b-moe-gptq": ("starling-lm", "Starling-LM 8x7B MoE GPTQ?"),
    "unknown/llava-1-6-vicuna-13b": ("llava-v1-6-vicuna-13b", "LLaVA-1.6 Vicuna 13B"),
    "unknown/gpt-3": ("gpt-3", "OpenAI GPT-3 (closed)"),
    "unknown/internvl2-5-72b": ("internvl2-5-72b", "InternVL2.5-72B (does 72B exist?)"),
    "unknown/skywork-prm-1-5b": ("skywork-prm-1-5b", "Skywork PRM 1.5B"),
    "unknown/videollama3-8b": ("videollama3-8b", "VideoLLaMA3-8B (does 8B exist?)"),
    "otter-9b": ("otter-9b", "Otter-9B?"),
    "unknown/otter-9b": ("otter-9b", "Otter-9B?"),
    "unknown/llama-3-1-nemotron-nano-vl-8b-v1": ("llama-3-1-nemotron-nano-vl-8b", "NVIDIA Nemotron-Nano-VL-8B"),
    "unknown/dall-e-mini": ("dalle-mini", "DALL-E Mini / craiyon"),
    "unknown/llava-video": ("llava-next-video", "LLaVA-NeXT-Video"),
    "unknown/gpt-3-davinci-002": ("davinci-002", "OpenAI davinci-002 (closed)"),
    "unknown/gpt-3-davinci-003": ("text-davinci-003", "OpenAI text-davinci-003 (closed)"),
    "unknown/llava-next-v-7b": ("llava-next-7b", "LLaVA-NeXT 7B (v vs m?)"),
    "unknown/gpt-3-1-3b-babbage-002": ("babbage-002", "OpenAI babbage-002 (closed)"),
    "unknown/instructgpt-curie-v1": ("text-curie-001", "OpenAI InstructGPT curie (closed)"),
    "unknown/minicpm-llama3-v-2-5": ("minicpm-llama3-v-2-5", "MiniCPM-Llama3-V-2.5"),
    "unknown/minicpm-llama3-v2-5": ("minicpm-llama3-v-2-5", "MiniCPM-Llama3-V-2.5"),
    "unknown/instructgpt-davinci-v2": ("text-davinci-002", "OpenAI InstructGPT davinci v2 (closed)"),
    "ai2/tulu-2-7b-rm-v0-nectar-binarized-3-8m-check": ("tulu-2-7b-rm", "AllenAI Tulu-2-7B reward model"),
    "unknown/bloomz-1-1b": ("bloomz-1b1", "BLOOMZ-1b1"),
    "unknown/bloomz-1-7b": ("bloomz-1b7", "BLOOMZ-1b7"),
    "unknown/bloomz-560m": ("bloomz-560m", "BLOOMZ-560m"),
    "unknown/bloomz": ("bloomz", "BLOOMZ-176B"),
}


def main() -> int:
    doc = yaml.safe_load(CORE.read_text())
    entries = [e for e in (doc["entries"] if isinstance(doc, dict) else doc)
               if isinstance(e, dict) and isinstance(e.get("id"), str)]
    canon_norm = {norm(e["id"]): e["id"] for e in entries}

    oracle = json.loads(ORACLE.read_text())["resolutions"]
    hf_repos = set()
    for v in oracle.values():
        fid = v.get("fixed_hf_model_id")
        if isinstance(fid, str) and "/" in fid:
            hf_repos.add(fid)
    hf_norm = {norm(r): r for r in hf_repos}

    hub_ids = set()
    if HUB.exists():
        hd = yaml.safe_load(HUB.read_text()) or {}
        for e in (hd.get("entries") if isinstance(hd, dict) else hd) or []:
            if isinstance(e, dict) and isinstance(e.get("id"), str):
                hub_ids.add(e["id"])
    hub_norm = {norm(i): i for i in hub_ids}

    def find(term: str, table: dict[str, str]) -> str | None:
        t = norm(term)
        if t in table:
            return table[t]
        hits = [v for k, v in table.items() if t in k or k.endswith("-" + t)]
        return hits[0] if hits else None

    for alias, (term, note) in GUESSES.items():
        on_hf = find(term, hf_norm)
        in_hub = find(term, hub_norm)
        as_canon = find(term, canon_norm)
        print(f"\n• {alias}   [{note}]")
        print(f"    search   : {term!r}")
        print(f"    on_HF    : {on_hf or '—'}")
        print(f"    hub_stats: {in_hub or '—'}")
        print(f"    canonical: {as_canon or '—'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
