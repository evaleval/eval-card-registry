#!/usr/bin/env python3
"""Relabel mislabeled parent edges + drop a spurious version edge
+ merge the baichuan org dup. No canonical_id changes (no re-pin); only edge
`relationship`/`axis` fixes and one `org_id` repoint. Reviewed explicit lists.

Dry-run by default; pass --apply to write seed/models/core.yaml +
seed/orgs.generated.yaml.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"
ORGS_GEN = ROOT / "seed" / "orgs.generated.yaml"

# --- reviewed reclassification of edges currently mislabeled `quantized` ------
# community finetunes — a finetune is NOT identity-preserving.
TO_FINETUNE = {
    "HelpingAI/Dhanishtha",
    "NikolaSigmoid/AceMath-1.5B-Instruct-dolphin-r1-200",
    "divyanshukunwar/SASTRI_1_9B",
    "ell44ot/gemma-2b-def",
    "meditsolutions/Llama-3.2-SUN-1B-Instruct",
    "meditsolutions/Llama-3.2-SUN-1B-chat",
    "mkurman/llama-3.2-MEDIT-3B-o1",
    "ngxson/MiniThinky-1B-Llama-3.2",
    "ngxson/MiniThinky-v2-1B-Llama-3.2",
    "prithivMLmods/Bellatrix-Tiny-1B-v2",
    "prithivMLmods/FastThink-0.5B-Tiny",
    "yasserrmd/Coder-GRPO-3B",
    "OpenLLM-France/Lucie-7B-Instruct-human-data",
}
TO_MERGE = {
    "vhab10/Llama-3.2-Instruct-3B-TIES",
    "vhab10/llama-3-8b-merged-linear",
}
# instruct variants mislabeled as quant -> training_stage (leaves the group walk)
TO_TRAINING_STAGE = {
    "HuggingFaceTB/SmolLM-135M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
}
# KEPT as quantized on purpose (NOT touched): *-instruct-turbo (Together FP8,
# real quant -> batch-3 serving/precision cleanup), *-litert-preview (LiteRT quant).

# spurious cross-org version edge to drop (wrong lineage; pollutes model_family_id)
DROP_VERSION_EDGE = {("mbzuai/k2-think-v2", "LLM360/K2-Think")}

# org merge: the baichuan dup -> the real HF namespace baichuan-inc
ORG_REPOINT = {"baichuan": "baichuan-inc"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    changes = []

    for e in entries:
        if not isinstance(e, dict):
            continue
        cid = str(e.get("id", ""))
        new_parents = []
        for p in (e.get("parents") or []):
            if not isinstance(p, dict):
                new_parents.append(p); continue
            pid = p.get("id")
            # drop spurious version edge
            if (cid, pid) in DROP_VERSION_EDGE and p.get("relationship") == "variant":
                changes.append((cid, f"DROP version edge -> {pid}"))
                continue
            if p.get("relationship") == "quantized":
                if cid in TO_FINETUNE:
                    p = {**p, "relationship": "finetune"}; p.pop("axis", None)
                    changes.append((cid, f"quantized->finetune ({pid})"))
                elif cid in TO_MERGE:
                    p = {**p, "relationship": "merge"}; p.pop("axis", None)
                    changes.append((cid, f"quantized->merge ({pid})"))
                elif cid in TO_TRAINING_STAGE:
                    p = {"id": pid, "relationship": "variant", "axis": "training_stage"}
                    changes.append((cid, f"quantized->variant/training_stage ({pid})"))
            new_parents.append(p)
        e["parents"] = new_parents
        # org repoint
        if isinstance(e.get("org_id"), str) and e["org_id"] in ORG_REPOINT:
            new = ORG_REPOINT[e["org_id"]]
            changes.append((cid, f"org_id {e['org_id']}->{new}"))
            if args.apply:
                e["org_id"] = new

    # drop the merged-away org row from orgs.generated.yaml
    org_doc = yaml.safe_load(ORGS_GEN.read_text()) if ORGS_GEN.exists() else []
    org_list = org_doc if isinstance(org_doc, list) else (org_doc or [])
    kept_orgs = [o for o in org_list if not (isinstance(o, dict) and o.get("id") in ORG_REPOINT)]
    dropped_orgs = [o.get("id") for o in org_list if isinstance(o, dict) and o.get("id") in ORG_REPOINT]

    print(f"edge/org changes: {len(changes)}")
    for c in changes:
        print(f"   {c[0]:55} {c[1]}")
    print(f"org rows to drop from orgs.generated.yaml: {dropped_orgs}")

    if args.apply:
        hdr = CORE.read_text().split("\n", 1)[0]
        CORE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000))
        ORGS_GEN.write_text(
            (ORGS_GEN.read_text().split("\n- ", 1)[0].rstrip() + "\n" if ORGS_GEN.exists() else "")
            + yaml.safe_dump(kept_orgs, sort_keys=False, allow_unicode=True, width=10_000)
        )
        print("APPLIED.")
    else:
        print("(dry-run; pass --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
