"""Unit tests for the models.dev dedup / group / mint / alias logic.

OFFLINE: drives the committed reference sidecars under
specs/model-resolution-rework/ plus the PINNED models.dev API snapshot at
tests/fixtures/modelsdev_api.snapshot.json. No network, no seed --local.
The snapshot and the underlying index are frozen from the SAME models.dev pull,
so the dedup count and the group roots reproduce EXACTLY (no drift band).

Validates:
  - the ported dedup (normalize -> canon_key_ordered -> safe_sig union-find)
    reproduces the committed 1,274-group underlying index;
  - head selection honours author-lab > re-host > null-org precedence;
  - the 663 EEE rescues map to their recorded underlying groups;
  - edge axis classification: training_stage / tier (no scale claim) /
    size (only on disclosed open-weight tokens);
  - every PROVIDER_TO_INFERENCE_PLATFORM target id exists in the curated
    inference_platforms catalog.
"""
from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = REPO_ROOT / "specs" / "model-resolution-rework"
API_CACHE = REPO_ROOT / "tests" / "fixtures" / "modelsdev_api.snapshot.json"
UNDERLYING_INDEX = SPEC_DIR / "modelsdev_underlying_index.json"
RESCUES = SPEC_DIR / "eee_modelsdev_rescues.json"
PLATFORMS = SPEC_DIR / "inference_platforms.proposed.json"


def _load_module():
    path = REPO_ROOT / "scripts" / "refresh_from_modelsdev.py"
    spec = importlib.util.spec_from_file_location("refresh_from_modelsdev", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


@pytest.fixture(scope="module")
def api():
    if not API_CACHE.exists():
        pytest.skip(f"pinned models.dev API snapshot missing at {API_CACHE}")
    return json.loads(API_CACHE.read_text())


@pytest.fixture(scope="module")
def underlying_index():
    if not UNDERLYING_INDEX.exists():
        pytest.skip(f"underlying index sidecar missing at {UNDERLYING_INDEX}")
    return json.loads(UNDERLYING_INDEX.read_text())


# ---------------------------------------------------------------------------
# 1. Dedup reproduces the committed 1,283 underlying groups.
# ---------------------------------------------------------------------------
def test_dedup_reproduces_underlying_group_count(mod, api, underlying_index):
    groups = mod.build_underlying_groups(api)
    n = len(groups)
    # The snapshot and the committed index are frozen from the SAME models.dev
    # pull, so the count is EXACT — no drift band.
    assert n == 1_283, f"expected 1283 underlying groups, got {n}"
    assert len(underlying_index) == 1_283


def test_dedup_group_roots_match_committed_index(mod, api, underlying_index):
    """The generator's group roots must reproduce the committed reference index
    EXACTLY — proving the ported normalize/canon_key/safe_sig logic is unchanged.

    The snapshot (tests/fixtures/modelsdev_api.snapshot.json) and the reference
    `modelsdev_underlying_index.json` are frozen from the SAME pull, so there is
    no snapshot drift to tolerate: any difference in the root set is a real
    normalize/canon_key/safe_sig regression. To re-pin against a refreshed API,
    replace the snapshot fixture AND regenerate the index from it with the same
    normalize/canon_key/safe_sig logic."""
    my_roots = set(mod.build_underlying_groups(api).keys())
    idx_roots = set(underlying_index.keys())
    assert my_roots == idx_roots, (
        f"root set diverged from the frozen index — "
        f"new={sorted(my_roots - idx_roots)[:8]} "
        f"gone={sorted(idx_roots - my_roots)[:8]}"
    )


@pytest.mark.parametrize(
    "root,expect_author,expect_org_hf",
    [
        ("claude-3-5-sonnet", True, "anthropic"),
        ("gpt-4o", True, "openai"),
        ("yi-large", False, None),          # re-host-only, org-unknown
        ("abliterated-model", False, None),  # community re-host, no author lab
    ],
)
def test_dedup_spotcheck_known_groups(mod, api, underlying_index, root, expect_author, expect_org_hf):
    groups = mod.build_underlying_groups(api)
    assert root in groups, f"group {root!r} not produced"
    head = mod.pick_underlying(root, groups[root])
    assert head["has_author_lab_entry"] is expect_author
    assert head["author_org"] == expect_org_hf
    # The committed index agrees with our head pick.
    idx_entry = underlying_index[root]
    assert idx_entry["has_author_lab_entry"] is expect_author
    assert idx_entry["author_org"] == expect_org_hf


# ---------------------------------------------------------------------------
# 2. Head selection precedence: author-lab > re-host > null.
# ---------------------------------------------------------------------------
def test_head_pick_author_lab_beats_rehost(mod):
    """A small synthetic group: the author-lab provider's org/name anchor wins
    over re-host providers, even when re-hosts disagree on date/name."""
    root = "claude-3-5-sonnet"
    recs = [
        # Re-host with a (wrong) earlier date and a different name.
        dict(provider="venice", raw="claude-3-5-sonnet", norm="claude-3-5-sonnet",
             key=root, family="claude", release="2023-01-01", name="Sonnet (venice)",
             open_weights=False, record={}),
        # The true author lab.
        dict(provider="anthropic", raw="claude-3-5-sonnet-20240620",
             norm="claude-3-5-sonnet-20240620", key=root, family="claude",
             release="2024-06-20", name="Claude 3.5 Sonnet", open_weights=False, record={}),
        # Another re-host.
        dict(provider="openrouter", raw="anthropic/claude-3.5-sonnet",
             norm="claude-3-5-sonnet", key=root, family="claude", release="2024-06-21",
             name="Claude 3.5 Sonnet", open_weights=False, record={}),
    ]
    head = mod.pick_underlying(root, recs)
    assert head["has_author_lab_entry"] is True
    assert head["author_org"] == "anthropic"
    # Display name comes from the author-lab record, not the re-host's vanity name.
    assert head["display_name"] == "Claude 3.5 Sonnet"
    # Head spelling is the author-lab's own spelling.
    assert head["head_spelling"] == "claude-3-5-sonnet-20240620"


def test_head_pick_rehost_only_infers_org_from_family(mod):
    """No author-lab provider in the group, but the family token implies an org
    (re-host > null): org inferred, has_author_lab_entry stays False."""
    root = "llama-3-1-8b"
    recs = [
        dict(provider="togetherai", raw="meta-llama/Llama-3.1-8B",
             norm="llama-3-1-8b", key=root, family="llama", release="2024-07-23",
             name="Llama 3.1 8B", open_weights=True, record={}),
        dict(provider="fireworks-ai", raw="llama-v3p1-8b",
             norm="llama-3-1-8b", key=root, family="llama", release=None,
             name="Llama 3.1 8B", open_weights=True, record={}),
    ]
    head = mod.pick_underlying(root, recs)
    assert head["has_author_lab_entry"] is False
    assert head["author_org"] == "meta-llama"   # inferred from family, HF-style
    assert head["open_weights"] is True          # any(True)


def test_head_pick_null_org_when_no_signal(mod):
    """No author lab and no family-org signal -> org stays null (the org-less
    bucket); head spelling falls back to the cleanest normalised spelling."""
    root = "abliterated-model"
    recs = [
        dict(provider="abliteration-ai", raw="abliterated-model",
             norm="abliterated-model", key=root, family=None, release="2026-01-06",
             name="Abliterated Model", open_weights=True, record={}),
    ]
    head = mod.pick_underlying(root, recs)
    assert head["has_author_lab_entry"] is False
    assert head["author_org"] is None
    assert head["head_spelling"] == "abliterated-model"


# ---------------------------------------------------------------------------
# 3. The 663 EEE rescues map to their recorded underlying groups.
# ---------------------------------------------------------------------------
def test_rescues_map_to_underlying_groups(mod, underlying_index):
    if not RESCUES.exists():
        pytest.skip(f"rescues sidecar missing at {RESCUES}")
    rescues = json.loads(RESCUES.read_text())
    assert len(rescues) == 663

    key_to_root = {r: r for r in underlying_index}
    mset_to_root: dict[str, list[str]] = defaultdict(list)
    for r in underlying_index:
        mset_to_root[mod.safe_sig(r)].append(r)

    def lookup(eee_id: str) -> str | None:
        n = mod.normalize_modelsdev_id(eee_id)
        k = mod.canon_key_ordered(n)
        if not k:
            return None
        if k in key_to_root:
            return k
        ms = mod.safe_sig(k)
        if ms in mset_to_root:
            return mset_to_root[ms][0]
        return None

    matched = sum(1 for eee_id, rec in rescues.items()
                  if lookup(eee_id) == rec["underlying_model_id"])
    # The ported normalize/canon must reproduce every recorded match.
    assert matched == 663, f"only {matched}/663 rescues reproduced"


def test_rescues_sample_spotcheck(mod, underlying_index):
    """Spot-check a handful of named rescues resolve to the expected group."""
    if not RESCUES.exists():
        pytest.skip("rescues sidecar missing")
    rescues = json.loads(RESCUES.read_text())
    sample = [eee for eee in ("01-ai/yi-large", "01-ai/Yi-1.5-34B-Chat") if eee in rescues]
    assert sample, "expected at least one known sample id in the rescues"
    for eee_id in sample:
        n = mod.normalize_modelsdev_id(eee_id)
        k = mod.canon_key_ordered(n)
        root = k if k in underlying_index else None
        if root is None:
            ms = mod.safe_sig(k)
            for r in underlying_index:
                if mod.safe_sig(r) == ms:
                    root = r
                    break
        assert root == rescues[eee_id]["underlying_model_id"]


# ---------------------------------------------------------------------------
# 4. Edge axis classification.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "token,expected",
    [
        ("instruct", ("variant", "training_stage")),
        ("chat", ("variant", "training_stage")),
        ("it", ("variant", "training_stage")),
        ("base", ("variant", "training_stage")),
        ("sonnet", ("variant", "tier")),
        ("haiku", ("variant", "tier")),
        ("opus", ("variant", "tier")),
        ("mini", ("variant", "tier")),
        ("nano", ("variant", "tier")),
        ("flash", ("variant", "tier")),
        ("pro", ("variant", "tier")),
        ("7b", ("variant", "size")),
        ("70b", ("variant", "size")),
        ("405b", ("variant", "size")),
        ("8x7b", ("variant", "size")),
        ("a16b", ("variant", "size")),
        ("thinking", ("variant", "mode")),
        ("vision", ("variant", "modality")),
        ("coder", ("variant", "domain")),
    ],
)
def test_classify_token_axis(mod, token, expected):
    assert mod._classify_token(token) == expected


def test_branded_tier_never_gets_size_edge(mod):
    """A branded tier (no disclosed scale) must classify as `tier`, NEVER
    `size`. Every tier token is guarded."""
    for tok in ("haiku", "sonnet", "opus", "mini", "nano", "flash", "pro",
                "small", "medium", "large"):
        rel, axis = mod._classify_token(tok)
        assert axis == "tier", f"{tok} mis-classified as {axis}"
        assert axis != "size"
        assert tok in mod._TIER_TOKENS


def test_size_token_only_for_disclosed_open_weight_tokens(mod):
    """A `size` edge is asserted ONLY for a genuinely-disclosed scale token
    (open-weight name params / MoE specs) — never for a branded tier or an
    unrelated token."""
    assert mod._is_size_token("70b") is True
    assert mod._is_size_token("8x7b") is True
    assert mod._is_size_token("a16b") is True
    assert mod._is_size_token("1.5b") is True
    assert mod._is_size_token("mini") is False
    assert mod._is_size_token("sonnet") is False
    assert mod._is_size_token("instruct") is False


def test_suffix_chain_llama_size_then_stage(mod):
    """A compound open-weight suffix `70b-instruct` classifies as a size edge
    followed by a training_stage edge (size first because the family root has no
    size token here)."""
    segments = mod._classify_suffix_segments("70b-instruct")
    assert segments == [
        ("variant", "size", "70b"),
        ("variant", "training_stage", "instruct"),
    ]


def test_instruct_suffix_is_training_stage_in_family_tree(mod):
    """End-to-end: a models.dev `-instruct` snapshot emits a variant/
    training_stage edge (NOT axis=mode) in the built family tree."""
    api = {
        "mistral": {
            "id": "mistral", "name": "Mistral", "models": {
                "mistral-7b-instruct": {
                    "id": "mistral-7b-instruct", "name": "Mistral 7B Instruct",
                    "release_date": "2023-09-27", "open_weights": True, "family": "mistral",
                },
            },
        },
    }
    out, _missing = mod._generate_models(api, {"mistralai"})
    by_id = {e["id"]: e for e in out}
    instruct = by_id["mistralai/mistral-7b-instruct"]
    assert instruct["parents"] == [{
        "id": "mistralai/mistral-7b", "relationship": "variant", "axis": "training_stage",
    }]


# ---------------------------------------------------------------------------
# 5. PROVIDER_TO_INFERENCE_PLATFORM ids all exist in the curated catalog.
# ---------------------------------------------------------------------------
def test_provider_to_inference_platform_ids_exist(mod):
    if not PLATFORMS.exists():
        pytest.skip(f"inference platforms sidecar missing at {PLATFORMS}")
    catalog = json.loads(PLATFORMS.read_text())
    valid_ids = {p["id"] for p in catalog["platforms"]}
    valid_providers = {p["models_dev_provider"] for p in catalog["platforms"]}

    assert mod.PROVIDER_TO_INFERENCE_PLATFORM, "map is empty"
    # Maps all 137 providers (the ~122 previously discarded + the author labs).
    assert len(mod.PROVIDER_TO_INFERENCE_PLATFORM) == len(catalog["platforms"]) == 137

    for provider, platform_id in mod.PROVIDER_TO_INFERENCE_PLATFORM.items():
        assert platform_id in valid_ids, f"platform id {platform_id!r} not in catalog"
        assert provider in valid_providers, f"provider {provider!r} not a known slug"


def test_strict_author_sourced_from_catalog(mod):
    """STRICT_AUTHOR is the kind=author_lab subset of the same catalog."""
    if not PLATFORMS.exists():
        pytest.skip("inference platforms sidecar missing")
    catalog = json.loads(PLATFORMS.read_text())
    author_providers = {
        p["models_dev_provider"] for p in catalog["platforms"]
        if p["kind"] == "author_lab"
    }
    assert mod.STRICT_AUTHOR == author_providers
    assert len(mod.STRICT_AUTHOR) == 29


# ---------------------------------------------------------------------------
# 6. Full-catalog generation: provider-preserving group -> mint -> alias.
# ---------------------------------------------------------------------------
def test_full_generation_seeds_closed_api_and_rehosts(mod, api):
    """The refactor seeds the FULL catalog (closed-API + re-hosts), not just
    author labs; re-host-only / minted groups are draft, author trees reviewed;
    provider spellings are aliased carrying their platform."""
    known = mod._load_known_org_ids()
    out, skipped = mod._generate_models(api, known)
    assert skipped == [], f"unexpected missing-org skips: {skipped[:5]}"

    by_id = {e["id"]: e for e in out}
    # Closed-API author lab is present and reviewed.
    assert "openai/gpt-4o" in by_id
    assert by_id["openai/gpt-4o"]["review_status"] == "reviewed"
    # A re-host-only / org-less group is minted as draft + org-unknown.
    assert "yi-large" in by_id
    yi = by_id["yi-large"]
    assert yi["org_id"] is None
    assert yi["review_status"] == "draft"
    assert "org-unknown" in yi["tags"]

    # No malformed (multi-slash) canonical ids.
    assert all(e["id"].count("/") <= 1 for e in out)

    # Provider-tagged aliases: a multi-provider author group carries platform
    # provenance folded into metadata.alias_platforms after finalize.
    fin = mod._finalize_entries([dict(e) for e in out])
    fin_by_id = {e["id"]: e for e in fin}
    gpt4o = fin_by_id["openai/gpt-4o"]
    meta = json.loads(gpt4o["metadata"])
    assert meta.get("alias_platforms"), "expected provider platform provenance"
    # Every tagged platform id is a real catalog id.
    catalog = json.loads(PLATFORMS.read_text())
    valid_ids = {p["id"] for p in catalog["platforms"]}
    for plat in meta["alias_platforms"].values():
        assert plat in valid_ids
    # No gnarly aliases survived finalize.
    for e in fin:
        for a in e["aliases"]:
            assert a.count("/") <= 1
            assert not a.startswith("@")


# ---------------------------------------------------------------------------
# 7. Mint-decision rule: defer HF-resolvable groups, mint off-HF ones.
# ---------------------------------------------------------------------------
HF_ORACLE_JSON = REPO_ROOT.parent / "hf_model_id_resolution.json"


def test_mint_decision_defers_on_hf_group_and_mints_off_hf(mod):
    """A models.dev group that IS a real HF repo defers to the HF id (spellings
    become aliases, no shadow mint); a genuinely off-HF group still mints.

    Uses a SYNTHETIC authority so the test is deterministic regardless of the
    live oracle contents, and exercises the hard `qwen-qwq-32b` vs `Qwen/QwQ-32B`
    case (differs in BOTH org spelling and brand-prefixed name)."""
    from eval_entity_resolver.normalization import normalize as _nz

    alias_index = mod._build_org_alias_index()
    # Synthetic authority: {dev_org: {normalized_name: real_hf_id}}.
    authority = {
        "alibaba": {_nz("QwQ-32B"): "Qwen/QwQ-32B"},
    }

    # (a) On-HF, hard case: models.dev key carries the brand prefix
    # (alibaba/qwen-qwq-32b), HF org is Qwen->alibaba, name qwen-qwq-32b->qwq-32b.
    deferred = mod._hf_defer_target(
        "alibaba/qwen-qwq-32b",
        "alibaba",
        ["alibaba/qwen-qwq-32b", "qwen-qwq-32b", "Qwen QwQ 32B"],
        alias_index,
        authority,
    )
    assert deferred == "Qwen/QwQ-32B", "on-HF brand-prefixed group must DEFER"

    # (b) On-HF, easy case: name already matches without prefix-strip.
    assert (
        mod._hf_defer_target(
            "alibaba/qwq-32b", "alibaba", ["QwQ-32B", "Qwen/QwQ-32B"], alias_index, authority
        )
        == "Qwen/QwQ-32B"
    )

    # (c) Off-HF closed-API group: nothing in authority -> MINT (None).
    assert (
        mod._hf_defer_target(
            "anthropic/claude-opus-4-5",
            "anthropic",
            ["claude-opus-4-5", "Claude Opus 4.5"],
            alias_index,
            authority,
        )
        is None
    )

    # (d) No false merge across DIFFERENT developers: a group named qwq-32b but
    # authored by openai must NOT defer to the alibaba HF id.
    assert (
        mod._hf_defer_target(
            "openai/qwq-32b", "openai", ["qwq-32b"], alias_index, authority
        )
        is None
    )

    # (e) Org-less group never defers (no org agreement possible).
    assert (
        mod._hf_defer_target("qwq-32b", None, ["qwq-32b"], alias_index, authority)
        is None
    )


def test_mint_decision_emits_hf_deferred_record_in_generation(mod, api):
    """End-to-end against the pinned snapshot + frozen oracle: the QwQ-32B shadow
    group defers to the real `Qwen/QwQ-32B` (an hf_deferred record carrying the
    models.dev spellings as aliases, NOT a shadow `alibaba/qwen-qwq-32b` mint),
    while closed-API families (Claude/Grok) still mint off-HF canonicals."""
    if not HF_ORACLE_JSON.exists():
        pytest.skip(f"HF oracle missing at {HF_ORACLE_JSON}")
    # Reset the module-cached authority so it builds from the real oracle.
    mod._HF_AUTHORITY = None

    known = mod._load_known_org_ids()
    out, skipped = mod._generate_models(api, known)
    assert skipped == [], f"unexpected missing-org skips: {skipped[:5]}"
    by_id = {e["id"]: e for e in out}

    def _is_deferred(e):
        return json.loads(e.get("metadata") or "{}").get("hf_deferred") is True

    # DEFER: the real HF id is present as an hf_deferred record; the shadow mint
    # id is NOT a canonical, and is carried as an alias on the HF entry instead.
    assert "Qwen/QwQ-32B" in by_id, "expected the real HF id to be emitted"
    qwq = by_id["Qwen/QwQ-32B"]
    assert _is_deferred(qwq)
    assert qwq["resolution_source"] == "models_dev"
    assert "alibaba/qwen-qwq-32b" not in by_id, "shadow dupe must NOT be minted"
    assert "alibaba/qwen-qwq-32b" in qwq["aliases"], "shadow spelling must be an alias"

    # MINT: genuinely off-HF closed-API families still mint {org}/{slug}.
    grok = [e for e in out if e["id"].startswith("xai/grok-3") and not _is_deferred(e)]
    assert grok, "off-HF Grok family must still mint"
    claude = [
        e for e in out if e["id"].startswith("anthropic/claude") and not _is_deferred(e)
    ]
    assert claude, "off-HF Claude family must still mint"

    # The defer path produced at least the known hand-fold shape.
    deferred_ids = {e["id"] for e in out if _is_deferred(e)}
    assert "Qwen/QwQ-32B" in deferred_ids
