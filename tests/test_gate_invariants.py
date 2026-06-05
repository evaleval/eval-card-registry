"""Acceptance-gate invariants for the model-resolution reconcile work.

These are PERMANENT regression tests: they encode the acceptance gate as pytest
cases that must hold across all future reconcile work. They run OFFLINE against
the committed `fixtures/` parquet warehouse and the frozen live-HF oracle
`hf_model_id_resolution.json` (at the evaleval workspace root).

The gate has two complementary surfaces:
  - the resolver behaviour (does every EEE id resolve, and do the 4,074 HF ids
    resolve ORG-AWARE to their HF-true `fixed_hf_model_id`); and
  - the canonical graph itself (dedup, no dangling parent edges, total
    group/family membership, null-at-origin lineage, Tier-3 honesty,
    no name-only cross-org merges).

Heavy cases (the full 6,720-id resolve sweep + the 4,074 oracle sweep) are
marked `@pytest.mark.slow` so they can be selected/deselected, but they stay
runnable in CI (the sweep is sub-second against the parquet fixtures).

The numbers asserted are the gate floor: COVERAGE must be 6,720/6,720 non-null;
ORACLE must be 4,074/4,074 org-aware-correct; case-insensitive canonical dups
must be 0; dangling parent edges must be 0. A regression that breaks any of
these MUST fail this module.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pytest
import yaml

from eval_entity_resolver.resolver import Resolver
from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES


# --------------------------------------------------------------------------
# Locations (offline; everything is committed)
# --------------------------------------------------------------------------
REGISTRY_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REGISTRY_ROOT / "fixtures"
ORGS_YAML = REGISTRY_ROOT / "seed" / "orgs.yaml"
# The frozen live-HF oracle lives at the evaleval workspace root.
ORACLE_PATH = REGISTRY_ROOT.parent / "hf_model_id_resolution.json"

# Gate floor numbers (the registry's measured baseline).
EXPECTED_TOTAL = 6720
EXPECTED_ORACLE = 4074  # fixed_exact + fixed_near_miss


# --------------------------------------------------------------------------
# Module-scoped fixtures (load the warehouse + oracle once)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def resolver() -> Resolver:
    """Resolver wired to the committed parquet fixtures (canonical graph +
    alias index). Same construction path the oracle-comparison script uses."""
    assert FIXTURES_DIR.exists(), f"missing fixtures dir: {FIXTURES_DIR}"
    return Resolver.from_parquet(str(FIXTURES_DIR))


@pytest.fixture(scope="module")
def models_df() -> pd.DataFrame:
    """The canonical_models table as materialised in fixtures (post-seed:
    the derived walk columns are already populated)."""
    df = pd.read_parquet(FIXTURES_DIR / "canonical_models.parquet")
    assert not df.empty
    return df


@pytest.fixture(scope="module")
def oracle() -> dict[str, dict]:
    assert ORACLE_PATH.exists(), f"missing oracle: {ORACLE_PATH}"
    return json.loads(ORACLE_PATH.read_text())["resolutions"]


@pytest.fixture(scope="module")
def hf_to_dev() -> dict[str, str]:
    """The two-tier HF-namespace → developer-org map: the static curated
    `_ORG_ALIASES` plus any `hf_org` aliases declared on curated orgs. Used
    for the ORG-AWARE oracle comparison (meta-llama ↔ meta etc.)."""
    mapping = {k.lower(): v for k, v in _ORG_ALIASES.items()}
    for entry in yaml.safe_load(ORGS_YAML.read_text()) or []:
        if isinstance(entry, dict) and isinstance(entry.get("hf_org"), str) and isinstance(entry.get("id"), str):
            mapping[entry["hf_org"].lower()] = entry["id"]
    return mapping


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _org_of(cid: str) -> str:
    return cid.split("/", 1)[0] if "/" in cid else ""


def _name_of(cid: str) -> str:
    return cid.split("/", 1)[1] if "/" in cid else cid


def _parse_parents(value) -> list[dict]:
    """`parents` is stored as a JSON string; may also arrive as a list or
    pandas NA. Normalise to a list of edge dicts."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value) or []
        except (ValueError, TypeError):
            return []
    # array-like (pyarrow list) or scalar NA
    try:
        if pd.isna(value):  # scalar NA
            return []
    except (ValueError, TypeError):
        pass  # array-like → not a scalar NA
    return [e for e in list(value) if isinstance(e, dict)]


def _parse_tags(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value) or []
        except (ValueError, TypeError):
            return []
    try:
        if pd.isna(value):
            return []
    except (ValueError, TypeError):
        pass
    return list(value)


# --------------------------------------------------------------------------
# Sanity: the oracle has the expected shape (guards against a swapped/empty file)
# --------------------------------------------------------------------------
def test_oracle_has_expected_population(oracle):
    assert len(oracle) == EXPECTED_TOTAL, (
        f"oracle key count drifted: {len(oracle)} != {EXPECTED_TOTAL}"
    )
    n_oracle = sum(
        1 for v in oracle.values()
        if v.get("resolution_status") in ("fixed_exact", "fixed_near_miss")
    )
    assert n_oracle == EXPECTED_ORACLE, (
        f"fixed_exact+fixed_near_miss count drifted: {n_oracle} != {EXPECTED_ORACLE}"
    )


# --------------------------------------------------------------------------
# 1. COVERAGE — every one of the 6,720 EEE ids resolves to a non-null canonical
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_coverage_all_eee_ids_resolve_non_null(resolver, oracle):
    """GATE part 1 (quantitative floor): every EEE id (incl. the 2,646
    not-on-HF tail that must gain non-null via Tier-2/3) resolves to a
    non-null canonical. A single no_match is a gate break."""
    no_match = [raw for raw in oracle if resolver.resolve(raw, "model").canonical_id is None]
    assert no_match == [], (
        f"{len(no_match)}/{len(oracle)} EEE ids resolved to no_match (gate floor "
        f"is 0). First few: {no_match[:10]}"
    )


# --------------------------------------------------------------------------
# 2. ORACLE — the 4,074 HF ids resolve ORG-AWARE to their fixed_hf_model_id
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_oracle_org_aware_match(resolver, oracle, hf_to_dev):
    """GATE part 1: each fixed_exact / fixed_near_miss id resolves to a
    canonical C where model-name(C) == model-name(fixed_hf_model_id)
    CASE-SENSITIVELY, and org(C) is ORG-AWARE-equal to the HF org (equal
    case-insensitively for community ids, or a curated namespace-alias of the
    resolved developer org for big-dev ids, e.g. meta-llama ↔ meta).

    The model-name comparison is case-sensitive ON PURPOSE — HF repo names are
    case-sensitive and the canonical id preserves HF casing."""
    fixed = {
        raw: meta for raw, meta in oracle.items()
        if meta.get("resolution_status") in ("fixed_exact", "fixed_near_miss")
    }

    # Raws whose oracle `fixed_near_miss` redirect points at a DIFFERENT model
    # (a wrong HF "did-you-mean"). They resolve to their own/corrected canonical,
    # so each is checked against THAT (must resolve away from the bad redirect)
    # rather than against the redirect.
    # Loaded from the sidecar the generator guard (generate_hf_oracle_seed) also
    # reads, so the same list drives prevention and this exemption.
    AUDIT_CORRECTED = frozenset(json.loads(
        (REGISTRY_ROOT / "specs" / "model-resolution-rework"
         / "audit_bad_nearmiss.json").read_text()
    ).get("raws", []))

    def org_aware_equal(resolved_org: str, hf_org: str) -> bool:
        # The resolved canonical's id-prefix is the REAL HF org (not a folded
        # dev slug). Two HF orgs of the same developer are both
        # valid (e.g. THUDM and zai-org both publish GLM); the invariant is
        # that they fold to the SAME canonical developer.
        if resolved_org.lower() == hf_org.lower():
            return True
        fold = lambda x: hf_to_dev.get(x.lower(), x.lower())
        return fold(resolved_org) == fold(hf_org)

    # Each audit-corrected raw must resolve to a DIFFERENT canonical than its bad
    # redirect does (i.e. it was actually re-pointed away from the wrong model).
    # Compared by resolved canonical, not name — the cross-uploader corrections
    # share the redirect's model NAME but differ in org.
    not_corrected = []
    for raw in (AUDIT_CORRECTED & set(fixed)):
        bad_id = fixed[raw]["fixed_hf_model_id"]
        cid = resolver.resolve(raw, "model").canonical_id
        bad_cid = resolver.resolve(bad_id, "model").canonical_id
        if cid is None or cid == bad_cid:
            not_corrected.append((raw, bad_id, cid))
    assert not_corrected == [], (
        f"{len(not_corrected)} audit-corrected raw(s) still resolve to the oracle's "
        f"bad redirect (the bad-redirect correction regressed): {not_corrected[:10]}"
    )

    # Every other oracle id must org-aware-match its fixed_hf_model_id.
    checkable = {raw: meta for raw, meta in fixed.items() if raw not in AUDIT_CORRECTED}
    passed = 0
    buckets: Counter = Counter()
    samples: list[tuple] = []
    for raw, meta in checkable.items():
        hf_id = meta["fixed_hf_model_id"]
        hf_org, hf_name = _org_of(hf_id), _name_of(hf_id)
        cid = resolver.resolve(raw, "model").canonical_id
        if cid is None:
            buckets["no_match"] += 1
            if len(samples) < 12:
                samples.append((raw, hf_id, None))
            continue
        name_ok = _name_of(cid) == hf_name           # case-sensitive
        org_ok = org_aware_equal(_org_of(cid), hf_org)
        if name_ok and org_ok:
            passed += 1
        else:
            key = "name+org" if (not name_ok and not org_ok) else ("name" if not name_ok else "org")
            buckets[f"{key}_mismatch"] += 1
            if len(samples) < 12:
                samples.append((raw, hf_id, cid))

    assert passed == len(checkable) == EXPECTED_ORACLE - len(AUDIT_CORRECTED & set(fixed)), (
        f"ORACLE org-aware match: {passed}/{len(checkable)} (floor "
        f"{EXPECTED_ORACLE - len(AUDIT_CORRECTED & set(fixed))} = {EXPECTED_ORACLE} - "
        f"{len(AUDIT_CORRECTED & set(fixed))} audit-corrected). Failure buckets: "
        f"{dict(buckets)}. Samples (raw, hf, resolved): {samples}"
    )


# --------------------------------------------------------------------------
# 3. DEDUP — 0 case-insensitive duplicate canonical ids
# --------------------------------------------------------------------------
def test_no_case_insensitive_duplicate_canonical_ids(models_df):
    """The casing migration MUST leave no lowercase+HF-cased duplicate pair:
    the HF-cased id wins, the old lowercase becomes an alias. Two canonical rows
    whose ids differ only by case are a dedup failure."""
    lc = Counter(i.lower() for i in models_df["id"])
    dups = {k: v for k, v in lc.items() if v > 1}
    # Surface the actual colliding ids for a useful failure message.
    detail = {}
    if dups:
        by_lower = defaultdict(list)
        for i in models_df["id"]:
            if i.lower() in dups:
                by_lower[i.lower()].append(i)
        detail = dict(by_lower)
    assert dups == {}, f"case-insensitive duplicate canonical ids: {detail}"


# --------------------------------------------------------------------------
# 4. NO DANGLING — every parents[].id references an existing canonical id
# --------------------------------------------------------------------------
def test_no_dangling_parent_edges(models_df):
    """Every `parents[].id` edge across canonical_models must point at a row
    that exists as a canonical id. A dangling parent edge breaks the
    group/family/lineage walks and the producer's lineage derivation."""
    idset = set(models_df["id"])
    dangling: list[tuple[str, str]] = []
    for cid, parents in zip(models_df["id"], models_df["parents"]):
        for edge in _parse_parents(parents):
            target = edge.get("id")
            if target is not None and target not in idset:
                dangling.append((cid, target))
    assert dangling == [], (
        f"{len(dangling)} dangling parent edges (parent id not a canonical). "
        f"First few: {dangling[:10]}"
    )


# --------------------------------------------------------------------------
# 5. MEMBERSHIP — group/family non-null (self for singletons);
#    lineage_origin_model_id null-at-origin (NOT self)
# --------------------------------------------------------------------------
def test_group_and_family_membership_is_total(models_df):
    """`model_group_id` and `model_family_id` are a TOTAL partition: NON-NULL
    for every model, equal to SELF for singletons / roots (NOT null at root,
    via the self-fallback). A null group/family is a membership break."""
    null_group = models_df[models_df["model_group_id"].isna()]
    null_family = models_df[models_df["model_family_id"].isna()]
    assert null_group.empty, (
        f"{len(null_group)} models have null model_group_id "
        f"(must be self at root). e.g. {null_group['id'].head(10).tolist()}"
    )
    assert null_family.empty, (
        f"{len(null_family)} models have null model_family_id "
        f"(must be self at root). e.g. {null_family['id'].head(10).tolist()}"
    )
    # And the self-fallback must actually fire for at least the obvious
    # singletons — guard against "non-null" being satisfied trivially by some
    # other value while self-at-root regressed. Many rows are their own group.
    self_group = (models_df["model_group_id"] == models_df["id"]).sum()
    assert self_group > 0, "no model is its own group root — self-fallback regressed"


def test_lineage_origin_model_id_is_null_at_origin_not_self(models_df):
    """`lineage_origin_model_id` (the id of the deepest non-variant ancestor)
    is NULL when self is the origin — NO self-fallback (unlike the org_id
    variant, which DOES self-fall-back). A row whose lineage_origin_model_id
    equals its own id is a regression of the null-at-origin semantics."""
    self_pointing = models_df[models_df["lineage_origin_model_id"] == models_df["id"]]
    assert self_pointing.empty, (
        f"{len(self_pointing)} models have lineage_origin_model_id == self "
        f"(must be null at origin). e.g. {self_pointing['id'].head(10).tolist()}"
    )
    # Non-trivial: SOME rows must actually carry a (non-null, non-self) origin
    # pointer — otherwise the column is uniformly null and the assertion above
    # is vacuous. Finetunes/quants of upstream weights have one.
    has_origin = models_df["lineage_origin_model_id"].notna().sum()
    assert has_origin > 0, (
        "no model carries a lineage_origin_model_id — the lineage walk produced "
        "nothing, so the null-at-origin check is trivially green"
    )


# --------------------------------------------------------------------------
# 6. NO NAME-ONLY CROSS-ORG MERGE — a same model-name under two orgs stays distinct
# --------------------------------------------------------------------------
def test_no_name_only_cross_org_merge(resolver, models_df):
    """Org-aware identity: a model NAME shared by two different orgs must NOT
    collapse to a single canonical. Concrete known case: the `Llama-3-Instruct-
    8B-SimPO` finetune exists independently under both `haoranxu` and
    `princeton-nlp`. Both ids must (a) exist as distinct canonicals and (b)
    resolve to THEMSELVES — not to each other or a shared merged id."""
    a = "haoranxu/Llama-3-Instruct-8B-SimPO"
    b = "princeton-nlp/Llama-3-Instruct-8B-SimPO"
    idset = set(models_df["id"])
    assert a in idset and b in idset, (
        "spot-check canonicals missing from fixtures — pick a fresh known "
        f"multi-org name. have a={a in idset} b={b in idset}"
    )
    # Same model name, different org → genuinely distinct identities.
    assert _name_of(a) == _name_of(b)
    assert _org_of(a) != _org_of(b)

    res_a = resolver.resolve(a, "model").canonical_id
    res_b = resolver.resolve(b, "model").canonical_id
    assert res_a == a, f"{a} resolved to {res_a}, not itself (cross-org merge?)"
    assert res_b == b, f"{b} resolved to {res_b}, not itself (cross-org merge?)"
    assert res_a != res_b, "two distinct-org same-name models merged to one canonical"


# --------------------------------------------------------------------------
# 7. TIER-3 HONESTY — inferred rows are draft; org-less inferred are org-unknown
# --------------------------------------------------------------------------
def test_tier3_inferred_rows_are_draft(models_df):
    """Every Tier-3 mint (`resolution_source=inferred`) must be `review_status
    = draft` — the registry never silently promotes a name-inferred guess to
    reviewed."""
    inferred = models_df[models_df["resolution_source"] == "inferred"]
    assert not inferred.empty, "no inferred rows in fixtures — Tier-3 mints vanished"
    non_draft = inferred[inferred["review_status"] != "draft"]
    assert non_draft.empty, (
        f"{len(non_draft)} inferred rows are not review_status=draft "
        f"(Tier-3 honesty break). e.g. {non_draft['id'].head(10).tolist()}"
    )


def test_orgless_inferred_rows_are_tagged_org_unknown(models_df):
    """Free-text mints WITHOUT an extractable org go to the org-less bucket and
    the registry never auto-guesses an org. The bucket is keyed by the `unknown`
    sentinel org (a `null` org_id materialises to it from the `unknown/` prefix
    at seed time). Every such inferred row MUST carry the `org-unknown` tag, so
    the bucket is honestly surfaced for review whether a consumer filters on the
    org FK or the tag."""
    inferred = models_df[models_df["resolution_source"] == "inferred"]
    orgless = inferred[inferred["org_id"].isna() | (inferred["org_id"] == "unknown")]
    assert not orgless.empty, (
        "no org-less inferred rows in fixtures — the org-less bucket vanished; "
        "if intentional, update this test"
    )
    missing_tag = [
        cid for cid, tags in zip(orgless["id"], orgless["tags"])
        if "org-unknown" not in _parse_tags(tags)
    ]
    assert missing_tag == [], (
        f"{len(missing_tag)} org-less inferred rows lack the 'org-unknown' tag "
        f"(org-less honesty break). e.g. {missing_tag[:10]}"
    )


# --------------------------------------------------------------------------
# ORG-INTEGRITY (org-canonicalization gate) — every model's org FK resolves,
# orgs are one-per-developer (no case-splits), and each oracle id is its own
# real-HF-repo canonical (not a folded org-slug form).
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def orgs_df() -> pd.DataFrame:
    df = pd.read_parquet(FIXTURES_DIR / "canonical_orgs.parquet")
    assert not df.empty
    return df


def test_no_dangling_org_fk(models_df, orgs_df):
    """Every `org_id` AND `lineage_origin_model_org_id` on a model must name a
    real `canonical_orgs` row. A dangling org FK = a model whose developer
    can't be resolved downstream (null `developer` in the producer)."""
    org_ids = set(orgs_df["id"].astype(str))
    org_fk = {x for x in models_df["org_id"].astype("string").dropna().unique() if x not in org_ids}
    lin_fk = {
        x for x in models_df["lineage_origin_model_org_id"].astype("string").dropna().unique()
        if x not in org_ids
    }
    assert org_fk == set(), f"{len(org_fk)} dangling org_id FK(s): {sorted(org_fk)[:10]}"
    assert lin_fk == set(), f"{len(lin_fk)} dangling lineage-org FK(s): {sorted(lin_fk)[:10]}"


def test_no_case_split_orgs(orgs_df):
    """One canonical_orgs row per developer: no two org ids may differ only by
    case (`Tencent` vs `tencent`), which would fragment the developer in every
    downstream org list/facet."""
    ci: Counter = Counter(str(i).lower() for i in orgs_df["id"])
    split = {lo: [i for i in orgs_df["id"] if str(i).lower() == lo] for lo, n in ci.items() if n > 1}
    assert split == {}, f"{len(split)} case-split org(s): {list(split.values())[:8]}"


@pytest.mark.slow
def test_oracle_canonical_id_is_real_hf_repo_id(resolver, oracle):
    """Real-HF-repo invariant: each oracle `fixed_hf_model_id` (the real HF repo
    id) must resolve to a canonical that IS that real id (directly), OR — for the
    rare oracle near-miss whose registry canonical sits under the original
    publisher's namespace — reach it via a confirmed alias. canonical_ids are
    never the synthetic org-folded form (`meta/Llama-…`) anymore."""
    fixed = [
        m["fixed_hf_model_id"] for m in oracle.values()
        if isinstance(m.get("fixed_hf_model_id"), str)
    ]
    unreachable = [
        fx for fx in fixed if resolver.resolve(fx, "model").canonical_id is None
    ]
    assert unreachable == [], (
        f"{len(unreachable)}/{len(fixed)} oracle real-HF ids unreachable: {unreachable[:10]}"
    )


def test_canonical_id_equals_metadata_hf_id(models_df):
    """Real-HF-repo invariant for the hub-stats tail: a model that records a real
    HF repo id in `metadata.hf_id` must USE it as its `canonical_id`. A mismatch
    where the real id is ABSENT means the canonical is a synthetic/slugified
    form that 404s on HF."""
    ids = set(models_df["id"].astype(str))
    viol = []
    for r in models_df.itertuples():
        md = getattr(r, "metadata", None)
        if not isinstance(md, str):
            continue
        try:
            h = json.loads(md).get("hf_id")
        except Exception:
            h = None
        if isinstance(h, str) and "/" in h and h != str(r.id) and h not in ids:
            viol.append((str(r.id), h))
    assert viol == [], (
        f"{len(viol)} models whose canonical_id != metadata.hf_id (real absent): {viol[:10]}"
    )


@pytest.mark.slow
def test_old_folded_form_resolves_to_real_hf_id(resolver, models_df):
    """Safety net (also guards the models.dev refresh cron): the OLD org-folded
    id (`alibaba/Qwen2.5-7B`) must resolve to the real-HF canonical
    (`Qwen/Qwen2.5-7B`) via the demoted alias left behind when the folded id was
    replaced by the real HF id. This is
    what makes `regenerate_catalog` (refresh_from_modelsdev) fold a future
    folded-slug mint onto the real canonical instead of minting a duplicate.
    A regression here = the cron would re-introduce folded canonicals."""
    ids = set(models_df["id"].astype(str))
    pairs = [
        ("alibaba/Qwen2.5-7B", "Qwen/Qwen2.5-7B"),
        ("meta/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
        ("deepseek/DeepSeek-V3", "deepseek-ai/DeepSeek-V3"),
    ]
    checked = 0
    for folded, real in pairs:
        if real not in ids:
            continue  # not in this fixture snapshot — skip, don't false-fail
        checked += 1
        got = resolver.resolve(folded, "model").canonical_id
        assert got == real, f"{folded!r} resolved to {got!r}, expected {real!r}"
    assert checked > 0, "no folded->real pairs present in fixtures to check"


def test_no_separator_split_orgs(orgs_df):
    """No two org ids differ ONLY by a separator/case (`prime-intellect` vs
    `PrimeIntellect`) — the plain case-insensitive check misses these, but they
    still fragment one developer into two rows downstream."""
    norm = lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower())
    by: dict = defaultdict(list)
    for oid in orgs_df["id"]:
        by[norm(oid)].append(str(oid))
    split = {k: v for k, v in by.items() if len(v) > 1}
    assert split == {}, f"{len(split)} separator/case-split org group(s): {list(split.values())[:8]}"


def test_no_real_hf_id_duplicated_by_slug(models_df):
    """A real HF id (`Qwen/Qwen2.5-7B-Instruct`, oracle/hf-sourced) must never be
    duplicated by a NON-real folded/slug variant (`alibaba/qwen-2-5-7b-instruct`)
    — the dup-merge collapses those. Two distinct REAL repos that
    happen to normalize alike (NAPS v-0.1.0 vs v0.1.0) are allowed; 0-real
    API/spelling dups are a known residual."""
    fixed = set()
    for v in json.loads(ORACLE_PATH.read_text())["resolutions"].values():
        fx = v.get("fixed_hf_model_id")
        if isinstance(fx, str) and "/" in fx:
            fixed.add(fx)
    hf_to_dev = {k.lower(): val for k, val in _ORG_ALIASES.items()}
    for e in yaml.safe_load(ORGS_YAML.read_text()) or []:
        if isinstance(e, dict) and isinstance(e.get("hf_org"), str) and isinstance(e.get("id"), str):
            hf_to_dev[e["hf_org"].lower()] = e["id"]

    def fold(org):
        return hf_to_dev.get(org.lower(), org)

    def ndup(s):
        s = s.lower()
        s = re.sub(r"([a-z])[-_ /]+(\d)", r"\1\2", s)
        s = re.sub(r"(\d)\.(\d)(?![bmkt])", r"\1-\2", s)
        return re.sub(r"[-_ /]+", "-", s)

    def has_real_hf_id(row):
        """True when the row's canonical_id is a verified real HF repo id —
        an oracle fixed id, an hf-sourced mint, or == its own metadata.hf_id."""
        cid = str(row.id)
        if cid in fixed:
            return True
        if isinstance(row.resolution_source, str) and row.resolution_source == "hf":
            return True
        md = getattr(row, "metadata", None)
        if isinstance(md, str):
            try:
                return json.loads(md).get("hf_id") == cid
            except Exception:
                return False
        return False

    clusters: dict = defaultdict(lambda: {"real": [], "nonreal": []})
    for row in models_df.itertuples():
        cid = str(row.id)
        if "/" not in cid:
            continue
        org, name = cid.split("/", 1)
        bucket = "real" if has_real_hf_id(row) else "nonreal"
        clusters[(fold(org), ndup(name))][bucket].append(cid)
    # Violation = a real id sharing a cluster with a NON-real slug variant.
    bad = [
        sorted(c["real"] + c["nonreal"])
        for c in clusters.values()
        if c["real"] and c["nonreal"]
    ]
    assert bad == [], f"{len(bad)} real-HF id(s) duplicated by a non-real slug: {bad[:8]}"


# --------------------------------------------------------------------------
# No minted models.dev canonical still shadows a real HF id.
#
# This is the RULE-AS-ASSERTION gate: it re-runs the SAME confident-match
# predicate the fold (scripts/fold_modelsdev_dupes.py) used — exact id / alias
# linkage / normalized-name-with-ORG-AGREEMENT / brand-prefix-stripped-with-org
# — over the materialised fixtures. If the fold cleaned everything, the predicate
# now finds NOTHING; a reintroduced mint-dupe makes it find a fold and fail.
#
# Why this catches what test_no_real_hf_id_duplicated_by_slug misses: that test
# clusters by a STRING normalisation of (folded-org, ndup(name)), so it cannot
# group a dupe that differs in BOTH org spelling AND name across the brand
# prefix (e.g. `alibaba/qwen-qwq-32b` vs real `Qwen/QwQ-32B` → dev `alibaba`):
# the names `qwen-qwq-32b` and `qwq-32b` land in different ndup() buckets. The
# fold predicate defers via the resolver's brand-prefix stripping + org
# agreement, so re-running it as the gate closes that exact blind spot.
# --------------------------------------------------------------------------
def _import_fold_module():
    scripts_dir = REGISTRY_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import fold_modelsdev_dupes as fold  # noqa: E402

    return fold


def _model_aliases_by_canonical() -> dict[str, list[str]]:
    """raw_value spellings declared for each model canonical, from the alias
    table — so the gate feeds the fold predicate the same alias surface the
    fold script reads off `core.yaml` entries."""
    apath = FIXTURES_DIR / "aliases.parquet"
    out: dict[str, list[str]] = defaultdict(list)
    if not apath.exists():
        return out
    adf = pd.read_parquet(apath)
    adf = adf[adf["entity_type"] == "model"]
    for row in adf.itertuples():
        cid = getattr(row, "canonical_id", None)
        rv = getattr(row, "raw_value", None)
        if isinstance(cid, str) and isinstance(rv, str) and rv:
            out[cid].append(rv)
    return out


# Mints the fold PREDICATE flags but which an adversarial review confirmed are
# NOT true dupes (the predicate's normalized/brand-stripped tier over-reaches
# here — a base vs a specific dated/quant/size variant, a family pointer vs a
# leaf, or a not-yet-real placeholder). These are intentionally NOT folded; the
# gate allowlists them so a legitimately-not-folded state stays green while any
# NEW confident mint-dupe still fails. Keep this list tight — every entry is a
# reviewed exception, not a blanket waiver.
_KNOWN_NON_DUPE_MINTS = frozenset({
    "alibaba/qwen-3-235b",                          # base vs ...-A22B-Instruct-2507 leaf
    "alibaba/qwen-3-30b",                           # base vs Qwen3-30B-A3B specific variant
    "alibaba/qwen3-235b-a22b-instruct-2507-tput",   # provider throughput tag vs ...-FP8 quant
    "google/gemma4",                                # family placeholder, not a real repo
    "olmo-3-1-32b",                                 # base vs Olmo-3.1-32B-Think leaf
    "openai/gpt-oss",                               # family pointer vs gpt-oss-120b leaf
    "zai/z-ai-glm-5",                               # near-spelling of zai-org/GLM-5, deferred
})


def test_no_minted_modelsdev_canonical_shadows_real_hf_id(models_df):
    """GATE: NO canonical with resolution_source=models_dev still has a CONFIDENT
    real-HF match — i.e. the fold found every mint-dupe and none can
    silently return. Uses the fold script's own predicate (shared, not a third
    matcher), so the gate and the cleanup agree by construction. A tight
    allowlist (`_KNOWN_NON_DUPE_MINTS`) carries the reviewed predicate
    over-reaches; any NEW confident mint-dupe outside it fails the gate."""
    fold = _import_fold_module()

    # reconstruct the entry-list shape decide_fold expects, from the fixtures
    alias_map = _model_aliases_by_canonical()
    entries: list[dict] = []
    for row in models_df.itertuples():
        cid = str(row.id)
        org_id = getattr(row, "org_id", None)
        dn = getattr(row, "display_name", None)
        rsrc = getattr(row, "resolution_source", None)
        md = getattr(row, "metadata", None)
        entries.append({
            "id": cid,
            "org_id": org_id if isinstance(org_id, str) else None,
            "display_name": dn if isinstance(dn, str) else None,
            "resolution_source": rsrc if isinstance(rsrc, str) else None,
            "aliases": alias_map.get(cid, []),
            "metadata": md if isinstance(md, str) else None,
        })

    hf_to_dev = fold.build_hf_to_dev()
    hf_ids, alias_to_hf, by_org_name, _ = fold.build_hf_targets(entries, hf_to_dev)

    mints = [e for e in entries if e["resolution_source"] == "models_dev"]
    assert mints, "no models_dev canonicals in fixtures — gate cannot run (reseed?)"

    leftover = []
    for m in mints:
        if m["id"] in _KNOWN_NON_DUPE_MINTS:
            continue
        f = fold.decide_fold(m, hf_ids, alias_to_hf, by_org_name, hf_to_dev)
        if f is not None:
            leftover.append((f["mint_id"], f["hf_target"], f["match_type"]))

    assert leftover == [], (
        f"{len(leftover)} models.dev-minted canonical(s) still confidently match "
        f"a real HF id (mint-dupe shadow not folded): {leftover[:12]}"
    )


# --------------------------------------------------------------------------
# org-conditional quant grouping: model_group_id never crosses a developer
# --------------------------------------------------------------------------
def test_model_group_id_does_not_cross_developer_org(models_df):
    """A model and its `model_group_id` root MUST share the same developer
    `org_id`. A first-party precision variant (same org) folds into the base
    group; a THIRD-PARTY / community quant (e.g. `unsloth/...-bnb-4bit`
    quantizing `microsoft/phi-4`) keeps its OWN group — its `quantized` edge
    still records the link via `lineage_origin_model_id`, but its scores never
    merge into the base lab's model. The same guard drops a spurious cross-org
    version edge. A group root under a different developer is the
    cross-org-identity-grouping bug this enforces against."""
    org_by_id = dict(
        zip(models_df["id"].astype(str), models_df["org_id"].astype("string"))
    )
    bad = []
    for cid, grp in zip(
        models_df["id"].astype(str), models_df["model_group_id"].astype("string")
    ):
        if grp is None or pd.isna(grp) or grp == cid:
            continue
        o_self = org_by_id.get(cid)
        o_grp = org_by_id.get(grp)
        if (
            o_self is not None and o_grp is not None
            and not pd.isna(o_self) and not pd.isna(o_grp)
            and o_self != o_grp
        ):
            bad.append((cid, grp, o_self, o_grp))
    assert bad == [], (
        f"{len(bad)} model(s) whose model_group_id root is a DIFFERENT developer "
        f"org (cross-org identity grouping is forbidden; a community quant must "
        f"keep its own group). First few: {bad[:10]}"
    )
