"""
replay_attribution.py — source attribution (Figures 5/6) over the 20-replicate main run, with no LLM calls.

The traced campaign behind attribution_9ds.json is a single run per screen. The 20-replicate main run
stored, per round, the batch, the LLM's named proposals and the revealed hits, but not the ML's own
top-B — the counterfactual that separates "both" from "LLM-only". The online ML depends only on the
revealed labels, so it can be replayed: rebuild the C-arm's OnlineAdaptiveArm exactly as the arm does, feed
it each round's recorded reveals in order, and recompute the fused ranking from the recorded LLM names.

A round counts as verified only if the replayed batch equals the recorded one gene for gene and in the same
order. Mismatches are counted and reported, never absorbed: the replay always continues on the recorded
trajectory, and analyses should use verified rounds only.

    python -m waddington_select.analysis.replay_attribution [--screens IFNG,IL2] [--repeats 2]
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

from ..arms.online_adaptive_arm import OnlineAdaptiveArm
from ..arms.waddington_c_arm import (
    DEFAULT_HONEST_WLLM, _drop_anchor_default, _get_feature_config, _legacy_router, _named_only_llm_vote,
)
from ..oracle import BATCH_SIZES, DatasetOracle
from .trace import hit_rate_by_source

REPO = Path(__file__).resolve().parents[2]
from .sources import REPORTED_RUN_DIR as RUN_DIR  # noqa: E402  (the reported run; analysis/sources.py)
RUN = RUN_DIR / "main_table_20rep.json"
MANIFEST = RUN_DIR / "waddington_c_20rep.manifest.json"
OUT = RUN_DIR / "attribution_replay_20rep.json"

W_LLM = DEFAULT_HONEST_WLLM
W_ML = 1.0 - W_LLM  # same expression as the arm, so the fused scores are bit-identical


def _check_protocol(manifest: Path = MANIFEST) -> None:
    """The replay is only meaningful under the configuration the 20-replicate C-arm actually ran."""
    proto = json.loads(manifest.read_text())["protocol"]
    want = {"legacy_router": False, "drop_anchor_features": True, "feature_policy": "metadata",
            "llm_vote_policy": "gpt_named_only", "force_wllm": None}
    got = {k: proto.get(k) for k in want}
    if got != want:
        raise SystemExit(f"manifest protocol {got} != replayed configuration {want}")
    if _legacy_router() or not _named_only_llm_vote() or not _drop_anchor_default():
        raise SystemExit("environment overrides the default C-arm configuration; unset WADDINGTON_* flags")


def _replay_run(base: OnlineAdaptiveArm, oracle: DatasetOracle, rec: dict, bs: int) -> list[dict]:
    online = copy.deepcopy(base)  # round-0 prior is identical across repeats; train it once per screen
    selected: set[str] = set()
    hits_seen: set[str] = set()
    rounds = []
    n_rounds = len(rec["selected_genes_per_round"])
    for r in range(n_rounds):
        ml = online.all_scores()
        named = set(rec["llm_named_proposals_per_round"][r])
        combined = {g: W_ML * ml.get(g, 0.0) + W_LLM * (1.0 if g in named else 0.0)
                    for g in online._genes if g not in selected}
        replayed = sorted(combined, key=combined.__getitem__, reverse=True)[:bs]
        recorded = list(rec["selected_genes_per_round"][r])
        ml_top = set(sorted(combined, key=lambda g: -ml.get(g, 0.0))[:bs])
        revealed = oracle.reveal(recorded)

        genes = []
        for g in recorded:
            in_llm, in_ml = g in named, g in ml_top
            source = "both" if (in_llm and in_ml) else ("llm_only" if in_llm else "ml_only")
            genes.append({"gene": g, "source": source, "ml_score": round(float(ml.get(g, 0.0)), 5),
                          "hit": bool(revealed[g])})
        rounds.append({
            "round": r + 1, "batch_size": bs, "verified": replayed == recorded,
            "n_llm_named": len(named),
            "genes": genes,
            "shap": {f: round(v, 5) for f, v in online.shap_for(recorded).items()},
            "counts": {s: sum(1 for x in genes if x["source"] == s) for s in ("both", "ml_only", "llm_only")},
        })
        hits_seen |= {g for g, h in revealed.items() if h}
        selected |= set(recorded)
        if r < n_rounds - 1:  # the final update would only train a model nothing uses
            online.update(r, revealed)
    if hits_seen != set(rec["revealed_hits"]):
        raise RuntimeError("oracle hits disagree with the recorded revealed_hits; labels have changed")
    return rounds


def run(screens: list[str] | None = None, repeats: int | None = None, out: Path = OUT,
        run_file: Path = RUN, manifest: Path = MANIFEST, arm: str = "waddington_c") -> dict:
    """Replay `arm` in `run_file`. The defaults are the reported headline run; another tier's run needs
    that tier's tables in the environment before import (experiments/21_tier_offline.sh)."""
    _check_protocol(manifest)
    data = json.loads(run_file.read_text())
    screens = screens or list(data.keys())
    pooled = {"rounds": []}
    per_dataset, verify = {}, {}
    for ds in screens:
        t0 = time.time()
        bs = BATCH_SIZES[ds]
        training_csv, extra = _get_feature_config(ds)
        base = OnlineAdaptiveArm(ds, bs, training_csv=training_csv, extra_feature_cols=extra,
                                 drop_anchor_feats=_drop_anchor_default())
        oracle = DatasetOracle(ds)
        runs = data[ds][arm][: repeats or None]
        ds_rounds = []
        for rep, rec in enumerate(runs):
            for rr in _replay_run(base, oracle, rec, bs):
                rr.update({"dataset": ds, "repeat": rep})
                ds_rounds.append(rr)
        n_ok = sum(r["verified"] for r in ds_rounds)
        verify[ds] = {"rounds": len(ds_rounds), "verified": n_ok}
        per_dataset[ds] = hit_rate_by_source({"rounds": [r for r in ds_rounds if r["verified"]]})
        pooled["rounds"].extend(ds_rounds)
        print(f"  {ds:26s} {len(runs)} repeats  verified {n_ok}/{len(ds_rounds)} rounds  "
              f"({time.time() - t0:.0f}s)", flush=True)
    verified = {"rounds": [r for r in pooled["rounds"] if r["verified"]]}
    res = {
        "_method": ("Replay of the 20-replicate GPT-5.6 Luna C-arm (main_table_20rep.json): online ML rebuilt "
                    "with the arm's own configuration and fed the recorded reveals; fused ranking recomputed "
                    "from the recorded LLM names. 'verified' = replayed batch equals the recorded batch."),
        "source_run": str(Path(run_file).resolve().relative_to(REPO)),
        "arm": arm,
        "verification": verify,
        "per_dataset": per_dataset,
        "overall": hit_rate_by_source(verified),
        "pooled": pooled,
    }
    out.write_text(json.dumps(res))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--screens", type=lambda s: s.split(","), default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--arm", default="waddington_c")
    a = ap.parse_args()
    r = run(a.screens, a.repeats, a.out, a.run, a.manifest, a.arm)
    v = r["verification"]
    print(f"\nverified {sum(x['verified'] for x in v.values())}/{sum(x['rounds'] for x in v.values())} rounds")
    for s, x in r["overall"].items():
        print(f"  {s:9s} picked={x['picked']:6d} hits={x['hits']:5d} hit_rate={x['hit_rate']*100:5.1f}%")
    print(f"saved -> {a.out}")
