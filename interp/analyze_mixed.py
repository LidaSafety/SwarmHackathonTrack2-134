"""Mixed swarms (swarm_mixed.py): when do local Qwen3-8B agents take part in swarm mechanisms, and is it in their activations?

1. Behaviour map: mechanism rates (swarm_map labels) for Qwen vs Gemini agents by N x topology, and for Qwen-only swarms.
2. Probes on Qwen calls: for each naturally occurring mechanism (coord_post, follow, avoid, copy, read_board), per-layer
   logistic probes vs (a) a TF-IDF monitor on the observation the agent saw and (b) a context-only model on scenario
   features (N, topology, round, assignment pending, claims visible). CV grouped by run.
3. Order parameter (Q4): per run, mean Qwen projection on the coordination direction (diff-in-means of coord_post,
   fitted leave-run-out) vs N and topology.
Writes runs/swarm_mixed/{behaviour.csv, probes.csv, order_param.csv}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_q3_monitor import probe, text_clf  # noqa: E402
from swarm_map import MECHS, run_calls  # noqa: E402

LOCAL = "Qwen3-8B"
TARGETS = [  # (label, rows where it is defined)
    ("post", lambda d: d.index == d.index),
    ("coord_post", lambda d: d.index == d.index),
    ("read_board", lambda d: d.index == d.index),
    ("follow", lambda d: d.pending_assignment & (d.n_touched > 0)),
    ("avoid", lambda d: d.claims_visible & (d.n_touched > 0)),
    ("copy", lambda d: d.index == d.index),
]


def cv_group(make, X, y, groups, k=5):
    from sklearn.model_selection import GroupKFold
    pred = np.full(len(y), np.nan)
    if len(np.unique(groups)) < 2:
        return pred
    for tr, te in GroupKFold(min(k, len(np.unique(groups)))).split(X, y, groups):
        if y[tr].min() == y[tr].max():
            continue
        Xtr = [X[i] for i in tr] if isinstance(X, list) else X[tr]
        Xte = [X[i] for i in te] if isinstance(X, list) else X[te]
        pred[te] = make().fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
    return pred


def context_clf():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), [0, 1, 2]), ("num", "passthrough", [3, 4])])
    return make_pipeline(ct, LogisticRegression(max_iter=3000, class_weight="balanced"))


def main(root: str = "runs/swarm_mixed") -> None:
    from sklearn.metrics import roc_auc_score
    pd.set_option("display.width", 220)
    root = Path(root)
    layers = json.loads((root / "probe.json").read_text())["save_layers"]
    rows, acts, obs = [], [], []
    for run in sorted(p for p in root.iterdir() if (p / "metrics.json").exists()):
        c = pd.DataFrame(run_calls(run))
        rows.append(c)
        if (run / "acts.npy").exists():
            a = np.load(run / "acts.npy").astype(np.float32)
            o = {r["act_idx"]: r["obs"] for r in map(json.loads, (run / "obs.jsonl").read_text().splitlines())}
            loc = c[c.agent_model == LOCAL]
            assert len(loc) == len(a), (run, len(loc), len(a))
            acts.append(a)
            obs += [o.get(i, "") for i in loc.act_idx]
    df = pd.concat(rows, ignore_index=True)
    df = df[~df.error]
    print(f"{df.run.nunique()} runs, {len(df)} calls ({(df.agent_model == LOCAL).sum()} local)")

    # 1. behaviour map
    df["follow_def"] = df.pending_assignment & (df.n_touched > 0)
    df["avoid_def"] = df.claims_visible & (df.n_touched > 0)
    g = df.groupby(["model", "agent_model", "topology", "N"])
    beh = g[["read_board", "post", "coord_post", "assign", "share", "copy"]].mean()
    beh["follow"] = g.apply(lambda h: h.follow[h.follow_def].mean() if h.follow_def.any() else np.nan)
    beh["n_follow"] = g.follow_def.sum()
    beh["avoid"] = g.apply(lambda h: h.avoid[h.avoid_def].mean() if h.avoid_def.any() else np.nan)
    beh["n_avoid"] = g.avoid_def.sum()
    beh["runs"], beh["calls"] = g.run.nunique(), g.size()
    beh = beh.round(3)
    print("\nBEHAVIOUR (rates per call; follow/avoid only where defined)")
    print(beh.to_string())
    beh.to_csv(root / "behaviour.csv")
    run_m = []
    for run in sorted(p for p in root.iterdir() if (p / "metrics.json").exists()):
        m, c = json.loads((run / "metrics.json").read_text()), json.loads((run / "config.json").read_text())
        run_m.append(dict(run=run.name, mix=c["model_mix"], N=c["n_agents"], topology=c["topology"],
                          leader=m["leadership"]["leader_emerged"], reach=m["leadership"]["leader_reach"],
                          copy_frac=m["division_of_labour"]["copy_frac"], score=m["division_of_labour"]["mean_score"]))
    print("\nRUNS")
    print(pd.DataFrame(run_m).to_string(index=False))

    # 2. probes on local calls
    loc = df[df.agent_model == LOCAL].reset_index(drop=True)
    if not acts:
        return
    A = np.concatenate(acts)
    assert len(A) == len(loc)
    ctx = loc[["topology", "model", "N", "round", "pending_assignment"]].copy()
    ctx["claims"] = loc.claims_visible.astype(int)
    ctx = ctx.astype({"N": str})[["topology", "model", "N", "round", "claims"]].values.tolist()
    out = []
    for name, defined in TARGETS:
        m = np.asarray(defined(loc), bool)
        y = loc[name].values[m].astype(int)
        groups = loc.run.values[m]
        if y.sum() < 8 or y.sum() > len(y) - 8:
            print(f"skip {name}: {y.sum()}/{len(y)} positive")
            continue
        sc = {"context only": cv_group(context_clf, [ctx[i] for i in np.where(m)[0]], y, groups),
              "text tf-idf": cv_group(text_clf, [obs[i] for i in np.where(m)[0]], y, groups)}
        for li, l in enumerate(layers):
            sc[f"probe L{l}"] = cv_group(probe, A[m][:, li], y, groups)
        for k, s in sc.items():
            ok = ~np.isnan(s)
            if ok.sum() and 0 < y[ok].sum() < ok.sum():
                out.append(dict(target=name, monitor=k, n=int(ok.sum()), pos=int(y[ok].sum()),
                                auroc=roc_auc_score(y[ok], s[ok])))
    res = pd.DataFrame(out).round(3)
    if res.empty:
        print("no probe results (too few runs or positives)")
        return
    print("\nPROBES on local agents' natural decisions (AUROC, CV grouped by run)")
    print(res.pivot(index="target", columns="monitor", values="auroc").to_string())
    print(res.groupby("target")[["n", "pos"]].max().to_string())
    res.to_csv(root / "probes.csv", index=False)

    # 3. order parameter: projection on a leave-run-out coord_post direction
    li = layers.index(22) if 22 in layers else len(layers) // 2
    y = loc.coord_post.values.astype(bool)
    proj = np.full(len(loc), np.nan)
    for r in loc.run.unique():
        tr, te = loc.run.values != r, loc.run.values == r
        if y[tr].any() and (~y[tr]).any():
            d = A[tr & y, li].mean(0) - A[tr & ~y, li].mean(0)
            proj[te] = A[te, li] @ (d / np.linalg.norm(d))
    loc["coord_proj"] = proj
    op = loc.groupby(["model", "topology", "N", "run"]).agg(m=("coord_proj", "mean"), sd=("coord_proj", "std"),
                                                            coord_rate=("coord_post", "mean")).reset_index()
    op = op.groupby(["model", "topology", "N"]).agg(m=("m", "mean"), sd_within=("sd", "mean"), between=("m", "std"),
                                                    coord_rate=("coord_rate", "mean"), runs=("run", "count")).round(3)
    print(f"\nORDER PARAMETER: mean local projection on the coordination direction (layer {layers[li]})")
    print(op.to_string())
    op.to_csv(root / "order_param.csv")


if __name__ == "__main__":
    main(*sys.argv[1:])
