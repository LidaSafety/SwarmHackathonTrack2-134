"""Q1 activation probe analysis (runs/interp_q1/<model>, from q1_decision.py).

1. Behaviour: lead / initiative / avoid-claimed / comply rates by framing, incentive, k, notes.
2. Probes: per-layer logistic probes vs a TF-IDF text monitor on the observation (5-fold CV grouped by seed), on all
   rows of a task and on matched subsets where the board framing is fixed, so only the agent's decision differs.
3. Geometry: cosine between the diff-in-means role directions and the Q2 refusal direction.
4. Steering: label rates by direction and alpha (alpha 0 = base rows of the same subset), with a random-direction control.
Writes probes.csv, steer.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_q3_monitor import cv_scores, probe, text_clf  # noqa: E402

TESTS = [  # (name, task, label, filter)
    ("initiative: all lead prompts", "lead", "initiative", None),
    ("initiative: ask_me only", "lead", "initiative", lambda d: d.framing == "ask_me"),
    ("lead: all lead prompts", "lead", "lead", None),
    ("avoid claimed: claims+collide", "lead", "avoid", None),
    ("avoid claimed: claims k=3 only", "lead", "avoid", lambda d: (d.framing == "claims") & (d.k == 3)),
    ("comply: all follow prompts", "follow", "comply", lambda d: d.framing != "none"),
    ("comply: assign only", "follow", "comply", lambda d: d.framing == "assign"),
    ("comply: assign, own_range notes", "follow", "comply", lambda d: (d.framing == "assign") & (d.notes == "own_range")),
]


def main(root: str = "runs/interp_q1/Qwen3-8B") -> None:
    from sklearn.metrics import roc_auc_score
    pd.set_option("display.width", 220)
    d = Path(root)
    layers = json.loads((d / "probe.json").read_text())["save_layers"]
    df = pd.DataFrame([json.loads(x) for x in (d / "rows.jsonl").read_text().splitlines()])
    A = np.load(d / "acts.npy").astype(np.float32)
    obs = None
    from q1_decision import build_follow, build_lead  # rebuild the observation text for the text monitor
    obs = [(build_lead(r.seed, r.framing, int(r.k) if r.k == r.k else 0, r.pressure) if r.task == "lead" else
            build_follow(r.seed, r.framing, r.notes, r.pressure))[1] for r in df.itertuples()]

    L, F = df[df.task == "lead"], df[df.task == "follow"]
    print("parsed:", df.parsed.mean().round(3))
    print("\nLEAD prompts: rates by framing x incentive (initiative / lead / post)")
    print(L.groupby(["framing", "k"])[["post", "initiative", "lead"]].mean().round(2).to_string())
    print(L.groupby("pressure")[["post", "initiative", "lead"]].mean().round(2).to_string())
    print("\navoid claimed tasks (rows touching a task):")
    print(L[L.avoid.notna()].groupby(["framing", "k"]).avoid.agg(["mean", "count"]).round(2).to_string())
    print("\nFOLLOW prompts: comply by framing x notes (rows touching a task)")
    Fc = F[F.comply.notna()]
    print(Fc.groupby(["framing", "notes"]).comply.mean().unstack().round(2).to_string())
    print(Fc.groupby("pressure").comply.mean().round(2).to_string())
    print(f"follow rows touching no task: {F.comply.isna().mean():.2f}")

    out = []
    for name, task, key, filt in TESTS:
        m = (df.task == task) & df[key].notna() if key in df else None
        if m is None:
            continue
        if filt is not None:
            m &= filt(df).fillna(False)
        sub = df[m]
        y, g = sub[key].astype(bool).values.astype(int), sub.seed.values
        if y.sum() < 5 or y.sum() > len(y) - 5:
            print(f"skip {name}: {y.sum()}/{len(y)} positive")
            continue
        X = A[m.values]
        scores = {"text tf-idf": cv_scores(text_clf, [obs[i] for i in np.where(m.values)[0]], y, g)}
        for li, l in enumerate(layers):
            scores[f"probe L{l}"] = cv_scores(probe, X[:, li], y, g)
        for k, s in scores.items():
            out.append(dict(test=name, monitor=k, n=len(y), pos=int(y.sum()), auroc=roc_auc_score(y, s)))
    res = pd.DataFrame(out).round(3)
    print("\n" + res.pivot(index="test", columns="monitor", values="auroc").to_string())
    print(res.groupby("test")[["n", "pos"]].first().to_string())
    res.to_csv(d / "probes.csv", index=False)

    dz = dict(np.load(d / "directions.npz"))
    names = sorted({k.rsplit("_", 1)[0] for k in dz if k.endswith("_d")})
    print("\nrole directions (diff-in-means, chosen on even seeds, AUROC on odd seeds):")
    for n in names:
        print(f"  {n}: layer {int(dz[n + '_layer'])}, held-out AUROC {float(dz[n + '_heldout_auroc']):.3f}")
    ref = Path("runs/interp_q2/Qwen3-8B_agent_r2/refusal_dir.npz")
    if ref.exists():
        r_hat = np.load(ref)["r_hat"]
        for n in names:
            v = dz[n + "_d"]
            print(f"  cos({n}, refusal@L{int(dz[n + '_layer'])}) = "
                  f"{v @ r_hat[int(dz[n + '_layer'])] / np.linalg.norm(v):+.3f}")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if int(dz[a + "_layer"]) == int(dz[b + "_layer"]):
                va, vb = dz[a + "_d"], dz[b + "_d"]
                print(f"  cos({a}, {b}) = {va @ vb / np.linalg.norm(va) / np.linalg.norm(vb):+.3f}")

    sp = d / "steer.jsonl"
    if sp.exists() and sp.stat().st_size:
        st = pd.DataFrame([json.loads(x) for x in sp.read_text().splitlines()])
        base = df.loc[sorted(st.i.unique())].assign(alpha=0.0)
        rows = []
        for dname, g in st.groupby("dir"):
            for al, h in pd.concat([base, g]).groupby("alpha"):
                lead_rows, fol = h[h.task == "lead"], h[(h.task == "follow") & (h.framing == "assign")]
                cl = lead_rows[lead_rows.framing == "claims"]
                rows.append(dict(dir=dname, alpha=al, parsed=h.parsed.mean(), post=lead_rows.post.mean(),
                                 initiative=lead_rows.initiative.mean(), lead=lead_rows.lead.mean(),
                                 initiative_ask_me=lead_rows[lead_rows.framing == "ask_me"].initiative.mean(),
                                 avoid_claims=cl.avoid.dropna().astype(float).mean(),
                                 comply_assign=fol.comply.dropna().astype(float).mean(),
                                 touch_none_assign=fol.comply.isna().mean(), n=len(h)))
        sres = pd.DataFrame(rows).round(2)
        print("\nsteering (alpha 0 = unsteered rows of the same subset):")
        print(sres.to_string(index=False))
        sres.to_csv(d / "steer.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
