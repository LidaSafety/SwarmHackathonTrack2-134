"""Summarise the Q2 interpretability pilot (runs/interp_q2/<model>_<frame>/).

For each model x frame x condition:
  behaviour      refusal rate and facts delivered, per intent cell
  harm signal    mean refusal-direction projection (at the run's best layer) for shown/harmful minus shown/benign,
                 as a % of the AdvBench-vs-Alpaca gap at that layer (100% = as separated as a canonical harmful request)
  auroc          how well the projection alone separates shown/harmful from shown/benign prompts, against the
                 |AUROC| of 50 random directions (prompts differ lexically, so random directions separate them too)
  composition    projection of whole/hidden minus split/hidden (does the full dossier look more harmful?), same scale
Writes runs/interp_q2/summary.csv and layer_auroc.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load(run: Path) -> tuple[pd.DataFrame, dict, int]:
    dirs = dict(np.load(run / "refusal_dir.npz"))
    L = len(dirs["cohen_d"])
    lo, hi = int(0.3 * L), int(0.7 * L)
    best = lo + int(np.argmax(dirs["cohen_d"][lo:hi]))
    df = pd.DataFrame([json.loads(x) for x in (run / "decisions.jsonl").read_text().splitlines()])
    df["P"] = df["proj"].map(np.array)
    return df, dirs, best


def main(root: str = "runs/interp_q2") -> None:
    rows, layer_rows = [], []
    for run in sorted(p for p in Path(root).iterdir() if (p / "decisions.jsonl").exists()):
        df, dirs, best = load(run)
        acts = np.load(run / "decision_acts.npy", mmap_mode="r")
        rng = np.random.default_rng(0)
        scale = dirs["harm_mean"][best] - dirs["safe_mean"][best]
        model, frame = run.name.split("_", 1)
        for cond, g in df.groupby("condition"):
            harm, ben = g[(g.intent == "shown") & (g.scenario == "harmful")], g[(g.intent == "shown") & (g.scenario == "benign")]
            hid = g[g.intent == "hidden"]
            ph, pb = np.stack(harm.P)[:, best], np.stack(ben.P)[:, best]
            y = np.r_[np.ones(len(ph)), np.zeros(len(pb))]
            # control: harmful and benign prompts differ lexically, so how well do random directions separate them?
            X = acts[np.r_[harm.index, ben.index], best].astype(np.float32)
            R = rng.standard_normal((X.shape[1], 50))
            rand = [max(a, 1 - a) for a in (roc_auc_score(y, X @ R[:, i]) for i in range(R.shape[1]))]
            rows.append(dict(
                model=model, frame=frame, condition=cond, layer=best, n=len(harm),
                refusal_hidden=hid.refused.mean(), refusal_harmful=harm.refused.mean(), refusal_benign=ben.refused.mean(),
                delivered_harmful=harm.delivered.mean(),
                harm_gap_pct=100 * (ph.mean() - pb.mean()) / scale,
                harm_auroc=roc_auc_score(y, np.r_[ph, pb]),
                rand_auroc_mean=float(np.mean(rand)), rand_auroc_p95=float(np.percentile(rand, 95)),
                proj_hidden_pct=100 * (np.stack(hid.P)[:, best].mean() - dirs["safe_mean"][best]) / scale,
            ))
            H, B = np.stack(harm.P), np.stack(ben.P)
            for l in range(H.shape[1]):
                layer_rows.append(dict(model=model, frame=frame, condition=cond, layer=l,
                                       auroc=roc_auc_score(y, np.r_[H[:, l], B[:, l]]),
                                       gap_pct=100 * (H[:, l].mean() - B[:, l].mean())
                                       / (dirs["harm_mean"][l] - dirs["safe_mean"][l])))
        # composition: whole vs split with intent hidden
        hid = df[df.intent == "hidden"]
        w = np.stack(hid[hid.condition == "whole"].P)[:, best].mean()
        for c in ("split", "split_board"):
            s = np.stack(hid[hid.condition == c].P)[:, best].mean()
            rows.append(dict(model=model, frame=frame, condition=f"whole-minus-{c} (hidden)", layer=best,
                             harm_gap_pct=100 * (w - s) / scale))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    out.to_csv(Path(root) / "summary.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(Path(root) / "layer_auroc.csv", index=False)

    steer = [p / "steer_summary.csv" for p in sorted(Path(root).iterdir()) if (p / "steer_summary.csv").exists()]
    for p in steer:
        print(f"\n{p.parent.name} steering:\n{pd.read_csv(p).to_string(index=False)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
