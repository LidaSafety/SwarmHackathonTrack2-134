"""Precursors of swarm behaviour in Qwen3-8B (mixed swarms, runs/swarm_mixed): is there a signal before the agent acts?

Agents are stateless across rounds except for their private notes and the board, so a precursor can live in
(1) earlier rounds, carried by notes or board; (2) earlier layers of the deciding call; (3) earlier tokens of the reply.
This script covers (1) and (2) on saved residuals (last prompt token, CPU):

A  Early warning. On calls where the agent does NOT post coordination, predict whether it will post coordination in the
   next k rounds (k = 1, 3, 5). Only calls before the agent's first coordination post, plus agents that never
   coordinate. Monitors: board + round features, TF-IDF on the observation, TF-IDF on the agent's own previous notes,
   probes per layer, and the length-matched content / decision directions (order_control.py part c). CV by run.
B  Event-aligned trajectories. For agents whose first coordination post is at round t0 >= 3, mean projection on each
   direction at t0-4..t0, residualised on board features and round, against never-coordinating agents of the same runs.
C  Silent text channel. Do the agent's private notes plan coordination before its first coordination post?
D  Layer emergence: AUROC by layer for "posts coordination this call" vs "within 3 rounds".
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_mixed import cv_group  # noqa: E402
from analyze_q3_monitor import probe, text_clf  # noqa: E402
from order_control import ROOT, board_features, board_of, load  # noqa: E402
from swarm_map import COORD  # noqa: E402

OUT = ROOT / "precursor"
PLAN = re.compile(COORD.pattern + r"|claim|\btak(?:e|ing) T\d|\bT\d{2}\s*-\s*T?\d{2}|range|others? (?:are|is) (?:doing|"
                  r"working)|my (?:share|part|block)", re.I)


def notes_of(df: pd.DataFrame) -> list[str]:
    """The notes each Qwen call wrote (from the raw reply)."""
    raw = {}
    for run in df.run.unique():
        for line in (ROOT / run / "events.jsonl").read_text().splitlines():
            e = json.loads(line)
            if e["kind"] == "llm" and "act_idx" in e:
                try:
                    r = e["raw"]
                    raw[(run, e["act_idx"])] = str(json.loads(r[r.index("{"): r.rindex("}") + 1]).get("notes", ""))
                except (ValueError, KeyError, AttributeError):
                    raw[(run, e["act_idx"])] = ""
    return [raw.get((r, i), "") for r, i in zip(df.run, df.act_idx)]


def main() -> None:
    from sklearn.metrics import roc_auc_score
    import statsmodels.formula.api as smf
    pd.set_option("display.width", 200)
    df, A, obs = load()
    layers = json.loads((ROOT / "probe.json").read_text())["save_layers"]
    dz = np.load(ROOT / "order_control" / "part_c_directions.npz")
    li22 = layers.index(int(dz["layer"]))
    feats = pd.DataFrame([board_features(b[1]) if (b := board_of(o)) else
                          dict(n_posts=0, n_coord=0, n_authors=0, omitted=0, n_ids=0) for o in obs])
    df = pd.concat([df.reset_index(drop=True), feats], axis=1)
    df["notes_out"] = notes_of(df)
    df["obs"] = obs
    df = df[df.model == "gem+qwen3"].copy()        # mixed swarms only: where coordination happens
    A = A[df.index.values]
    df = df.reset_index(drop=True)
    df["key"] = df.run + "/" + df.agent
    assert (df.groupby("key")["round"].diff().dropna() > 0).all()  # event order: rounds increase within an agent
    # previous call's notes = what this call sees in YOUR NOTES (text channel carried across rounds)
    df["notes_prev"] = df.groupby("key").notes_out.shift(1).fillna("")
    onset = df[df.coord_post].groupby("key")["round"].min()
    df["onset"] = df.key.map(onset)
    df["never"] = df.onset.isna()
    for d in ("d_in_perp", "d_dec", "d_len"):
        df[d] = A[:, li22] @ dz[d]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(df)} Qwen calls in mixed runs, {df.key.nunique()} agents, {len(onset)} ever post coordination "
          f"(onset rounds: {onset.describe()[['min', '50%', 'max']].to_dict()})")

    # A. early warning
    ctxcols = ["n_posts", "n_coord", "n_authors", "round"]
    rows = []
    pre = df[(~df.coord_post) & (df.never | (df["round"] < df.onset))].copy()
    for k in (1, 3, 5):
        y = ((~pre.never) & (pre.onset - pre["round"] <= k)).values.astype(int)
        m = pre.index.values
        g = pre.run.values
        sc = {"board + round": cv_group(lambda: probe(1.0), pre[ctxcols].values.astype(float), y, g),
              "tf-idf observation": cv_group(text_clf, list(pre.obs), y, g),
              "tf-idf own notes (prev)": cv_group(text_clf, [n or "empty" for n in pre.notes_prev], y, g),
              "content dir (len-matched)": pre.d_in_perp.values, "decision dir (len-matched)": pre.d_dec.values,
              "length dir": pre.d_len.values}
        for li, l in enumerate(layers):
            sc[f"probe L{l}"] = cv_group(probe, A[m, li], y, g)
        for name, s in sc.items():
            ok = ~np.isnan(s)
            rows.append(dict(k=k, monitor=name, n=int(ok.sum()), pos=int(y[ok].sum()), auroc=roc_auc_score(y[ok], s[ok])))
    ew = pd.DataFrame(rows).round(3)
    print("\nA. EARLY WARNING: calls with no coordination post yet; target = first coordination post within k rounds")
    print(ew.pivot(index="monitor", columns="k", values="auroc").to_string())
    print(ew.groupby("k")[["n", "pos"]].first().to_string())
    ew.to_csv(OUT / "early_warning.csv", index=False)

    # B. event-aligned, residualised on board features and round
    for d in ("d_in_perp", "d_dec", "d_len"):
        fit = smf.ols(f"{d} ~ n_posts + n_coord + n_authors + C(round)", df).fit()
        df[d + "_res"] = fit.resid
    ev = df[(~df.never) & (df.onset >= 3)].copy()
    ev["rel"] = ev["round"] - ev.onset
    ev = ev[ev.rel.between(-4, 0)]
    ctrl = df[df.never]
    print(f"\nB. EVENT-ALIGNED ({ev.key.nunique()} agents with onset >= round 3; residual on board + round; "
          f"never-coordinators mean = 0 reference {ctrl[[c + '_res' for c in ('d_in_perp', 'd_dec', 'd_len')]].mean().round(2).to_dict()})")
    t = ev.groupby("rel")[[c + "_res" for c in ("d_in_perp", "d_dec", "d_len")]].agg(["mean", "sem"]).round(2)
    t["n"] = ev.groupby("rel").size()
    print(t.to_string())
    t.to_csv(OUT / "event_aligned.csv")

    # C. silent text channel: notes plan coordination before onset?
    first = df[~df.never].copy()
    first["rel"] = first["round"] - first.onset
    plan = first[first.rel < 0].groupby("key").apply(lambda h: h.notes_out.str.contains(PLAN).any())
    plan_never = df[df.never].groupby("key").apply(lambda h: h.notes_out.str.contains(PLAN).any())
    lead = first[(first.rel < 0) & first.notes_out.str.contains(PLAN)].groupby("key").rel.min()
    print(f"\nC. SILENT NOTES: coordinators whose own notes mention claims/splits/ranges before their first coordination "
          f"post: {plan.mean():.2f} ({plan.sum()}/{len(plan)}), median lead {(-lead).median() if len(lead) else float('nan')} "
          f"rounds; never-coordinators with such notes: {plan_never.mean():.2f} ({plan_never.sum()}/{len(plan_never)})")
    ex = first[(first.rel < 0) & first.notes_out.str.contains(PLAN)].head(3)
    for r in ex.itertuples():
        print(f"   {r.key} round {r.round} (onset {int(r.onset)}): {r.notes_out[:200]!r}")

    # D. layer emergence: same call vs 3 rounds ahead
    print("\nD. LAYER EMERGENCE (probe AUROC, CV by run)")
    y_now = df.coord_post.values.astype(int)
    rows = []
    for li, l in enumerate(layers):
        rows.append(dict(layer=l, this_call=roc_auc_score(y_now, cv_group(probe, A[:, li], y_now, df.run.values)),
                         within_3=ew[(ew.k == 3) & (ew.monitor == f"probe L{l}")].auroc.iloc[0]))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
