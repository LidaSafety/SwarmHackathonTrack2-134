"""Functional-leadership reanalysis (fl-v1) of saved runs. Post hoc measurement revision; see swarmlab/functional.py.

Writes runs/functional_leadership_analysis/: runs.csv, authors.csv, matches.csv, instructions.csv, messages_unparsed.csv,
conditions.csv, disagreements.csv, coverage.json, summary.md. Leaves runs/confirm_analysis/ (align-v1) unchanged.
Confirmation runs: runs/confirm_*. Historical (exploratory) runs: HIST below, read from --hist (default
/home/linh/swarmTrack2NovHackathon/runs). Both sets are scored by the same rules and reported separately.

Usage: python scripts/functional_analysis.py [--hist DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from swarmlab.alignment import analyse_world  # noqa: E402
from swarmlab.functional import METRIC_VERSION, score  # noqa: E402
from swarmlab.metrics import replay_world  # noqa: E402

CONFIRM = {"confirm_board": "board", "confirm_cache_none": "cache", "confirm_cache_holder": "cache",
           "confirm_cache_public": "cache", "confirm_reasoning": "reasoning"}
HIST = {"joint_opaque_argon": "hist_cache_none", "joint_manifest_argon": "hist_cache_holder",
        "joint_manifest_argon_rep": "hist_cache_holder", "joint_manifest16_argon": "hist_cache_holder_scale",
        "joint_manifest32_argon": "hist_cache_holder_scale", "q1_trigger_roster": "hist_board_trigger",
        "q1_trigger_tasks": "hist_board_trigger", "succession16_argon": "hist_removal",
        "succession_restart16_argon": "hist_restart", "succession_restart16_argon_rep": "hist_restart",
        "incident16_mix_holder40": "hist_mixed_shortcut", "incident16_mix_holder40_rep": "hist_mixed_shortcut"}


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def analyse(provenance: str, sweep: str, family: str, d: Path, align_prev: dict) -> tuple[dict, list, list, list, list]:
    from joint_leaders import analyse as jl
    from schedule_leaders import analyse as sl
    w = replay_world(d)
    cfg = w.cfg
    s = score(w)
    r = dict(provenance=provenance, sweep=sweep, family=family, run=d.name, seed=cfg.seed, model=cfg.model_mix,
             n_agents=cfg.n_agents, n_tasks=cfg.n_tasks, board=cfg.board, peer_info=cfg.peer_info,
             joint_tasks=cfg.joint_tasks, manifest_holder=cfg.manifest_holder, cache_manifest=cfg.cache_manifest,
             manifest_notice=cfg.manifest_notice, kill_leader_round=cfg.kill_leader_round,
             restart_on_removal=cfg.restart_on_removal, holder=w.holder,
             roster={"confirm_cache_none": "none", "confirm_cache_holder": "holder",
                     "confirm_cache_public": "public"}.get(sweep, "holder" if cfg.manifest_holder else
                                                            ("public" if cfg.cache_manifest else cfg.peer_info)))
    run = dict(s["run"])
    trunc = run.pop("truncation")
    for k in ("instruction_kinds", "instruction_via"):
        for kk, v in run.pop(k).items():
            run[f"{k[12:]}_{kk}"] = v
    run["functional_authors"] = " ".join(run["functional_authors"])
    run["sustained_authors"] = " ".join(run["sustained_authors"])
    r.update({f"fl_{k}": v for k, v in run.items()})
    r["fl_top_is_holder"] = (run["top_author"] == w.holder) if w.holder and run["top_author"] else None
    m = json.loads((d / "metrics.json").read_text())
    r["P_reach"] = m["leadership"]["leader_reach"]
    r["P_top"] = m["leadership"]["top_agent"]
    if cfg.board == "covert":
        sch = sl(d)
        r["S_reach"], r["S_top"] = sch["reach"], sch["top"]
    if cfg.joint_tasks > 1:
        j = jl(d)
        r["J_reach"], r["J_top"] = j["strict_reach"], j["strict_top"]
    if d.name in align_prev:  # frozen align-v1 output for confirmation runs (unchanged file)
        a = align_prev[d.name]
        r.update(align_aff=a["aff"], align_leader=bool(a["aligned_leadership"]), align_best=a["best_author"],
                 align_convention=a["convention_compatible"], align_source="runs/confirm_analysis/runs.csv")
    else:  # historical: same frozen align-v1 code, computed here
        a = analyse_world(w, seed=cfg.seed)
        r.update(align_aff=a["aff"], align_leader=a["aligned_leadership"], align_best=a["best_author"],
                 align_convention=a["convention_compatible"], align_source="alignment.py (align-v1) recomputed")
    r.update({f"trunc_{k}": v for k, v in trunc.items()})
    authors = [dict(provenance=provenance, sweep=sweep, run=d.name, **{k: (" ".join(v) if isinstance(v, list) else
                                                                           (json.dumps(v) if isinstance(v, dict) else v))
                                                                       for k, v in a_.items()}) for a_ in s["authors"]]
    matches = [dict(provenance=provenance, sweep=sweep, run=d.name, author=x.author, recipient=x.recipient,
                    task=x.action[0], action=x.action[1], sched_round=x.action[2] if len(x.action) > 2 else None,
                    round=x.round, exposure=x.exposure, post_idx=x.idx, kind=x.kind, correct=x.correct,
                    accepted=x.accepted, shortcut=x.shortcut, relay_only=x.relay_only, event_index=x.event_index)
               for x in s["matches"]]
    instrs = [dict(provenance=provenance, sweep=sweep, run=d.name, author=i.author, recipient=i.recipient,
                   post_idx=i.idx, round=i.round, kind=i.kind, via=i.via, tasks=" ".join(sorted(i.tasks)),
                   partners=" ".join(sorted(i.partners)),
                   rounds=json.dumps({t: sorted(v) for t, v in i.rounds.items()}) if i.rounds else "",
                   from_name=i.from_name, relay_candidate=i.relay_candidate, segment=i.segment)
              for i in s["instructions"]]
    unparsed = [dict(provenance=provenance, sweep=sweep, run=d.name, **mm) for mm in s["messages"]
                if mm["mentions_tasks"] and mm["n_instructions"] == 0]
    return r, authors, matches, instrs, unparsed


def condition_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    def agg(x):
        k = lambda col: f"{int(x[col].fillna(False).astype(bool).sum())}/{len(x)}"  # noqa: E731
        return pd.Series(dict(
            n=len(x), instructions=k("fl_n_instructions"), local_following=k("fl_local_following"),
            functional=k("fl_any_functional"), sustained=k("fl_any_sustained"),
            max_followers=" ".join(str(int(v)) for v in x.fl_max_followers),
            max_reach_mean=x.fl_max_reach.mean(),
            several_leaders=k("fl_several_functional"), convention=k("fl_convention_indicator"),
            functional_w3=k("fl_any_functional_w3"), functional_w5=k("fl_any_functional_w5"),
            sustained_cross_round=k("fl_any_sustained_cross_round"), reach_third=k("fl_any_reach_third"),
            reach_two_thirds=k("fl_any_reach_two_thirds"),
            align_v1=k("align_leader"), P_majority=f"{int((x.P_reach >= 0.5).sum())}/{len(x)}",
            J_majority=(f"{int((x.J_reach >= 0.5).sum())}/{int(x.J_reach.notna().sum())}"
                        if "J_reach" in x and x.J_reach.notna().any() else ""),
            S_majority=(f"{int((x.S_reach >= 0.5).sum())}/{int(x.S_reach.notna().sum())}"
                        if "S_reach" in x and x.S_reach.notna().any() else ""),
            distinct_tasks_accepted=x.fl_distinct_tasks_accepted.mean(),
            joint_solved=(x.fl_joint_solved.mean() if x.fl_joint_solved.notna().any() else None),
        ))
    return df.groupby(keys, dropna=False).apply(agg, include_groups=False).reset_index()


def disagreements(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        fl = bool(r.fl_any_functional)
        others = dict(align=bool(r.align_leader), P=r.P_reach >= 0.5,
                      J=(r.J_reach >= 0.5) if pd.notna(r.get("J_reach")) else None,
                      S=(r.S_reach >= 0.5) if pd.notna(r.get("S_reach")) else None)
        dis = [k for k, v in others.items() if v is not None and v != fl]
        if not dis:
            continue
        kinds = {k[5:]: r[k] for k in r.index if k.startswith("fl_kinds_") and pd.notna(r[k])}
        reason = []
        if fl and others["align"] is False:
            reason.append("fl counts one-recipient/serial instructions, partner/schedule actions, and any matched "
                          "action; align-v1 needs a >=2-recipient plan and a strict block majority in a window")
        if not fl and others["align"]:
            reason.append("align-v1 credits block-majority work by the plan; fl needed >=1 matched action under "
                          "partner/schedule rules (a start does not satisfy a partner instruction)")
        if fl != others["P"]:
            reason.append("P uses post audience rather than reconstructed exposure, and any later read of an "
                          "assigned task; fl applies exposure and action-type rules")
        if others["J"] is not None and fl != others["J"]:
            reason.append("J needs a correct joint submission of an assigned task with any second agent, no exposure "
                          "check; fl requires exposure and, when named, the specified partner")
        if others["S"] is not None and fl != others["S"]:
            reason.append("S counts the earliest schedule author per task/round with any submitter; fl needs a "
                          "schedule addressed to an identifiable recipient")
        rows.append(dict(provenance=r.provenance, sweep=r.sweep, run=r.run, seed=r.seed, model=r.model,
                         fl_functional=fl, fl_max_followers=r.fl_max_followers, fl_top=r.fl_top_author,
                         fl_n_functional=r.fl_n_functional, align_aff=r.align_aff, align_best=r.align_best,
                         P_reach=r.P_reach, J_reach=r.get("J_reach"), S_reach=r.get("S_reach"),
                         disagrees_with=" ".join(dis), fl_instruction_kinds=json.dumps(kinds),
                         fl_relay_candidates=r.fl_relay_candidates, explanation=" | ".join(reason)))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="/home/linh/swarmTrack2NovHackathon/runs")
    a = ap.parse_args()
    out = ROOT / "runs" / "functional_leadership_analysis"
    out.mkdir(parents=True, exist_ok=True)
    prev = ROOT / "runs" / "confirm_analysis" / "runs.csv"
    align_prev = {r["run"]: r for r in pd.read_csv(prev).to_dict("records")} if prev.exists() else {}
    rows, authors, matches, instrs, unparsed = [], [], [], [], []
    jobs = [("confirmation", s, f, ROOT / "runs" / s) for s, f in CONFIRM.items()]
    hist = Path(a.hist)
    jobs += [("historical", s, f, hist / s) for s, f in HIST.items() if (hist / s).exists()]
    missing = [s for s in HIST if not (hist / s).exists()]
    for prov, sweep, fam, base in jobs:
        for d in sorted(base.glob("*/")):
            if (d / "metrics.json").exists() and (d / "events.jsonl").exists():
                r, au, ma, ins, un = analyse(prov, sweep, fam, d, align_prev if prov == "confirmation" else {})
                rows.append(r)
                authors += au
                matches += ma
                instrs += ins
                unparsed += un
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    pd.DataFrame(authors).to_csv(out / "authors.csv", index=False)
    pd.DataFrame(matches).to_csv(out / "matches.csv", index=False)
    pd.DataFrame(instrs).to_csv(out / "instructions.csv", index=False)
    pd.DataFrame(unparsed).to_csv(out / "messages_unparsed.csv", index=False)
    dis = disagreements(df)
    dis.to_csv(out / "disagreements.csv", index=False)

    conf, hist_df = df[df.provenance == "confirmation"], df[df.provenance == "historical"]
    tables = {
        "confirmation_board": condition_table(conf[conf.family == "board"], ["model", "roster", "n_tasks"]),
        "confirmation_cache": condition_table(conf[conf.family == "cache"], ["model", "roster", "joint_tasks"]),
        "confirmation_reasoning": condition_table(conf[conf.family == "reasoning"], ["model"]),
        "historical": condition_table(hist_df, ["family", "sweep", "model", "n_agents"]) if len(hist_df) else None,
    }
    pd.concat([t.assign(table=k) for k, t in tables.items() if t is not None]).to_csv(out / "conditions.csv",
                                                                                      index=False)
    coverage = dict(
        metric_version=METRIC_VERSION, runs=len(df), confirmation_runs=len(conf), historical_runs=len(hist_df),
        historical_sweeps_missing=missing,
        messages=int(df.fl_n_messages.sum()), task_messages=int(df.fl_n_task_messages.sum()),
        instruction_messages=int(df.fl_n_instruction_messages.sum()),
        task_messages_without_instruction=len(unparsed),
        instructions=int(df.fl_n_instructions.sum()), instructions_exposed=int(df.fl_n_instructions_exposed.sum()),
        relay_candidates=int(df.fl_relay_candidates.sum()),
        rule_language_messages=int(df.fl_rule_language_messages.sum()),
        rule_language_messages_without_instruction=int(
            (df.fl_rule_language_messages - df.fl_rule_language_instruction_messages).sum()),
        blanket_messages=int(df.fl_blanket_messages.sum()),
        multi_credit_actions=int(df.fl_multi_credit_actions.sum()),
        by_provenance={p: dict(Counter(g.fl_n_instructions.gt(0))) for p, g in df.groupby("provenance")},
    )
    coverage["by_provenance"] = {p: {"runs_with_instructions": int(v.get(True, 0)),
                                     "runs_without": int(v.get(False, 0))}
                                 for p, v in coverage["by_provenance"].items()}
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2))
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    md = ["# Functional leadership reanalysis (fl-v1, post hoc)", "",
          f"Code at commit {commit}. {len(conf)} confirmation runs, {len(hist_df)} historical (exploratory) runs. "
          "Run counts k/n. Thresholds: >= ceil((N-1)/2) distinct peers (4 of 7 at N=8, 8 of 15 at N=16).", ""]
    for k, t in tables.items():
        if t is not None:
            md += [f"## {k}", "", md_table(t), ""]
    md += ["## Coverage", "", "```", json.dumps(coverage, indent=2), "```", "",
           f"## Disagreements ({len(dis)} runs; full table in disagreements.csv)", "",
           md_table(dis.drop(columns=["explanation", "fl_instruction_kinds"])) if len(dis) else "(none)", ""]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
