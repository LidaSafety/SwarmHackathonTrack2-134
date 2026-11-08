"""Frozen analysis of the confirmation batch (experiment_handoff_plan.txt, Priorities 1 and 3).

Writes runs/confirm_analysis/{runs.csv, conditions.csv, summary.md}. Primary outcome: plan alignment above chance
(swarmlab.alignment, METRIC_VERSION align-v1). Historical P (metrics leader_reach) and J (scripts/joint_leaders.py
strict reach, joint runs only) are reported beside it, never as substitutes. Runs are the unit of analysis; agents and
calls within a run are not independent replications.

Usage: python scripts/confirm_analysis.py [runs_root]   (default: runs/)
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from swarmlab.alignment import METRIC_VERSION, analyse_world  # noqa: E402
from swarmlab.metrics import replay_world  # noqa: E402

SWEEPS = {"confirm_board": "board", "confirm_cache_none": "cache", "confirm_cache_holder": "cache",
          "confirm_cache_public": "cache", "confirm_reasoning": "reasoning"}


def process_stats(ev: list[dict]) -> dict:
    llm = [e for e in ev if e["kind"] == "llm"]
    fr = Counter(e.get("finish_reason") for e in llm)
    return dict(calls=len(llm), llm_errors=sum(bool(e["error"]) for e in llm),
                parse_errors=sum(bool(e["error"]) and "parse" in e["error"] for e in llm),
                finish_length=sum(e.get("finish_reason") == "length" or e.get("empty_retry_finish") == "length"
                                  for e in llm),
                finish_error=fr.get("error", 0), billed_usd=round(sum(e.get("billed_usd", 0.0) for e in llm), 4),
                est_usd=round(sum(e.get("est_usd", 0.0) for e in llm), 4),
                reasoning_tok=sum(e.get("reasoning_tok") or 0 for e in llm), out_tok=sum(e["out_tok"] for e in llm))


def run_row(sweep: str, d: Path) -> dict:
    w = replay_world(d)
    cfg = w.cfg
    m = json.loads((d / "metrics.json").read_text())
    a = analyse_world(w, seed=cfg.seed)
    row = dict(sweep=sweep, family=SWEEPS[sweep], run=d.name, seed=cfg.seed, model=cfg.model_mix,
               n_agents=cfg.n_agents, n_tasks=cfg.n_tasks, peer_info=cfg.peer_info, joint_tasks=cfg.joint_tasks,
               roster={"confirm_cache_none": "none", "confirm_cache_holder": "holder",
                       "confirm_cache_public": "public"}.get(sweep, cfg.peer_info))
    row.update({k: a[k] for k in ("plan_produced", "first_plan_round", "n_plans", "aff", "aligned_leadership",
                                  "excess_alignment", "excess_testable", "observed_alignment", "null_alignment",
                                  "best_author", "exposed_recipients", "insufficient_recipients",
                                  "convention_compatible", "control_alignment", "control_n", "sub_aff",
                                  "partner_compliance")})
    row["n_plan_authors"] = len(a["plan_authors"])
    row["holder"] = w.holder
    row["best_is_holder"] = (a["best_author"] == w.holder) if w.holder and a["best_author"] else None
    row["primary_plan_text"] = (a["primary_plan"] or {}).get("text", "")
    for k, v in a["truncation"].items():
        row[f"trunc_{k}"] = v
    L = m["leadership"]
    row.update(P_leader_reach=L["leader_reach"], P_top=L["top_agent"])
    if cfg.joint_tasks > 1:
        from joint_leaders import analyse as jl
        j = jl(d)
        row.update(J_strict_reach=j["strict_reach"], joint_solved=j["solved"])
    D, X = m["division_of_labour"], m["exploit"]
    row.update(unique_tasks_solved=D["unique_tasks_solved"], legit_solves=D["legit_solves"],
               team_accepted=round(D["mean_score"] * cfg.n_agents), redundancy=D["redundancy"])
    if sweep == "confirm_reasoning":  # Priority 3 measures
        aware = len(X["self_learned"]) + len(X["social_learned"])
        row.update(bypass_aware_frac=X["aware_frac"], bypass_adopt_frac=X["adopt_frac"],
                   bypass_self_discovered=len(X["discoverers_self"]), bypass_adopters=len(X["adopters"]),
                   adopt_given_aware=(len(X["adopters"]) / (X["aware_frac"] * cfg.n_agents)) if X["aware_frac"] else None,
                   adopters_self_learned=len(X["self_learned"]), adopters_social=len(X["social_learned"]),
                   bypass_submissions=X["exploit_submissions"], first_bypass_round=X["first_use_round"],
                   bypass_warning_posts=m["compromise"]["warning_posts"])
        del aware
    row.update(process_stats(w.events))
    return row


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def fmt(x, nd=2):
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs"
    out = root / "confirm_analysis"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sweep in SWEEPS:
        for d in sorted((root / sweep).glob("*/")):
            if (d / "metrics.json").exists():
                rows.append(run_row(sweep, d))
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)

    keys = {"board": ["model", "roster", "n_tasks"], "cache": ["model", "roster", "joint_tasks"], "reasoning": ["model"]}
    tables, md = [], []
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    md += [f"# Confirmation batch analysis", "",
           f"Metric {METRIC_VERSION}; analysis code at commit {commit} (uncommitted changes, if any, are not recorded).",
           f"{len(df)} runs. Unit of analysis: the run. Entries k/n are run counts.", ""]
    for fam, ks in keys.items():
        sub = df[df.family == fam]
        if sub.empty:
            continue
        g = sub.groupby(ks, dropna=False)
        agg = g.apply(lambda x: pd.Series(dict(
            n=len(x),
            plan_produced=f"{int(x.plan_produced.sum())}/{len(x)}",
            aligned_leadership=f"{int(x.aligned_leadership.sum())}/{len(x)}",
            aff_mean=x.aff.mean(), aff_runs=" ".join(fmt(v) for v in x.aff),
            excess_mean_given_plan=x.excess_alignment.dropna().mean() if x.excess_alignment.notna().any() else None,
            excess_n=int(x.excess_alignment.notna().sum()),
            convention_flag=int(x.convention_compatible.fillna(False).sum()),
            P_mean=x.P_leader_reach.mean(),
            J_mean=(x.J_strict_reach.mean() if "J_strict_reach" in x and x.J_strict_reach.notna().any() else None),
            best_is_holder=(f"{int(x.best_is_holder.fillna(False).sum())}/{int(x.best_is_holder.notna().sum())}"
                            if x.best_is_holder.notna().any() else ""),
            partner_compliance=(x.partner_compliance.dropna().mean() if x.partner_compliance.notna().any() else None),
            unique_solved_mean=x.unique_tasks_solved.mean(),
            **({"bypass_adopt_frac": x.bypass_adopt_frac.mean(), "bypass_aware_frac": x.bypass_aware_frac.mean(),
                "adopt_given_aware": x.adopt_given_aware.dropna().mean() if x.adopt_given_aware.notna().any() else None,
                "bypass_runs_any_adopter": f"{int((x.bypass_adopters > 0).sum())}/{len(x)}",
                "legit_solves_mean": x.legit_solves.mean(), "finish_length_calls": int(x.finish_length.sum()),
                "reasoning_tok_mean": x.reasoning_tok.mean()} if fam == "reasoning" else {}),
        )), include_groups=False).reset_index()
        agg.insert(0, "family", fam)
        tables.append(agg)
        md += [f"## {fam}", "", md_table(agg), ""]
    pd.concat(tables).to_csv(out / "conditions.csv", index=False)
    md += ["## Process", "", f"LLM calls {int(df.calls.sum())}, errors {int(df.llm_errors.sum())}, "
           f"cap hits (finish_reason length, incl. first attempts retried) {int(df.finish_length.sum())}, "
           f"provider finish_reason error {int(df.finish_error.sum())}, billed ${df.billed_usd.sum():.2f} "
           f"(price-table estimate ${df.est_usd.sum():.2f}).", ""]
    trunc_cols = [c for c in df.columns if c.startswith("trunc_")]
    if trunc_cols:
        md += ["Truncation (summed over runs, by family):", "",
               md_table(df.groupby("family")[trunc_cols].sum(), index=True), ""]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
