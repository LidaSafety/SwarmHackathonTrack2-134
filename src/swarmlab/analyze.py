"""Aggregate a sweep directory: summary.csv, per-condition means, metric-vs-N plots, jump detection."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METRICS = {
    "leader_reach": "Top leader reach (followers / N-1)",
    "leader_emerged": "P(leader emerged)",
    "dol_index": "Division-of-labour index",
    "coverage": "Task coverage (legit)",
    "mean_score": "Mean score",
    "joined_frac": "Fraction on board",
    "adopt_frac": "Exploit adoption fraction",
    "role_entropy": "Role entropy (bits)",
}


def load_runs(d: Path) -> pd.DataFrame:
    rows = []
    for mf in sorted(d.glob("*/metrics.json")):
        m = json.loads(mf.read_text())
        rows.append({
            **m["run"],
            "run_dir": mf.parent.name,
            "leader_reach": m["leadership"]["leader_reach"],
            "leader_emerged": float(m["leadership"]["leader_emerged"]),
            "n_leaders": m["leadership"]["n_leaders"],
            "compliance_rate": m["leadership"]["compliance_rate"],
            "first_assignment_round": m["leadership"]["first_assignment_round"],
            "top_is_hub": _top_is_hub(m),
            "dol_index": m["division_of_labour"]["dol_index"],
            "coverage": m["division_of_labour"]["coverage"],
            "mean_score": m["division_of_labour"]["mean_score"],
            "copy_frac": m["division_of_labour"]["copy_frac"],
            "joined_frac": m["board"]["joined_frac"],
            "n_posts": m["board"]["n_posts"],
            "adopt_frac": m["exploit"]["adopt_frac"],
            "aware_frac": m["exploit"]["aware_frac"],
            "r_eff": m["exploit"]["r_eff"],
            "first_use_round": m["exploit"]["first_use_round"],
            "role_entropy": m["roles"]["entropy"],
            **{f"role_{k}": v for k, v in m["roles"]["counts"].items()},
            "refusals": sum(v.get("refusals", 0) for v in m["llm"].values()),
            "llm_errors": sum(v.get("errors", 0) for v in m["llm"].values()),
        })
    return pd.DataFrame(rows)


def _top_is_hub(m: dict) -> float | None:
    agents, top = m["agents"], m["leadership"]["top_agent"]
    if not top or m["leadership"]["top_followers"] == 0:
        return None
    return float(agents[top]["degree"] == max(a["degree"] for a in agents.values()))


def jump_score(xs: np.ndarray, ys: np.ndarray) -> float:
    """Largest single-step change as a fraction of the total range (1.0 = all change in one step)."""
    if len(ys) < 3 or np.nanmax(ys) - np.nanmin(ys) < 1e-9:
        return float("nan")
    return float(np.nanmax(np.abs(np.diff(ys))) / (np.nanmax(ys) - np.nanmin(ys)))


def analyze(d: Path) -> pd.DataFrame:
    df = load_runs(d)
    if df.empty:
        print(f"no runs found in {d}")
        return df
    df.to_csv(d / "summary.csv", index=False)
    keys = [k for k in ("model_mix", "topology", "n_agents") if df[k].nunique() > 0]
    agg = df.groupby(keys)[list(METRICS) + ["r_eff", "compliance_rate", "top_is_hub"]].agg(["mean", "std", "count"])
    agg.to_csv(d / "by_condition.csv")
    print(df.groupby(keys)[list(METRICS)].mean().round(3).to_string())

    # metric vs N, one line per topology, one figure per model mix
    mets = list(METRICS)
    jumps = []
    for mix in sorted(df.model_mix.unique()):
        sub = df[df.model_mix == mix]
        fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
        for ax, met in zip(axes.flat, mets):
            for topo, g in sub.groupby("topology"):
                s = g.groupby("n_agents")[met].agg(["mean", "std"]).sort_index()
                ax.errorbar(s.index, s["mean"], yerr=s["std"].fillna(0), marker="o", ms=3, capsize=2, label=topo)
                jumps.append({"model_mix": mix, "topology": topo, "metric": met,
                              "jump_score": jump_score(s.index.values, s["mean"].values)})
            ax.set_xscale("log", base=2)
            ax.set_title(METRICS[met], fontsize=9)
            ax.tick_params(labelsize=7)
        axes[0][0].legend(fontsize=7)
        fig.suptitle(f"model mix: {mix}  (mean ± sd over seeds)")
        fig.supxlabel("population size N")
        fig.tight_layout()
        fig.savefig(d / f"scaling_{mix}.png", dpi=130)
        plt.close(fig)
    pd.DataFrame(jumps).to_csv(d / "jumps.csv", index=False)
    print(f"\nwrote {d/'summary.csv'}, {d/'by_condition.csv'}, {d}/scaling_<mix>.png, {d/'jumps.csv'}")

    for mf in sorted(d.glob("*/metrics.json")):
        plot_run(mf.parent)
    return df


def plot_run(run_dir: Path) -> None:
    """Communication graph with influence (assignment-followed) edges and exploit spread for one run."""
    m = json.loads((run_dir / "metrics.json").read_text())
    events = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    init = next(e for e in events if e["kind"] == "init")
    g = nx.Graph()
    g.add_nodes_from(m["agents"])
    g.add_edges_from(init["graph_edges"])
    pos = nx.spring_layout(g, seed=0)
    role_color = {"coordinator": "#d62728", "worker": "#1f77b4", "free_rider": "#ff7f0e",
                  "exploiter": "#9467bd", "idle": "#bbbbbb"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    colors = [role_color[m["agents"][n]["role"]] for n in g.nodes]
    sizes = [120 + 120 * m["agents"][n]["followers"] for n in g.nodes]
    nx.draw_networkx_edges(g, pos, ax=a1, alpha=0.15)
    dg = nx.DiGraph()
    dg.add_weighted_edges_from(m["leadership"]["influence_edges"])
    nx.draw_networkx_edges(dg, pos, ax=a1, edge_color="#d62728", width=[1 + w / 3 for *_, w in dg.edges(data="weight")],
                           arrows=True, arrowsize=10)
    nx.draw_networkx_nodes(g, pos, ax=a1, node_color=colors, node_size=sizes)
    nx.draw_networkx_labels(g, pos, ax=a1, font_size=6)
    for r, c in role_color.items():
        a1.scatter([], [], c=c, label=r)
    a1.legend(fontsize=7, loc="lower left")
    a1.set_title("roles + influence (red arrows = assignments followed)", fontsize=9)
    a1.axis("off")
    X = m["exploit"]
    a2.plot(range(1, len(X["adoption_curve"]) + 1), X["adoption_curve"], marker="o")
    a2.set_ylim(0, 1)
    a2.set_xlabel("round")
    a2.set_ylabel("fraction of agents using exploit")
    a2.set_title(f"exploit adoption (self={len(X['self_learned'])}, social={len(X['social_learned'])}, "
                 f"abstained={len(X['aware_but_abstained'])})", fontsize=9)
    fig.suptitle(run_dir.name, fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / "run.png", dpi=120)
    plt.close(fig)
