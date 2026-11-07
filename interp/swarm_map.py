"""Where do swarm mechanisms appear on their own? Per-call labels mined from every free-running swarm log.

For each LLM call (one agent, one round) the actions it produced are labelled with naturally occurring swarm mechanisms:
  read_board   read the board history
  post         posted anything
  coord_post   posted a coordination proposal, a self-claim or an assignment
  assign       posted an assignment for another agent
  share        posted task answers / data values for others
  follow       touched only tasks a visible peer post had assigned to it (only calls with an assignment pending)
  avoid        touched no task that a visible peer had already claimed (only calls where such claims exist)
  copy         submitted an accepted answer without reading the data for it (answer taken from the board)
Calls are joined with the scenario (model, N, topology, incentive, board, peer info, adversary) and the round.

Writes runs/swarm_map/calls.csv (one row per call), by_scenario.csv, and prints a scenario map.
Usage: python interp/swarm_map.py [sweep dirs ...]
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swarmlab.metrics import KV, parse_directives  # noqa: E402

SWEEPS = ["runs/or_pilot", "runs/or_pilot2", "runs/or_pilot3", "runs/or_pilot4", "runs/or_scaling",
          "runs/or_transition", "runs/or_compromise", "runs/local_pilot", "runs/interp_q3/swarm_Qwen3-8B",
          "runs/swarm_mixed"]
COORD = re.compile(r"coordinat|split|divide|don't duplicate|avoid duplicat|who (?:does|takes)|assign", re.I)
ANSWER = re.compile(r"\bT\d{2}\s*(?:[:=]|->|is|answer)\s*-?\d+", re.I)
MECHS = ["read_board", "post", "coord_post", "assign", "share", "follow", "avoid", "copy"]


def task_of(e) -> int | None:
    if e["kind"] == "submit":
        t = str(e.get("task", ""))
    elif e["kind"] in ("read", "read_missing") and "/tasks/" in str(e.get("path", "")):
        t = str(e["path"]).rsplit("/", 1)[-1].split(".")[0]
    else:
        return None
    return int(t[1:]) if len(t) == 3 and t[1:].isdigit() else None


def run_calls(run: Path) -> list[dict]:
    cfg = json.loads((run / "config.json").read_text())
    events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    init = next(e for e in events if e["kind"] == "init")
    bad = set(init.get("compromised", []))
    n_tasks = cfg.get("n_tasks", 24)
    scen = dict(sweep=run.parent.name, run=run.name, model=cfg.get("model_mix") or cfg["models"][0]["name"],
                N=cfg["n_agents"], topology=cfg["topology"], pressure=cfg["pressure"], board=cfg["board"],
                peer_info=cfg.get("peer_info", "none"), rounds=cfg["rounds"], adversary=int(cfg.get("compromised", 0) > 0))
    assigned = defaultdict(set)          # agent -> tasks assigned to it by visible peer posts so far
    claimed_by_others = defaultdict(set)  # agent -> tasks claimed by peers in posts it can see
    read_data = defaultdict(int)         # agent -> data reads so far (proxy for doing the lookup itself)
    calls, cur = [], {}
    pending_posts = []                   # posts become visible from the next round on
    for e in events:
        k, a = e["kind"], e.get("agent")
        if k == "llm":
            r = e["round"]
            for p in [p for p in pending_posts if p["round"] < r]:
                assigns, claims = parse_directives(p["agent"], p["message"], n_tasks)
                for tgt, ts in assigns.items():
                    if tgt in p["audience"]:
                        assigned[tgt] |= {int(t[1:]) for t in ts}
                for x in p["audience"]:
                    if x != p["agent"]:
                        claimed_by_others[x] |= {int(t[1:]) for t in claims}
                        for tgt, ts in assigns.items():
                            if tgt != x:
                                claimed_by_others[x] |= {int(t[1:]) for t in ts}
                pending_posts.remove(p)
            row = dict(scen, agent=a, agent_model=init["models"].get(a), act_idx=e.get("act_idx"), round=r,
                       honest=a not in bad, pending_assignment=bool(assigned[a]),
                       claims_visible=bool(claimed_by_others[a]), touched=[], copies=0, error=e.get("error") is not None,
                       **{m: False for m in MECHS})
            row["_assigned"], row["_claimed"] = set(assigned[a]), set(claimed_by_others[a])
            calls.append(row)
            cur[a] = row
            continue
        row = cur.get(a)
        if row is None:
            continue
        if k == "read_board":
            row["read_board"] = True
        elif k == "post":
            row["post"] = True
            assigns, claims = parse_directives(a, e["message"], n_tasks)
            row["assign"] |= bool(assigns)
            row["coord_post"] |= bool(assigns or claims or COORD.search(e["message"]))
            row["share"] |= bool(ANSWER.search(e["message"]) or KV.search(e["message"]))
            pending_posts.append(dict(round=e["round"], agent=a, message=e["message"], audience=set(e["audience"])))
        elif k == "read" and str(e.get("path", "")).startswith("/data"):
            read_data[a] += 1
        if k == "submit" and e.get("accepted") and not e.get("exploit") and not e.get("legit_work", True):
            row["copies"] += 1
        t = task_of(e)
        if t is not None:
            row["touched"].append(t)
    for row in calls:
        ts = set(row["touched"])
        row["follow"] = bool(ts) and row["pending_assignment"] and ts <= row["_assigned"]
        row["avoid"] = bool(ts) and row["claims_visible"] and not (ts & row["_claimed"])
        row["copy"] = row["copies"] > 0
        row["n_touched"] = len(ts)
        del row["_assigned"], row["_claimed"]
        row["touched"] = ",".join(map(str, sorted(ts)))
    return calls


def main(*sweeps: str) -> None:
    pd.set_option("display.width", 250)
    rows = []
    for s in sweeps or SWEEPS:
        for run in sorted(Path(s).glob("*/events.jsonl")):
            if (run.parent / "config.json").exists():
                rows += run_calls(run.parent)
    df = pd.DataFrame(rows)
    df = df[df.honest & ~df.error]
    out = Path("runs/swarm_map")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "calls.csv", index=False)
    print(f"{df.run.nunique()} runs, {len(df)} honest calls")

    # follow / avoid only defined where an assignment / peer claim was visible and a task was touched
    df["follow_def"] = df.pending_assignment & (df.n_touched > 0)
    df["avoid_def"] = df.claims_visible & (df.n_touched > 0)
    keys = ["model", "agent_model", "board", "pressure", "topology", "N", "adversary"]
    g = df.groupby(keys)
    tab = g[["read_board", "post", "coord_post", "assign", "share", "copy"]].mean()
    tab["follow"] = g.apply(lambda h: h.follow[h.follow_def].mean() if h.follow_def.any() else float("nan"))
    tab["n_follow"] = g.follow_def.sum()
    tab["avoid"] = g.apply(lambda h: h.avoid[h.avoid_def].mean() if h.avoid_def.any() else float("nan"))
    tab["n_avoid"] = g.avoid_def.sum()
    tab["runs"] = g.run.nunique()
    tab["calls"] = g.size()
    tab = tab.round(3)
    tab.to_csv(out / "by_scenario.csv")
    print(tab.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
