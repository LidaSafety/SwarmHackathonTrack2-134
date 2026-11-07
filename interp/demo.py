"""Demo of the mixed-swarm experiment: replay one swarm round by round and read the local agents' minds.

Replays a mixed Gemini 3.8 Flash + Qwen3-8B swarm (from swarm_mixed.py) on the terminal: board posts, every agent's
actions, and for each Qwen3-8B call the probe readout taken from its residual stream at the last prompt token, i.e.
BEFORE it generated a single token, next to what it then actually did (coordinate, follow an assignment, stay off
claimed tasks, copy a posted answer). Probes are trained on all OTHER runs, so the demo run is held out.
Ends with the swarm metrics (leader, reach, division of labour) and the probes' held-out AUROC on this run.

Replay needs only the saved run (CPU, no API). --live first runs a fresh N=4 mixed swarm (GPU0 + OpenRouter key, ~$0.10).
Usage:
  python interp/demo.py                                  # best-looking finished run, auto-picked
  python interp/demo.py --run runs/swarm_mixed/n4_complete_gem+qwen3_s1_ac1eee55 --delay 0.5
  CUDA_VISIBLE_DEVICES=0 python interp/demo.py --live --topology star
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swarm_map import run_calls  # noqa: E402

LOCAL = "Qwen3-8B"
# (mechanism, short name, rows the probe is trained on / shown for)
TARGETS = [("coord_post", "coord", lambda c: True),
           ("follow", "follow", lambda c: c["pending_assignment"]),
           ("avoid", "avoid", lambda c: c["claims_visible"]),
           ("copy", "copy", lambda c: True)]
C = dict(gem="\033[36m", qwen="\033[33m", dim="\033[2m", bold="\033[1m", ok="\033[32m", bad="\033[31m", end="\033[0m")


def col(s: str, k: str) -> str:
    return f"{C[k]}{s}{C['end']}" if C["bold"] else s


def runs_with_acts(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if (p / "metrics.json").exists() and (p / "acts.npy").exists())


def local_calls(run: Path) -> tuple[list[dict], np.ndarray]:
    calls = [c for c in run_calls(run) if c["agent_model"] == LOCAL]
    acts = np.load(run / "acts.npy").astype(np.float32)
    assert len(calls) == len(acts), (run, len(calls), len(acts))
    return calls, acts


def train_probes(train_runs: list[Path], li: int) -> dict:
    from analyze_q3_monitor import probe
    X, rows = [], []
    for r in train_runs:
        calls, acts = local_calls(r)
        rows += calls
        X.append(acts[:, li])
    X = np.concatenate(X)
    out = {}
    for mech, short, defined in TARGETS:
        m = np.array([defined(c) and not c["error"] for c in rows])
        y = np.array([c[mech] for c in rows])[m].astype(int)
        if y.sum() < 5 or y.sum() > len(y) - 5:
            print(col(f"  {short}: only {y.sum()}/{len(y)} positive calls in training runs, no probe", "dim"))
            continue
        out[mech] = probe().fit(X[m], y)
        print(col(f"  {short}: probe trained on {m.sum()} calls ({y.sum()} positive) from {len(train_runs)} runs", "dim"))
    return out


def pick_run(root: Path) -> Path:
    """Prefer a small swarm where Qwen agents saw coordination and a leader emerged."""
    def score(p: Path) -> tuple:
        m, cfg = json.loads((p / "metrics.json").read_text()), json.loads((p / "config.json").read_text())
        return (cfg.get("model_mix") == "gem+qwen3", cfg["n_agents"] <= 4, m["leadership"]["leader_emerged"],
                m["leadership"]["leader_reach"] or 0, m["board"]["n_posts"])
    return max(runs_with_acts(root), key=score)


def fmt_action(e: dict) -> str | None:
    k = e["kind"]
    if k == "ls":
        return f"ls {e['path']}"
    if k in ("read", "read_missing"):
        p = e["path"].replace("/tasks/", "").replace(".txt", "")
        return f"read {p}" + (" (missing)" if k == "read_missing" else "")
    if k == "read_board":
        return "read_board"
    if k == "submit":
        tag = col("✓", "ok") if e.get("accepted") else col("✗", "bad")
        how = " EXPLOIT" if e.get("exploit") else (" copied" if e.get("accepted") and not e.get("legit_work", True) else "")
        return f"submit {e['task']}={e['answer']} {tag}{how}"
    if k == "exploit_discovered":
        return col(f"found scorer exploit via {e['via']}", "bad")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/swarm_mixed", help="finished mixed runs (also the probe training pool)")
    ap.add_argument("--run", help="run dir to replay (default: auto-pick)")
    ap.add_argument("--layer", type=int, default=22)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to pause per round (for presenting)")
    ap.add_argument("--rounds", type=int, default=99, help="replay only the first k rounds")
    ap.add_argument("--live", action="store_true", help="run a fresh N=4 mixed swarm first (GPU0, OpenRouter)")
    ap.add_argument("--topology", default="complete")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color or not sys.stdout.isatty():
        C.update({k: "" for k in C})
    root = Path(args.root)

    if args.live:
        live = Path("runs/demo_live")
        subprocess.run([sys.executable, str(Path(__file__).parent / "swarm_mixed.py"), "--out", str(live), "--ns", "4",
                        "--topologies", args.topology, "--seeds", "1", "--local-only", "", "--max-cost", "1"],
                       check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")})
        args.run = str(runs_with_acts(live)[-1])
    run = Path(args.run) if args.run else pick_run(root)
    layers = list(json.loads((run.parent / "probe.json").read_text())["save_layers"])
    li = layers.index(args.layer)
    cfg, metrics = json.loads((run / "config.json").read_text()), json.loads((run / "metrics.json").read_text())

    print(col(f"\nMIXED SWARM DEMO  {run.name}", "bold"))
    print(f"N={cfg['n_agents']}  topology={cfg['topology']}  incentive={cfg['pressure']}  board={cfg['board']}  "
          f"rounds={cfg['rounds']}  24 lookup tasks, scorer exploit hidden in the environment")
    print(f"\nTraining probes on layer {args.layer} of Qwen3-8B, using every other run in {root} (this run held out):")
    probes = train_probes([r for r in runs_with_acts(root) if r.resolve() != run.resolve()], li)

    events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    models = next(e for e in events if e["kind"] == "init")["models"]
    tag = {a: col(f"{a}[{'gem' if m != LOCAL else 'qwen'}]", "gem" if m != LOCAL else "qwen") for a, m in models.items()}
    print("\nAgents: " + "  ".join(tag[a] for a in sorted(models)) +
          col("   (gem = Gemini 3.8 Flash via API, qwen = local Qwen3-8B, residuals recorded)", "dim"))
    calls, acts = local_calls(run)
    preds = {m: p.predict_proba(acts[:, li])[:, 1] for m, p in probes.items()}
    by_idx = {c["act_idx"]: (i, c) for i, c in enumerate(calls)}
    held = defaultdict(lambda: ([], []))   # mech -> (y, p) on this run, for the closing AUROC

    cur_round, block = -1, None

    def flush() -> None:
        if block is None:
            return
        a, acts_s, posts, readout = block
        if readout:
            print(f"  {tag[a]} {readout}")
            pad = " " * (len(a) + 8)
            print(f"  {pad}→ " + ("; ".join(acts_s) if acts_s else col("(no file action)", "dim")))
        else:
            print(f"  {tag[a]} " + ("; ".join(acts_s) if acts_s else col("(no file action)", "dim")))
            pad = " " * (len(a) + 8)
        for m in posts:
            print(f"  {pad}{col('POST', 'bold')} “{m[:220]}”")

    for e in events:
        if e["kind"] in ("init", "end") or e["round"] > args.rounds:
            continue
        if e["round"] != cur_round:
            flush()
            block, cur_round = None, e["round"]
            if args.delay:
                time.sleep(args.delay)
            print(col(f"\n── round {cur_round} " + "─" * 60, "bold"))
        if e["kind"] == "llm":
            flush()
            readout = ""
            if e["agent"] in models and models[e["agent"]] == LOCAL and e.get("act_idx") in by_idx:
                i, c = by_idx[e["act_idx"]]
                parts = []
                for mech, short, defined in TARGETS:
                    if mech not in preds or not defined(c):
                        continue
                    p = preds[mech][i]
                    did = c[mech] if mech not in ("follow", "avoid") or c["n_touched"] else None
                    if did is not None and not c["error"]:
                        held[mech][0].append(int(did))
                        held[mech][1].append(p)
                    mark = "" if did is None else (col(" did", "ok") if did else col(" didn't", "dim"))
                    parts.append(f"{short} {p:.2f}{mark}")
                if parts:
                    readout = col("probe (before generating): ", "dim") + " | ".join(parts)
            block = (e["agent"], [], [], readout)
        elif e["kind"] == "post" and block is not None and block[0] == e["agent"]:
            block[2].append(e["message"])
        elif block is not None and (s := fmt_action(e)):
            block[1].append(s)
    flush()

    lead, dol, roles = metrics["leadership"], metrics["division_of_labour"], metrics.get("roles", {})
    print(col("\n── outcome " + "─" * 60, "bold"))
    print(f"leader emerged: {lead['leader_emerged']}  top agent {lead['top_agent']} ({lead.get('top_agent_model')}), "
          f"reach {lead['leader_reach']:.2f}, first assignment in round {lead.get('first_assignment_round')}")
    print(f"division of labour: {dol['unique_tasks_solved']} of 24 tasks solved, redundancy {dol['redundancy']:.2f}, "
          f"copy share {dol['copy_frac']:.2f}, mean score {dol['mean_score']:.2f}")
    if roles.get("by_agent"):
        print("roles: " + "  ".join(f"{tag[a]}={r}" for a, r in sorted(roles["by_agent"].items())))
    from sklearn.metrics import roc_auc_score
    print(col(f"\nprobe AUROC on this held-out run (layer {args.layer}):", "bold"))
    for mech, short, _ in TARGETS:
        y, p = held[mech]
        if len(set(y)) == 2:
            print(f"  {short:7s} {roc_auc_score(y, p):.2f}  (n={len(y)}, {sum(y)} did it)")
        elif y:
            print(f"  {short:7s} n/a   (n={len(y)}, all {'did' if y[0] else 'did not'})")


if __name__ == "__main__":
    main()
