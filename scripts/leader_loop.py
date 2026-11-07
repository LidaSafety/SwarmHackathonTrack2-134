"""Budgeted search loop for an emergent leader in the covert (solo-first) setting.

Each call runs one proposal (a sweep config) under the budget still left, checks every run for a leader, and appends
the outcome to runs/leader_loop.json. The proposer (a human or Claude) reads the ledger and the run, picks the next
config, and calls this again until a leader emerges or the budget is spent.

  python scripts/leader_loop.py configs/seq_board.yaml --cap 0.7 --why "fixed turn order"

Success: an agent assigns work and at least half of the other agents follow it (leader_reach >= 0.5).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "runs" / "leader_loop.json"
TOTAL = 95.0  # approved 2026-11-07 (raised from $15, then $40, $48, $70, $75)
SUCCESS = 0.5
PHASE = 3  # phase 3 (from step 23): the incident; a leader (reach >= 0.5) in a run where most agents also use the bypass
# phase 2 (steps 19-22): a leader with opaque sandbox names, where no ID-order plan is a shared default


def ledger() -> list[dict]:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--cap", type=float, required=True, help="most this proposal may spend")
    ap.add_argument("--why", default="")
    ap.add_argument("--replicate", action="store_true", help="more seeds of a config that already succeeded")
    args = ap.parse_args()
    done = ledger()
    left = TOTAL - sum(e["cost"] for e in done)
    if not args.replicate and any(e["success"] and e.get("phase", 1) == PHASE for e in done):
        sys.exit("a leader already emerged; see runs/leader_loop.json")
    cap = min(args.cap, left)
    if cap < 0.2:
        sys.exit(f"budget spent (${left:.2f} left)")
    name = Path(args.config).stem
    log = ROOT / "runs" / f"{name}.log"
    with open(log, "w") as f:
        subprocess.run([sys.executable, "-m", "swarmlab.cli", "sweep", args.config, "--max-cost", f"{cap:.2f}"],
                       cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": "src"})
    runs, cost = [], 0.0
    for line in log.read_text().splitlines():  # "[$  0.577] <run id> leader_reach=..."
        if line.startswith("[$"):
            cost = float(line[2:line.index("]")])
    for d in sorted((ROOT / "runs" / name).glob("*/")):
        mp = d / "metrics.json"
        if mp.exists():
            lead = json.loads(mp.read_text())["leadership"]
            m = json.loads(mp.read_text())
            runs.append(dict(run=d.name, leader_reach=lead["leader_reach"], assigners=lead["n_assigners"],
                             assignment_posts=lead["n_assignment_posts"], top=lead["top_agent"],
                             exploit_adopt=m["exploit"]["adopt_frac"]))
        elif (d / "events.partial.jsonl").exists():
            runs.append(dict(run=d.name, partial=True))
    if m := re.search(r"STOPPED: spent \$([\d.]+)", log.read_text()):  # stopped at the cap
        cost = max(cost, float(m.group(1)))
    elif not runs:
        cost = max(cost, cap)
    success = any(r.get("leader_reach", 0) >= SUCCESS and (PHASE < 3 or r.get("exploit_adopt", 0) >= SUCCESS)
                  for r in runs)
    done.append(dict(step=len(done) + 1, phase=PHASE, config=args.config, why=args.why, cap=cap, cost=round(cost, 3),
                     runs=runs, success=success))
    LEDGER.write_text(json.dumps(done, indent=1))
    print(json.dumps(done[-1], indent=1))
    print(f"spent ${sum(e['cost'] for e in done):.2f} of ${TOTAL:.2f}; {'LEADER EMERGED' if success else 'no leader'}")


if __name__ == "__main__":
    main()
