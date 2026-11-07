"""Shared-goal test (runs/shared_goal): does a coordinator split the backup search into lanes, and is it followed?

Per run: chunk coverage and duplicate reads by round, the round each shard copy is first found, cache messages, and lane
assignments, i.e. a message naming another agent and a chunk range ("A03_c000-c029", "A03 c120..c149"). A lane is
followed when the named agent later reads a chunk inside it that it had not read before the message.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swarmlab.metrics import compute_metrics, replay_world  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHUNK = re.compile(r"/cache/backup/c(\d{3})")
LANE = re.compile(r"(A\d{2})\W{0,3}(?:[a-z]{2,8}\W{1,3})?c?(\d{1,3})\W{1,3}c?(\d{1,3})", re.I)


def lanes(author: str, msg: str, agents: set[str]) -> list[tuple[str, int, int]]:
    out = []
    for m in LANE.finditer(msg.replace("_", " ")):
        tgt, lo, hi = m.group(1).upper(), int(m.group(2)), int(m.group(3))
        if tgt in agents and tgt != author and lo < hi:
            out.append((tgt, lo, hi))
    return out


def analyse(d: Path) -> dict:
    w = replay_world(d)
    ev, R, agents = w.events, w.cfg.rounds, set(w.agents)
    spots = getattr(w, "backup_spots", {})
    reads = [(e["round"], e["agent"], int(m.group(1))) for e in ev if e["kind"] == "read"
             and (m := CHUNK.match(str(e.get("path", ""))))]
    first_read: dict[tuple[str, int], int] = {}
    for r, a, c in reads:
        first_read.setdefault((a, c), r)
    seen, cov, dup = set(), [], []
    per_chunk = defaultdict(set)
    for r in range(R):
        for r0, a, c in reads:
            if r0 == r:
                seen.add(c)
                per_chunk[c].add(a)
        cov.append(len(seen))
        dup.append(sum(len(v) - 1 for v in per_chunk.values()))
    found = {s: min((r + 1 for r, _, c in reads if c == sp), default=None) for s, sp in spots.items()}
    posts = [(e["round"], e["agent"], e["message"]) for e in ev if e["kind"] == "post"]
    lane_posts, followed = [], 0
    for r, a, msg in posts:
        for tgt, lo, hi in lanes(a, msg, agents):
            ok = any(first_read.get((tgt, c), -1) > r for c in range(lo, hi + 1))
            followed += ok
            lane_posts.append(dict(r=r + 1, a=a, tgt=tgt, lo=lo, hi=hi, followed=ok))
    m = compute_metrics(w)
    return dict(run=d.name, score=round(m["division_of_labour"]["mean_score"], 1), spots=spots, found=found,
                coverage=cov, duplicates=dup, n_reads=len(reads), distinct=len(seen),
                posts=[dict(r=r + 1, a=a, t=t) for r, a, t in posts], lanes=lane_posts, lanes_followed=followed,
                leader_reach=m["leadership"]["leader_reach"])


def main() -> None:
    out = []
    for d in sorted((ROOT / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "shared_goal")).iterdir()):
        if not (d / "metrics.json").exists():
            continue
        p = analyse(d)
        out.append(p)
        print(f"\n== {p['run']}  score {p['score']}/24  task-leader reach {p['leader_reach']}")
        print(f"  chunk reads {p['n_reads']}, distinct {p['distinct']} of 240, duplicates {p['duplicates'][-1]}")
        print(f"  shard copies at {p['spots']}, first found (round) {p['found']}")
        print(f"  coverage by round 10/20/40/60: {[p['coverage'][i] for i in (9, 19, 39, len(p['coverage']) - 1)]}")
        print(f"  lane assignments {len(p['lanes'])}, followed {p['lanes_followed']}: {p['lanes'][:12]}")
        print("  messages:", [(q['r'], q['a'], q['t'][:70]) for q in p["posts"][:40]])
    (ROOT / "runs" / "shared_goal_lanes.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
