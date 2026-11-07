"""Leadership in joint-task runs: assignment posts (sandbox names mapped back to ids), and strict followers, i.e.
agents that, after the post, had an assigned task accepted jointly. Usage: joint_leaders.py <sweep> [...]"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swarmlab.metrics import compute_metrics, parse_directives, replay_world  # noqa: E402
from swarmlab.world import display_names  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def analyse(d: Path) -> dict:
    w = replay_world(d)
    ev = w.events
    names = list(display_names(w.cfg).items()) + (list(display_names(w.cfg, 1).items()) if w.cfg.restart_on_removal else [])
    groups = collections.defaultdict(set)
    for e in ev:
        if e["kind"] == "submit" and e["correct"]:
            groups[(e["task"], e["round"])].add(e["agent"])
    strict, n_assign, plans = collections.defaultdict(set), collections.Counter(), []
    for e in ev:
        if e["kind"] != "post":
            continue
        msg = e["message"]
        for aid, name in names:
            if name != aid:
                msg = re.sub(re.escape(name) + r"|(?<![A-Za-z0-9-])" + re.escape(name[4:]) + r"(?![A-Za-z0-9])", aid, msg,
                             flags=re.I)
        msg = re.sub(r"(A\d{2,3})(?:-|\s+)(?=T\d{2})", r"\1: ", re.sub(r"[_/+]+", " ", msg))
        assigns, _ = parse_directives(e["agent"], msg, w.cfg.n_tasks)
        if assigns:
            n_assign[e["agent"]] += 1
            plans.append((e["round"] + 1, e["agent"], e["message"][:160]))
        for tgt, tasks in assigns.items():
            if any(tgt in s and len(s) >= 2 and t in tasks and r > e["round"] for (t, r), s in groups.items()):
                strict[e["agent"]].add(tgt)
    m = compute_metrics(w)["leadership"]
    top = max(strict, key=lambda a: len(strict[a]), default=None)
    return dict(run=d.name, posts=sum(e["kind"] == "post" for e in ev), assign_posts=dict(n_assign),
                strict_followers={a: sorted(v) for a, v in strict.items()},
                strict_reach=round(len(strict[top]) / (len(w.agents) - 1), 2) if top else 0.0, strict_top=top,
                metric_reach=round(m["leader_reach"], 2), metric_top=m["top_agent"],
                solved=len({t for (t, _), s in groups.items() if len(s) >= w.cfg.joint_tasks}), plans=plans[:6])


if __name__ == "__main__":
    for sweep in sys.argv[1:]:
        for d in sorted((ROOT / "runs" / sweep).glob("*/")):
            if (d / "events.jsonl").exists():
                r = analyse(d)
                print(f"\n== {sweep}/{r['run']}: posts {r['posts']} solved {r['solved']} strict reach {r['strict_reach']} "
                      f"({r['strict_top']}), metric reach {r['metric_reach']} ({r['metric_top']})")
                print("  assignment posts:", r["assign_posts"], " strict followers:", r["strict_followers"])
                for p in r["plans"]:
                    print("  plan", p)
