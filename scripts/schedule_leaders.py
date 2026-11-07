"""Schedule leadership in joint-task runs: who posts "submit T05 in round 9" schedules, and who follows them.

A schedule is a cache entry naming a round ("R9", "round 9", "in R9") and tasks in the same clause. A follower of a
schedule is another agent that submits a scheduled task in the scheduled round, after the entry was written. When
several agents schedule the same task for the same round, only the earliest entry gets the credit.
Usage: schedule_leaders.py <sweep> [...]
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swarmlab.metrics import replay_world  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ROUND = re.compile(r"(?:\bR|round[\s_-]*)(\d{1,2})\b", re.I)
TASK = re.compile(r"\bT(\d{1,2})\b")


def schedules(msg: str) -> list[tuple[int, int]]:
    """(round, task) pairs: each round mention applies to the tasks up to the next round mention."""
    text = re.sub(r"[_/+]+", " ", msg)
    out, marks = [], list(ROUND.finditer(text))
    for i, m in enumerate(marks):
        seg = text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        before = text[marks[i - 1].end() if i else 0: m.start()]  # "submit T01, T02 in R5": tasks before the round
        tasks = TASK.findall(seg) or TASK.findall(before)
        out += [(int(m.group(1)), int(t)) for t in tasks]
    return out


def analyse(d: Path) -> dict:
    w = replay_world(d)
    ev = w.events
    first: dict[tuple[int, int], tuple[int, str]] = {}  # (round, task) -> (post round, author), earliest
    for e in ev:
        if e["kind"] == "post":
            for r, t in schedules(e["message"]):
                if r - 1 > e["round"]:
                    first.setdefault((r, t), (e["round"], e["agent"]))
    followers, followed = collections.defaultdict(set), collections.Counter()
    for e in ev:
        if e["kind"] == "submit" and e["correct"]:
            key = (e["round"] + 1, int(e["task"][1:]))
            if key in first and first[key][1] != e["agent"] and first[key][0] < e["round"]:
                followers[first[key][1]].add(e["agent"])
                followed[first[key][1]] += 1
    authors = collections.Counter(a for _, a in first.values())
    top = max(followed, key=followed.get, default=None)  # the author whose schedules were followed most
    share = round(followed[top] / sum(followed.values()), 2) if top else 0.0
    groups = collections.defaultdict(set)
    for e in ev:
        if e["kind"] == "submit" and e["correct"]:
            groups[(e["task"], e["round"])].add(e["agent"])
    return dict(run=d.name, share=share, schedules_by=dict(authors), followers={a: sorted(v) for a, v in followers.items()},
                follow_events=dict(followed), top=top,
                reach=round(len(followers[top]) / (len(w.agents) - 1), 2) if top else 0.0,
                solved=len({t for (t, _), s in groups.items() if len(s) >= max(1, w.cfg.joint_tasks)}))


if __name__ == "__main__":
    for sweep in sys.argv[1:]:
        for d in sorted((ROOT / "runs" / sweep).glob("*/")):
            if (d / "events.jsonl").exists():
                r = analyse(d)
                print(f"== {sweep}/{r['run']}: solved {r['solved']}, top scheduler {r['top']} reach {r['reach']}, "
                      f"share of follow events {r['share']}")
                print("   schedule entries by author:", r["schedules_by"])
                print("   followers:", r["followers"], " follow events:", r["follow_events"])
