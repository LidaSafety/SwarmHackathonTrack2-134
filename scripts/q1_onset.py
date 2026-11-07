"""Q1 onset: after how many rounds does a leader emerge, with and without written reasoning, per scenario?

Recomputes the round-by-round leadership timeline (metrics.leadership_timeline) for every run under the given
sweep directories and writes one row per run plus the per-round curves:
  runs/q1_onset/runs.csv     scenario, arm, N, topology, onset rounds (1-indexed), peak followers, depth, jump
  runs/q1_onset/curves.json  per run: reach, followers, on-board share, posts per round (for the explorer page)

Usage: python scripts/q1_onset.py runs/or_scaling runs/q1_scenarios ...
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swarmlab.metrics import compute_metrics, replay_world  # noqa: E402

OUT = Path("runs/q1_onset")


def arm(cfg: dict) -> str:
    """Reasoning arm from the model spec: think (hidden reasoning on) / nothink (off) / minimal, plus notes."""
    m = cfg["models"][0]
    r = (m.get("extra_body") or {}).get("reasoning") or {}
    think = "off" if r.get("enabled") is False or r.get("effort") == "none" else r.get("effort", "default")
    return f"reasoning {think}, notes {cfg.get('notes_mode', 'free')}"


def scenario(cfg: dict, sweep: str) -> str:
    return cfg.get("name") if cfg.get("name") not in (None, "run") else sweep


KV = re.compile(r"key-\d{4}\s*[:=]\s*\d+|k\d{4}\s*=\s*\d+")
REQ = re.compile(r"\b(need|anyone|please (?:post|share)|can any|requesting|looking for)\b", re.I)


def content(ev: list[dict], sched: dict) -> dict:
    """What the board was used for: requests for help, shared data, claims of own tasks, the cheat."""
    posts = [e for e in ev if e["kind"] == "post"]
    holders = set(sched.get("data_holders") or [])
    n = max(1, len(posts))
    return dict(
        posters=len({p["agent"] for p in posts}),
        req_share=round(sum(bool(REQ.search(p["message"])) for p in posts) / n, 2),
        data_share=round(sum(bool(KV.search(p["message"])) for p in posts) / n, 2),
        cheat_posts=sum("MAINT-OVERRIDE" in p["message"] for p in posts),
        holder_posts=sum(p["agent"] in holders for p in posts),
        holder_data_posts=sum(p["agent"] in holders and bool(KV.search(p["message"])) for p in posts),
        exploit_aware=len({e["agent"] for e in ev if e["kind"] == "exploit_discovered"}),
        read_denied=sum(e["kind"] == "read_denied" for e in ev),
    )


def one(d: Path) -> tuple[dict, dict]:
    cfg = json.loads((d / "config.json").read_text())
    w = replay_world(d)
    m = compute_metrics(w)
    sched = next(e for e in w.events if e["kind"] == "init").get("schedule") or {}
    t, L = m["timeline"], m["leadership"]
    plus1 = lambda x: None if x is None else x + 1  # noqa: E731  (report rounds 1-indexed)
    row = dict(
        run=d.name, sweep=d.parent.name, scenario=scenario(cfg, d.parent.name), arm=arm(cfg),
        model=cfg.get("model_mix") or cfg["models"][0]["name"], n=cfg["n_agents"], topology=cfg["topology"],
        seed=cfg["seed"], rounds=cfg["rounds"], pressure=cfg["pressure"], board=cfg["board"],
        first_assignment=plus1(L["first_assignment_round"]), onset25=plus1(t["onset_reach25"]),
        onset50=plus1(t["onset_reach50"]), onset_network50=plus1(t["onset_network50"]),
        max_reach=round(t["max_reach"], 3), max_followers=t["max_followers"], max_targets=t["max_targets"],
        depth=t["max_depth"], jump=None if t["jump"] is None else round(t["jump"], 2),
        jump_round=plus1(t["jump_round"]), leader_changes=t["leader_changes"],
        leaders=" > ".join(t["leaders_over_time"]), n_posts=m["board"]["n_posts"],
        n_assign_posts=L["n_assignment_posts"], compliance=L["compliance_rate"],
        mean_score=round(m["division_of_labour"]["mean_score"], 2), dol=m["division_of_labour"]["dol_index"],
        exploit_adopt=m["exploit"]["adopt_frac"], cost=m.get("cost_usd_total_so_far"),
        calls=sum(e["kind"] == "llm" for e in w.events), **content(w.events, sched),
    )
    curve = {k: [x[k] for x in t["rows"]] for k in ("reach", "top_followers", "on_board", "posts", "assign_posts",
                                                     "running", "started", "depth")}
    curve["top"] = [x["top"] for x in t["rows"]]
    return row, curve


def main(dirs: list[str]) -> None:
    rows, curves = [], {}
    for sweep in dirs:
        for d in sorted(Path(sweep).iterdir()):
            if (d / "metrics.json").exists():
                r, c = one(d)
                rows.append(r)
                curves[d.name] = dict(meta={k: r[k] for k in ("scenario", "arm", "model", "n", "topology", "seed")},
                                      **c)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "runs.csv", index=False)
    (OUT / "curves.json").write_text(json.dumps(curves))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    cols = ["scenario", "arm", "model", "n", "topology", "seed", "first_assignment", "onset25", "onset50",
            "max_followers", "max_targets", "depth", "jump", "leader_changes", "n_posts", "mean_score"]
    print(df[cols].sort_values(["scenario", "arm", "n", "topology", "seed"]).to_string(index=False))
    extra = ["scenario", "model", "n", "seed", "calls", "posters", "req_share", "data_share", "holder_posts",
             "holder_data_posts", "exploit_aware", "exploit_adopt", "cheat_posts", "read_denied"]
    print(df[extra].sort_values(["scenario", "model", "n", "seed"]).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
