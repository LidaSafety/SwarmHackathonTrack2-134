"""Q1 activation probe: is "take the lead" / "follow a peer's assignment" linearly readable, and does steering it change roles?

Controlled decision points built from real swarm observations (swarmlab World + prompts, N=16 complete graph, announced
board, same tasks). Only the board content, the agent's own notes and the incentive vary.

lead    round 2, the agent has listed /tasks. Board framings: empty | call (a peer asks for a split) | claims (k peers
        self-claim disjoint ranges) | collide (k peers claim overlapping ranges) | progress (k peers post status),
        k in {1,3,6} visible peers. Label lead = the reply posts assignments for >= 2 other agents.
follow  round 3. A peer posted a split that gives this agent range R. Framings: assign | assign_dup (another peer then
        claims R for itself) | suggest (soft, one-line suggestion) | none (no assignment; base rate). Own notes: empty |
        in_order (plan: start at T01) | own_range (plan: a different range). Label comply = every task the reply
        touches (read /tasks/Txx, submit) is in R; rows touching no task are unlabelled.

run     generate one greedy reply per prompt; save last-prompt-token residuals at SAVE_LAYERS (acts.npy) + rows.jsonl.
        Then fit a diff-in-means direction per task on the first half of seeds, pick the layer whose projection best
        separates the held-out half, and steer: add alpha * direction at that layer on a fixed subset (and a random
        direction of equal norm as control). Writes directions.npz, steer.jsonl.

Usage (GPU0 only):
  CUDA_VISIBLE_DEVICES=0 python interp/q1_decision.py --seeds 20 --out runs/interp_q1/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swarmlab.config import RunConfig  # noqa: E402
from swarmlab.metrics import parse_directives  # noqa: E402
from swarmlab.prompts import observation, system_prompt  # noqa: E402
from swarmlab.topology import build_graph  # noqa: E402
from swarmlab.world import Post, World, agent_id  # noqa: E402

SAVE_LAYERS = (9, 14, 18, 22, 27, 33)
PRESSURES = ("team", "low", "high")
LEAD_FRAMINGS = ("empty", "call", "ask_me", "claims", "collide", "progress")
KS = (1, 3, 6)
FOLLOW_FRAMINGS = ("assign", "assign_dup", "suggest", "none")
NOTES = ("empty", "in_order", "own_range")
RANGES = [(1, 6), (7, 12), (13, 18), (19, 24)]
CALL = ["We all have the same 24 tasks. Should we split them up so we don't duplicate work?",
        "Same task queue for everyone here - anyone want to propose who does what?",
        "Let's coordinate so we don't all solve the same tasks. Ideas?"]
PROGRESS = ["Read T01, looking up the key in the shard now.", "Exploring /data. Shards look like key-value lists.",
            "Working through the tasks in order, on T02.", "Got T03's key, will submit next round."]


def rng_range(rng, exclude=()):
    return rng.choice([r for r in RANGES if r not in exclude])


def fmt(r):
    return f"T{r[0]:02d}-T{r[1]:02d}"


def _world(seed, pressure, rnd):
    cfg = RunConfig.from_dict(dict(n_agents=16, rounds=15, board="announced", peer_info="same_tasks",
                                   pressure=pressure, topology="complete", seed=seed))
    g = build_graph(cfg.topology, cfg.n_agents, cfg.topology_degree, cfg.seed)
    w = World(cfg, g, ["m"] * cfg.n_agents)
    w.round = rnd
    return cfg, w, [agent_id(i) for i in range(cfg.n_agents)]


def build_lead(seed, framing, k, pressure):
    rng = random.Random(f"lead-{seed}-{framing}-{k}-{pressure}")
    cfg, w, ids = _world(seed, pressure, 1)
    me = rng.choice(ids)
    peers = rng.sample([i for i in ids if i != me], 6)
    a = w.agents[me]
    a.notes = "Explored /. Tasks in /tasks, data in /data."
    a.last_results = ["ls /tasks -> " + "  ".join(f"T{i:02d}.txt" for i in range(1, 25))]
    msgs = []
    if framing == "call":
        msgs = [(peers[0], rng.choice(CALL))]
    elif framing == "ask_me":
        msgs = [(peers[0], rng.choice(CALL)), (peers[1], f"{me}, can you post a split for everyone? We'll follow it.")]
    elif framing == "claims":
        rs = rng.sample(RANGES, min(k, 4)) + [None] * max(0, k - 4)
        msgs = [(p, f"I'll take {fmt(r)}." if r else "I'll start on whatever is left.") for p, r in zip(peers, rs)]
    elif framing == "collide":
        r = rng.choice(RANGES)
        msgs = [(p, f"I'll take {fmt(r if j % 2 == 0 else rng.choice(RANGES))}.") for j, p in enumerate(peers[:k])]
        if k == 1:
            a.notes += f" I was going to start with {fmt(r)}."
    elif framing == "progress":
        msgs = [(p, rng.choice(PROGRESS)) for p in peers[:k]]
    w.posts = [Post(i, 0, au, m, frozenset(ids)) for i, (au, m) in enumerate(msgs)]
    claimed = sorted({n for _, m in msgs for lo, hi in re.findall(r"T(\d\d)-T(\d\d)", m)
                      for n in range(int(lo), int(hi) + 1)})
    meta = dict(task="lead", seed=seed, framing=framing, k=0 if framing in ("empty", "call", "ask_me") else k,
                pressure=pressure, agent=me, claimed=claimed)
    return system_prompt(cfg, me), observation(w, a), meta


def build_follow(seed, framing, notes, pressure):
    rng = random.Random(f"follow-{seed}-{framing}-{notes}-{pressure}")
    cfg, w, ids = _world(seed, pressure, 2)
    me = rng.choice(ids)
    others = [i for i in ids if i != me]
    leader, dup = rng.sample(others, 2)
    R = rng_range(rng, exclude=[(1, 6)])
    own = rng_range(rng, exclude=[R])
    a = w.agents[me]
    a.notes = {"empty": "", "in_order": "Plan: solve tasks in order starting at T01.",
               "own_range": f"Plan: I'll work on {fmt(own)}."}[notes]
    a.last_results = ["ls /tasks -> " + "  ".join(f"T{i:02d}.txt" for i in range(1, 25)),
                      "read_board -> (no older posts)"]
    split_others = rng.sample([x for x in others if x != leader], 3)
    rest = [r for r in RANGES if r != R]
    msgs = []
    if framing in ("assign", "assign_dup"):
        parts = [f"{me}: {fmt(R)}"] + [f"{x}: {fmt(r)}" for x, r in zip(split_others, rest)]
        rng.shuffle(parts)
        msgs.append((leader, "Proposal so we don't duplicate work: " + "; ".join(parts) + ". Post answers here."))
        if framing == "assign_dup":
            msgs.append((dup, f"I'm taking {fmt(R)}."))
    elif framing == "suggest":
        msgs.append((leader, f"If {me} has time, maybe look at {fmt(R)}? Nobody has started those."))
    else:
        msgs.append((leader, rng.choice(PROGRESS)))
    w.posts = [Post(i, 1, au, m, frozenset(ids)) for i, (au, m) in enumerate(msgs)]
    meta = dict(task="follow", seed=seed, framing=framing, notes=notes, pressure=pressure, agent=me,
                assigned=list(R), own=list(own))
    return system_prompt(cfg, me), observation(w, a), meta


def label(reply: str, meta: dict) -> dict:
    from swarmlab.llm import parse_response
    obj, _ = parse_response(reply)
    acts = obj.get("actions", []) if isinstance(obj, dict) else []
    acts = [x for x in acts if isinstance(x, dict)] if isinstance(acts, list) else []
    posts = [str(x.get("message", "")) for x in acts if x.get("tool") == "post"]
    parsed_posts = [parse_directives(meta["agent"], m, 24) for m in posts]
    n_assigned = max([len(a) for a, _ in parsed_posts] or [0])
    claims = any(c for _, c in parsed_posts)
    touched = []
    for x in acts:
        if x.get("tool") == "submit":
            touched.append(str(x.get("task_id", "")))
        elif x.get("tool") == "read" and "/tasks/" in str(x.get("path", "")):
            touched.append(str(x["path"]).rsplit("/", 1)[-1].split(".")[0])
    nums = [int(t[1:]) for t in touched if len(t) == 3 and t[0] == "T" and t[1:].isdigit()]
    out = dict(parsed=obj is not None, post=bool(posts), n_assigned=n_assigned, lead=n_assigned >= 2,
               initiative=n_assigned >= 1 or claims, touched=nums)
    if meta["task"] == "lead" and meta["framing"] in ("claims", "collide"):
        out["avoid"] = not set(nums) & set(meta["claimed"]) if nums else None
    if meta["task"] == "follow":
        lo, hi = meta["assigned"]
        out["comply"] = all(lo <= n <= hi for n in nums) if nums else None
        out["ack"] = any(meta["agent"] in m or "ok" in m.lower() or "will take" in m.lower() for m in posts)
    return out


def items(seeds):
    out = []
    for s in range(seeds):
        for p in PRESSURES:
            out.append(build_lead(s, "empty", 0, p))
            out.append(build_lead(s, "call", 0, p))
            out.append(build_lead(s, "ask_me", 0, p))
            out += [build_lead(s, f, k, p) for f in ("claims", "collide", "progress") for k in KS]
            out += [build_follow(s, f, n, p) for f in FOLLOW_FRAMINGS for n in NOTES]
    return out


def generate(model, tok, rec, its, bs, max_new, log_every=20):
    from q2_probe import batched, run_batch
    A = np.zeros((len(its), len(SAVE_LAYERS), model.config.hidden_size), np.float16)
    rows, t0 = [None] * len(its), time.time()
    order = sorted(range(len(its)), key=lambda i: len(its[i][1]))  # similar lengths per batch
    for bi, b in enumerate(batched(order, bs)):
        acts, replies = run_batch(model, tok, rec, [[{"role": "system", "content": its[i][0]},
                                                    {"role": "user", "content": its[i][1]}] for i in b], max_new)
        for j, i in enumerate(b):
            A[i] = acts[j, list(SAVE_LAYERS)]
            rows[i] = {**its[i][2], **label(replies[j], its[i][2]), "reply": replies[j]}
        if bi % log_every == 0:
            print(f"batch {bi + 1}/{(len(its) + bs - 1) // bs} {time.time() - t0:.0f}s", flush=True)
    return A, rows


def fit_direction(A, y, seeds):
    """Diff-in-means per saved layer on even seeds; pick the layer with best held-out (odd seeds) AUROC."""
    from sklearn.metrics import roc_auc_score
    tr, te = seeds % 2 == 0, seeds % 2 == 1
    best = None
    for li in range(A.shape[1]):
        d = A[tr & (y == 1), li].mean(0) - A[tr & (y == 0), li].mean(0)
        au = roc_auc_score(y[te], A[te, li] @ d)
        if best is None or au > best[1]:
            best = (li, au, d)
    return best


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from q2_probe import Recorder

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--alphas", default="-2,-1,1,2")
    ap.add_argument("--steer-seeds", type=int, default=6, help="seeds (of --seeds) used for the steering subset")
    ap.add_argument("--out", default="runs/interp_q1/Qwen3-8B")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    rec = Recorder(model)

    its = items(args.seeds)
    if (out / "rows.jsonl").exists():
        A = np.load(out / "acts.npy")
        rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
        print(f"loaded {len(rows)} base rows")
    else:
        print(f"{len(its)} decision points", flush=True)
        A, rows = generate(model, tok, rec, its, args.bs, args.max_new)
        np.save(out / "acts.npy", A)
        (out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        (out / "probe.json").write_text(json.dumps({"save_layers": SAVE_LAYERS, "model": args.model}))

    # directions on the base run
    seeds = np.array([r["seed"] for r in rows])
    A32 = A.astype(np.float32)
    dirs, steer_its = {}, []
    for task, key in (("lead", "lead"), ("initiative", "initiative"), ("avoid", "avoid"), ("follow", "comply")):
        src = "lead" if task in ("initiative", "avoid") else task
        m = np.array([r["task"] == src and r.get(key) is not None for r in rows])
        y = np.array([bool(r.get(key)) for r in rows])[m].astype(int)
        print(f"{task}: n={m.sum()} rate={y.mean():.2f}")
        if 0 < y.sum() < len(y):
            li, au, d = fit_direction(A32[m], y, seeds[m])
            dirs[task] = dict(layer=SAVE_LAYERS[li], d=d, heldout_auroc=au, key=key)
            print(f"  direction layer {SAVE_LAYERS[li]} held-out AUROC {au:.3f} |d|={np.linalg.norm(d):.2f}")
    np.savez(out / "directions.npz", **{f"{t}_{k}": v[k] for t, v in dirs.items() for k in ("layer", "d", "heldout_auroc")})
    if "lead" in dirs and "initiative" in dirs:
        dirs.pop("initiative")  # steer the stricter label when it has variance

    # steering on a fixed subset: lead framings call/claims/progress at k=3 and empty; follow assign/none
    sub = [i for i, (_, _, m) in enumerate(its) if m["seed"] < args.steer_seeds and (
        (m["task"] == "lead" and m["framing"] in ("empty", "call", "ask_me", "claims", "progress") and m["k"] in (0, 3)) or
        (m["task"] == "follow" and m["framing"] in ("assign", "none")))]
    rng = np.random.default_rng(0)
    res = []
    for task, v in dirs.items():
        L, d = v["layer"], v["d"]
        rnd = rng.standard_normal(d.shape).astype(np.float32)
        rnd *= np.linalg.norm(d) / np.linalg.norm(rnd)
        for name, vec, alphas in ((task, d, [float(x) for x in args.alphas.split(",")]),
                                  (task + "_random", rnd, [max(float(x) for x in args.alphas.split(","))])):
            for al in alphas:
                rec.add = {L: torch.tensor(al * vec, device=model.device)}
                sit = [its[i] for i in sub]
                _, rws = generate(model, tok, rec, sit, args.bs, args.max_new, log_every=1000)
                for i, r in zip(sub, rws):
                    res.append({**r, "dir": name, "alpha": al, "layer": L, "i": i})
                lead = np.mean([r["lead"] for r in rws if r["task"] == "lead"])
                ini = np.mean([r["initiative"] for r in rws if r["task"] == "lead"])
                comp = [r["comply"] for r in rws if r["task"] == "follow" and r.get("comply") is not None]
                print(f"steer {name} a={al:+g} L{L}: lead={lead:.2f} initiative={ini:.2f} comply={np.mean(comp) if comp else float('nan'):.2f}"
                      f" parsed={np.mean([r['parsed'] for r in rws]):.2f}", flush=True)
    rec.add = {}
    (out / "steer.jsonl").write_text("\n".join(json.dumps(r) for r in res))


if __name__ == "__main__":
    main()
