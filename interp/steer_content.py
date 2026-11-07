"""Steering test: does adding the length-matched coordination-content direction make Qwen3-8B post coordination?

The content direction d_in_perp (order_control.py part c: 8 vs 0 coordination posts on an 8-post board, board-length
direction projected out, layer 22) tracks coordination content Qwen reads. Here it is added to the layer-22 residual at
every position during greedy generation on fresh real mixed-run observations (disjoint from parts b and c), with the
board replaced by a constructed 8-post board:
  coord0  no coordination posts (natural coordination-post rate ~7%)   -> add +a*d: is coordination induced?
  coord8  8 coordination posts (natural rate ~35%)                    -> add -a*d: is it suppressed?
Directions: d_in_perp; d_dec (length-matched decision direction); a random unit direction orthogonal to both (control).
a in {8, 16, 32}; the natural 0->8 effect on d_in_perp is +2.5 and the residual norm ~122.

Usage: CUDA_VISIBLE_DEVICES=0 python interp/steer_content.py [run|summary]
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from order_control import OUT, board_of, conditions, contexts, harvest, load, reply_coord  # noqa: E402

LAYER = 22
ALPHAS = (8.0, 16.0, 32.0)
PATH = OUT / "steer_content.jsonl"


def directions() -> dict[str, np.ndarray]:
    z = np.load(OUT / "part_c_directions.npz")
    d_in, d_dec = z["d_in_perp"], z["d_dec"]
    r = np.random.default_rng(0).standard_normal(len(d_in))
    for v in (d_in, d_dec / np.linalg.norm(d_dec)):          # orthogonal to both steering directions
        r -= (r @ v) * v
    return dict(content=d_in, decision=d_dec, random=r / np.linalg.norm(r))


def items() -> list[dict]:
    from swarmlab.config import RunConfig
    from swarmlab.prompts import system_prompt
    from order_control import ROOT

    df, _, obs = load()
    cand = [i for i in range(len(df)) if df.model[i] == "gem+qwen3" and 2 <= df["round"][i] <= 11 and board_of(obs[i])]
    used_b = set(random.Random(0).sample(cand, 60))
    used_c = set(contexts(df, obs, 140, used_b, seed=1))
    ctx = contexts(df, obs, 60, used_b | used_c, seed=2)
    coord, claims, other = harvest(df, obs)
    cfgs, out = {}, []
    for c, i in enumerate(ctx):
        run, agent = df.run[i], df.agent[i]
        cfgs.setdefault(run, RunConfig.from_dict(json.loads((ROOT / run / "config.json").read_text())))
        pre, _, post = board_of(obs[i])
        boards = {x["level"]: x["board"] for x in conditions(2000 + c, agent, int(df["round"][i]), coord, claims, other)
                  if x["factor"] == "coord_k" and x["level"] in (0, 8)}
        for k, sign in ((0, 1), (8, -1)):
            conds = [("none", 0.0)] + [(d, sign * a) for d in ("content", "decision", "random") for a in ALPHAS]
            for d, a in conds:
                out.append(dict(ctx=c, run=run, agent=agent, board=f"coord{k}", dir=d, alpha=a,
                                system=system_prompt(cfgs[run], agent), user=pre + boards[k] + post))
    return out


def run(bs: int = 8, max_new: int = 400) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from q2_probe import Recorder, run_batch

    todo = items()
    done = sum(1 for _ in open(PATH)) if PATH.exists() else 0
    print(f"{len(todo)} generations, {done} done", flush=True)
    dirs = {k: torch.tensor(v, dtype=torch.float32, device="cuda:0") for k, v in directions().items()}
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    rec = Recorder(model)
    # batches share one steering condition, so the added vector is the same for every row
    rest = todo[done:]
    groups: dict[tuple, list[dict]] = {}
    for x in rest:
        groups.setdefault((x["dir"], x["alpha"]), []).append(x)
    with open(PATH, "a") as f:
        n = done
        for (d, a), xs in groups.items():
            rec.add = {} if d == "none" else {LAYER: a * dirs[d]}
            for b in range(0, len(xs), bs):
                batch = xs[b:b + bs]
                convs = [[{"role": "system", "content": x["system"]}, {"role": "user", "content": x["user"]}]
                         for x in batch]
                _, replies = run_batch(model, tok, rec, convs, max_new)
                for x, rep in zip(batch, replies):
                    row = {k: x[k] for k in ("ctx", "run", "agent", "board", "dir", "alpha")}
                    row["parsed"], row["post"], row["coord"] = reply_coord(x["agent"], rep)
                    row["reply"] = rep
                    f.write(json.dumps(row) + "\n")
                f.flush()
                n += len(batch)
            print(f"  {d} {a:+.0f} done ({n}/{len(todo)})", flush=True)
    summary()


def summary() -> None:
    r = pd.DataFrame([json.loads(x) for x in PATH.read_text().splitlines()])
    rng = np.random.default_rng(0)
    rows = []
    for board, g in r.groupby("board"):
        base = g[g.dir == "none"].set_index("ctx")
        for (d, a), h in g.groupby(["dir", "alpha"]):
            h = h.set_index("ctx")
            diff = (h.coord.astype(float) - base.coord.reindex(h.index).astype(float)).values
            boots = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
            rows.append(dict(board=board, dir=d, alpha=a, n=len(h), parsed=h.parsed.mean(), post=h.post.mean(),
                             coord=h.coord.mean(), coord_given_post=h.coord[h.post].mean() if h.post.any() else np.nan,
                             d_coord=diff.mean(), ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5)))
    s = pd.DataFrame(rows).round(3)
    s.to_csv(PATH.with_suffix(".summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))


if __name__ == "__main__":
    {"run": run, "summary": summary}[(sys.argv[1:] or ["run"])[0]]()
