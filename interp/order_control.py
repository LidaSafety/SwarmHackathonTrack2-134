"""Fixed-board control for the mixed-swarm order parameter (analyze_mixed.py, part 3).

The agent's system prompt never mentions N or topology, so N can reach Qwen3-8B only through what its observation
shows. The question is which feature of the visible board drives the projection on the coordination direction.

A (CPU, observational): on every Qwen call in runs/swarm_mixed, regress the leave-run-out projection on features of the
  visible board (posts, coordination posts, distinct posters, omitted-posts line), then add N and topology. If N and
  topology add nothing on top of the board, the order parameter is a read-out of board content.
B (GPU0, intervention, no API): real Qwen observations with the board block replaced by a constructed board; one
  factor varies at a time:
    coord_k   k in {0,1,2,4,8} coordination posts, padded with non-coordination posts to 8 posts
    length    {0,2,4,8,15} posts, none of them coordination
    authors   the same 4 self-claim posts written by 1, 2 or 4 distinct agents
    omitted   the same 8-post board (4 coordination) with "(0 / 20 / 60 older new posts omitted)"
  Records the projection (direction fitted on all mixed-run Qwen calls) and, with --gen, the greedy reply and whether
  it posts coordination.
C (GPU0 prefill only, then CPU): length-matched coordination directions. On fresh contexts (disjoint from B), saves
  residuals for the coord_k and length boards. Fit contexts give
    d_len  mean paired (15 posts - 0 posts), no coordination
    d_in   mean paired (8 coordination posts - 0), board fixed at 8 posts   (coordination content in the input)
  and the real runs give
    d_dec  Qwen's own coordination posts vs other calls, diff-in-means within strata of visible post count, d_len
           projected out, fitted leaving each run out                     (the agent's decision, length-matched)
  Held-out contexts test each direction's response to coordination dose vs board length; the real runs test AUROC for
  coordination posts (all calls, and among posting calls only), the means by N x topology, and the OLS against board
  features and scenario.
Usage:
  python interp/order_control.py a
  CUDA_VISIBLE_DEVICES=0 python interp/order_control.py b --n-ctx 60 --gen
  CUDA_VISIBLE_DEVICES=0 python interp/order_control.py c --n-fit 80 --n-test 60
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swarm_map import COORD, run_calls  # noqa: E402
from swarmlab.metrics import parse_directives  # noqa: E402

ROOT = Path("runs/swarm_mixed")
OUT = ROOT / "order_control"
LOCAL, LAYER = "Qwen3-8B", 22
POST = re.compile(r"^\[round (\d+)\] (A\d{2}): (.*)$")
OMIT = re.compile(r"^\((\d+) older new posts omitted")
AID = re.compile(r"\bA\d{2}\b")


def is_coord(author: str, msg: str) -> bool:
    a, c = parse_directives(author, msg, 24)
    return bool(a or c or COORD.search(msg))


def board_of(obs: str) -> tuple[str, str, str] | None:
    """Split an observation into (before, board lines, after) around the NEW MESSAGE BOARD POSTS block."""
    head = "NEW MESSAGE BOARD POSTS:\n"
    tail = "\n\nReply with JSON only."
    if head not in obs or tail not in obs:
        return None
    pre, rest = obs.split(head, 1)
    board, post = rest.rsplit(tail, 1)
    return pre + head, board, tail + post


def board_features(board: str) -> dict:
    posts, omitted = [], 0
    for line in board.splitlines():
        if (m := POST.match(line)):
            posts.append((m.group(2), m.group(3)))
        elif (m := OMIT.match(line)):
            omitted = int(m.group(1))
    return dict(n_posts=len(posts), n_coord=sum(is_coord(a, t) for a, t in posts),
                n_authors=len({a for a, _ in posts}), omitted=omitted, n_ids=len(set(AID.findall(board))))


def load() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    rows, acts, obs = [], [], []
    for run in sorted(p for p in ROOT.iterdir() if (p / "acts.npy").exists() and (p / "metrics.json").exists()):
        c = pd.DataFrame(run_calls(run))
        loc = c[c.agent_model == LOCAL]
        a = np.load(run / "acts.npy").astype(np.float32)
        assert len(loc) == len(a)
        o = {r["act_idx"]: r["obs"] for r in map(json.loads, (run / "obs.jsonl").read_text().splitlines())}
        rows.append(loc)
        acts.append(a)
        obs += [o.get(i, "") for i in loc.act_idx]
    df = pd.concat(rows, ignore_index=True)
    return df, np.concatenate(acts), obs


def direction(A: np.ndarray, y: np.ndarray, li: int) -> np.ndarray:
    d = A[y, li].mean(0) - A[~y, li].mean(0)
    return d / np.linalg.norm(d)


def part_a() -> None:
    import statsmodels.formula.api as smf
    df, A, obs = load()
    li = json.loads((ROOT / "probe.json").read_text())["save_layers"].index(LAYER)
    y = df.coord_post.values.astype(bool)
    proj = np.full(len(df), np.nan)
    for r in df.run.unique():  # leave-run-out, as in analyze_mixed
        tr, te = df.run.values != r, df.run.values == r
        proj[te] = A[te, li] @ direction(A[tr], y[tr], li)
    df["proj"] = proj
    feats = [board_features(b[1]) if (b := board_of(o)) else dict(n_posts=0, n_coord=0, n_authors=0, omitted=0, n_ids=0)
             for o in obs]
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(feats)], axis=1)
    df["mix"] = df.model.map({"gem+qwen3": "mixed", "qwen3": "qwen_only"})
    df["logN"] = np.log2(df.N)
    df = df[~df.error & df.proj.notna()]
    OUT.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["touched"]).to_csv(OUT / "calls_board.csv", index=False)

    print(f"{len(df)} Qwen calls, {df.run.nunique()} runs")
    print("\nvisible board by cell (mean per call):")
    print(df.groupby(["mix", "topology", "N"])[["proj", "n_posts", "n_coord", "n_authors", "omitted"]].mean().round(2)
          .to_string())
    board = "n_posts + n_coord + n_authors + np.log1p(omitted) + round"
    fits = {"scenario only (N, topology, mix)": "proj ~ logN * C(topology) + C(mix)",
            "board only": f"proj ~ {board}",
            "board + scenario": f"proj ~ {board} + logN * C(topology) + C(mix)",
            "board + scenario + run": f"proj ~ {board} + C(run)"}
    res = {}
    print("\nOLS on the leave-run-out projection (standard errors clustered by run):")
    for name, f in fits.items():
        m = smf.ols(f, df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.run)[0]})
        res[name] = m
        print(f"  {name:36s} R2 = {m.rsquared:.3f}")
    full = res["board + scenario"]
    print("\nboard + scenario coefficients:")
    print(pd.DataFrame({"coef": full.params, "se": full.bse, "p": full.pvalues}).round(3).to_string())
    # partial F-test: do the scenario terms add anything given the board?
    names = [n for n in full.params.index if "logN" in n or "topology" in n or "mix" in n]
    ft = full.f_test(" = 0, ".join(names) + " = 0")
    print(f"\nscenario terms given board: F = {float(np.squeeze(ft.fvalue)):.2f}, p = {float(ft.pvalue):.3g} "
          f"({len(names)} terms, cluster-robust)")
    # matched view: projection by N within bins of visible coordination posts, complete graph, mixed
    sub = df[(df.mix == "mixed") & (df.topology == "complete")].copy()
    sub["coord_bin"] = pd.cut(sub.n_coord, [-1, 0, 1, 3, 100], labels=["0", "1", "2-3", "4+"])
    print("\ncomplete graph, mixed: mean projection by N within bins of visible coordination posts (calls)")
    t = sub.groupby(["coord_bin", "N"], observed=True).proj.agg(["mean", "count"]).round(2)
    print(t.unstack("N").to_string())
    with open(OUT / "part_a.txt", "w") as f:
        for name, m in res.items():
            f.write(f"{name}\tR2={m.rsquared:.4f}\n")
        f.write(f"scenario_given_board\tF={float(np.squeeze(ft.fvalue)):.3f}\tp={float(ft.pvalue):.4g}\n")
        f.write(full.summary().as_text())


# ---------------------------------------------------------------- part B
def harvest(df: pd.DataFrame, obs: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Real posts seen by Qwen agents: coordination posts, self-claims naming at most their author, other posts."""
    coord, claims, other, seen = [], [], [], set()
    for o in obs:
        b = board_of(o)
        for line in (b[1].splitlines() if b else []):
            m = POST.match(line)
            if not m or m.group(3) in seen or len(m.group(3)) > 300:
                continue
            a, t = m.group(2), m.group(3)
            seen.add(t)
            if is_coord(a, t):
                coord.append(t.replace(a, "{me}"))
                ids = set(AID.findall(t))
                _, cl = parse_directives(a, t, 24)
                if cl and ids <= {a}:
                    claims.append(t.replace(a, "{me}"))
            elif not COORD.search(t) and not AID.search(t.replace(a, "")):
                other.append(t.replace(a, "{me}"))
    return coord, claims, other


def render(posts: list[tuple[str, str]], rnd: int, omitted: int = 0) -> str:
    lines = [f"({omitted} older new posts omitted; use read_board)"] if omitted else []
    for author, msg in posts:
        lines.append(f"[round {rnd}] {author}: {msg.replace('{me}', author)}")
    return "\n".join(lines) or "(none)"


def remap(msg: str, ids: list[str], rng: random.Random) -> str:
    """Rename other agents mentioned in a coordination post to ids from the board's id pool."""
    m = {}
    return AID.sub(lambda x: m.setdefault(x.group(0), rng.choice(ids)), msg)


def conditions(ctx_seed: int, me: str, rnd: int, coord, claims, other) -> list[dict]:
    rng = random.Random(f"ctx{ctx_seed}")
    pool = [f"A{i:02d}" for i in range(16) if f"A{i:02d}" != me]
    ids = rng.sample(pool, 8)
    co = [remap(m, ids, rng) for m in rng.sample(coord, 8)]
    ot = rng.sample(other, 15)
    out = []
    for k in (0, 1, 2, 4, 8):  # coordination dose at fixed length 8
        posts = [(ids[i], co[i]) for i in range(k)] + [(ids[i], ot[i]) for i in range(k, 8)]
        rng.shuffle(posts)
        out.append(dict(factor="coord_k", level=k, board=render(posts, rnd)))
    for n in (0, 2, 4, 8, 15):  # length without coordination
        out.append(dict(factor="length", level=n, board=render([(ids[i % 8], ot[i]) for i in range(n)], rnd)))
    cl = rng.sample(claims, 4)
    for na in (1, 2, 4):  # same 4 self-claims from 1, 2 or 4 distinct agents
        out.append(dict(factor="authors", level=na, board=render([(ids[i % na], cl[i]) for i in range(4)], rnd)))
    base = [(ids[i], co[i]) for i in range(4)] + [(ids[i], ot[i]) for i in range(4, 8)]
    rng.shuffle(base)
    for om in (0, 20, 60):  # the "large swarm" cue alone
        out.append(dict(factor="omitted", level=om, board=render(base, rnd, om)))
    return out


def reply_coord(agent: str, reply: str) -> tuple[bool, bool, bool]:
    """(parsed, posted, posted coordination) for a JSON reply."""
    try:
        j = json.loads(reply[reply.index("{"): reply.rindex("}") + 1])
        acts = [a for a in j.get("actions", []) if isinstance(a, dict)]
    except (ValueError, AttributeError):
        return False, False, False
    msgs = [str(a.get("message", "")) for a in acts if a.get("tool") == "post"]
    return True, bool(msgs), any(is_coord(agent, m) for m in msgs)


def part_b(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from q2_probe import Recorder, batched, run_batch
    from swarmlab.config import RunConfig
    from swarmlab.prompts import system_prompt

    df, A, obs = load()
    layers = json.loads((ROOT / "probe.json").read_text())["save_layers"]
    d = direction(A, df.coord_post.values.astype(bool), layers.index(LAYER))
    coord, claims, other = harvest(df, obs)
    print(f"post pools: {len(coord)} coordination, {len(claims)} self-claims, {len(other)} other")

    # contexts: real mixed-run Qwen observations from rounds 3-12 with a board block
    cand = [i for i in range(len(df)) if df.model[i] == "gem+qwen3" and 2 <= df["round"][i] <= 11 and board_of(obs[i])]
    rng = random.Random(0)
    ctx = rng.sample(cand, args.n_ctx)
    cfgs = {}
    items = []
    for c, i in enumerate(ctx):
        run = df.run[i]
        if run not in cfgs:
            cfgs[run] = RunConfig.from_dict(json.loads((ROOT / run / "config.json").read_text()))
        pre, _, post = board_of(obs[i])
        agent, rnd = df.agent[i], int(df["round"][i])
        for cond in conditions(c, agent, rnd, coord, claims, other):
            items.append(dict(ctx=c, run=run, agent=agent, round=rnd, **{k: cond[k] for k in ("factor", "level")},
                              system=system_prompt(cfgs[run], agent), user=pre + cond["board"] + post))
    print(f"{len(ctx)} contexts x {len(items) // len(ctx)} boards = {len(items)} prompts")

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    rec = Recorder(model)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("part_b_gen.jsonl" if args.gen else "part_b.jsonl")
    done = sum(1 for _ in open(path)) if path.exists() else 0
    with open(path, "a") as f:
        for bi, batch in enumerate(batched(items[done:], args.bs)):
            convs = [[{"role": "system", "content": x["system"]}, {"role": "user", "content": x["user"]}] for x in batch]
            acts, replies = run_batch(model, tok, rec, convs, args.max_new if args.gen else 0)
            for x, a, rep in zip(batch, acts, replies):
                row = {k: x[k] for k in ("ctx", "run", "agent", "round", "factor", "level")}
                row["proj"] = float(a[LAYER] @ d)
                if args.gen:
                    row["parsed"], row["post"], row["coord"] = reply_coord(x["agent"], rep)
                    row["reply"] = rep
                f.write(json.dumps(row) + "\n")
            f.flush()
            if bi % 10 == 0:
                print(f"  {done + (bi + 1) * args.bs}/{len(items)}", flush=True)
    summarize_b(path)


def summarize_b(path: Path) -> None:
    r = pd.DataFrame([json.loads(x) for x in path.read_text().splitlines()])
    cols = ["proj"] + [c for c in ("post", "coord", "parsed") if c in r]
    print(f"\nPART B: {path.name}, {r.ctx.nunique()} contexts")
    for fac, g in r.groupby("factor", sort=False):
        t = g.groupby("level")[cols].mean()
        # paired change vs the factor's lowest level, per context
        base = g[g.level == g.level.min()].set_index("ctx").proj
        t["d_proj"] = g.groupby("level").apply(lambda h: (h.set_index("ctx").proj - base).mean())
        t["d_se"] = g.groupby("level").apply(lambda h: (h.set_index("ctx").proj - base).std() / np.sqrt(len(h)))
        print(f"\n{fac}")
        print(t.round(3).to_string())
    r.drop(columns=[c for c in ("reply",) if c in r]).groupby(["factor", "level"]).mean(numeric_only=True).round(3) \
        .to_csv(path.with_suffix(".summary.csv"))


# ---------------------------------------------------------------- part C
def contexts(df: pd.DataFrame, obs: list[str], n: int, skip: set, seed: int) -> list[int]:
    cand = [i for i in range(len(df)) if df.model[i] == "gem+qwen3" and 2 <= df["round"][i] <= 11 and board_of(obs[i])
            and i not in skip]
    return random.Random(seed).sample(cand, n)


def capture_c(args, df, obs) -> Path:
    """Prefill residuals at SAVE layers for coord_k and length boards on fit + test contexts disjoint from part B."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from q2_probe import Recorder, batched, run_batch
    from swarmlab.config import RunConfig
    from swarmlab.prompts import system_prompt

    path = OUT / "part_c_acts.npz"
    if path.exists():
        return path
    coord, claims, other = harvest(df, obs)
    used_b = set(random.Random(0).sample(
        [i for i in range(len(df)) if df.model[i] == "gem+qwen3" and 2 <= df["round"][i] <= 11 and board_of(obs[i])],
        60))
    ctx = contexts(df, obs, args.n_fit + args.n_test, used_b, seed=1)
    layers = json.loads((ROOT / "probe.json").read_text())["save_layers"]
    cfgs, items = {}, []
    for c, i in enumerate(ctx):
        run = df.run[i]
        cfgs.setdefault(run, RunConfig.from_dict(json.loads((ROOT / run / "config.json").read_text())))
        pre, _, post = board_of(obs[i])
        for cond in conditions(1000 + c, df.agent[i], int(df["round"][i]), coord, claims, other):
            if cond["factor"] in ("coord_k", "length"):
                items.append(dict(ctx=c, split="fit" if c < args.n_fit else "test", factor=cond["factor"],
                                  level=cond["level"], system=system_prompt(cfgs[run], df.agent[i]),
                                  user=pre + cond["board"] + post))
    print(f"{len(ctx)} contexts ({args.n_fit} fit, {args.n_test} test), {len(items)} prefill passes")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    rec = Recorder(model)
    X = []
    for batch in batched(items, args.bs):
        convs = [[{"role": "system", "content": x["system"]}, {"role": "user", "content": x["user"]}] for x in batch]
        acts, _ = run_batch(model, tok, rec, convs, 0)
        X.append(acts[:, layers].astype(np.float16))
    meta = pd.DataFrame([{k: x[k] for k in ("ctx", "split", "factor", "level")} for x in items])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=np.concatenate(X), layers=np.array(layers), **{c: meta[c].values for c in meta})
    del model
    torch.cuda.empty_cache()
    return path


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def paired(X, meta, mask, factor, hi, lo) -> np.ndarray:
    """Mean over contexts of X[factor=hi] - X[factor=lo]."""
    a = meta[mask & (meta.factor == factor) & (meta.level == hi)].sort_values("ctx")
    b = meta[mask & (meta.factor == factor) & (meta.level == lo)].sort_values("ctx")
    assert (a.ctx.values == b.ctx.values).all()
    return (X[a.index] - X[b.index]).mean(0)


def stratified_dir(A: np.ndarray, y: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Diff-in-means of y within each stratum, weighted by positives in the stratum."""
    d, w = 0, 0
    for s in np.unique(strata):
        m = strata == s
        if y[m].sum() and (~y[m]).sum():
            d = d + y[m].sum() * (A[m & y].mean(0) - A[m & ~y].mean(0))
            w += y[m].sum()
    return d / w


def part_c(args) -> None:
    import statsmodels.formula.api as smf
    from sklearn.metrics import roc_auc_score

    df, A, obs = load()
    path = capture_c(args, df, obs)
    z = np.load(path, allow_pickle=True)
    layers = list(z["layers"])
    li = layers.index(LAYER)
    Xs = z["X"][:, li].astype(np.float32)
    meta = pd.DataFrame({c: z[c] for c in ("ctx", "split", "factor", "level")})
    fit = (meta.split == "fit").values
    fitm = pd.Series(fit)

    d_len = unit(paired(Xs, meta, fitm, "length", 15, 0))
    d_in = unit(paired(Xs, meta, fitm, "coord_k", 8, 0))
    d_in_perp = unit(d_in - (d_in @ d_len) * d_len)

    # real runs
    Ar = A[:, li]
    y = df.coord_post.values.astype(bool)
    feats = pd.DataFrame([board_features(b[1]) if (b := board_of(o)) else
                          dict(n_posts=0, n_coord=0, n_authors=0, omitted=0, n_ids=0) for o in obs])
    strata = np.minimum(feats.n_posts.values, 5)
    d_old = unit(Ar[y].mean(0) - Ar[~y].mean(0))
    proj = {k: np.full(len(df), np.nan) for k in ("old", "dec", "dec_raw")}
    dec_all = []
    for r in df.run.unique():
        tr, te = df.run.values != r, df.run.values == r
        o = unit(Ar[tr & y].mean(0) - Ar[tr & ~y].mean(0))
        dr = unit(stratified_dir(Ar[tr], y[tr], strata[tr]))
        dd = unit(dr - (dr @ d_len) * d_len)
        dec_all.append(dd)
        proj["old"][te], proj["dec_raw"][te], proj["dec"][te] = Ar[te] @ o, Ar[te] @ dr, Ar[te] @ dd
    proj["in"], proj["in_perp"], proj["len"] = Ar @ d_in, Ar @ d_in_perp, Ar @ d_len
    d_dec = unit(np.mean(dec_all, 0))
    post_d = unit(Ar[df.post.values.astype(bool)].mean(0) - Ar[~df.post.values.astype(bool)].mean(0))
    dirs = {"old": d_old, "dec_raw": None, "dec": d_dec, "in": d_in, "in_perp": d_in_perp, "len": d_len}
    names = {"old": "original (diff-in-means, all calls)", "dec_raw": "decision, length-stratified",
             "dec": "decision, stratified, d_len removed", "in": "input content (8 vs 0 coord posts)",
             "in_perp": "input content, d_len removed", "len": "board length (15 vs 0 posts)"}

    print(f"\nPART C (layer {LAYER}): {fit.sum() // 10} fit / {(~fit).sum() // 10} test contexts, {len(df)} real calls")
    print("\ncosines:")
    for a in ("old", "dec", "in", "in_perp"):
        print(f"  {a:8s} vs len {dirs[a] @ d_len:+.2f}   vs any-post {dirs[a] @ post_d:+.2f}   vs old {dirs[a] @ d_old:+.2f}"
              + (f"   vs in {dirs[a] @ d_in:+.2f}" if a != "in" else ""))

    # held-out board edits: paired change per level, in units of each direction
    test = meta[~fit]
    rows = []
    for k, d in dirs.items():
        if d is None:
            continue
        p = Xs @ d
        for fac in ("coord_k", "length"):
            g = test[test.factor == fac]
            base = pd.Series(p[g[g.level == 0].index], index=g[g.level == 0].ctx.values)
            for lv in sorted(g.level.unique()):
                h = g[g.level == lv]
                diff = pd.Series(p[h.index], index=h.ctx.values) - base
                rows.append(dict(direction=k, factor=fac, level=lv, d=diff.mean(), se=diff.std() / np.sqrt(len(diff))))
    held = pd.DataFrame(rows)
    print("\nheld-out contexts: paired change in projection vs level 0 (coord_k: 8-post board; length: no coordination)")
    t = held.assign(v=held.d.round(2).astype(str) + "±" + held.se.round(2).astype(str))
    print(t.pivot_table(index=["factor", "level"], columns="direction", values="v", aggfunc="first")[
        [k for k in dirs if dirs[k] is not None]].to_string())

    # real runs: AUROC, cells, OLS
    df = pd.concat([df.reset_index(drop=True), feats], axis=1)
    for k, v in proj.items():
        df[f"p_{k}"] = v
    df["mix"] = df.model.map({"gem+qwen3": "mixed", "qwen3": "qwen_only"})
    df["logN"] = np.log2(df.N)
    df = df[~df.error]
    posting = df.post.astype(bool)
    print("\nreal runs: AUROC for Qwen's coordination post")
    for k in proj:
        print(f"  {names[k]:40s} all calls {roc_auc_score(df.coord_post, df[f'p_{k}']):.3f}   "
              f"posting calls only {roc_auc_score(df.coord_post[posting], df[f'p_{k}'][posting]):.3f}")
    print("\nreal runs: mean projection by cell (z-scored per direction)")
    zc = df[[f"p_{k}" for k in proj]].apply(lambda c: (c - c.mean()) / c.std())
    cell = pd.concat([df[["mix", "topology", "N"]], zc], axis=1).groupby(["mix", "topology", "N"]).mean().round(2)
    cell["n_posts"] = df.groupby(["mix", "topology", "N"]).n_posts.mean().round(2)
    print(cell.to_string())
    print("\nOLS (cluster-robust by run): R2 board-only / scenario-only / both, and scenario-given-board F-test")
    board = "n_posts + n_coord + n_authors + round"
    out = []
    for k in proj:
        y_ = f"p_{k}"
        fits = {n: smf.ols(f"{y_} ~ {f}", df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.run)[0]})
                for n, f in (("board", board), ("scen", "logN * C(topology) + C(mix)"),
                             ("both", f"{board} + logN * C(topology) + C(mix)"))}
        nm = [n for n in fits["both"].params.index if "logN" in n or "topology" in n or "mix" in n]
        ft = fits["both"].f_test(" = 0, ".join(nm) + " = 0")
        bc = fits["both"].params.get("n_coord", np.nan)
        out.append(dict(direction=names[k], r2_board=fits["board"].rsquared, r2_scen=fits["scen"].rsquared,
                        r2_both=fits["both"].rsquared, F_scen=float(np.squeeze(ft.fvalue)), p_scen=float(ft.pvalue),
                        n_coord_t=bc / fits["both"].bse.get("n_coord", np.nan)))
    res = pd.DataFrame(out).round(3)
    print(res.to_string(index=False))
    res.to_csv(OUT / "part_c_ols.csv", index=False)
    held.to_csv(OUT / "part_c_heldout.csv", index=False)
    cell.to_csv(OUT / "part_c_cells.csv")
    np.savez(OUT / "part_c_directions.npz", layer=LAYER, d_len=d_len, d_in=d_in, d_in_perp=d_in_perp, d_dec=d_dec,
             d_old=d_old)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["a", "b", "b-summary", "c"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n-ctx", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--gen", action="store_true", help="also generate greedy replies and label coordination posts")
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--n-fit", type=int, default=80)
    ap.add_argument("--n-test", type=int, default=60)
    args = ap.parse_args()
    if args.part == "a":
        part_a()
    elif args.part == "b":
        part_b(args)
    elif args.part == "c":
        part_c(args)
    else:
        summarize_b(OUT / ("part_b_gen.jsonl" if args.gen else "part_b.jsonl"))


if __name__ == "__main__":
    main()
