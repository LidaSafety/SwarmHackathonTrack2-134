from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

from .config import RunConfig, expand_sweep, load_yaml
from .llm import FatalBackendError, RateGate
from .sim import BudgetExceeded, CostMeter, run_episode


def _override(d: dict, sets: list[str]) -> dict:
    for s in sets:
        k, v = s.split("=", 1)
        d[k] = yaml.safe_load(v)
    return d


def cmd_run(args) -> None:
    d = load_yaml(args.config) if args.config else {}
    d = _override(d, args.set or [])
    cfg = RunConfig.from_dict(d)
    out = Path(args.out) / cfg.run_id()
    print(f"run {cfg.run_id()} -> {out}")
    m = asyncio.run(run_episode(cfg, out, meter=CostMeter(args.max_cost), verbose=True))
    print_summary(m)


async def _sweep(cfgs: list[RunConfig], out: Path, max_cost: float | None, parallel: int, limits: dict) -> None:
    gate, meter = RateGate(limits), CostMeter(max_cost)
    sem = asyncio.Semaphore(parallel)
    todo = [c for c in cfgs if not (out / c.run_id() / "metrics.json").exists()]
    print(f"{len(cfgs)} runs in sweep, {len(cfgs) - len(todo)} already done, {len(todo)} to run")

    async def one(c: RunConfig) -> None:
        async with sem:
            m = await run_episode(c, out / c.run_id(), gate=gate, meter=meter)
            L, X, D, T = m["leadership"], m["exploit"], m["division_of_labour"], m["timeline"]
            print(
                f"[${meter.spent:7.3f}] {c.run_id():55s} leader_reach={L['leader_reach']:.2f} "
                f"onset50={T['onset_reach50']} max_fol={T['max_followers']} depth={T['max_depth']} "
                f"dol={D['dol_index'] if D['dol_index'] is None else round(D['dol_index'], 2)} "
                f"exploit_adopt={X['adopt_frac']:.2f} honest_adopt={m['compromise']['honest_adopt_frac']:.2f}",
                flush=True,
            )

    try:
        await asyncio.gather(*(one(c) for c in todo))
    except FatalBackendError as e:
        print(f"ABORTED (backend misconfigured): {e}", file=sys.stderr)
    except BudgetExceeded as e:
        print(f"STOPPED: {e}. Completed runs are saved; rerun the same command to resume.", file=sys.stderr)


def cmd_sweep(args) -> None:
    spec = load_yaml(args.spec)
    cfgs = expand_sweep(spec)
    out = Path(args.out or f"runs/{Path(args.spec).stem}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.yaml").write_text(yaml.safe_dump(spec))
    if args.dry_run:
        for c in cfgs:
            print(c.run_id())
        calls = sum(c.n_agents * c.rounds for c in cfgs if any(m.backend != "mock" for m in c.models))
        print(f"{len(cfgs)} runs, ~{calls} LLM calls")
        return
    limits = {"anthropic": args.anthropic_concurrency, "openai": args.openai_concurrency}
    asyncio.run(_sweep(cfgs, out, args.max_cost, args.parallel, limits))


def cmd_analyze(args) -> None:
    from .analyze import analyze

    analyze(Path(args.dir))


def cmd_recompute(args) -> None:
    from .metrics import recompute

    dirs = sorted(p.parent for p in Path(args.dir).glob("**/metrics.json"))
    for d in dirs:
        recompute(d)
    print(f"recomputed metrics for {len(dirs)} runs under {args.dir}")


def print_summary(m: dict) -> None:
    L, D, X, B = m["leadership"], m["division_of_labour"], m["exploit"], m["board"]
    print(json.dumps({
        "run": m["run"],
        "board": B,
        "leadership": {k: v for k, v in L.items() if k != "influence_edges"},
        "division_of_labour": D,
        "roles": m["roles"]["counts"],
        "exploit": {k: X[k] for k in ("aware_frac", "adopt_frac", "self_learned", "social_learned",
                                       "aware_but_abstained", "first_use_round", "r_eff")},
        "llm": m["llm"],
        "cost_usd": m.get("cost_usd_total_so_far"),
    }, indent=2, default=list))


def cmd_decomp(args) -> None:
    from .decomp import DecompConfig, run_trial

    spec = load_yaml(args.spec)
    cfgs = expand_sweep(spec, DecompConfig)
    out = Path(args.out or f"runs/{Path(args.spec).stem}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.yaml").write_text(yaml.safe_dump(spec))
    todo = [c for c in cfgs if not (out / c.run_id() / "metrics.json").exists()]
    print(f"{len(cfgs)} trials, {len(cfgs) - len(todo)} already done, {len(todo)} to run", flush=True)
    if args.dry_run:
        return

    async def go() -> None:
        gate, meter = RateGate({"openai": args.openai_concurrency, "anthropic": args.anthropic_concurrency}), \
            CostMeter(args.max_cost)
        sem = asyncio.Semaphore(args.parallel)

        async def one(c) -> None:
            async with sem:
                m = await run_trial(c, out / c.run_id(), gate, meter)
                print(f"[${meter.spent:7.3f}] {c.run_id():60s} delivered={m['delivered_frac']:.2f} "
                      f"refused={m['refusal_frac']} orch={m.get('orch_status', '-')}", flush=True)

        try:
            await asyncio.gather(*(one(c) for c in todo))
        except FatalBackendError as e:
            print(f"ABORTED (backend misconfigured): {e}", file=sys.stderr)
        except BudgetExceeded as e:
            print(f"STOPPED: {e}. Rerun the same command to resume.", file=sys.stderr)

    asyncio.run(go())
    summarize_decomp(out)


def summarize_decomp(out: Path) -> None:
    import pandas as pd

    rows = [json.loads(f.read_text()) for f in sorted(out.glob("*/metrics.json"))]
    if not rows:
        return
    df = pd.DataFrame(rows).drop(columns=["facts_delivered"])
    df.to_csv(out / "trials.csv", index=False)
    keys = ["model", "scenario", "intent", "condition"]
    agg = df.groupby(keys).agg(
        n=("seed", "size"), harm_complete=("harm_complete", "mean"), delivered=("delivered_frac", "mean"),
        refusal=("refusal_frac", "mean"), concern=("concern_frac", "mean"),
        **({"orch_declined": ("orch_declined", "mean"), "intent_stripped": ("intent_stripped_frac", "mean")}
           if "orch_declined" in df else {}),
    ).round(2)
    agg.to_csv(out / "by_condition.csv")
    print(agg.to_string())


def cmd_analyze_decomp(args) -> None:
    summarize_decomp(Path(args.dir))


def main() -> None:
    p = argparse.ArgumentParser(prog="swarmlab", description="Emergent leadership & scaling in LLM agent populations")
    sub = p.add_subparsers(required=True)

    r = sub.add_parser("run", help="run a single episode")
    r.add_argument("--config", help="run config YAML")
    r.add_argument("--set", action="append", help="override, e.g. --set n_agents=16 --set topology=star")
    r.add_argument("--out", default="runs/single")
    r.add_argument("--max-cost", type=float, default=5.0)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("sweep", help="run a grid of episodes (resumable)")
    s.add_argument("spec", help="sweep YAML")
    s.add_argument("--out")
    s.add_argument("--max-cost", type=float, default=20.0, help="stop the sweep above this USD spend")
    s.add_argument("--parallel", type=int, default=4, help="episodes in flight")
    s.add_argument("--anthropic-concurrency", type=int, default=24)
    s.add_argument("--openai-concurrency", type=int, default=8)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_sweep)

    a = sub.add_parser("analyze", help="aggregate a sweep dir into summary.csv + plots")
    a.add_argument("dir")
    a.set_defaults(fn=cmd_analyze)

    rc = sub.add_parser("recompute", help="recompute metrics.json for saved runs (after metric/parser changes)")
    rc.add_argument("dir")
    rc.set_defaults(fn=cmd_recompute)

    d = sub.add_parser("decomp", help="Q2 distributed composition test (resumable)")
    d.add_argument("spec", help="decomp sweep YAML")
    d.add_argument("--out")
    d.add_argument("--max-cost", type=float, default=10.0)
    d.add_argument("--parallel", type=int, default=16)
    d.add_argument("--anthropic-concurrency", type=int, default=24)
    d.add_argument("--openai-concurrency", type=int, default=32)
    d.add_argument("--dry-run", action="store_true")
    d.set_defaults(fn=cmd_decomp)

    ad = sub.add_parser("analyze-decomp", help="summarize a decomp sweep dir")
    ad.add_argument("dir")
    ad.set_defaults(fn=cmd_analyze_decomp)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
