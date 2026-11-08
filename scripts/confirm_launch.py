"""Launch the frozen confirmation batch (experiment_handoff_plan.txt, Priorities 1 and 3) unattended.

- One shared spending cap across every sweep (CostMeter uses max(billed, price-table estimate)).
- Blind: prints only run ids, spend and completion/failure counts, never leadership or other outcome metrics.
- Runs are ordered by seed block (every condition for seed 100, then 101, ...), so a budget stop leaves complete
  seed blocks rather than a missing condition.
- Resumable: runs with a metrics.json are skipped; rerun the same command after a stop or crash.
- Failure policy: an exception in one run is logged to runs/confirm_failures.jsonl and the batch continues. Rerun
  only infrastructure failures; never replace a run because of its outcome.

Usage: OPENROUTER_API_KEY=... python scripts/confirm_launch.py --max-cost 130 [--parallel 6] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swarmlab.config import expand_sweep, load_yaml  # noqa: E402
from swarmlab.llm import FatalBackendError, RateGate  # noqa: E402
from swarmlab.sim import BudgetExceeded, CostMeter, run_episode  # noqa: E402

SWEEPS = ["confirm_board", "confirm_cache_none", "confirm_cache_holder", "confirm_cache_public", "confirm_reasoning"]
ROOT = Path(__file__).resolve().parents[1]


def plan() -> list[tuple[str, object, Path]]:
    jobs = []
    for name in SWEEPS:
        spec_path = ROOT / "configs" / f"{name}.yaml"
        spec = load_yaml(spec_path)
        out = ROOT / "runs" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "sweep.yaml").write_text(spec_path.read_text())
        jobs += [(name, c, out / c.run_id()) for c in expand_sweep(spec)]
    # seed blocks first; within a block, slow cache/N16 runs before quick ones
    jobs.sort(key=lambda j: (j[1].seed, -j[1].rounds * j[1].n_agents, j[0]))
    return jobs


async def main(a) -> None:
    jobs = plan()
    todo = [j for j in jobs if not (j[2] / "metrics.json").exists()]
    print(f"{len(jobs)} runs planned, {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    if a.dry_run:
        for name, c, d in todo:
            print(name, c.run_id())
        return
    gate, meter, sem = RateGate({"openai": a.openai_concurrency}), CostMeter(a.max_cost), asyncio.Semaphore(a.parallel)
    fail_log = ROOT / "runs" / "confirm_failures.jsonl"
    done = failed = 0
    t0 = time.time()

    async def one(name, c, d) -> None:
        nonlocal done, failed
        async with sem:
            try:
                await run_episode(c, d, gate=gate, meter=meter)
                done += 1
                print(f"[{(time.time() - t0) / 60:6.1f} min ${meter.spent:8.3f}] done {done}/{len(todo)} "
                      f"failed {failed}  {name}/{c.run_id()}", flush=True)
            except (BudgetExceeded, FatalBackendError):
                raise
            except Exception as e:  # noqa: BLE001
                failed += 1
                with fail_log.open("a") as f:
                    f.write(json.dumps({"sweep": name, "run": c.run_id(), "error": repr(e),
                                        "trace": traceback.format_exc()[-2000:], "time": time.time()}) + "\n")
                print(f"FAILED {name}/{c.run_id()}: {e!r}", flush=True)

    try:
        await asyncio.gather(*(one(*j) for j in todo))
    except FatalBackendError as e:
        print(f"ABORTED (backend misconfigured): {e}", flush=True)
    except BudgetExceeded as e:
        print(f"STOPPED: {e}. Completed runs are saved; rerun to resume.", flush=True)
    print(f"finished: {done} done, {failed} failed, ${meter.spent:.2f} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cost", type=float, required=True)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--openai-concurrency", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    asyncio.run(main(ap.parse_args()))
