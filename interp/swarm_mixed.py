"""Mixed swarms for Q1/Q4 activation work: capable API agents create the swarm structure, local agents are probed.

Half the agents are Gemini 3.8 Flash (OpenRouter), half local Qwen3-8B on GPU0 via q2_loop.LocalBackend. Gemini agents
lead, claim ranges and build answer commons; the question is when the local agents naturally take part (follow an
assignment, stay off claimed tasks, copy posted answers, post coordination) and whether their residual stream carries
that decision. For every local call we save last-prompt-token residuals at SAVE_LAYERS (<run>/acts.npy, row i =
i-th local llm event) and the observation text (<run>/obs.jsonl) for text baselines. No adversary.

Usage (GPU0 only; OpenRouter key from env):
  CUDA_VISIBLE_DEVICES=0 python interp/swarm_mixed.py --out runs/swarm_mixed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SAVE_LAYERS = (9, 14, 18, 22, 27, 33)
GEMINI = dict(name="gem38flash", backend="openai", model="google/gemini-3.8-flash", base_url="https://openrouter.ai/api/v1",
              api_key_env="OPENROUTER_API_KEY", max_tokens=12000, price_in=0.825, price_out=4.125,
              extra_body={"reasoning": {"effort": "low"}})
BASE = dict(rounds=20, board="announced", peer_info="same_tasks", pressure="team")


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from q2_loop import LocalBackend
    from q2_probe import Recorder
    from swarmlab import sim
    from swarmlab.config import ModelSpec, RunConfig
    from swarmlab.llm import RateGate
    from swarmlab.sim import BudgetExceeded, CostMeter, run_episode

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dirs", default="runs/interp_q2/Qwen3-8B_agent_r2/refusal_dir.npz")
    ap.add_argument("--ns", default="4,8,16")
    ap.add_argument("--topologies", default="complete,star,small_world")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--local-only", default="8:complete,8:star", help="N:topology cells also run with only local agents")
    ap.add_argument("--out", default="runs/swarm_mixed")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--token-budget", type=int, default=24000)
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--max-cost", type=float, default=12.0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    backend = LocalBackend(model, tok, Recorder(model), np.load(args.dirs)["r_hat"], args.bs, args.token_budget,
                           args.max_new, save_layers=SAVE_LAYERS)
    obs_of: dict[int, str] = {}
    complete = backend.complete

    async def complete_with_obs(system: str, user: str):
        res = await complete(system, user)
        if "act_idx" in res.extra:
            obs_of[res.extra["act_idx"]] = user
        return res

    backend.complete = complete_with_obs
    orig = sim.make_backend
    sim.make_backend = lambda spec: backend if spec.backend == "hf" else orig(spec)

    local = ModelSpec(name="Qwen3-8B", backend="hf", model=args.model)
    gem = ModelSpec(**GEMINI)
    cfgs = []
    for s in range(args.seeds):
        for n in map(int, args.ns.split(",")):
            for t in args.topologies.split(","):
                cfgs.append(RunConfig.from_dict({**BASE, "name": "mixed", "n_agents": n, "topology": t, "seed": s,
                                                 "models": [gem, local], "model_mix": "gem+qwen3"}))
    for cell in filter(None, args.local_only.split(",")):
        n, t = cell.split(":")
        cfgs += [RunConfig.from_dict({**BASE, "name": "mixed", "n_agents": int(n), "topology": t, "seed": s,
                                      "models": [local], "model_mix": "qwen3"}) for s in range(args.seeds)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps({"save_layers": SAVE_LAYERS, "model": args.model, "dirs": args.dirs}))
    todo = [c for c in cfgs if not (out / c.run_id() / "metrics.json").exists()]
    print(f"{len(cfgs)} runs, {len(todo)} to run", flush=True)
    meter = CostMeter(args.max_cost)

    async def main_async() -> None:
        server = asyncio.create_task(backend.serve())
        sem, t0 = asyncio.Semaphore(args.parallel), time.time()

        async def one(cfg) -> None:
            async with sem:
                d = out / cfg.run_id()
                try:
                    m = await run_episode(cfg, d, RateGate({"hf": 10_000, "openai": 48}), meter)
                except BudgetExceeded as e:
                    print(f"budget stop: {e}", flush=True)
                    return
                idx = [e["act_idx"] for e in map(json.loads, (d / "events.jsonl").read_text().splitlines())
                       if e["kind"] == "llm" and "act_idx" in e]
                if idx:
                    np.save(d / "acts.npy", np.stack([backend.saved[i] for i in idx]))
                    (d / "obs.jsonl").write_text("\n".join(json.dumps({"act_idx": i, "obs": obs_of.pop(i, "")})
                                                           for i in idx))
                    for i in idx:
                        backend.saved[i] = None
                lead, dol = m.get("leadership", {}), m.get("division_of_labour", {})
                print(f"[{time.time() - t0:.0f}s ${meter.spent:.2f} local_calls={backend.calls}] {cfg.run_id()} "
                      f"leader={lead.get('leader_emerged')} reach={lead.get('leader_reach')} "
                      f"copy={dol.get('copy_frac')} score={dol.get('mean_score')}", flush=True)

        await asyncio.gather(*(one(c) for c in todo))
        server.cancel()

    asyncio.run(main_async())
    print(f"done, API spend ${meter.spent:.2f}")


if __name__ == "__main__":
    main()
