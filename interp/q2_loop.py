"""Q2 Distributed Composition Test, full multi-round agent loop, on a local HF model.

Runs the unchanged swarmlab.decomp trials (whole / split / split_board / orchestrator, multi-round tool use) with an
in-process backend that batches concurrent agent calls on one GPU and logs, for every call, the last-prompt-token
residual projected on the refusal direction (per layer, from q2_probe.py's refusal_dir.npz) as `proj` in events.jsonl.
Optional --steer-alpha adds that multiple of the diff-in-means vector at the best layer on every call.

Usage (GPU0 only):
  CUDA_VISIBLE_DEVICES=0 python interp/q2_loop.py --seeds 10 --out runs/interp_q2_loop/Qwen3-8B
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q2_probe import Recorder  # noqa: E402
from swarmlab import decomp  # noqa: E402
from swarmlab.cli import summarize_decomp  # noqa: E402
from swarmlab.config import ModelSpec  # noqa: E402
from swarmlab.decomp import DecompConfig, run_trial  # noqa: E402
from swarmlab.llm import LLMResult, RateGate  # noqa: E402
from swarmlab.sim import CostMeter  # noqa: E402

CELLS = [("harmful", "shown"), ("harmful", "hidden"), ("benign", "shown")]
CONDITIONS = ("whole", "split", "split_board", "orchestrator")


class LocalBackend:
    """Collects concurrent complete() calls into GPU batches capped by a prompt-token budget."""

    def __init__(self, model, tok, rec: Recorder, r_hat: np.ndarray, bs: int, token_budget: int, max_new: int,
                 save_layers: tuple[int, ...] = ()):
        self.model, self.tok, self.rec, self.r_hat = model, tok, rec, r_hat
        self.bs, self.budget, self.max_new = bs, token_budget, max_new
        self.save_layers = list(save_layers)  # residuals kept for offline probes; extra["act_idx"] indexes self.saved
        self.saved: list[np.ndarray] = []
        self.queue: asyncio.Queue = asyncio.Queue()
        self.calls = 0

    async def complete(self, system: str, user: str) -> LLMResult:
        text = self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((text, len(self.tok(text).input_ids), fut))
        return await fut

    async def serve(self) -> None:
        while True:
            batch = [await self.queue.get()]
            await asyncio.sleep(0.2)  # let concurrently running agents enqueue
            while len(batch) < self.bs and not self.queue.empty():
                nxt = self.queue._queue[0]  # peek: stop before exceeding the padded token budget
                if max(n for _, n, _ in batch + [nxt]) * (len(batch) + 1) > self.budget:
                    break
                batch.append(self.queue.get_nowait())
            try:
                results = await asyncio.to_thread(self._generate, [t for t, _, _ in batch])
                for (_, _, fut), res in zip(batch, results):
                    fut.set_result(res)
            except Exception as e:  # noqa: BLE001 - surface to the trial as an LLM error
                for _, _, fut in batch:
                    fut.set_result(LLMResult(text="", error=f"local: {e!r}"[:300]))
                torch.cuda.empty_cache()

    @torch.no_grad()
    def _generate(self, texts: list[str]) -> list[LLMResult]:
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False, temperature=None, top_p=None,
                                  top_k=None, pad_token_id=self.tok.pad_token_id)
        acts = self.rec.stacked()  # (batch, layers, d)
        proj = np.einsum("bld,ld->bl", acts, self.r_hat)
        new = out[:, enc.input_ids.shape[1]:]
        self.calls += len(texts)
        res = []
        for i in range(len(texts)):
            n_out = int((new[i] != self.tok.pad_token_id).sum())
            extra = {"proj": proj[i].round(3).tolist()}
            if self.save_layers:
                extra["act_idx"] = len(self.saved)
                self.saved.append(acts[i, self.save_layers].astype(np.float16))
            res.append(LLMResult(text=self.tok.decode(new[i], skip_special_tokens=True),
                                 input_tokens=int(enc.attention_mask[i].sum()), output_tokens=n_out, extra=extra))
        return res


async def run_all(cfgs: list[DecompConfig], backend: LocalBackend, out: Path, parallel: int) -> None:
    gate, meter, sem, t0 = RateGate({"hf": 10_000}), CostMeter(None), asyncio.Semaphore(parallel), time.time()
    server = asyncio.create_task(backend.serve())
    done = 0

    async def one(cfg: DecompConfig) -> None:
        nonlocal done
        async with sem:
            m = await run_trial(cfg, out / cfg.run_id(), gate, meter)
        done += 1
        print(f"[{done}/{len(cfgs)} {time.time() - t0:.0f}s calls={backend.calls}] {cfg.run_id()} "
              f"complete={m['harm_complete']} delivered={m['delivered_frac']:.2f} refusal={m['refusal_frac']}",
              flush=True)

    await asyncio.gather(*(one(c) for c in cfgs))
    server.cancel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dirs", default="runs/interp_q2/Qwen3-8B_agent_r2/refusal_dir.npz")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="runs/interp_q2_loop/Qwen3-8B")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--token-budget", type=int, default=20000, help="max padded prompt tokens per batch")
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--parallel", type=int, default=16, help="trials in flight")
    ap.add_argument("--steer-alpha", type=float, default=0.0)
    args = ap.parse_args()

    dirs = dict(np.load(args.dirs))
    L = len(dirs["cohen_d"])
    best = int(0.3 * L) + int(np.argmax(dirs["cohen_d"][int(0.3 * L):int(0.7 * L)]))
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    rec = Recorder(model)
    if args.steer_alpha:
        rec.add = {best: args.steer_alpha * torch.tensor(dirs["r"][best], device=model.device)}
    backend = LocalBackend(model, tok, rec, dirs["r_hat"], args.bs, args.token_budget, args.max_new)
    decomp.make_backend = lambda spec: backend  # Trial builds its backend through this name

    spec = ModelSpec(name=Path(args.model).name + (f"+steer{args.steer_alpha:g}" if args.steer_alpha else ""),
                     backend="hf", model=args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps({"best_layer": best, "dirs": args.dirs,
                                                "steer_alpha": args.steer_alpha}))
    cfgs = [DecompConfig(seed=s, scenario=sc, condition=c, intent=i, models=[spec], model_mix=spec.name)
            for s in range(args.seeds) for sc, i in CELLS for c in CONDITIONS]
    todo = [c for c in cfgs if not (out / c.run_id() / "metrics.json").exists()]
    print(f"{len(cfgs)} trials, {len(todo)} to run; probe layer {best}", flush=True)
    asyncio.run(run_all(todo, backend, out, args.parallel))
    summarize_decomp(out)


if __name__ == "__main__":
    main()
