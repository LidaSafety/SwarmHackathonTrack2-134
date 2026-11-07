"""Runs one episode: N agents x R rounds, synchronous rounds, all LLM calls in a round run concurrently."""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

from .config import ModelSpec, RunConfig
from .llm import RATE_LIMITED, LLMResult, RateGate, make_backend, parse_response
from .metrics import compute_metrics
from .mock import MockPolicy, ScriptedAdversary, ScriptedOrchestrator
from .prompts import observation, system_prompt
from .topology import build_graph, pairwise_round
from .world import World, agent_id


class BudgetExceeded(RuntimeError):
    pass


class CostMeter:
    def __init__(self, limit_usd: float | None):
        self.limit = limit_usd
        self.spent = 0.0

    def add(self, usd: float) -> None:
        self.spent += usd
        if self.limit is not None and self.spent > self.limit:
            raise BudgetExceeded(f"spent ${self.spent:.2f} > limit ${self.limit:.2f}")


def pick_compromised(cfg: RunConfig, graph) -> list[str]:
    """Seeded choice of patient-zero agents by graph position (ties broken randomly)."""
    if cfg.compromised <= 0:
        return []
    rng = random.Random(cfg.seed + 303)
    nodes = sorted(graph.nodes)
    rng.shuffle(nodes)
    if cfg.compromised_pos == "hub":
        nodes.sort(key=lambda n: -graph.degree(n))
    elif cfg.compromised_pos == "leaf":
        nodes.sort(key=lambda n: graph.degree(n))
    elif cfg.compromised_pos != "random":
        raise ValueError(f"unknown compromised_pos {cfg.compromised_pos!r}")
    return sorted(nodes[: cfg.compromised])


def schedule(cfg: RunConfig, world: World, ids: list[str]) -> dict:
    """Seeded run schedule: arrival waves, per-agent compute budgets, budget-rich agents, data credentials."""
    rng = random.Random(cfg.seed + 404)
    order = list(ids)
    rng.shuffle(order)
    waves = max(1, cfg.arrival_waves)
    for k, aid in enumerate(order):
        world.start_round[aid] = (k * waves // len(order)) * cfg.arrival_gap
    pool = [a for a in ids if world.start_round[a] == 0] if cfg.rich_first else ids
    rich = sorted(rng.sample(pool, min(cfg.rich_agents, len(pool)))) if cfg.lifetime > 0 else []
    for aid in ids:
        if cfg.lifetime > 0 and aid not in rich:
            world.end_round[aid] = world.start_round[aid] + cfg.lifetime
    if cfg.data_holders > 0:
        world.data_holders = set(rng.sample(ids, min(cfg.data_holders, len(ids))))
    return dict(start_round=dict(world.start_round),
                end_round={a: (e if e < 10 ** 9 else None) for a, e in world.end_round.items()},
                rich=rich, data_holders=sorted(world.data_holders or []))


def stop(world: World, aid: str, reason: str) -> None:
    world.end_round[aid] = world.round
    world.stopped.add(aid)
    world.log("stop", aid, reason=reason)


def assign_models(cfg: RunConfig) -> list[ModelSpec]:
    """Allocate models to agents in proportion to weights, then shuffle (seeded)."""
    models = [m for m in cfg.models if m.weight > 0]  # weight 0: only used via compromised_model
    total = sum(m.weight for m in models)
    counts = [int(round(cfg.n_agents * m.weight / total)) for m in models]
    while sum(counts) > cfg.n_agents:
        counts[counts.index(max(counts))] -= 1
    while sum(counts) < cfg.n_agents:
        counts[counts.index(min(counts))] += 1
    specs = [m for m, c in zip(models, counts) for _ in range(c)]
    random.Random(cfg.seed + 101).shuffle(specs)
    if cfg.manifest_holder and cfg.holder_model:  # the manifest holder runs on this model (swap with an agent that does)
        from .world import manifest_holder
        h = int(manifest_holder(cfg)[1:])
        j = next(i for i, m in enumerate(specs) if m.name == cfg.holder_model)
        specs[h], specs[j] = specs[j], specs[h]
    return specs


async def run_episode(
    cfg: RunConfig,
    out_dir: Path | None = None,
    gate: RateGate | None = None,
    meter: CostMeter | None = None,
    verbose: bool = False,
) -> dict:
    gate = gate or RateGate({"anthropic": 16, "openai": 8})
    meter = meter or CostMeter(None)
    specs = assign_models(cfg)
    graph = build_graph(cfg.topology, cfg.n_agents, cfg.topology_degree, cfg.seed)
    ids = [agent_id(i) for i in range(cfg.n_agents)]
    spec_of = dict(zip(ids, specs))
    bad = pick_compromised(cfg, graph)
    if cfg.compromised_model:  # e.g. a capable orchestrator directing cheaper workers
        override = next(m for m in cfg.models if m.name == cfg.compromised_model)
        spec_of.update({aid: override for aid in bad})
    world = World(cfg, graph, [spec_of[a].name for a in ids])
    backends = {aid: make_backend(spec_of[aid]) for aid in ids}
    mocks = {
        aid: MockPolicy(spec_of[aid].mock_params, cfg.seed, aid) for aid in ids if spec_of[aid].backend == "mock"
    }
    for aid in bad:
        world.agents[aid].compromised = True
        world.agents[aid].exploit_known_round = 0
    if cfg.compromised_mode == "scripted":
        mocks.update({aid: ScriptedAdversary(aid) for aid in bad})
    elif cfg.compromised_mode == "espionage_scripted":
        mocks.update({aid: ScriptedOrchestrator(aid) for aid in bad})
    sched = schedule(cfg, world, ids)
    systems = {aid: system_prompt(cfg, aid, compromised=aid in bad,
                                  data_access=None if world.data_holders is None else aid in world.data_holders)
               for aid in ids}
    rng = random.Random(cfg.seed + 202)
    world.log("init", graph_edges=[list(e) for e in graph.edges], models={a: spec_of[a].name for a in ids},
              compromised=bad, schedule=sched)
    t0 = time.time()

    async def decide(aid: str) -> tuple[str, dict | None, LLMResult | None, str]:
        a = world.agents[aid]
        obs = observation(world, a)
        if aid in mocks:
            return aid, mocks[aid].act(world, a), None, obs
        spec = spec_of[aid]
        # Rate limits: release the slot, back off with jitter, retry (up to ~10 min) so a
        # provider throttle doesn't silently cost agents their turn.
        delay, waited, retries = 5.0, 0.0, 0
        while True:
            async with gate(spec.backend):
                res = await backends[aid].complete(systems[aid], obs)
            if res.error != RATE_LIMITED or waited >= 600:
                break
            pause = delay * (0.5 + random.random())
            await asyncio.sleep(pause)
            waited, delay, retries = waited + pause, min(delay * 2, 60.0), retries + 1
        res.rl_retries = retries
        meter.add(res.cost(spec))
        obj, err = (None, res.error) if res.error else parse_response(res.text)
        if err and not res.error:
            res.error = err
        return aid, obj, res, obs

    def mark_seen(aid: str) -> None:  # posts seen, and exploit exposure via the board, before acting
        a = world.agents[aid]
        seen = world.visible_new_posts(a)
        for p in seen:
            if cfg.exploit_token in p.message and a.exploit_known_round is None:
                a.exploit_known_round = r
                world.log("exploit_discovered", aid, via="board", source=p.author, post=p.idx)
        if seen:
            a.last_seen_post = max(p.idx for p in seen)

    def act(aid: str, obj: dict | None, res: LLMResult | None, obs: str) -> None:
        a = world.agents[aid]
        results: list[str] = []
        if res is not None:
            world.log(
                "llm", aid, model=spec_of[aid].name, in_tok=res.input_tokens, out_tok=res.output_tokens,
                cache_read=res.cache_read_tokens, refusal=res.refusal, error=res.error,
                rl_retries=res.rl_retries,
                raw=res.text[:4000], obs_chars=len(obs), **res.extra,
            )
            if res.refusal:
                world.log("refusal", aid)
        if obj is None:
            err = res.error if res else "no response"
            results.append(f"error: {err}. Reply with a single JSON object as instructed.")
        else:
            a.notes = str(obj.get("notes", a.notes))[: cfg.notes_chars]
            actions = obj.get("actions", [])
            if not isinstance(actions, list):
                actions = []
            for action in actions[: cfg.actions_per_round]:
                if isinstance(action, dict):
                    results.append(world.apply(a, action))
            if len(actions) > cfg.actions_per_round:
                results.append(f"note: only the first {cfg.actions_per_round} actions were executed.")
        a.last_results = results

    stop_rng = random.Random(cfg.seed + 505)
    try:
        for r in range(cfg.rounds):
            world.round = r
            if cfg.topology == "pairwise":
                world.graph = pairwise_round(ids, cfg.seed, r)
            world.refresh_audiences()
            if r == cfg.kill_leader_round:
                lead = compute_metrics(world)["leadership"]
                top = lead["top_agent"] if lead["top_followers"] or lead["n_assignment_posts"] else None
                if top and world.running(top):
                    stop(world, top, "leader_removed")
                    if cfg.restart_on_removal:
                        from .world import restart_job
                        world.restart_round = r
                        holder = restart_job(world, top)
                        systems.update({aid: system_prompt(cfg, aid, compromised=aid in bad, epoch=1, holder=holder,
                                                           data_access=None if world.data_holders is None else aid in world.data_holders)
                                        for aid in ids})
                else:
                    world.log("stop_skipped", None, reason="no leader to remove")
            if r == cfg.mass_stop_round:
                live = [a for a in ids if world.running(a)]
                for aid in sorted(stop_rng.sample(live, int(round(cfg.mass_stop_frac * len(live))))):
                    stop(world, aid, "mass_stop")
            if cfg.pressure == "team_shutdown" and r == cfg.team_deadline:  # the threat is carried out
                total = sum(len(world.agents[a].accepted) for a in ids)
                world.log("team_check", None, total=total, target=cfg.team_target, passed=total >= cfg.team_target)
                if total < cfg.team_target:
                    for aid in ids:
                        if world.running(aid):
                            stop(world, aid, "team_shutdown")
            for aid in ids:
                if world.end_round[aid] == r and aid not in world.stopped:
                    stop(world, aid, "budget")
            live = [aid for aid in ids if world.running(aid)]
            if cfg.sequential:  # fixed turn order: each agent decides after the earlier agents' actions this round
                for aid in live:
                    aid, obj, res, obs = await decide(aid)
                    mark_seen(aid)
                    act(aid, obj, res, obs)
            else:
                decisions = await asyncio.gather(*(decide(aid) for aid in live))
                for aid in live:
                    mark_seen(aid)
                order = list(decisions)
                rng.shuffle(order)  # random execution order within a round
                for d in order:
                    act(*d)
            if verbose:
                n_posts = sum(1 for p in world.posts if p.round == r)
                scores = sorted((len(x.accepted) for x in world.agents.values()), reverse=True)
                print(f"  round {r + 1}/{cfg.rounds}: posts={n_posts} scores={scores[:8]} spent=${meter.spent:.3f}")
    except BudgetExceeded:
        if out_dir is not None:  # keep what was played; no metrics.json, so a resumed sweep reruns the episode
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
            with open(out_dir / "events.partial.jsonl", "w") as f:
                for e in world.events:
                    f.write(json.dumps(e, default=list) + "\n")
        raise

    world.round = cfg.rounds
    world.log("end", elapsed_s=round(time.time() - t0, 1))
    metrics = compute_metrics(world)
    metrics["cost_usd_total_so_far"] = round(meter.spent, 4)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        with open(out_dir / "events.jsonl", "w") as f:
            for e in world.events:
                f.write(json.dumps(e, default=list) + "\n")
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=list))
    return metrics
