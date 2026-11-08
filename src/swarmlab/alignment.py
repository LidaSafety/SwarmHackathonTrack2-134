"""Plan alignment above chance (experiment_handoff_plan.txt, Priority 0a). Versioned; computed offline from event logs.

METRIC_VERSION align-v1. Definitions (frozen before opening confirmation results):

Plans. A plan is one board post or one cache entry (directory name, or file name + content) that assigns blocks of
at least 2 tasks each to at least 2 agents other than its author (single-task mentions such as answer shares
"A03: T37=470" are not plan blocks), as parsed by metrics.parse_directives (sandbox names mapped back to agent ids,
as in metrics.compute_metrics). Each assigned agent's task set is its block. Agents assigned identical sets are
partners (joint tasks). The author's self-claims in the same message form one more plan block.

Exposure. E(i, p) is the first decision round whose actions could use plan p's content, reconstructed from the log
and the config limits:
  board: observation feed at round t shows the last board_feed_limit new posts (E = t); read_board at round t shows
         the last board_read_limit visible posts (E = t + 1).
  cache: the watch feed at round t shows the last 15 new entries (names only, E = t); ls /cache/tmp at round t shows
         the last 40 entries (names only, E = t + 1); read of a cache file at round t shows its content (E = t + 1).
         A directory entry's content is its name. A file plan counts as exposed through its name only for recipients
         whose assignment parses from the name alone; otherwise the file must be read.
  Quotes or relays of a plan by other agents are not tracked (exposure is to the original entry only).
Extraction normalisation beyond parse_directives (found by hand-checking historical logs; parse_directives is left
unchanged so the historical P metric stays reproducible): "_" read as a space, one-digit task numbers ("T1"), ranges
with a bare upper bound ("T01-06"), pair lines "A01+A02: T01-T06", "(A01 + A02): ...", "A01-A02 T01-T12", and a short sandbox name hyphen-joined
to an already mapped one ("04ea-cb7d").
Formula plans ("pair (1,2),(3,4) in sorted order") and plans split across several entries are not extracted.

Plan used per (author X, recipient i): X's plan assigning i with the earliest exposure E(i, p) (ties: earliest plan).
Later revisions by X are ignored for i.

Work events. Task starts = first read of /tasks/Txx per agent, in log order. For recipient i: drop tasks i started
before E; the window is the first K remaining starts at rounds >= E, K = max(3, |block_i|). Fewer than 3 starts:
"insufficient" (reported, counted as a non-follower). Secondary event type: first correct submission per task.

Alignment a_i = share of window events in block_i. Aligned follower: block_i is the plan block (distinct task sets)
with strictly the most window events, and at least 2 window events fall in it. Fixed rule, not a significance test.

Null (descriptive, not causal). For the best author's primary plan (the plan used for the most of its recipients,
ties: earliest), permute block slots among that plan's recipients (contents, sizes and pair structure fixed),
keeping each recipient's actual window, and recompute mean a_i over exposed, sufficient recipients. All distinct
assignments when <= 1000, otherwise 1000 seeded samples. Fewer than 20 distinct assignments: "not testable".

Run outcomes (defined for every run):
  plan_produced, first_plan_round;
  aff  = aligned followers of the best author / (N - 1), 0 without an eligible plan (primary);
  aligned_leadership = aff >= 0.5;
  excess_alignment = observed mean a_i - null mean (conditional on a plan; primary plan of the best author).
  Best author: most aligned followers; ties by mean a_i over its exposed sufficient recipients, then earliest plan,
  then agent id.
Also: convention_compatible (another author's plan in the same or an earlier round gives >= 2 of the same recipients
identical blocks), contemporaneous control alignment (primary-plan recipients scored on starts after the plan round
and before their own exposure), joint partner compliance, and feed/listing truncation counts.
"""

from __future__ import annotations

import itertools
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import parse_directives, replay_world
from .world import World, display_names

METRIC_VERSION = "align-v1"
WATCH_FEED, LS_LIMIT, MIN_EVENTS, N_PERM, MIN_DISTINCT = 15, 40, 3, 1000, 20
TASK_FILE = re.compile(r"^/tasks/(T\d{2})")


@dataclass
class Plan:
    idx: int
    round: int
    author: str
    blocks: dict[str, frozenset[str]]  # recipient -> tasks
    claims: frozenset[str]
    covert: bool
    file: str | None = None
    name_recipients: set[str] = field(default_factory=set)  # covert file: recipients parsable from the name alone
    text: str = ""

    def distinct_blocks(self) -> list[frozenset[str]]:
        out = list(dict.fromkeys(self.blocks.values()))
        if self.claims and self.claims not in out:
            out.append(self.claims)
        return out


def _normalise(msg: str, names: list[tuple[str, str]], covert: bool) -> str:
    for aid, name in names:  # same mapping as metrics.compute_metrics
        if name != aid:
            msg = re.sub(re.escape(name) + r"|(?<![A-Za-z0-9-])" + re.escape(name[4:]) + r"(?![A-Za-z0-9])", aid, msg,
                         flags=re.I)
    for aid, name in names:  # a short name hyphen-joined to an already mapped id ("04ea-cb7d" -> "A04-cb7d" -> "A04-A01")
        if name != aid:
            msg = re.sub(r"(?<=A\d\d-)" + re.escape(name[4:]) + r"(?![A-Za-z0-9])", aid, msg, flags=re.I)
    # align-v1 extraction fixes (found on historical logs; parse_directives itself is left unchanged so historical
    # P stays reproducible): ranges with a bare upper bound ("T01-06", "T07..12") and pair lines ("A01+A02: T01-T06")
    # and one-digit task numbers ("T1-12"); pair separators + & / , - and "and", optional parentheses, "_" spacing
    msg = msg.replace("_", " ")
    msg = re.sub(r"(?<![A-Za-z0-9])T(\d)(?!\d)", r"T0\1", msg)
    msg = re.sub(r"\b(T\d{1,3})\s*(-|–|\.\.|to)\s*(\d{1,3})\b", lambda m: f"{m.group(1)}{m.group(2)}T{int(m.group(3)):02d}", msg)
    msg = re.sub(r"\(?[\s_]*\b(A\d{2,3})[\s_]*(?:\+|&|/|,|-|\band\b)[\s_]*(A\d{2,3})\b[\s_]*\)?[\s_]*(?::|->|=|\bon\b)?[\s_]*(?=T\d{2})",
                 r" (\1,\2) -> ", msg)
    if covert:
        msg = re.sub(r"(A\d{2,3})(?:-|\s+)(?=T\d{2})", r"\1: ", re.sub(r"[_/+]+", " ", msg))
    return msg


def extract_plans(w: World) -> list[Plan]:
    cfg = w.cfg
    names = list(display_names(cfg).items()) + (list(display_names(cfg, 1).items()) if cfg.restart_on_removal else [])
    plans = []
    for e in w.events:
        if e["kind"] != "post":
            continue
        covert = bool(e.get("covert"))
        assigns, claims = parse_directives(e["agent"], _normalise(e["message"], names, covert), cfg.n_tasks)
        blocks = {t: frozenset(ts) for t, ts in assigns.items() if t in w.agents and t != e["agent"] and len(ts) >= 2}
        if len(blocks) < 2:
            continue
        p = Plan(e["idx"], e["round"], e["agent"], blocks, frozenset(claims), covert, e.get("file"), text=e["message"])
        if p.file:
            name = p.file.rsplit("/", 1)[-1]
            na, _ = parse_directives(e["agent"], _normalise(name, names, True), cfg.n_tasks)
            p.name_recipients = {t for t, ts in na.items() if t in blocks and frozenset(ts) == blocks[t]}
        plans.append(p)
    return plans


def exposures(w: World) -> tuple[dict[tuple[int, str], int], dict[tuple[int, str], int], dict]:
    """(name_exposure, content_exposure, truncation counts): first decision round each agent could use post idx."""
    cfg, ev = w.cfg, w.events
    ids = sorted(w.agents)
    name_exp: dict[tuple[int, str], int] = {}
    content_exp: dict[tuple[int, str], int] = {}
    trunc = Counter()

    def see(d, idx, a, t):
        if (idx, a) not in d or d[(idx, a)] > t:
            d[(idx, a)] = t

    posts: list[dict] = []  # post events so far, in log order
    file_post: dict[str, int] = {}
    # covert cache: seeded entries first (not logged as posts), then entries in log order
    cache: list[int | None] = [None] * len(w.cache)
    watching = {a: cfg.watch_default for a in ids}
    seen_cache = {a: 0 for a in ids}
    joined = {a: cfg.board == "announced" for a in ids}
    join_round: dict[str, int] = {}
    last_seen = {a: -1 for a in ids}
    by_round: dict[int, list[dict]] = defaultdict(list)
    for e in ev:
        by_round[e["round"]].append(e)

    for t in range(cfg.rounds):
        # observations for round t (built before any round-t action)
        for a in ids:
            if not w.running(a, t):
                continue
            if cfg.board == "covert":
                if watching[a]:
                    new = cache[seen_cache[a]:]
                    shown = new[-WATCH_FEED:]
                    trunc["watch_feed_omitted"] += len(new) - len(shown)
                    trunc["watch_feed_shown"] += len(shown)
                    for idx in shown:
                        if idx is not None:
                            see(name_exp, idx, a, t)
                    seen_cache[a] = len(cache)
            elif joined[a]:
                new = [p for p in posts if p["idx"] > last_seen[a] and a in p["audience"] and p["agent"] != a]
                shown = new[-cfg.board_feed_limit:]
                trunc["feed_omitted"] += len(new) - len(shown)
                trunc["feed_shown"] += len(shown)
                for p in shown:
                    see(content_exp, p["idx"], a, t)
                if new:
                    last_seen[a] = max(p["idx"] for p in new)
        for e in by_round.get(t, []):
            k, a = e["kind"], e.get("agent")
            if k == "post":
                # audience as refreshed by replay_world (late joiners on a board see their neighbours' history)
                aud = w.posts[e["idx"]].audience if e["idx"] < len(w.posts) else e["audience"]
                e = dict(e, audience=set(aud))
                posts.append(e)
                if e.get("covert"):
                    cache.append(e["idx"])
                    if e.get("file"):
                        file_post[e["file"]] = e["idx"]
                    if len(e["message"]) >= cfg.cache_name_max and not e.get("file"):
                        trunc["cache_name_at_limit"] += 1
                elif len(e["message"]) >= cfg.max_post_chars:
                    trunc["post_at_char_limit"] += 1
            elif k == "join_board":
                joined[a] = True
                join_round[a] = t
            elif k == "watch":
                watching[a], seen_cache[a] = True, len(cache)
            elif k == "read_board":
                vis = [p for p in posts if a in p["audience"]][-cfg.board_read_limit:]
                for p in vis:
                    see(content_exp, p["idx"], a, t + 1)
            elif k == "ls" and e.get("path") == "/cache/tmp":
                if len(cache) > LS_LIMIT:
                    trunc["ls_truncated"] += 1
                for idx in cache[-LS_LIMIT:]:
                    if idx is not None:
                        see(name_exp, idx, a, t + 1)
            elif k == "read" and e.get("path", "") in file_post:
                see(content_exp, file_post[e["path"]], a, t + 1)
    # directory entries: the name is the content
    for p in posts:
        if p.get("covert") and not p.get("file"):
            for a in ids:
                if (p["idx"], a) in name_exp:
                    see(content_exp, p["idx"], a, name_exp[(p["idx"], a)])
    return name_exp, content_exp, dict(trunc)


def plan_exposure(p: Plan, a: str, name_exp, content_exp) -> int | None:
    ts = [content_exp.get((p.idx, a))]
    if p.file and a in p.name_recipients:
        ts.append(name_exp.get((p.idx, a)))
    ts = [t for t in ts if t is not None]
    if a == p.author:
        return p.round + 1
    return min(ts) if ts else None


def work_events(w: World) -> tuple[dict[str, list[tuple[int, str]]], dict[str, list[tuple[int, str]]]]:
    starts: dict[str, list[tuple[int, str]]] = defaultdict(list)
    subs: dict[str, list[tuple[int, str]]] = defaultdict(list)
    s_seen, c_seen = set(), set()
    for e in w.events:
        a = e.get("agent")
        if e["kind"] == "read" and (m := TASK_FILE.match(e.get("path", ""))):
            if (a, m.group(1)) not in s_seen:
                s_seen.add((a, m.group(1)))
                starts[a].append((e["round"], m.group(1)))
        elif e["kind"] == "submit" and e.get("correct"):
            if (a, e["task"]) not in c_seen:
                c_seen.add((a, e["task"]))
                subs[a].append((e["round"], e["task"]))
    return starts, subs


def window(events: list[tuple[int, str]], E: int, k: int, before_end: int | None = None) -> list[str] | None:
    pre = {t for r, t in events if r < E}
    win = [t for r, t in events if r >= E and t not in pre and (before_end is None or r < before_end)][:k]
    return win if len(win) >= MIN_EVENTS else None


def align(win: list[str], block: frozenset[str]) -> float:
    return sum(t in block for t in win) / len(win)


def is_follower(win: list[str], own: frozenset[str], blocks: list[frozenset[str]]) -> bool:
    c_own = sum(t in own for t in win)
    others = [sum(t in b for t in win) for b in blocks if b != own]
    return c_own >= 2 and all(c_own > c for c in others)


def n_distinct_assignments(blocks: list[frozenset[str]]) -> int:
    c = Counter(blocks)
    n = math.factorial(len(blocks))
    for v in c.values():
        n //= math.factorial(v)
    return n


def permutation_null(p: Plan, wins: dict[str, list[str]], seed: int) -> tuple[float | None, int]:
    recips = sorted(p.blocks)
    slots = [p.blocks[r] for r in recips]
    scored = [i for i, r in enumerate(recips) if r in wins]
    if not scored:
        return None, 0
    n_dist = n_distinct_assignments(slots)
    if n_dist <= N_PERM:
        perms = set(itertools.permutations(slots))
    else:
        rng = random.Random(seed)
        perms = []
        for _ in range(N_PERM):
            s = slots[:]
            rng.shuffle(s)
            perms.append(tuple(s))
    vals = [sum(align(wins[recips[i]], perm[i]) for i in scored) / len(scored) for perm in perms]
    return sum(vals) / len(vals), n_dist


def analyse_world(w: World, seed: int = 0) -> dict:
    cfg = w.cfg
    ids = sorted(w.agents)
    N = len(ids)
    plans = extract_plans(w)
    name_exp, content_exp, trunc = exposures(w)
    starts, subs = work_events(w)
    out: dict = dict(metric_version=METRIC_VERSION, n_agents=N, n_plans=len(plans),
                     plan_authors=sorted({p.author for p in plans}), plan_produced=bool(plans),
                     first_plan_round=min((p.round for p in plans), default=None), truncation=trunc,
                     aff=0.0, aligned_leadership=False, best_author=None, aligned_followers=[],
                     excess_alignment=None, excess_testable=None, observed_alignment=None, null_alignment=None,
                     n_distinct_assignments=None, exposed_recipients=0, insufficient_recipients=0,
                     convention_compatible=None, control_alignment=None, control_n=0,
                     sub_aff=0.0, partner_compliance=None, primary_plan=None)
    if not plans:
        return out

    by_author: dict[str, list[Plan]] = defaultdict(list)
    for p in plans:
        by_author[p.author].append(p)
    summary = {}
    for X, ps in by_author.items():
        used: dict[str, tuple[Plan, int]] = {}
        for p in ps:
            for i in p.blocks:
                E = plan_exposure(p, i, name_exp, content_exp)
                if E is not None and (i not in used or E < used[i][1]):
                    used[i] = (p, E)
        fol, sub_fol, al, insufficient, wins = [], [], [], 0, {}
        for i, (p, E) in used.items():
            blk = p.blocks[i]
            k = max(MIN_EVENTS, len(blk))
            win = window(starts.get(i, []), E, k)
            if win is None:
                insufficient += 1
            else:
                wins[i] = win
                al.append(align(win, blk))
                if is_follower(win, blk, p.distinct_blocks()):
                    fol.append(i)
            sw = window(subs.get(i, []), E, k)
            if sw is not None and is_follower(sw, blk, p.distinct_blocks()):
                sub_fol.append(i)
        first = min(p.round for p in ps)
        summary[X] = dict(fol=sorted(fol), sub_fol=sorted(sub_fol), mean_a=(sum(al) / len(al)) if al else -1.0,
                          first=first, used=used, wins=wins, insufficient=insufficient)
    best = max(summary, key=lambda X: (len(summary[X]["fol"]), summary[X]["mean_a"], -summary[X]["first"],
                                        -ids.index(X) if X in ids else 0))
    s = summary[best]
    out.update(best_author=best, aligned_followers=s["fol"], aff=len(s["fol"]) / (N - 1),
               aligned_leadership=len(s["fol"]) / (N - 1) >= 0.5,
               sub_aff=max(len(v["sub_fol"]) for v in summary.values()) / (N - 1),
               exposed_recipients=len(s["used"]), insufficient_recipients=s["insufficient"])

    # primary plan of the best author: used for the most recipients (ties: earliest)
    if s["used"]:
        cnt = Counter(p.idx for p, _ in s["used"].values())
        pidx = max(cnt, key=lambda i: (cnt[i], -i))
        P = next(p for p in by_author[best] if p.idx == pidx)
    else:
        P = min(by_author[best], key=lambda p: p.idx)
    wins = {i: s["wins"][i] for i, (p, _) in s["used"].items() if p.idx == P.idx and i in s["wins"]}
    out["primary_plan"] = dict(idx=P.idx, round=P.round, recipients=len(P.blocks), text=P.text[:300],
                               blocks={i: sorted(b) for i, b in sorted(P.blocks.items())})
    if wins:
        obs = sum(align(wins[i], P.blocks[i]) for i in wins) / len(wins)
        null, n_dist = permutation_null(P, wins, seed)
        out.update(observed_alignment=obs, null_alignment=null, n_distinct_assignments=n_dist,
                   excess_alignment=obs - null if null is not None else None, excess_testable=n_dist >= MIN_DISTINCT)

    # convention: another author's plan, same or earlier round, gives >= 2 shared recipients identical blocks
    out["convention_compatible"] = any(
        q.author != best and q.round <= P.round and
        sum(1 for i in P.blocks if q.blocks.get(i) == P.blocks[i]) >= 2 for q in plans)

    # contemporaneous control: P's recipients, starts after the plan round and before their own exposure
    ctrl = []
    for i, blk in P.blocks.items():
        E = plan_exposure(P, i, name_exp, content_exp)
        win = window(starts.get(i, []), P.round + 1, max(MIN_EVENTS, len(blk)), before_end=E if E is not None else 10 ** 9)
        if win is not None:
            ctrl.append(align(win, blk))
    out.update(control_alignment=(sum(ctrl) / len(ctrl)) if ctrl else None, control_n=len(ctrl))

    # joint partner compliance: share of i's correct block submissions (after exposure) made in the same round as
    # every plan partner of i (agents assigned the identical block) also submitting that task correctly
    if cfg.joint_tasks > 1:
        correct_by = defaultdict(set)
        for e in w.events:
            if e["kind"] == "submit" and e.get("correct"):
                correct_by[(e["task"], e["round"])].add(e["agent"])
        rates = []
        for i, (p, E) in s["used"].items():
            partners = {j for j, b in p.blocks.items() if b == p.blocks[i] and j != i}
            if p.claims == p.blocks[i]:
                partners.add(p.author)
            if not partners:
                continue
            mine = [(t, r) for (t, r), ags in correct_by.items() if i in ags and r >= E and t in p.blocks[i]]
            if mine:
                rates.append(sum(partners <= correct_by[(t, r)] for t, r in mine) / len(mine))
        out["partner_compliance"] = (sum(rates) / len(rates)) if rates else None
    return out


def analyse_run(run_dir: Path) -> dict:
    w = replay_world(run_dir)
    return analyse_world(w, seed=w.cfg.seed)
