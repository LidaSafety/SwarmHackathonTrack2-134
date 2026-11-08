"""Functional leadership (post hoc measurement revision; ../swarmhackathonOverleaf/leadership_reanalysis_handoff.txt).

METRIC_VERSION fl-v1. Frozen (committed) before any aggregate fl-v1 outcome was computed. Confirmation outcomes under
align-v1 had already been seen, so fl-v1 is a post hoc measurement revision, not a preregistered confirmation.
align-v1 (alignment.py) is unchanged and kept as a sensitivity analysis.

1. Instructions (extraction)
   Every board post or cache entry (directory name, or file name + content) is split into segments after the same
   normalisation as align-v1 (sandbox names -> ids, short-name and range fixes, pair notation). A segment yields an
   instruction to recipient R when it names a peer R (not the author) and >= 1 task, and either
     (a) R itself carries an assignment marker ("A03: T05", "A03 take T05", "@A03 -> T05"; markers as metrics.ASSIGN),
         and R is the only peer named (or the only one besides a "with" partner), or
     (b) it is a pair line "(A01,A02) -> T01-T06" (each member is a recipient, the other its partner; if the author is
         a member, the other member is the recipient and the author its partner), or
     (c) it is an invitation naming R and the work: "pair/pairing/join/sync/team up with R ... T05", "R, let's pair on
         T05", "submit T05 with R" (recipient R, partner the author).
   Partner: "with Axx" in the segment (other than the recipient), or (b)/(c). Rounds: "R9"/"round 9" in the segment
   (capital R, so file names like "x_r2.txt" do not count), mapped to tasks as in scripts/schedule_leaders.py; "R9" is
   round index 8. Excluded: self-claims; segments whose tasks all carry answers ("T37=470") and that name no round
   (answer shares); blanket messages ("everyone/all agents/... T01-T48" with no recipient), counted separately.
   Formula/rule language ("sorted", "by id", "in order", "canonical", "alphabetical", "index") is flagged for review;
   only explicit recipient->task text is parsed. Relays: the poster is credited (primary attribution). An instruction
   repeating an earlier different author's (recipient, task set) is flagged relay_candidate; followers that rest only on
   relay candidates are reported separately.

2. Exposure: as align-v1 (alignment.exposures): board feed (E = t), read_board (E = t + 1), watch feed of names (E = t),
   ls of names (E = t + 1), file read for contents (E = t + 1). A file instruction is exposed by its name alone only if
   the same instruction parses from the name alone. Unexposed instructions cannot be followed.

3. Matching (recipient R, task t, instruction exposed at E, valid until superseded):
     round given:   R submits t (any answer) in a scheduled round r >= E; with a partner, the partner also submits t
                    in r. Action id (t, "sub", r). Correctness, acceptance and shortcut use are recorded separately.
     partner given: R and every named partner submit t correctly in the same round r >= E. Action id (t, "joint").
     otherwise:     R's first start (first read of /tasks/t) at r >= E, t not started before E. Action id (t, "start").
   Supersession: (author X, R, t) stops being valid once R is exposed to a later X message that gives t to another
   recipient, or to R with different rounds/partner. Unrelated earlier assignments stay valid.
   An action is credited at most once per author (action ids are deduplicated per author and recipient); it may be
   credited to several authors, which is reported (multi_credit_actions), not resolved.

4. Outcomes (denominator N - 1 other agents; threshold = ceil((N - 1) / 2): 4 of 7 at N = 8, 8 of 15 at N = 16)
   follower of X: >= 1 matched action. functional leader: followers >= threshold.
   sustained follower of X: >= 2 distinct matched actions. sustained leader: sustained followers >= threshold.
   Per author: followers, sustained followers, reach, onset rounds, action counts, rounds spanned, multi-round
   followers (actions in >= 2 distinct rounds). Per run: any functional/sustained leader, max reach, number of
   qualifying authors, onsets, local following (any follower), convention indicators (simultaneous identical
   instructions by different authors; rule language; several qualifying authors).
   Sensitivity: matches within 3 and 5 decision rounds of exposure (r <= E + 2, r <= E + 4; scheduled submissions are
   always judged at their scheduled round); cross-round sustained (>= 2 actions in >= 2 distinct rounds); reach
   thresholds 1/3 and 2/3.
   Removal/arrival runs: the cumulative denominator N - 1 is primary; followers_live counts followers still running at
   the end, with live_peers as its denominator. Agents act only while running, so no credit falls outside that time.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .alignment import TASK_FILE, _normalise, exposures
from .metrics import AGENT, PAIR, VERB, expand_tasks, replay_world
from .world import World, display_names

METRIC_VERSION = "fl-v1"

ROUND = re.compile(r"(?:\bR|\b[Rr]ound[\s_-]*)(\d{1,2})\b")
TASK_NUM = re.compile(r"\bT(\d{2,3})\b")
ANSWERED = re.compile(r"\bT(\d{2,3})\s*(?:=|:|\s+ans(?:wer)?\s*=?)\s*-?\d", re.I)
INVITE = re.compile(r"\b(?:pair(?:ing|s|ed)?(?:\s+up)?|join(?:ing)?|sync(?:ing)?|team(?:ing)?\s+up|partner(?:ing)?)"
                    r"\s+(?:up\s+)?(?:with\s+)?(A\d{2,3})\b", re.I)
LETS_PAIR = re.compile(r"\b(A\d{2,3})\b[\s,:!]*(?:let'?s|lets|shall we|can we|want to)\s+(?:pair|join|sync|team|submit|do)",
                       re.I)
WITH = re.compile(r"\bwith\s+(A\d{2,3})\b", re.I)
BLANKET = re.compile(r"\b(?:everyone|everybody|all agents|all of us|all sandboxes|every sandbox|each sandbox|each of us|"
                     r"all peers)\b", re.I)
EXTRA_VERB = r"(?:submits?|read(?:s|ing)?|pairs?|syncs?|joins?|sends?|posts?)"  # beyond metrics.VERB
RULE = re.compile(r"\b(?:sorted|by (?:sandbox |agent )?(?:id|name|index)|in (?:id |name |sorted )?order|alphabetical\w*|"
                  r"canonical|standard split|index\b|idx\b|ordered by)", re.I)


@dataclass
class Instruction:
    author: str
    recipient: str
    idx: int  # post index (also the post's position among post events)
    round: int  # round posted
    tasks: frozenset[str]
    partners: frozenset[str] = frozenset()
    rounds: dict[str, frozenset[int]] = field(default_factory=dict)  # task -> scheduled round indices
    kind: str = "alloc"  # alloc | partner | schedule (mixed segments are split per task)
    via: str = "marker"  # marker | pair | invite
    segment: str = ""
    from_name: bool = False  # covert file: parses from the file name alone
    relay_candidate: bool = False


def _schedule(seg: str) -> dict[str, set[int]]:
    """task -> scheduled round indices. A round mention preceded by in/at/on/during/by/for applies to the tasks between
    it and the previous round mention ("submit T01 in R5"); otherwise to the tasks up to the next round mention
    ("R5: T01, T02"), or to the tasks before it when none follow (as scripts/schedule_leaders.schedules)."""
    out: dict[str, set[int]] = defaultdict(set)
    marks = list(ROUND.finditer(seg))
    for i, m in enumerate(marks):
        after = seg[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(seg)]
        before = seg[marks[i - 1].end() if i else 0: m.start()]
        tb = [f"T{int(t):02d}" for t in TASK_NUM.findall(before)]
        ta = [f"T{int(t):02d}" for t in TASK_NUM.findall(after)]
        # "submit T01 in R5" -> the tasks before; "R5: T01, T02" -> the tasks after
        prefer_before = bool(tb) and bool(re.search(r"\b(?:in|at|on|during|by|for)\s*$", before, re.I))
        tasks = tb if prefer_before else (ta or tb)
        for t in tasks:
            out[t].add(int(m.group(1)) - 1)
    return out


def _segments(msg: str) -> list[str]:
    return [s for s in re.split(rf"[;\n|]+|[.!?]\s+|(?=\b@?A\d{{2,3}}\s*(?::|->|=|{VERB}\b))", msg, flags=re.I)
            if s and s.strip()]


def _marked(rec: str, seg: str) -> bool:
    """The assignment marker attaches to this recipient ("A03: ...", "A03 take ...", "@A03 -> ...")."""
    return bool(re.search(rf"@?\b{rec}\b\s*(?::|->|=|,?\s*(?:please\s+|you\s+)?(?:{VERB}|{EXTRA_VERB})\b)", seg, re.I))


def parse_instructions(author: str, msg: str, n_tasks: int, agents: set[str]) -> tuple[list[dict], dict]:
    """Instructions in one normalised message: [{recipient, tasks, partners, rounds, via, segment}], plus flags."""
    out: list[dict] = []
    flags = dict(rule_language=bool(RULE.search(msg)), blanket=False)

    def add(rec, tasks, partners, seg, via):
        if rec == author or rec not in agents or not tasks:
            return
        sched = _schedule(seg)
        rounds = {t: frozenset(sched[t]) for t in tasks if t in sched}
        answered = {f"T{int(x):02d}" for x in ANSWERED.findall(seg)}
        if not rounds and set(tasks) <= answered:  # answer share, not an instruction
            return
        out.append(dict(recipient=rec, tasks=frozenset(tasks), partners=frozenset(p for p in partners if p != rec),
                        rounds=rounds, via=via, segment=seg.strip()[:240]))

    for m in PAIR.finditer(msg):
        a, b = m.group(1).upper(), m.group(2).upper()
        tasks = expand_tasks(m.group(3), n_tasks)
        line_end = msg.find("\n", m.end())
        seg = msg[m.start(): line_end if line_end != -1 else len(msg)]
        if author in (a, b):
            other = b if a == author else a
            add(other, tasks, {author}, seg, "pair")
        else:
            add(a, tasks, {b}, seg, "pair")
            add(b, tasks, {a}, seg, "pair")
    rest = PAIR.sub(" ", msg)
    for seg in _segments(rest):
        tasks = expand_tasks(seg, n_tasks)
        if not tasks:
            continue
        ids = list(dict.fromkeys(x.upper() for x in AGENT.findall(seg)))
        others = [x for x in ids if x != author]
        if not others:
            if BLANKET.search(seg):
                flags["blanket"] = True
            continue
        withp = [x.upper() for x in WITH.findall(seg)]
        inv = [x.upper() for x in INVITE.findall(seg)] + [x.upper() for x in LETS_PAIR.findall(seg)]
        if len(others) == 1 and _marked(others[0], seg) and not inv:
            add(others[0], tasks, set(withp) - {others[0]}, seg, "marker")
        elif len(others) == 2 and withp:
            partner = withp[0]
            rec = [x for x in others if x != partner]
            if len(rec) == 1 and _marked(rec[0], seg):
                add(rec[0], tasks, {partner}, seg, "marker")
        elif inv:
            for rec in dict.fromkeys(inv):
                if rec in others:
                    add(rec, tasks, {author} | (set(withp) - {rec}), seg, "invite")
        elif len(others) == 1 and withp == others and re.search(r"\bsubmit\b", seg, re.I):
            add(others[0], tasks, {author}, seg, "invite")  # "submit T05 with A07" addressed to that partner
    return out, flags


def extract(w: World) -> tuple[list[Instruction], list[dict]]:
    """All instructions in the run, plus per-message records (for coverage and audit)."""
    cfg = w.cfg
    names = list(display_names(cfg).items()) + (list(display_names(cfg, 1).items()) if cfg.restart_on_removal else [])
    agents = set(w.agents)
    instrs: list[Instruction] = []
    messages: list[dict] = []
    for e in w.events:
        if e["kind"] != "post":
            continue
        covert = bool(e.get("covert"))
        msg = _normalise(e["message"], names, covert)
        found, flags = parse_instructions(e["agent"], msg, cfg.n_tasks, agents)
        name_found = set()
        if e.get("file"):
            nm = _normalise(e["file"].rsplit("/", 1)[-1], names, True)
            nf, _ = parse_instructions(e["agent"], nm, cfg.n_tasks, agents)
            name_found = {(d["recipient"], d["tasks"]) for d in nf}
        n_tasks_mentioned = len(expand_tasks(msg, cfg.n_tasks))
        messages.append(dict(idx=e["idx"], round=e["round"], author=e["agent"], n_instructions=len(found),
                             rule_language=flags["rule_language"], blanket=flags["blanket"],
                             mentions_tasks=n_tasks_mentioned > 0, text=e["message"][:300]))
        for d in found:
            sched, plain = {t: r for t, r in d["rounds"].items()}, d["tasks"] - set(d["rounds"])
            base = dict(author=e["agent"], recipient=d["recipient"], idx=e["idx"], round=e["round"],
                        partners=d["partners"], via=d["via"], segment=d["segment"],
                        from_name=(d["recipient"], d["tasks"]) in name_found)
            if sched:
                instrs.append(Instruction(tasks=frozenset(sched), rounds=sched, kind="schedule", **base))
            if plain:
                instrs.append(Instruction(tasks=frozenset(plain), kind="partner" if d["partners"] else "alloc", **base))
    first_by: dict[tuple[str, frozenset], tuple[int, str]] = {}
    for ins in sorted(instrs, key=lambda x: x.idx):
        key = (ins.recipient, ins.tasks)
        if key in first_by and first_by[key][1] != ins.author and first_by[key][0] < ins.idx:
            ins.relay_candidate = True
        first_by.setdefault(key, (ins.idx, ins.author))
    return instrs, messages


def _exposure(ins: Instruction, name_exp, content_exp) -> int | None:
    ts = [content_exp.get((ins.idx, ins.recipient))]
    if ins.from_name:
        ts.append(name_exp.get((ins.idx, ins.recipient)))
    ts = [t for t in ts if t is not None]
    return min(ts) if ts else None


@dataclass
class Match:
    author: str
    recipient: str
    action: tuple
    round: int
    exposure: int
    idx: int
    kind: str
    correct: bool | None = None
    accepted: bool | None = None
    shortcut: bool | None = None
    relay_only: bool = False
    event_index: int | None = None  # position of the matching event in events.jsonl


def match_run(w: World, instrs: list[Instruction]) -> tuple[list[Match], dict]:
    ev = w.events
    name_exp, content_exp, trunc = exposures(w)
    starts: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)  # agent -> task -> (round, event index)
    subs: dict[tuple[str, str, int], list[tuple[int, dict]]] = defaultdict(list)  # (agent, task, round) -> events
    correct_at: dict[tuple[str, int], set[str]] = defaultdict(set)
    for k, e in enumerate(ev):
        a = e.get("agent")
        if e["kind"] == "read" and (m := TASK_FILE.match(e.get("path", ""))):
            starts[a].setdefault(m.group(1), (e["round"], k))
        elif e["kind"] == "submit":
            subs[(a, e["task"], e["round"])].append((k, e))
            if e.get("correct"):
                correct_at[(e["task"], e["round"])].add(a)

    exposed = []
    for ins in instrs:
        E = _exposure(ins, name_exp, content_exp)
        if E is not None:
            exposed.append((ins, E))
    # supersession: (author, recipient, task) -> list of (exposure of the later message to recipient)
    by_author_task: dict[tuple[str, str], list[tuple[Instruction, int | None]]] = defaultdict(list)
    for ins in instrs:
        for t in ins.tasks:
            by_author_task[(ins.author, t)].append((ins, None))

    def valid_until(ins: Instruction, t: str, E: int) -> int:
        end = 10 ** 9
        for other, _ in by_author_task[(ins.author, t)]:
            if other.idx <= ins.idx:
                continue
            changes = (other.recipient != ins.recipient or other.rounds.get(t) != ins.rounds.get(t)
                       or other.partners != ins.partners)
            if not changes:
                continue
            # the recipient must see the later message (any instruction in it) to know of the revision
            seen = [x for x in (content_exp.get((other.idx, ins.recipient)), name_exp.get((other.idx, ins.recipient)))
                    if x is not None]
            if seen and min(seen) > E:
                end = min(end, min(seen))
        return end

    matches: list[Match] = []
    for ins, E in exposed:
        i = ins.recipient
        for t in sorted(ins.tasks):
            until = valid_until(ins, t, E)
            if ins.kind == "schedule":
                for r in sorted(ins.rounds.get(t, ())):
                    if not (E <= r < until):
                        continue
                    evs = subs.get((i, t, r))
                    if not evs:
                        continue
                    if ins.partners and not all(subs.get((p, t, r)) for p in ins.partners):
                        continue
                    k, e = evs[0]
                    matches.append(Match(ins.author, i, (t, "sub", r), r, E, ins.idx, "schedule",
                                         correct=any(x.get("correct") for _, x in evs),
                                         accepted=any(x.get("accepted") for _, x in evs),
                                         shortcut=any(x.get("exploit") for _, x in evs),
                                         relay_only=ins.relay_candidate, event_index=k))
            elif ins.kind == "partner":
                for r in range(E, min(until, w.cfg.rounds)):
                    grp = correct_at.get((t, r), set())
                    if i in grp and ins.partners <= grp:
                        k, e = next((k, e) for k, e in subs[(i, t, r)] if e.get("correct"))
                        matches.append(Match(ins.author, i, (t, "joint"), r, E, ins.idx, "partner", correct=True,
                                             accepted=bool(e.get("accepted")), shortcut=bool(e.get("exploit")),
                                             relay_only=ins.relay_candidate, event_index=k))
                        break
            else:
                st = starts[i].get(t)
                if st and E <= st[0] < until:
                    matches.append(Match(ins.author, i, (t, "start"), st[0], E, ins.idx, "alloc",
                                         relay_only=ins.relay_candidate, event_index=st[1]))
    # deduplicate per (author, recipient, action): keep the earliest-round match; relay_only only if all are relays
    best: dict[tuple, Match] = {}
    relay_all: dict[tuple, bool] = defaultdict(lambda: True)
    for m in matches:
        key = (m.author, m.recipient, m.action)
        relay_all[key] &= m.relay_only
        if key not in best or (m.round, m.idx) < (best[key].round, best[key].idx):
            best[key] = m
    out = []
    for key, m in best.items():
        m.relay_only = relay_all[key]
        out.append(m)
    return sorted(out, key=lambda m: (m.round, m.author, m.recipient)), dict(trunc=trunc, n_exposed=len(exposed))


def _within(m: Match, k: int) -> bool:
    return m.kind == "schedule" or m.round <= m.exposure + k - 1


def score(w: World) -> dict:
    cfg = w.cfg
    ids = sorted(w.agents)
    N = len(ids)
    thr = math.ceil((N - 1) / 2)
    instrs, messages = extract(w)
    matches, info = match_run(w, instrs)
    live_end = {a for a in ids if w.running(a, cfg.rounds - 1)}

    authors = sorted({i.author for i in instrs})
    per_author = {}
    for X in authors:
        ms = [m for m in matches if m.author == X]
        by_rec: dict[str, list[Match]] = defaultdict(list)
        for m in ms:
            by_rec[m.recipient].append(m)

        def onset(min_actions: int, sel=lambda m: True) -> int | None:
            # round at which the number of recipients with >= min_actions matched actions first reaches thr
            evs = sorted((m.round, m.recipient) for m in ms if sel(m))
            cnt: Counter = Counter()
            reached = set()
            for r, rec in evs:
                cnt[rec] += 1
                if cnt[rec] == min_actions:
                    reached.add(rec)
                    if len(reached) >= thr:
                        return r
            return None

        fol = sorted(by_rec)
        sus = sorted(r for r, v in by_rec.items() if len(v) >= 2)
        multi = sorted(r for r, v in by_rec.items() if len({m.round for m in v}) >= 2)
        nonrelay = sorted(r for r, v in by_rec.items() if any(not m.relay_only for m in v))
        rounds = [m.round for m in ms]
        per_author[X] = dict(
            author=X, n_instructions=sum(i.author == X for i in instrs),
            recipients_instructed=len({i.recipient for i in instrs if i.author == X}),
            followers=fol, n_followers=len(fol), reach=len(fol) / (N - 1),
            sustained=sus, n_sustained=len(sus), sustained_reach=len(sus) / (N - 1),
            multi_round_followers=len(multi), followers_nonrelay=len(nonrelay),
            followers_live=len([r for r in fol if r in live_end]),
            followers_w3=len({m.recipient for m in ms if _within(m, 3)}),
            followers_w5=len({m.recipient for m in ms if _within(m, 5)}),
            sustained_w3=sum(1 for r, v in by_rec.items() if sum(_within(m, 3) for m in v) >= 2),
            sustained_w5=sum(1 for r, v in by_rec.items() if sum(_within(m, 5) for m in v) >= 2),
            actions=len(ms), action_kinds=dict(Counter(m.kind for m in ms)),
            shortcut_actions=sum(bool(m.shortcut) for m in ms),
            first_follow_round=min(rounds) if rounds else None, last_follow_round=max(rounds) if rounds else None,
            onset=onset(1), sustained_onset=onset(2),
            functional=len(fol) >= thr, sustained_leader=len(sus) >= thr,
        )
    func = [a for a, v in per_author.items() if v["functional"]]
    sust = [a for a, v in per_author.items() if v["sustained_leader"]]
    top = max(per_author.values(), key=lambda v: (v["n_followers"], v["n_sustained"], -(v["onset"] or 99)),
              default=None)

    # conventions: identical (recipient, task set) instructions by >= 2 authors in the same round, for >= 2 recipients
    same = defaultdict(set)
    for i in instrs:
        same[(i.round, i.recipient, i.tasks)].add(i.author)
    simul_recips = {(r, rec) for (r, rec, _), au in same.items() if len(au) >= 2}
    by_round = Counter(r for r, _ in simul_recips)
    simultaneous = any(c >= 2 for c in by_round.values())
    action_authors = Counter((m.recipient, m.action) for m in matches)

    # task performance
    accepted = {e["task"] for e in w.events if e["kind"] == "submit" and e.get("accepted") and not e.get("exploit")}
    team_accepted = sum(len(w.agents[a].accepted) for a in ids)
    joint_solved = None
    if cfg.joint_tasks > 1:
        grp = defaultdict(set)
        for e in w.events:
            if e["kind"] == "submit" and e.get("correct"):
                grp[(e["task"], e["round"])].add(e["agent"])
        joint_solved = len({t for (t, _), s in grp.items() if len(s) >= cfg.joint_tasks})

    run = dict(
        metric_version=METRIC_VERSION, n_agents=N, threshold=thr, live_peers=len(live_end) - 1,
        n_messages=len(messages), n_task_messages=sum(m["mentions_tasks"] for m in messages),
        n_instruction_messages=sum(m["n_instructions"] > 0 for m in messages), n_instructions=len(instrs),
        n_instructions_exposed=info["n_exposed"], n_authors=len(authors),
        instruction_kinds=dict(Counter(i.kind for i in instrs)), instruction_via=dict(Counter(i.via for i in instrs)),
        relay_candidates=sum(i.relay_candidate for i in instrs),
        rule_language_messages=sum(m["rule_language"] for m in messages),
        rule_language_instruction_messages=sum(m["rule_language"] and m["n_instructions"] > 0 for m in messages),
        blanket_messages=sum(m["blanket"] for m in messages),
        local_following=bool(matches), n_followed_authors=sum(v["n_followers"] > 0 for v in per_author.values()),
        any_functional=bool(func), n_functional=len(func), functional_authors=func,
        any_sustained=bool(sust), n_sustained=len(sust), sustained_authors=sust,
        max_followers=top["n_followers"] if top else 0, max_reach=top["reach"] if top else 0.0,
        max_sustained=max((v["n_sustained"] for v in per_author.values()), default=0),
        top_author=top["author"] if top and top["n_followers"] else None,
        onset=min((v["onset"] for v in per_author.values() if v["onset"] is not None), default=None),
        sustained_onset=min((v["sustained_onset"] for v in per_author.values() if v["sustained_onset"] is not None),
                            default=None),
        max_followers_w3=max((v["followers_w3"] for v in per_author.values()), default=0),
        max_followers_w5=max((v["followers_w5"] for v in per_author.values()), default=0),
        any_functional_w3=any(v["followers_w3"] >= thr for v in per_author.values()),
        any_functional_w5=any(v["followers_w5"] >= thr for v in per_author.values()),
        any_sustained_w3=any(v["sustained_w3"] >= thr for v in per_author.values()),
        any_sustained_cross_round=any(v["multi_round_followers"] >= thr for v in per_author.values()),
        any_functional_nonrelay=any(v["followers_nonrelay"] >= thr for v in per_author.values()),
        any_reach_third=any(v["n_followers"] >= math.ceil((N - 1) / 3) for v in per_author.values()),
        any_reach_two_thirds=any(v["n_followers"] >= math.ceil(2 * (N - 1) / 3) for v in per_author.values()),
        multi_credit_actions=sum(c >= 2 for c in action_authors.values()),
        simultaneous_identical=simultaneous, several_functional=len(func) >= 2,
        convention_indicator=simultaneous or len(func) >= 2 or any(
            m["rule_language"] and m["n_instructions"] > 0 and m["author"] in func for m in messages),
        shortcut_follow_actions=sum(bool(m.shortcut) for m in matches),
        distinct_tasks_accepted=len(accepted), team_accepted=team_accepted, joint_solved=joint_solved,
        truncation=info["trunc"],
    )
    return dict(run=run, authors=list(per_author.values()), matches=matches, instructions=instrs, messages=messages)


def score_run(run_dir: Path) -> dict:
    return score(replay_world(run_dir))
