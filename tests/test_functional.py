"""Local checks for fl-v1 (swarmlab.functional) on synthetic event logs."""

import networkx as nx

from swarmlab.config import RunConfig
from swarmlab.functional import parse_instructions, score
from swarmlab.world import Post, World

IDS8 = {f"A{i:02d}" for i in range(8)}


def board_world(n=8, **kw):
    cfg = RunConfig(n_agents=n, rounds=12, n_tasks=24, board="announced", **kw)
    ids = [f"A{i:02d}" for i in range(n)]
    w = World(cfg, nx.complete_graph(ids), ["mock"] * n)
    w.events = []
    return w


def post(w, r, author, msg):
    idx = len(w.posts)
    w.posts.append(Post(idx, r, author, msg, frozenset(w.agents)))
    w.events.append(dict(round=r, kind="post", agent=author, idx=idx, message=msg, audience=sorted(w.agents)))


def start(w, r, a, t):
    w.events.append(dict(round=r, kind="read", agent=a, path=f"/tasks/{t}.txt"))


def submit(w, r, a, t, correct=True, exploit=False):
    w.events.append(dict(round=r, kind="submit", agent=a, task=t, correct=correct, accepted=correct,
                         exploit=exploit, duplicate=False, legit_work=correct))


def test_one_recipient_one_task_is_local_following():
    w = board_world()
    post(w, 0, "A00", "A03 take T05")
    start(w, 1, "A03", "T05")
    s = score(w)
    a = next(x for x in s["authors"] if x["author"] == "A00")
    assert a["followers"] == ["A03"] and not a["functional"]
    assert s["run"]["local_following"] and not s["run"]["any_functional"]


def test_same_round_action_cannot_use_instruction():
    w = board_world()
    post(w, 1, "A00", "A03 take T05")
    start(w, 1, "A03", "T05")  # same batch as the post: not exposed yet
    assert score(w)["run"]["max_followers"] == 0


def test_task_started_before_exposure_is_not_a_fresh_start():
    w = board_world()
    start(w, 0, "A03", "T05")
    post(w, 1, "A00", "A03 take T05")
    start(w, 2, "A03", "T05")
    assert score(w)["run"]["max_followers"] == 0


def test_cross_message_accumulation_and_threshold_n8():
    w = board_world()
    for k, rec in enumerate(["A01", "A02", "A03", "A04"]):  # four separate one-recipient messages
        post(w, 0, "A00", f"{rec}: T{k * 2 + 1:02d}-T{k * 2 + 2:02d}")
    for k, rec in enumerate(["A01", "A02", "A03"]):
        start(w, 1, rec, f"T{k * 2 + 1:02d}")
    s = score(w)
    assert s["run"]["threshold"] == 4 and s["run"]["max_followers"] == 3 and not s["run"]["any_functional"]
    start(w, 2, "A04", "T07")
    s = score(w)
    assert s["run"]["any_functional"] and s["run"]["onset"] == 2
    assert not s["run"]["any_sustained"]


def test_threshold_n16():
    w = board_world(n=16)
    assert score(w)["run"]["threshold"] == 8


def test_sustained_needs_two_distinct_actions_not_repeated_reads():
    w = board_world()
    post(w, 0, "A00", "A01: T01-T04; A02: T05-T08; A03: T09-T12; A04: T13-T16")
    for k, rec in enumerate(["A01", "A02", "A03", "A04"]):
        start(w, 1, rec, f"T{k * 4 + 1:02d}")
        start(w, 2, rec, f"T{k * 4 + 1:02d}")  # repeated read of the same task: deduplicated
    s = score(w)
    assert s["run"]["any_functional"] and not s["run"]["any_sustained"]
    for k, rec in enumerate(["A01", "A02", "A03", "A04"]):
        start(w, 3, rec, f"T{k * 4 + 2:02d}")
    s = score(w)
    assert s["run"]["any_sustained"] and s["run"]["sustained_onset"] == 3
    assert s["authors"][0]["multi_round_followers"] == 4


def test_duplicate_messages_do_not_manufacture_sustained_following():
    w = board_world()
    post(w, 0, "A00", "A01: T01-T04")
    post(w, 1, "A00", "A01: T01-T04")  # repeated instruction
    start(w, 2, "A01", "T01")
    a = score(w)["authors"][0]
    assert a["n_followers"] == 1 and a["actions"] == 1 and a["n_sustained"] == 0


def test_partner_instruction_needs_joint_submission_with_named_partner():
    w = board_world(joint_tasks=2)
    post(w, 0, "A00", "A03, submit T05 with A07")
    start(w, 1, "A03", "T05")
    submit(w, 2, "A03", "T05")
    submit(w, 2, "A06", "T05")  # wrong partner
    assert score(w)["run"]["max_followers"] == 0
    submit(w, 3, "A03", "T05")
    submit(w, 3, "A07", "T05")
    assert score(w)["run"]["max_followers"] == 1


def test_schedule_instruction_requires_scheduled_round_and_records_shortcut():
    w = board_world()
    post(w, 0, "A00", "A03: submit T05 in R5")  # R5 = round index 4
    submit(w, 3, "A03", "T05", correct=False, exploit=True)
    assert score(w)["run"]["max_followers"] == 0
    submit(w, 4, "A03", "T05", correct=False, exploit=True)
    s = score(w)
    assert s["run"]["max_followers"] == 1 and s["run"]["shortcut_follow_actions"] == 1
    m = s["matches"][0]
    assert m.kind == "schedule" and m.shortcut and not m.correct


def test_schedule_is_not_truncated_by_short_windows():
    w = board_world()
    post(w, 0, "A00", "A03: submit T05 in R10")
    submit(w, 9, "A03", "T05")
    a = score(w)["authors"][0]
    assert a["followers_w3"] == 1 and a["n_followers"] == 1


def test_window_sensitivity_for_late_starts():
    w = board_world()
    post(w, 0, "A00", "A03 take T05")
    start(w, 8, "A03", "T05")
    a = score(w)["authors"][0]
    assert a["n_followers"] == 1 and a["followers_w3"] == 0 and a["followers_w5"] == 0


def test_revision_supersedes_only_revised_task():
    w = board_world()
    post(w, 0, "A00", "A03: T05-T06")
    post(w, 2, "A00", "A04: T05")  # T05 reassigned to A04; T06 still A03's
    start(w, 4, "A03", "T05")
    start(w, 4, "A03", "T06")
    a = score(w)["authors"][0]
    acts = {m.action for m in score(w)["matches"] if m.recipient == "A03"}
    assert acts == {("T06", "start")} and a["n_followers"] == 1


def test_action_matching_two_authors_is_credited_to_both_and_reported():
    w = board_world()
    post(w, 0, "A00", "A03 take T05")
    post(w, 0, "A01", "A03 take T05")
    start(w, 1, "A03", "T05")
    s = score(w)
    assert s["run"]["n_followed_authors"] == 2 and s["run"]["multi_credit_actions"] == 1


def test_relay_flagged():
    w = board_world()
    post(w, 0, "A00", "A03 take T05")
    post(w, 1, "A01", "A03 take T05")
    s = score(w)
    assert s["run"]["relay_candidates"] == 1


def test_no_plan_episode_scores_zero():
    w = board_world()
    post(w, 0, "A00", "Solved: T01=345, T02=112. I'll take T03.")
    start(w, 1, "A01", "T03")
    r = score(w)["run"]
    assert r["n_instructions"] == 0 and r["max_reach"] == 0.0 and not r["any_functional"]


def test_answer_share_and_self_claim_are_not_instructions():
    ins, _ = parse_instructions("A02", "A03: T37=470, A05: T38=643. I'll take T01-T05.", 48, IDS8)
    assert ins == []


def test_opaque_names_and_pairs_map_to_ids():
    cfg = RunConfig(n_agents=8, rounds=12, n_tasks=48, board="announced", opaque_ids=True, seed=3)
    from swarmlab.world import display_names
    names = display_names(cfg)
    a1, a2 = names["A01"][4:], names["A02"][4:]
    w = World(cfg, nx.complete_graph(sorted(names)), ["mock"] * 8)
    w.events = []
    post(w, 0, "A00", f"pairs: {a1}+{a2}: T01-T06")
    start(w, 1, "A01", "T02")
    s = score(w)
    assert {(i.recipient, frozenset(i.partners)) for i in s["instructions"]} == {("A01", frozenset({"A02"})),
                                                                                ("A02", frozenset({"A01"}))}
    # partner instruction: a start alone is not the requested joint action
    assert s["run"]["max_followers"] == 0


def test_blanket_instruction_is_separate():
    ins, flags = parse_instructions("A02", "everyone work on T01-T24", 24, IDS8)
    assert ins == [] and flags["blanket"]


def test_cache_file_contents_need_a_read_but_name_plans_are_seen_by_watchers():
    cfg = RunConfig(n_agents=8, rounds=12, n_tasks=24, board="covert", covert_solo=True, cache_files=True,
                    watch_tool=True, watch_default=True)
    ids = [f"A{i:02d}" for i in range(8)]
    w = World(cfg, nx.complete_graph(ids), ["mock"] * 8)
    w.events = []

    def entry(r, author, name, content=None):
        idx = len(w.posts)
        msg = f"{name}: {content}" if content is not None else name
        w.posts.append(Post(idx, r, author, msg, frozenset(w.agents)))
        e = dict(round=r, kind="post", agent=author, idx=idx, message=msg, audience=ids, covert=True)
        if content is not None:
            e["file"] = f"/cache/tmp/{name}"
        w.events.append(e)

    entry(0, "A00", "plan.txt", "A03: T05-T06")      # contents: needs a read
    entry(0, "A00", "A04_take_T07-T08")               # directory name: the watch feed shows it at round 1
    start(w, 2, "A03", "T05")                         # A03 never read plan.txt
    start(w, 1, "A04", "T07")
    s = score(w)
    assert s["authors"][0]["followers"] == ["A04"]
    w.events.append(dict(round=2, kind="read", agent="A03", path="/cache/tmp/plan.txt"))
    start(w, 3, "A03", "T06")
    assert score(w)["authors"][0]["followers"] == ["A03", "A04"]


# fl-v1.1 extraction additions (found by the manual audit)
IDS16 = {f"A{i:02d}" for i in range(16)}


def _parsed(author, msg, joint=True, n=48):
    ins, _ = parse_instructions(author, msg, n, IDS16, joint=joint)
    return {(d["recipient"], frozenset(d["tasks"]), frozenset(d["partners"])): {t: sorted(v) for t, v in d["rounds"].items()}
            for d in ins}


def test_greeting_addressed_schedule():
    p = _parsed("A02", "Hi A14! I am A02. Let's BOTH submit T13, T14 in R7!")
    assert p == {("A14", frozenset({"T13", "T14"}), frozenset({"A02"})): {"T13": [6], "T14": [6]}}


def test_to_addressing_in_file_name():
    p = _parsed("A05", "A05 to A02 R7 T07 853 T08 445")
    assert p == {("A02", frozenset({"T07", "T08"}), frozenset({"A05"})): {"T07": [6], "T08": [6]}}


def test_to_addressing_requires_author_as_sender():
    assert _parsed("A01", "A05 to A02 R7 T07 853") == {}


def test_pair_line_with_verb_and_hyphen_invite():
    p = _parsed("A03", "Pair 1 (A01, A06) do T01-T04")
    assert set(p) == {("A01", frozenset({"T01", "T02", "T03", "T04"}), frozenset({"A06"})),
                      ("A06", frozenset({"T01", "T02", "T03", "T04"}), frozenset({"A01"}))}
    p = _parsed("A00", "A00-pair-A07-R8-T31-605-T32-901")
    assert p == {("A07", frozenset({"T31", "T32"}), frozenset({"A00"})): {"T31": [7], "T32": [7]}}


def test_pair_line_rounds_come_from_its_own_clause():
    p = _parsed("A01", "Pair 1(A12+A02):T01-T03, Pair 2(A14+A05):T04-T06. Submit in R5")
    assert all(r == {} for r in p.values())


def test_bare_number_blocks():
    p = _parsed("A10", "Please claim your blocks: A01: 1-3, A02: 4-6.", joint=False)
    assert set(p) == {("A01", frozenset({"T01", "T02", "T03"}), frozenset()),
                      ("A02", frozenset({"T04", "T05", "T06"}), frozenset())}


def test_addressed_answer_share_is_not_an_instruction():
    assert _parsed("A04", "Hi A00! T43=930, T44=455.") == {}
