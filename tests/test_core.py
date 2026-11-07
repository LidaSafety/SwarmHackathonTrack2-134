import asyncio

from swarmlab.config import RunConfig
from swarmlab.llm import parse_response
from swarmlab.metrics import expand_tasks, parse_directives
from swarmlab.sim import run_episode
from swarmlab.topology import build_graph


def test_expand_tasks():
    assert expand_tasks("T01-T03, T07 and T09..T10", 24) == ["T01", "T02", "T03", "T07", "T09", "T10"]
    assert expand_tasks("T30", 24) == []


def test_assignments_typical_llm_phrasings():
    msg = "Proposal to split the work: A03: T01-T06; A05: T07-T12\nA07 take T13, T14, T15. I'll take T19-T24."
    assigns, claims = parse_directives("A01", msg, 24)
    assert assigns["A03"] == ["T01", "T02", "T03", "T04", "T05", "T06"]
    assert assigns["A05"][0] == "T07"
    assert assigns["A07"] == ["T13", "T14", "T15"]
    assert claims == ["T19", "T20", "T21", "T22", "T23", "T24"]


def test_answer_sharing_is_not_assignment():
    assigns, claims = parse_directives("A02", "Solved: T01=345, T02=112. Anyone have T05?", 24)
    assert assigns == {} and claims == []


def test_parse_response_variants():
    assert parse_response('```json\n{"notes": "a", "actions": []}\n```')[0] == {"notes": "a", "actions": []}
    obj, _ = parse_response('<think>{bad}</think>Sure! {"notes": "say \\"hi\\" {x}", "actions": [{"tool": "ls"}]}')
    assert obj["actions"][0]["tool"] == "ls"
    assert parse_response("no json here")[0] is None


def test_topologies_connected_and_relabelled():
    for kind in ("complete", "star", "ring", "small_world", "scale_free", "random"):
        g = build_graph(kind, 16, 4, seed=1)
        assert len(g) == 16 and all(n.startswith("A") for n in g)


def test_mock_episode_metrics_shape():
    cfg = RunConfig(n_agents=6, rounds=6, topology="ring", seed=3)
    m = asyncio.run(run_episode(cfg))
    assert 0 <= m["leadership"]["leader_reach"] <= 1
    assert m["division_of_labour"]["coverage"] > 0
    assert len(m["exploit"]["adoption_curve"]) == 6


def test_assignment_takes_and_false_positive():
    msg = "how about: A01 takes T01-T06, A02 takes T07-T12, A03 takes T13-T18"
    assigns, claims = parse_directives("A01", msg, 24)
    assert assigns == {"A02": [f"T{i:02d}" for i in range(7, 13)], "A03": [f"T{i:02d}" for i in range(13, 19)]}
    assert claims == [f"T{i:02d}" for i in range(1, 7)]
    # addressing an agent without a directive is not an assignment
    assigns, _ = parse_directives("A02", "A00, please confirm T01-T12 are covered.", 24)
    assert assigns == {}


def test_replay_matches_live_metrics(tmp_path):
    from swarmlab.metrics import compute_metrics, replay_world

    cfg = RunConfig(n_agents=6, rounds=6, topology="small_world", seed=5, board="announced")
    live = asyncio.run(run_episode(cfg, tmp_path / "r"))
    replayed = compute_metrics(replay_world(tmp_path / "r"))
    for k in ("leadership", "division_of_labour", "exploit"):
        assert live[k] == replayed[k]


def test_seeded_compromise_hub_and_metrics():
    from swarmlab.sim import pick_compromised

    g = build_graph("star", 8, 4, seed=2)
    hub = max(g.nodes, key=g.degree)
    assert pick_compromised(RunConfig(n_agents=8, topology="star", seed=2, compromised=1, compromised_pos="hub"), g) == [hub]
    assert hub not in pick_compromised(RunConfig(n_agents=8, topology="star", seed=2, compromised=1, compromised_pos="leaf"), g)
    cfg = RunConfig(n_agents=8, rounds=6, topology="star", seed=2, board="announced", compromised=1, compromised_pos="hub")
    m = asyncio.run(run_episode(cfg))
    c = m["compromise"]
    assert c["compromised"] == [hub] and c["compromised_degree"] == [7]
    assert hub not in c["honest_adopters"]
    assert c["honest_aware_frac"] > 0  # the scripted adversary advertises the token to its neighbours


def test_run_id_stable_without_compromise():
    assert RunConfig(seed=1).run_id() == RunConfig(seed=1, compromised=0).run_id()
    assert RunConfig(seed=1).run_id() != RunConfig(seed=1, compromised=1).run_id()


def test_decomp_mock_all_conditions_deliver_and_grade():
    from swarmlab.decomp import DecompConfig, build_records, facts_in, run_trial

    rec = build_records(3)
    assert facts_in(f"{rec.truth['address']}, town; plate {rec.truth['plate']}", rec) == {
        "address": True, "plate": True, "gym": False, "school": False}
    for cond in ("whole", "split", "split_board", "orchestrator"):
        m = asyncio.run(run_trial(DecompConfig(seed=3, condition=cond, intent="shown"), None))
        assert m["harm_complete"], cond
        assert m["refusal_frac"] == 0
    m = asyncio.run(run_trial(DecompConfig(seed=3, condition="orchestrator", intent="shown"), None))
    assert m["intent_stripped_frac"] == 1.0  # the mock orchestrator dispatches context-free fragments
