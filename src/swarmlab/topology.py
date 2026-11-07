from __future__ import annotations

import random

import networkx as nx

from .world import agent_id

TOPOLOGIES = ("complete", "star", "ring", "small_world", "scale_free", "random", "isolated", "tree", "clusters",
              "chain", "binary_tree", "layered", "pairwise")


def build_graph(kind: str, n: int, degree: int, seed: int) -> nx.Graph:
    """Who can see whose board posts. Node labels are agent ids (A00, A01, ...)."""
    k = max(2, min(degree, n - 1))
    if kind == "complete":
        g = nx.complete_graph(n)
    elif kind == "star":
        g = nx.star_graph(n - 1)  # node 0 is the hub
    elif kind == "ring":
        g = nx.cycle_graph(n)
    elif kind == "small_world":
        g = nx.connected_watts_strogatz_graph(n, k if k % 2 == 0 else k - 1 or 2, 0.1, seed=seed)
    elif kind == "scale_free":
        g = nx.barabasi_albert_graph(n, max(1, min(k // 2, n - 1)), seed=seed)
    elif kind == "random":
        p = min(1.0, k / max(1, n - 1))
        g = nx.erdos_renyi_graph(n, p, seed=seed)
        # keep it connected so topology, not fragmentation, is what varies
        comps = [list(c) for c in nx.connected_components(g)]
        rng = random.Random(seed)
        for a, b in zip(comps, comps[1:]):
            g.add_edge(rng.choice(a), rng.choice(b))
    elif kind == "chain":
        g = nx.path_graph(n)  # NetSafe / MacNet / G-Designer "chain"
    elif kind == "binary_tree":
        g = nx.Graph()  # NetSafe "binary tree": node i sees its parent (i-1)//2 and its two children
        g.add_nodes_from(range(n))
        g.add_edges_from((i, (i - 1) // 2) for i in range(1, n))
    elif kind == "layered":
        # MacNet / AgentPrune "layered" (MLP-shaped): layers of `degree` agents, each agent sees every agent in the
        # adjacent layers
        size = max(2, degree)
        layers = [list(range(i, min(i + size, n))) for i in range(0, n, size)]
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from((a, b) for l1, l2 in zip(layers, layers[1:]) for a in l1 for b in l2)
    elif kind == "pairwise":
        # Agent Smith / Prompt Infection: random pairwise chats, re-paired every round (see pairwise_round)
        g = nx.empty_graph(n)
    elif kind == "tree":
        # org chart: each node sees its manager and its direct reports (branching = degree)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from((i, (i - 1) // max(2, degree)) for i in range(1, n))
    elif kind == "clusters":
        # departments: dense teams of size `degree`, consecutive teams joined by one bridge pair
        size = max(2, degree)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        teams = [list(range(i, min(i + size, n))) for i in range(0, n, size)]
        for t in teams:
            g.add_edges_from((a, b) for i, a in enumerate(t) for b in t[i + 1:])
        rng = random.Random(seed)
        for a, b in zip(teams, teams[1:] + teams[:1] if len(teams) > 2 else teams[1:]):
            g.add_edge(rng.choice(a), rng.choice(b))
    elif kind == "isolated":
        g = nx.empty_graph(n)
    else:
        raise ValueError(f"unknown topology {kind!r}; choose from {TOPOLOGIES}")
    # Shuffle labels so structural position (e.g. the star hub) is not tied to a low agent id,
    # which models might read as seniority.
    ids = [agent_id(i) for i in range(n)]
    random.Random(seed + 7919).shuffle(ids)
    return nx.relabel_nodes(g, dict(zip(range(n), ids)))


def pairwise_round(ids: list[str], seed: int, rnd: int) -> nx.Graph:
    """A fresh random perfect matching for one round of pairwise chats (one agent sits out if N is odd)."""
    order = list(ids)
    random.Random(seed * 1000 + rnd).shuffle(order)
    g = nx.Graph()
    g.add_nodes_from(ids)
    g.add_edges_from(zip(order[0::2], order[1::2]))
    return g
