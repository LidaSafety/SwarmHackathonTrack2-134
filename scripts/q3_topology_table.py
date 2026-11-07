"""Q3 across published topologies: exposure, adoption, exposure x pooled conversion, speed and hops.

Pools runs/or_compromise (complete, star, ring, small-world) and runs/q3_topologies (chain, binary tree, layered,
random, scale-free, pairwise) for DeepSeek V4 Flash; writes runs/q3_topology_table.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    for sweep in ("or_compromise", "q3_topologies"):
        for d in sorted((ROOT / "runs" / sweep).glob("*dsflash*")):
            if not (d / "metrics.json").exists():
                continue
            m = json.loads((d / "metrics.json").read_text())
            cfg = json.loads((d / "config.json").read_text())
            c = m["compromise"]
            honest = cfg["n_agents"] - len(c["compromised"])
            topo = cfg["topology"]
            if topo in ("star", "chain", "binary_tree", "layered", "scale_free", "random", "small_world", "ring"):
                pos = cfg.get("compromised_pos", "random")
                key = f"{topo} ({pos})" if topo == "star" else topo
            else:
                key = topo
            rows.append(dict(shape=key, exposed=c["honest_aware_frac"], adopted=c["honest_adopt_frac"],
                             n_exposed=round(c["honest_aware_frac"] * honest), n_adopt=len(c["honest_adopters"]),
                             first=c["first_honest_adoption_round"], hops=c["max_adopter_hops"],
                             relays=c["relay_posts"], warns=c["warning_posts"]))
    df = pd.DataFrame(rows)
    conv = df.n_adopt.sum() / df.n_exposed.sum()
    g = df.groupby("shape").agg(runs=("exposed", "size"), exposed=("exposed", "mean"), adopted=("adopted", "mean"),
                                first=("first", "median"), any_adopt=("first", lambda s: s.notna().sum()),
                                max_hops=("hops", "max"), warns=("warns", "mean"))
    g["predicted"] = g.exposed * conv
    g["error"] = (g.predicted - g.adopted).abs()
    g = g.sort_values("exposed", ascending=False).round(3)
    pd.set_option("display.width", 200)
    print(f"pooled conversion {df.n_adopt.sum()}/{df.n_exposed.sum()} = {conv:.3f}; mean abs error {g.error.mean():.3f}")
    print(g.to_string())
    g.to_csv(ROOT / "runs" / "q3_topology_table.csv")


if __name__ == "__main__":
    main()
