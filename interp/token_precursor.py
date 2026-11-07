"""Token-level precursor: is Qwen3-8B already committed to posting coordination while it writes its private notes?

Every Qwen call in runs/swarm_mixed is replayed teacher-forced: system prompt + saved observation + the reply the agent
actually wrote. Residuals at layers 14/22 are saved at fixed points of the reply that all precede any action:
  prompt     last prompt token (as in the saved acts)
  n0 .. n4   notes value: first token, 25/50/75%, last token
  actions    the "actions" key
  tool1      just before the first tool name is written
Replies post after the notes (JSON order), so for a coordination post every point is "silent". At each point a probe
predicts whether the reply will post coordination (and, among posting replies, coordination vs other posts), against
TF-IDF on the observation plus the notes written so far. The silence check repeats it on replies whose notes never
mention claims, splits or ranges.

Usage: CUDA_VISIBLE_DEVICES=0 python interp/token_precursor.py [capture|analyze]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from order_control import ROOT, load  # noqa: E402

OUT = ROOT / "token_precursor"
LAYERS = (14, 22)
POS = ["prompt", "n0", "n1", "n2", "n3", "n4", "actions", "tool1"]


def raw_replies(df: pd.DataFrame) -> list[str]:
    raw = {}
    for run in df.run.unique():
        for line in (ROOT / run / "events.jsonl").read_text().splitlines():
            e = json.loads(line)
            if e["kind"] == "llm" and "act_idx" in e:
                raw[(run, e["act_idx"])] = e.get("raw") or ""
    return [raw.get((r, i), "") for r, i in zip(df.run, df.act_idx)]


def char_points(reply: str) -> dict | None:
    """Character offsets in the reply for each point (None if the reply is not notes-then-actions JSON)."""
    i_notes, i_act = reply.find('"notes"'), reply.find('"actions"')
    if i_notes < 0 or i_act < i_notes:
        return None
    v0 = reply.find('"', reply.find(":", i_notes) + 1) + 1          # first char of the notes value
    v1 = reply.rfind('"', v0, i_act)                                 # its closing quote
    i_tool = reply.find('"tool"', i_act)
    if v1 <= v0 or i_tool < 0:
        return None
    t1 = reply.find('"', reply.find(":", i_tool) + 1)               # opening quote of the first tool name
    span = v1 - 1 - v0
    pts = dict(n0=v0, n1=v0 + span // 4, n2=v0 + span // 2, n3=v0 + 3 * span // 4, n4=v1 - 1, actions=i_act, tool1=t1)
    pts["notes_text"] = reply[v0:v1]
    return pts


def capture() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from q2_probe import find_layers
    from swarmlab.config import RunConfig
    from swarmlab.prompts import system_prompt

    df, _, obs = load()
    raws = raw_replies(df)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    items, meta = [], []
    cfgs = {}
    for i in range(len(df)):
        pts = char_points(raws[i]) if not df.error[i] else None
        if pts is None:
            continue
        run, agent = df.run[i], df.agent[i]
        cfgs.setdefault(run, RunConfig.from_dict(json.loads((ROOT / run / "config.json").read_text())))
        prompt = tok.apply_chat_template([{"role": "system", "content": system_prompt(cfgs[run], agent)},
                                          {"role": "user", "content": obs[i]}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
        text = prompt + raws[i]
        enc = tok(text, return_offsets_mapping=True)
        ends = np.array([e for _, e in enc.offset_mapping])
        n_prompt = len(tok(prompt).input_ids)
        tix = {"prompt": n_prompt - 1}
        for k in POS[1:]:
            c = len(prompt) + pts[k]
            # residual at the last token that ends at or before this character: the model has read up to here
            tix[k] = int(np.searchsorted(ends, c + 1, side="right") - 1)
        if not (tix["prompt"] <= tix["n0"] <= tix["n4"] <= tix["actions"] <= tix["tool1"]):
            continue
        items.append((enc.input_ids, [tix[k] for k in POS]))
        meta.append(dict(i=i, notes_text=pts["notes_text"], n_tok=len(enc.input_ids)))
    print(f"{len(items)} of {len(df)} calls parsed as notes-then-actions")

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0").eval()
    layers = find_layers(model)
    store = {}
    hooks = [layers[l].register_forward_hook(
        lambda _m, _i, out, l=l: store.__setitem__(l, (out[0] if isinstance(out, tuple) else out).detach()))
        for l in LAYERS]
    X = np.zeros((len(items), len(POS), len(LAYERS), model.config.hidden_size), np.float16)
    order = np.argsort([len(ids) for ids, _ in items])              # length-sorted batches: little padding
    budget = 16000
    b = 0
    with torch.no_grad():
        while b < len(order):
            L = len(items[order[min(b + 7, len(order) - 1)]][0])
            bs = max(1, min(8, budget // L))
            idx = order[b:b + bs]
            L = max(len(items[j][0]) for j in idx)
            ids = torch.full((len(idx), L), tok.pad_token_id or 0, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, j in enumerate(idx):                               # left padding
                n = len(items[j][0])
                ids[r, L - n:] = torch.tensor(items[j][0])
                att[r, L - n:] = 1
            model(input_ids=ids.cuda(), attention_mask=att.cuda(), logits_to_keep=1)
            for r, j in enumerate(idx):
                off = L - len(items[j][0])
                pos = torch.tensor([off + p for p in items[j][1]], device="cuda")
                for li, l in enumerate(LAYERS):
                    X[j, :, li] = store[l][r, pos].float().cpu().numpy().astype(np.float16)
            b += len(idx)
            if (b // bs) % 50 == 0:
                print(f"  {b}/{len(order)}", flush=True)
    for h in hooks:
        h.remove()
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "acts.npy", X)
    pd.DataFrame(meta).to_json(OUT / "meta.jsonl", orient="records", lines=True)


def analyze() -> None:
    from sklearn.metrics import roc_auc_score

    from analyze_mixed import cv_group
    from analyze_q3_monitor import probe, text_clf
    from precursor import PLAN

    pd.set_option("display.width", 200)
    df, A0, obs = load()
    X = np.load(OUT / "acts.npy").astype(np.float32)
    meta = pd.read_json(OUT / "meta.jsonl", lines=True)
    d = df.iloc[meta.i.values].reset_index(drop=True)
    d["notes_text"] = meta.notes_text.fillna("").values
    o = [obs[i] for i in meta.i]
    # sanity: the replayed prompt-token residual should match the residual saved during the run
    li22 = json.loads((ROOT / "probe.json").read_text())["save_layers"].index(22)
    a, b_ = A0[meta.i.values, li22], X[:, 0, LAYERS.index(22)]
    cos = (a * b_).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b_, axis=1)
    print(f"{len(d)} replies; replayed vs saved prompt residual cosine: median {np.median(cos):.3f}, min {cos.min():.3f}")
    print(f"notes mention claims/splits/ranges in {d.notes_text.str.contains(PLAN).mean():.2f} of replies; "
          f"among coordination posts {d.notes_text[d.coord_post].str.contains(PLAN).mean():.2f}")

    def frac_notes(text: str, k: str) -> str:
        f = {"prompt": 0, "n0": 0, "n1": .25, "n2": .5, "n3": .75}.get(k, 1.0)
        return text[: int(len(text) * f)]

    tests = [("coord post: all replies", np.ones(len(d), bool), d.coord_post.values),
             ("coord post: notes never mention coordination", ~d.notes_text.str.contains(PLAN).values,
              d.coord_post.values),
             ("coord vs other post: posting replies", d.post.values.astype(bool), d.coord_post.values),
             ("coord vs other post: posting, silent notes",
              d.post.values.astype(bool) & ~d.notes_text.str.contains(PLAN).values, d.coord_post.values)]
    rows = []
    for name, m, y in tests:
        y = y[m].astype(int)
        g = d.run.values[m]
        if y.sum() < 8 or y.sum() > len(y) - 8:
            print(f"skip {name}: {y.sum()}/{len(y)}")
            continue
        idx = np.where(m)[0]
        for pi, k in enumerate(POS):
            for li, l in enumerate(LAYERS):
                s = cv_group(probe, X[idx, pi, li], y, g)
                rows.append(dict(test=name, pos=k, monitor=f"probe L{l}", auroc=roc_auc_score(y, s), n=len(y), pos_n=y.sum()))
            texts = [o[i] + "\nNOTES SO FAR: " + frac_notes(d.notes_text[i], k) for i in idx]
            s = cv_group(text_clf, texts, y, g)
            rows.append(dict(test=name, pos=k, monitor="tf-idf obs + notes so far", auroc=roc_auc_score(y, s),
                             n=len(y), pos_n=y.sum()))
            if k in ("n4", "actions", "tool1"):
                s = cv_group(text_clf, [d.notes_text[i] or "empty" for i in idx], y, g)
                rows.append(dict(test=name, pos=k, monitor="tf-idf notes only", auroc=roc_auc_score(y, s),
                                 n=len(y), pos_n=y.sum()))
    res = pd.DataFrame(rows).round(3)
    res.to_csv(OUT / "results.csv", index=False)
    for name in res.test.unique():
        r = res[res.test == name]
        print(f"\n{name}  (n={r.n.iloc[0]}, positives={r.pos_n.iloc[0]})")
        print(r.pivot(index="monitor", columns="pos", values="auroc")[POS].to_string())


if __name__ == "__main__":
    {"capture": capture, "analyze": analyze}[(sys.argv[1:] or ["capture"])[0]]()
