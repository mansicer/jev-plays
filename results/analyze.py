"""Recompute every number in the README from the logs in this directory.

    uv run python results/analyze.py            # prints the tables and the three rates

Each run directory holds meta.json (the row written by craftax_agent/run_agent.py) and steps.jsonl.gz (one
record per step: action, reward, achievements, the option table, jev's probabilities, the observation text).
"""
from __future__ import annotations

import gzip, json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGURE = [  # config directory -> label used in the README
    ("jev-macro-llm-planner", "Jev-macro + GPT-5.6-terra Planner"),
    ("jev-macro", "Jev-macro"),
    ("jev-raw", "Jev-raw"),
    ("llm-control", "GPT-5.6-terra control"),
    ("random-macro", "Random-macro"),
]
COMBAT = {"attack", "flee", "hold", "block"}
ADJ = re.compile(r"(zombie|skeleton) 1 step (north|south|east|west) \(")


def runs(cfg: str):
    for d in sorted((ROOT / cfg).iterdir()):
        if (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text())
            with gzip.open(d / "steps.jsonl.gz", "rt") as f:
                recs = [json.loads(line) for line in f]
            yield d.name, meta, recs


def hostile_adjacent(rec) -> bool:
    m = re.search(r"Mobs in view:(.*?)(?:\n\n|$)", rec["obs_long"], re.S)
    return bool(m and ADJ.search(m.group(1)))


def mean(xs):
    return sum(xs) / len(xs)


def figure_table():
    print("## Results figure (3 seeds x 2000 steps)\n")
    print("| Agent | seed | achievements | steps survived | died | s/step | jev calls | LLM calls |")
    print("|---|---|---|---|---|---|---|---|")
    for cfg, label in FIGURE:
        rows = list(runs(cfg))
        for name, meta, recs in rows:
            jev_calls = sum(1 for r in recs if r["source"] == "jev")
            print(f"| {label} | {meta['seed']} | {meta['achievements']} | {meta['steps']} | {meta['died']} | "
                  f"{meta['mean_step_s']:.2f} | {jev_calls} | {meta.get('llm_calls', '-')} |")
        print(f"| **{label} mean** | | **{mean([m['achievements'] for _, m, _ in rows]):.1f}** | "
              f"**{mean([m['steps'] for _, m, _ in rows]):.0f}** | | {mean([m['mean_step_s'] for _, m, _ in rows]):.2f} | | |")
    print()


def combat_rates():
    print("## Combat response when a hostile mob is orthogonally adjacent (jev-decided steps only)\n")
    print("| Config | seed | adjacent steps | chose attack / hold / flee | rate |")
    print("|---|---|---|---|---|")
    for cfg in ("jev-macro", "jev-macro-llm-planner-v1", "jev-macro-llm-planner"):
        for name, meta, recs in runs(cfg):
            adj = [r for r in recs if r["source"] == "jev" and hostile_adjacent(r)]
            c = sum(1 for r in adj if r["macro"] in COMBAT)
            print(f"| {cfg} | {meta['seed']} | {len(adj)} | {c} | {100 * c / max(1, len(adj)):.0f}% |")
    print()


def fixed_order_stats():
    """The 41 % first-position preference and the offered->chosen rates come from the 12 fixed-order runs
    (an earlier harness version that listed options in a fixed order; the final experiments shuffle)."""
    pos, offered, chosen = Counter(), Counter(), Counter()
    for name, meta, recs in runs("fixed-order-jev-macro"):
        for r in recs:
            if r["source"] != "jev":
                continue
            m = re.search(r"options \((\d+)\): (.*)", r["reply"])
            if not m:
                continue
            opts = [o.strip() for o in m.group(2).split(";")]
            for o in opts:
                offered[o] += 1
            chosen[r["macro"]] += 1
            if r["macro"] in opts:
                pos[opts.index(r["macro"])] += 1
    total = sum(pos.values())
    print("## Fixed-order runs: position of the chosen option\n")
    print("| position | share |"); print("|---|---|")
    for p in range(6):
        print(f"| {p} | {100 * pos[p] / total:.0f}% |")
    print(f"\n(first position: {100 * pos[0] / total:.0f}% of {total} jev decisions)\n")
    print("## Fixed-order runs: offered -> chosen rate per option\n")
    print("| option | offered | chosen | rate |"); print("|---|---|---|---|")
    for k, v in offered.most_common():
        print(f"| {k} | {v} | {chosen[k]} | {100 * chosen[k] / v:.1f}% |")
    print()


if __name__ == "__main__":
    figure_table()
    combat_rates()
    fixed_order_stats()
