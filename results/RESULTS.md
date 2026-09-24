# Results data

Every number in the README comes from the runs in this directory. `python results/analyze.py` recomputes all of them
from the logs; the tables below are its output as of 2026-09-21.

Layout: `<config>/<run_id>/meta.json` is the row written by `craftax_agent/run_agent.py` (achievements, steps, death,
tokens, timing, unlocked achievements, objectives); `steps.jsonl.gz` holds one record per step with the action, reward,
achievement count, the option table shown to jev, jev's probabilities and the observation text.

| Directory | Configuration | Used for |
|---|---|---|
| `jev-macro-llm-planner/` | Jev-Macro + LLM planner (objective + standing rules + plan, event-triggered) | figure, "fights every time", 15.7 |
| `jev-macro/` | Jev-Macro, no planner, shuffled option table | figure, 14.7, combat 95–100 % |
| `jev-raw/` | Jev-Action-Sim: jev over primitive actions with one-step facts and simulated outcomes | figure, 14.3 |
| `llm-control/` | LLM chooses every action (5-turn memory, scratchpad, achievement board), 500-step horizon | figure, 12.0 |
| `random-macro/` | uniform random over the same macro table | figure, 5.0 |
| `jev-macro-llm-planner-v1/` | Jev-Macro + LLM planner giving only an objective and a plan, 25-step cadence | "combat response fell to 48–77 %", scored below jev alone |
| `fixed-order-jev-macro/` | 12 earlier Jev-Macro runs (500 steps, options in a fixed order, harness survival rules of that version) | "took the first option 41 % of the time", "92–100 % when a fact matches" |

Replays are bit-exact only with the pinned `jax==0.4.38` / `craftax==1.6.1` (the RNG stream changed in later JAX releases); `tests/test_replay.py` checks this against `random-macro/20260921-132601-classic-s1`.

Definitions used by `analyze.py`:

- **achievements**: distinct Craftax-Classic achievements unlocked in the episode (max 22).
- **steps survived**: steps until death; every run here died before the 2000-step horizon (500 for `llm-control`).
- **combat response**: among steps where jev made the decision and a zombie or skeleton stood on an orthogonally adjacent
  tile (parsed from the observation text), the share that chose `attack`, `hold`, `flee` or `block`.
- **position of the chosen option**: in the fixed-order runs, the index of jev's chosen option in the list it was shown.
- **offered → chosen rate**: per option key, how often jev chose it when it was in the list (fixed-order runs).

## Results figure (3 seeds x 2000 steps)

| Agent | seed | achievements | steps survived | died | s/step | jev calls | LLM calls |
|---|---|---|---|---|---|---|---|
| Jev-macro + GPT-5.6-terra Planner | 0 | 11 | 186 | True | 1.03 | 171 | 18 |
| Jev-macro + GPT-5.6-terra Planner | 1 | 19 | 527 | True | 1.25 | 406 | 55 |
| Jev-macro + GPT-5.6-terra Planner | 2 | 17 | 220 | True | 1.47 | 197 | 31 |
| **Jev-macro + GPT-5.6-terra Planner mean** | | **15.7** | **311** | | 1.25 | | |
| Jev-macro | 0 | 14 | 387 | True | 0.25 | 209 | - |
| Jev-macro | 1 | 14 | 236 | True | 0.31 | 181 | - |
| Jev-macro | 2 | 16 | 341 | True | 0.27 | 191 | - |
| **Jev-macro mean** | | **14.7** | **321** | | 0.27 | | |
| Jev-raw | 1 | 15 | 258 | True | 0.46 | 258 | - |
| Jev-raw | 0 | 12 | 219 | True | 0.37 | 173 | - |
| Jev-raw | 2 | 16 | 351 | True | 0.36 | 280 | - |
| **Jev-raw mean** | | **14.3** | **276** | | 0.40 | | |
| GPT-5.6-terra control | 0 | 9 | 154 | True | 3.61 | 0 | - |
| GPT-5.6-terra control | 1 | 12 | 93 | True | 3.46 | 0 | - |
| GPT-5.6-terra control | 2 | 15 | 174 | True | 3.29 | 0 | - |
| **GPT-5.6-terra control mean** | | **12.0** | **140** | | 3.45 | | |
| Random-macro | 0 | 9 | 228 | True | 0.00 | 0 | - |
| Random-macro | 1 | 3 | 41 | True | 0.00 | 0 | - |
| Random-macro | 2 | 3 | 95 | True | 0.00 | 0 | - |
| **Random-macro mean** | | **5.0** | **121** | | 0.00 | | |

## Combat response when a hostile mob is orthogonally adjacent (jev-decided steps only)

| Config | seed | adjacent steps | chose attack / hold / flee | rate |
|---|---|---|---|---|
| jev-macro | 0 | 7 | 7 | 100% |
| jev-macro | 1 | 17 | 17 | 100% |
| jev-macro | 2 | 4 | 4 | 100% |
| jev-macro-llm-planner-v1 | 0 | 35 | 27 | 77% |
| jev-macro-llm-planner-v1 | 1 | 16 | 7 | 44% |
| jev-macro-llm-planner-v1 | 2 | 14 | 8 | 57% |
| jev-macro-llm-planner | 0 | 4 | 4 | 100% |
| jev-macro-llm-planner | 1 | 29 | 29 | 100% |
| jev-macro-llm-planner | 2 | 6 | 6 | 100% |

## Fixed-order runs: position of the chosen option

| position | share |
|---|---|
| 0 | 41% |
| 1 | 17% |
| 2 | 6% |
| 3 | 4% |
| 4 | 8% |
| 5 | 15% |

(first position: 41% of 2839 jev decisions)

## Fixed-order runs: offered -> chosen rate per option

| option | offered | chosen | rate |
|---|---|---|---|
| stone | 2256 | 494 | 21.9% |
| water | 2065 | 43 | 2.1% |
| explore | 2026 | 141 | 7.0% |
| tree | 1943 | 496 | 25.5% |
| iron | 1694 | 126 | 7.4% |
| shelter | 1269 | 118 | 9.3% |
| place_stone | 1081 | 29 | 2.7% |
| place_furnace | 1055 | 22 | 2.1% |
| coal | 1039 | 58 | 5.6% |
| flee | 989 | 458 | 46.3% |
| cow | 939 | 23 | 2.4% |
| grass_sapling | 932 | 2 | 0.2% |
| sleep | 710 | 0 | 0.0% |
| table | 480 | 441 | 91.9% |
| furnace | 301 | 64 | 21.3% |
| attack | 274 | 253 | 92.3% |
| make_wood_pickaxe | 123 | 12 | 9.8% |
| make_wood_sword | 123 | 12 | 9.8% |
| make_stone_pickaxe | 87 | 12 | 13.8% |
| make_stone_sword | 87 | 12 | 13.8% |
| place_table | 69 | 12 | 17.4% |
| make_iron_pickaxe | 14 | 4 | 28.6% |
| make_iron_sword | 14 | 5 | 35.7% |
| block | 3 | 2 | 66.7% |

