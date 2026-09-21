"""Run-log layout shared by the web engine and the CLI runner.

    craftax_agent/runs/<tag>/<YYYYMMDD-HHMMSS>-<variant>-s<seed>/    steps.jsonl · system_prompt.txt · meta.json · [frames/]
    craftax_agent/runs/<tag>/summary.json                            CLI batch rows, merged by (variant, seed)
    craftax_agent/runs/_smoke/<tag>/…                                CLI runs shorter than 100 steps
    craftax_agent/runs/_archive/                                     retired variants

<tag> is the policy name (llm, random, manual, jev_macro, jev_action_map, jev_action_sim) plus "_llm" when a jev
policy runs with the LLM objective planner. meta.json says whether the run came from the web engine or the CLI.
"""
import time
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
SMOKE_STEPS = 100          # CLI runs below this go to runs/_smoke/ and are not summarised


def run_tag(policy: str, planner: bool = False, chooser: str = "jev", objective: str = "llm") -> str:
    """runs/<tag>/: jev_macro, jev_macro_llm (LLM planner), match_macro_llm / random_macro (code-only choosers),
    jev_macro_sched (scripted objective sequence)."""
    if policy == "jev_macro" and chooser != "jev":
        policy = f"{chooser}_macro"
    if not policy.startswith("jev") and not policy.endswith("_macro"):
        return policy
    return policy + ("_llm" if planner else "_sched" if objective == "sched" else "")


def new_run_id(variant: str, seed: int) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{variant}-s{seed}"


def run_dir(tag: str, run_id: str, smoke: bool = False) -> Path:
    """Directory for a run; a second run with the same id in the same second gets a -2, -3 … suffix."""
    base = (RUNS / "_smoke" if smoke else RUNS) / tag
    d, n = base / run_id, 1
    while d.exists():
        n += 1
        d = base / f"{run_id}-{n}"
    return d
