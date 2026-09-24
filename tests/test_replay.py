"""No API key: replaying a logged episode's actions reproduces its achievements and death step exactly.
The same determinism produces the comparison GIF and the numbers in results/."""
import gzip
import json
from pathlib import Path

from craftax_agent.craftax_text_env import CraftaxTextEnv

RUN = Path(__file__).resolve().parent.parent / "results" / "random-macro" / "20260921-132601-classic-s1"


def test_replay_matches_log():
    meta = json.loads((RUN / "meta.json").read_text())
    with gzip.open(RUN / "steps.jsonl.gz", "rt") as f:
        recs = [json.loads(line) for line in f]
    env = CraftaxTextEnv(meta["variant"], max_timesteps=10000, render=False)
    env.reset(meta["seed"])
    done = False
    for rec in recs:
        _, _, done, info = env.step(rec["action"])
        assert info["unlocked"] == rec["unlocked"], f"step {rec['step']} diverged"
        assert env.n_achievements() == rec["achievements"]
        if done:
            break
    assert env.n_achievements() == meta["achievements"]
    assert done == meta["died"] and rec["step"] + 1 == meta["steps"]
