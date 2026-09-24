"""Real Jev calls (needs TYPESAFE_API_KEY): the macro policy runs 20 steps and jev makes the decisions."""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
pytestmark = [pytest.mark.jev,
              pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY not set")]


def test_jev_macro_decides():
    from craftax_agent.craftax_text_env import CraftaxTextEnv
    from craftax_agent.jev_policy import JevMacroPolicy

    env = CraftaxTextEnv("classic", max_timesteps=10000, render=False)
    env.reset(0)
    pol = JevMacroPolicy(env)
    sources = []
    for _ in range(20):
        action, reply, info = pol.decide()
        assert action in env.action_names
        _, reward, done, ex = env.step(action)
        pol.note_outcome(info["macro"], action, reward, ex["unlocked"])
        sources.append(info["source"])
        if done:
            break
    assert "jev" in sources, sources
    jev_steps = [s for s in sources if s == "jev"]
    assert len(jev_steps) >= 5
