"""No API key: the text wrapper builds both variants and steps with random legal actions."""
import random

import pytest

from craftax_agent.craftax_text_env import VARIANTS, CraftaxTextEnv


@pytest.mark.parametrize("variant", VARIANTS)
def test_random_steps(variant):
    env = CraftaxTextEnv(variant, max_timesteps=200, render=True)
    obs = env.reset(0)
    assert obs["image"].size[0] > 0 and "Available actions" in obs["text"]["short_term_context"]
    rng = random.Random(0)
    for _ in range(50):
        legal = env.legal_actions()
        assert legal and set(legal) <= set(env.action_names)
        obs, reward, done, info = env.step(rng.choice(legal))
        assert 0 <= env.n_achievements() <= env.n_ach_total
        if done:
            break
