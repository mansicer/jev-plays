"""No-LLM smoke test: random legal actions in Craftax-Classic and Craftax via the text wrapper."""
import sys, random, time
from craftax_agent.craftax_text_env import VARIANTS, CraftaxTextEnv

for variant in (sys.argv[1:] or VARIANTS):
    t = time.time()
    env = CraftaxTextEnv(variant, max_timesteps=200, render=True)
    obs = env.reset(0)
    print(f"[{variant}] built in {time.time() - t:.0f}s, {len(env.action_names)} actions, "
          f"{env.n_ach_total} achievements, frame {obs['image'].size}")
    total = 0.0
    for step in range(50):
        obs, r, done, info = env.step(random.choice(env.legal_actions()))
        total += r
        if done:
            break
    print(f"[{variant}] 50 random steps: reward {total:+.1f}, achievements {env.n_achievements()}, score {env.score()}")
    print(obs["text"]["long_term_context"].split("\n\n")[2])
