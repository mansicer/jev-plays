"""Uniform-random baseline over legal actions — shared by the web engine and the CLI runner."""
import random


class RandomPolicy:
    def __init__(self, env):
        self.env = env

    @property
    def name(self) -> str:
        return "random"

    def decide(self, obs=None):
        acts = self.env.legal_actions()
        a = random.choice(acts)
        return a, f"(random policy) chose {a} from {len(acts)} legal actions", {"latency_s": 0.0, "macro": a, "source": "random"}

    def note_outcome(self, *a):
        pass

    def reset(self):
        pass
