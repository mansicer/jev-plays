"""LLM policy for Craftax — the one implementation shared by the web engine (server/app.py) and the CLI runner.

Same interface as the jev policies: decide() -> (action, reply, info), note_outcome(...), reset().
Memory (scratchpad + replayed turns) is an LLMMemory, the same component the objective planner uses; `use_goals`
injects the code-maintained achievement board + status notes (goals.py) into the observation.
Everything the model sees stays descriptive: the manual (env.system_prompt), the observation, the board.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from craftax_agent.goals import board as goal_board, board_text
from craftax_agent.llm_memory import LLMMemory

ASK = ("Reply with exactly one action from the available list inside <action></action> tags. "
       "Keep any reasoning to one short sentence.")
NOTE_TOPICS = ("current goal, next 2-3 steps, remembered locations with coordinates (e.g. 'stone at (28,27)'), "
               "and what failed")


@dataclass
class LLMPolicy:
    env: object
    memory: LLMMemory = field(default_factory=LLMMemory)
    use_goals: bool = True
    model: str = ""
    client: object = None
    step: int = 0                 # env steps taken in this episode (ticked by note_outcome)

    def __post_init__(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL") or None,
                                 max_retries=6)   # transient connection errors must not kill a 500-step run
        self.model = self.model or os.environ["OPENAI_MODEL_NAME"]

    @property
    def name(self) -> str:
        return "LLM"

    # memory options / state, exposed flat for the server and the runner
    @property
    def history_n(self) -> int: return self.memory.history_n
    @history_n.setter
    def history_n(self, v: int): self.memory.history_n = v
    @property
    def use_scratchpad(self) -> bool: return self.memory.use_scratchpad
    @use_scratchpad.setter
    def use_scratchpad(self, v: bool): self.memory.use_scratchpad = v
    @property
    def scratchpad(self) -> str: return self.memory.scratchpad
    @property
    def scratchpad_step(self) -> int: return self.memory.scratchpad_step
    @property
    def turns(self) -> list: return self.memory.turns

    def reset(self):
        self.clear_memory()
        self.step = 0

    def clear_memory(self):
        """Forget notes and replayed turns without resetting the world."""
        self.memory.clear()

    def note_outcome(self, macro_key: str, action: str, reward: float, unlocked: list[str]):
        self.step += 1

    # ---- prompt construction ----
    @staticmethod
    def history_message(obs: dict) -> str:
        """What a past turn keeps when replayed: the observation only (no action list, board, notes, ask)."""
        t = obs["text"]
        return t["long_term_context"] + "\n\n" + t["short_term_context"].split("Available actions:")[0].rstrip()

    def user_message(self, obs: dict) -> str:
        t = obs["text"]
        parts = [t["long_term_context"], t["short_term_context"]]
        if self.use_goals:
            parts.append(board_text(goal_board(self.env)))
        if self.memory.use_scratchpad:
            parts.append(self.memory.notes_block())
        parts.append(ASK + self.memory.ask_suffix(NOTE_TOPICS))
        return "\n\n".join(parts)

    # ---- decision ----
    def decide(self, obs: dict | None = None) -> tuple[str, str, dict]:
        """obs: the current observation dict from env.reset()/env.step(); recomputed from the env if omitted."""
        if obs is None:
            obs = self.env.observe()
        msgs = self.memory.messages(self.user_message(obs))
        t0 = time.perf_counter()
        resp = self.client.responses.create(model=self.model, instructions=self.env.system_prompt, input=msgs)
        reply = resp.output_text or ""
        m = re.search(r"<action>(.*?)</action>", reply, re.I | re.S)
        action = m.group(1).strip() if m else reply.strip()
        self.memory.record(self.history_message(obs), reply, self.step)
        u = resp.usage
        return action, reply, {"latency_s": round(time.perf_counter() - t0, 2),
                               "macro": action, "source": "llm",
                               "input_tokens": getattr(u, "input_tokens", None),
                               "output_tokens": getattr(u, "output_tokens", None),
                               "model": self.model, "response_id": resp.id,
                               "context_turns": (len(msgs) - 1) // 2,
                               "scratchpad": self.memory.scratchpad,
                               "notes": goal_board(self.env)["notes"] if self.use_goals else []}
