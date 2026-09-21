"""Memory shared by every LLM call site (the LLM policy and the objective planner): the same two mechanisms,
the same options, the same replay rules.

    history_n      how many past (message, reply) turns are replayed as multi-turn input (0 = memoryless)
    use_scratchpad the model rewrites private notes each turn inside <scratchpad>, fed back verbatim next turn

Replayed turns are stripped of everything the current turn restates: the caller passes the replayable core of each
message (`hist_msg`) separately from the full message, and the <scratchpad> of every reply but the latest is removed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SCRATCHPAD_ASK = ("\nThen rewrite your notes inside <scratchpad></scratchpad> (max ~80 words): {topics}. "
                  "The notes are the only thing that persists between turns; write the full updated version each "
                  "time, not a diff.")
_SP = re.compile(r"\s*<scratchpad>(.*?)</scratchpad>\s*", re.I | re.S)


@dataclass
class LLMMemory:
    history_n: int = 5
    use_scratchpad: bool = True
    scratchpad: str = ""          # model-written notes, carried verbatim into the next turn
    scratchpad_step: int = -1     # step at which the notes were last updated
    turns: list = field(default_factory=list)   # [(replayable message, assistant reply)]

    def clear(self):
        self.scratchpad, self.scratchpad_step, self.turns = "", -1, []

    # ---- prompt pieces ----
    def notes_block(self) -> str:
        """The scratchpad injection for the current turn ('' when disabled)."""
        if not self.use_scratchpad:
            return ""
        return "Your private notes from previous turns (scratchpad):\n<scratchpad>\n" + (self.scratchpad or "(empty)") + "\n</scratchpad>"

    def ask_suffix(self, topics: str) -> str:
        return SCRATCHPAD_ASK.format(topics=topics) if self.use_scratchpad else ""

    def messages(self, user_msg: str) -> list[dict]:
        """Responses-API input: the last `history_n` turns replayed, then this turn."""
        msgs = []
        turns = self.turns[-self.history_n:] if self.history_n > 0 else []
        for i, (u, a) in enumerate(turns):
            if i < len(turns) - 1:                       # the current turn carries the newest notes
                a = _SP.sub("\n", a).strip()
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": user_msg})
        return msgs

    # ---- after the reply ----
    def record(self, hist_msg: str, reply: str, step: int):
        sp = _SP.search(reply)
        if self.use_scratchpad and sp and sp.group(1).strip():
            self.scratchpad, self.scratchpad_step = sp.group(1).strip(), step
        self.turns.append((hist_msg, reply))
        self.turns = self.turns[-max(self.history_n, 10):]   # keep a little slack so raising N later has data

    def state(self) -> dict:
        """For the web snapshot / logs."""
        return {"scratchpad": self.scratchpad, "scratchpad_step": self.scratchpad_step, "history_n": self.history_n,
                "use_scratchpad": self.use_scratchpad, "turns_available": len(self.turns)}
