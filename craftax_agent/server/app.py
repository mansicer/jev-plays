"""Local real-time Craftax agent engine.

FastAPI + WebSocket server that owns one vanilla Craftax (Craftax-Symbolic-v1) environment,
runs a policy loop (LLM / random / manual) and streams every step — rendered frame,
text observation, model reply, reward, achievements — to the browser UI in static/.

Run:  .venv/bin/python -m uvicorn craftax_agent.server.app:app --port 8765
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from craftax_agent.craftax_text_env import VARIANTS, CraftaxTextEnv
from craftax_agent.goals import board as goal_board
from craftax_agent.random_policy import RandomPolicy
from craftax_agent.runlog import new_run_id, run_dir, run_tag

STATIC = Path(__file__).parent / "static"
DEFAULT_VARIANT = os.environ.get("CRAFTAX_VARIANT", "classic")
JEV_POLICIES = ("jev_macro", "jev_action_map", "jev_action_sim")


def new_tokens():
    """Per model family: LLM (policy or planner calls) and jev (System One calls)."""
    return {k: {"input": 0, "output": 0, "steps": 0} for k in ("llm", "jev")}


def img_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- session

@dataclass
class Session:
    seed: int = 0
    variant: str = DEFAULT_VARIANT   # classic | full
    policy: str = "llm"          # llm | random | manual | one of JEV_POLICIES
    planner: bool = False        # jev policies: add the LLM objective planner (System Two)
    delay: float = 0.0           # seconds between auto steps
    running: bool = False
    step: int = 0
    total_reward: float = 0.0
    history: list = field(default_factory=list)
    env: CraftaxTextEnv | None = None
    system_prompt: str = ""
    obs: dict = None
    manual_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    clients: set = field(default_factory=set)
    task: asyncio.Task | None = None
    busy: bool = False
    status: str = "booting"
    episode_tokens: dict = field(default_factory=new_tokens)
    total_tokens: dict = field(default_factory=new_tokens)
    episodes_started: int = 0
    episodes_with_tokens: int = 0
    # ---- LLM options: one set, applied to every LLM call site (the LLM policy and the jev objective planner);
    #      each site keeps its own LLMMemory (its own turns and notes) ----
    history_n: int = 5
    use_scratchpad: bool = True
    use_goals: bool = True
    llm_policy: object = None

    def _memories(self):
        out = [self.llm_policy.memory] if self.llm_policy is not None else []
        planner = getattr(self.jev_policy, "planner", None)
        if planner is not None:
            out.append(planner.memory)
        return out

    def apply_llm_options(self):
        for m in self._memories():
            m.history_n, m.use_scratchpad = self.history_n, self.use_scratchpad
        if self.llm_policy is not None:
            self.llm_policy.use_goals = self.use_goals

    def _llm(self):
        from craftax_agent.llm_memory import LLMMemory
        from craftax_agent.llm_policy import LLMPolicy
        if self.llm_policy is None or self.llm_policy.env is not self.env:
            self.llm_policy = LLMPolicy(self.env, memory=LLMMemory(self.history_n, self.use_scratchpad), use_goals=self.use_goals)
        return self.llm_policy

    def active_memory(self):
        """The LLMMemory the current policy uses: the planner's for a jev policy with the planner on, else the LLM policy's."""
        if self.policy in JEV_POLICIES:
            planner = getattr(self.jev_policy, "planner", None) if self.planner else None
            return planner.memory if planner is not None else None
        return self.llm_policy.memory if self.policy == "llm" and self.llm_policy is not None else None

    # ---- env ----
    @property
    def state(self):
        return self.env.state

    @property
    def actions(self) -> list[str]:
        return self.env.action_names

    def build(self, variant: str | None = None):
        if variant:
            self.variant = variant
        self.env = CraftaxTextEnv(self.variant, max_timesteps=10000, render=True, pixel_size=16)
        self.system_prompt = self.env.system_prompt

    run_id: str = ""
    run_dir: Path | None = None    # created on the first step, so untouched episodes leave no folder

    def reset(self, seed: int):
        self.finish_run()
        self.seed = seed
        self.obs = self.env.reset(seed)
        self.run_id = new_run_id(self.variant, seed)
        self.run_dir = None
        self.policies_used = []
        self.step = 0
        self.total_reward = 0.0
        self.history = []
        self.episode_tokens = new_tokens()
        self.episodes_started += 1
        self._llm().reset()
        if self.jev_policy is not None:
            self.jev_policy.reset()

    def env_step(self, action: str):
        self.obs, reward, done, info = self.env.step(action)
        return reward, done, info["unlocked"], info["hp_delta"]

    # ---- run log (craftax_agent/runs/<tag>/<run_id>/, see runlog.py) ----
    policies_used: list = field(default_factory=list)
    def run_meta(self, final: bool = False) -> dict:
        return {"source": "web", "run_id": self.run_id, "variant": self.variant, "seed": self.seed, "policy": self.policy,
                "planner": self.planner, "model": os.environ.get("OPENAI_MODEL_NAME"), "history_n": self.history_n,
                "scratchpad": self.use_scratchpad, "goals": self.use_goals,
                "policies_used": self.policies_used, "steps": len(self.history), "total_reward": round(self.total_reward, 3),
                "achievements": self.env.n_achievements(), "achievements_total": self.env.n_ach_total,
                "score": self.env.score(), "max_score": self.env.max_score,
                "tokens": self.episode_tokens, "finished": final, "status": self.status}

    def log_step(self, rec: dict, obs_before: dict):
        if self.run_dir is None:   # grouped by the policy that took the first step (jev + planner -> "<policy>_llm")
            self.run_dir = run_dir(run_tag(self.policy, self.planner), self.run_id)
            self.run_id = self.run_dir.name
            (self.run_dir / "frames").mkdir(parents=True, exist_ok=True)
        if self.policy not in self.policies_used:
            self.policies_used.append(self.policy)
            (self.run_dir / "system_prompt.txt").write_text(self.system_prompt)
        t = obs_before["text"]
        row = {**rec, "obs_long": t["long_term_context"], "obs_short": t["short_term_context"]}
        with (self.run_dir / "steps.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        if obs_before.get("image") is not None:
            obs_before["image"].save(self.run_dir / "frames" / f"{rec['step']:04d}.png")
        if rec.get("done") and self.obs.get("image") is not None:   # final frame after the last action
            self.obs["image"].save(self.run_dir / "frames" / f"{rec['step'] + 1:04d}.png")
        (self.run_dir / "meta.json").write_text(json.dumps(self.run_meta(final=bool(rec.get("done"))), indent=1))

    def finish_run(self):
        if self.run_dir is not None and self.obs is not None:
            if self.obs.get("image") is not None:
                self.obs["image"].save(self.run_dir / "frames" / f"{self.step:04d}.png")
            (self.run_dir / "meta.json").write_text(json.dumps(self.run_meta(final=True), indent=1))

    # ---- policies ----
    def llm_decide(self) -> tuple[str, str, dict]:
        return self._llm().decide(self.obs)

    # ---- jev (System One) policies: "jev" = jev + code fallback, "dual" = jev + LLM escalation ----
    jev_policy: object = None

    def jev_decide(self) -> tuple[str, str, dict]:
        from craftax_agent.jev_policy import JevActionPolicy, JevMacroPolicy, ObjectivePlanner
        want = (self.policy, self.planner)
        if self.jev_policy is None or self.jev_policy.env is not self.env or getattr(self.jev_policy, "_policy_id", None) != want:
            from craftax_agent.llm_memory import LLMMemory
            planner = ObjectivePlanner(self.env, memory=LLMMemory(self.history_n, self.use_scratchpad)) if self.planner else None
            if self.policy == "jev_macro":
                self.jev_policy = JevMacroPolicy(self.env, planner=planner)
            else:
                self.jev_policy = JevActionPolicy(self.env, mode={"jev_action_map": "map", "jev_action_sim": "sim"}[self.policy],
                                                  planner=planner)
            self.jev_policy._policy_id = want
        action, reply, info = self.jev_policy.decide()
        planner = getattr(self.jev_policy, "planner", None)
        info["model"] = self.jev_policy.model + (f" + {planner.model}" if planner else "")
        return action, reply, info

    def random_decide(self) -> tuple[str, str, dict]:
        return RandomPolicy(self.env).decide()

    # ---- messaging ----
    def snapshot(self) -> dict:
        t = self.obs["text"]
        mem = self.active_memory()
        return {"type": "state", "step": self.step, "seed": self.seed, "policy": self.policy, "planner": self.planner,
                "run_id": self.run_id,
                "variant": self.variant, "actions": self.actions,
                "running": self.running, "busy": self.busy, "status": self.status, "delay": self.delay,
                "frame": img_b64(self.obs["image"]),
                "obs_long": t["long_term_context"], "obs_short": t["short_term_context"],
                "legal_actions": self.env.legal_actions(),
                "total_reward": self.total_reward,
                "achievements": self.env.n_achievements(), "achievements_total": self.env.n_ach_total,
                "score": self.env.score(), "max_score": self.env.max_score,
                "last": self.history[-1] if self.history else None,
                "history": self.history[-200:],
                "tokens": {"episode": self.episode_tokens, "total": self.total_tokens,
                           "episodes": self.episodes_with_tokens},
                "goals": {**goal_board(self.env), "enabled": self.use_goals},
                "memory": {**(mem.state() if mem else {"scratchpad": "", "scratchpad_step": -1, "turns_available": 0}),
                           "history_n": self.history_n, "use_scratchpad": self.use_scratchpad,
                           "owner": "planner" if self.policy in JEV_POLICIES else "llm"},
                "model": os.environ.get("OPENAI_MODEL_NAME")}

    async def broadcast(self, msg: dict):
        data = json.dumps(msg, ensure_ascii=False)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def do_step(self, forced_action: str | None = None):
        """One decision + env step; safe to call from the loop or a single-step command."""
        if self.busy:
            return
        self.busy = True
        self.status = "deciding"
        await self.broadcast({"type": "status", "status": self.status, "busy": True})
        try:
            if forced_action is not None:
                action, reply, info = forced_action, "(manual)", {"latency_s": 0.0}
            elif self.policy == "llm":
                action, reply, info = await asyncio.to_thread(self.llm_decide)
            elif self.policy in JEV_POLICIES:
                action, reply, info = await asyncio.to_thread(self.jev_decide)
            elif self.policy == "random":
                action, reply, info = self.random_decide()
            else:  # manual: wait for the browser
                self.status = "waiting for your action"
                await self.broadcast({"type": "status", "status": self.status, "busy": True})
                action = await self.manual_queue.get()
                reply, info = "(manual)", {"latency_s": 0.0}
            self.status = "stepping"
            obs_before = self.obs
            reward, done, unlocked, hp_delta = await asyncio.to_thread(self.env_step, action)
            # the active policy sees every env step, including ones the browser took over (manual / forced),
            # so its step counter, stuck signature and objective ttl stay in sync with the world
            pol = self.jev_policy if self.policy in JEV_POLICIES else self.llm_policy if self.policy == "llm" else None
            if pol is not None and pol.env is self.env:
                pol.note_outcome(info.get("macro", "manual"), action, reward, unlocked)
            self.total_reward += reward
            spent = {"llm": (info.get("input_tokens"), info.get("output_tokens")),
                     "jev": (info.get("jev_input_tokens"), info.get("jev_output_tokens"))}
            for kind, (pt, ct) in spent.items():
                if not (pt or ct):
                    continue
                if not any(b["steps"] for b in self.episode_tokens.values()):
                    self.episodes_with_tokens += 1   # count only episodes that actually spent tokens
                for bucket in (self.episode_tokens[kind], self.total_tokens[kind]):
                    bucket["input"] += pt or 0
                    bucket["output"] += ct or 0
                    bucket["steps"] += 1
            rec = {"step": self.step, "action": action, "reward": reward, "done": done,
                   "achievements": self.env.n_achievements(), "score": self.env.score(), "reply": reply,
                   "unlocked": unlocked, "hp_delta": round(hp_delta, 2), **info}
            self.history.append(rec)
            self.log_step(rec, obs_before)
            self.step += 1
            if done:
                self.running = False
                self.status = "episode over"
            else:
                self.status = "running" if self.running else "paused"
        except Exception as e:  # surface API / env errors in the UI instead of dying
            self.running = False
            self.status = f"error: {type(e).__name__}: {e}"[:300]
        finally:
            self.busy = False
        await self.broadcast(self.snapshot())

    async def loop(self):
        while self.running:
            await self.do_step()
            if self.running:
                await asyncio.sleep(self.delay)


S = Session()
app = FastAPI(title="Craftax Agent Engine")


@app.on_event("startup")
async def _startup():
    S.status = "warming up JAX (≈60 s)"
    await asyncio.to_thread(S.build)
    await asyncio.to_thread(S.reset, 0)
    S.status = "paused"


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/actions")
async def actions():
    return {"actions": S.actions, "variant": S.variant}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    S.clients.add(ws)
    try:
        if S.obs is not None:
            await ws.send_text(json.dumps(S.snapshot(), ensure_ascii=False))
        else:
            await ws.send_text(json.dumps({"type": "status", "status": S.status, "busy": True}))
        while True:
            msg = json.loads(await ws.receive_text())
            cmd = msg.get("cmd")
            if cmd == "config":
                if "policy" in msg and msg["policy"] in ("llm", "random", "manual", *JEV_POLICIES):
                    S.policy = msg["policy"]
                if "planner" in msg:
                    S.planner = bool(msg["planner"])
                if "delay" in msg:
                    S.delay = max(0.0, float(msg["delay"]))
                if "history_n" in msg:
                    S.history_n = max(0, min(20, int(msg["history_n"])))
                if "scratchpad" in msg:
                    S.use_scratchpad = bool(msg["scratchpad"])
                if "goals" in msg:
                    S.use_goals = bool(msg["goals"])
                S.apply_llm_options()
                await S.broadcast(S.snapshot())
            elif cmd == "start":
                if not S.running and not S.busy and S.obs is not None:
                    S.running = True
                    S.status = "running"
                    S.task = asyncio.create_task(S.loop())
                    await S.broadcast({"type": "status", "status": S.status, "busy": S.busy})
            elif cmd == "pause":
                S.running = False
                S.status = "pausing…" if S.busy else "paused"
                await S.broadcast({"type": "status", "status": S.status, "busy": S.busy})
            elif cmd == "step":
                if not S.running:
                    asyncio.create_task(S.do_step())
            elif cmd == "act":  # manual action from the browser
                a = msg.get("action")
                if a in S.actions:
                    if S.policy == "manual" and S.busy:
                        await S.manual_queue.put(a)
                    elif not S.running and not S.busy:
                        asyncio.create_task(S.do_step(forced_action=a))
            elif cmd in ("reset", "reset_all", "variant"):
                S.running = False
                if S.task:
                    S.task.cancel()
                while S.busy:
                    await asyncio.sleep(0.05)
                S.status = "resetting"
                await S.broadcast({"type": "status", "status": S.status, "busy": True})
                if cmd == "variant" and msg.get("variant") in VARIANTS and msg["variant"] != S.variant:
                    S.status = f"switching to {msg['variant']} (JIT warm-up ≈60 s)"
                    await S.broadcast({"type": "status", "status": S.status, "busy": True})
                    await asyncio.to_thread(S.build, msg["variant"])
                    S.total_tokens = new_tokens()      # stats are per environment
                    S.episodes_with_tokens = 0
                    S.episodes_started = 0
                if cmd == "reset_all":  # wipe session-wide statistics too
                    S.total_tokens = new_tokens()
                    S.episodes_with_tokens = 0
                    S.episodes_started = 0
                await asyncio.to_thread(S.reset, int(msg.get("seed", S.seed)))
                S.status = "paused"
                await S.broadcast(S.snapshot())
    except WebSocketDisconnect:
        pass
    finally:
        S.clients.discard(ws)
