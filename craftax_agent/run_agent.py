"""Batch runner for every policy on Craftax — same seeds, same env, one JSONL per run.

    python -m craftax_agent.run_agent --policy llm            --steps 2000 --seeds 0 1 2 [--history-n 5] [--no-scratchpad] [--no-goals]
    python -m craftax_agent.run_agent --policy jev_macro      --steps 2000 --seeds 0 1 2   # Jev-Macro
    python -m craftax_agent.run_agent --policy jev_action_map --steps 2000 --seeds 0 1 2   # Jev-Action-Map
    python -m craftax_agent.run_agent --policy jev_action_sim --steps 2000 --seeds 0 1 2   # Jev-Action-Sim
    python -m craftax_agent.run_agent --policy jev_macro --planner --ttl 25 ...            # any jev policy + LLM objective planner
    python -m craftax_agent.run_agent --policy random         --steps 2000 --seeds 0 1 2

The LLM policy is the same LLMPolicy the web engine uses (craftax_agent/llm_policy.py): memory (scratchpad),
history replay and the goal board are options, defaults match the web engine. `--render` saves a frame per step
so build_viewer.py can make a replay page. Reads TYPESAFE_API_KEY / OPENAI_* from .env.
Logs craftax_agent/runs/<tag>/<date>-<variant>-s<seed>/ (steps.jsonl, system_prompt.txt, meta.json) and merges a row into
runs/<tag>/summary.json; runs shorter than 100 steps go to runs/_smoke/ and are not summarised (layout: runlog.py).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from craftax_agent.craftax_text_env import VARIANTS, CraftaxTextEnv
from craftax_agent.random_policy import RandomPolicy
from craftax_agent.runlog import RUNS, SMOKE_STEPS, new_run_id, run_dir, run_tag


POLICIES = {"llm": "LLM", "jev_macro": "Jev-Macro", "jev_action_map": "Jev-Action-Map", "jev_action_sim": "Jev-Action-Sim",
            "random": "random"}
JEV_POLICIES = ("jev_macro", "jev_action_map", "jev_action_sim")



def make_policy(policy: str, env: CraftaxTextEnv, args):
    from craftax_agent.llm_memory import LLMMemory
    memory = LLMMemory(history_n=args.history_n, use_scratchpad=not args.no_scratchpad)   # LLM policy and planner alike
    if policy == "llm":
        from craftax_agent.llm_policy import LLMPolicy
        return LLMPolicy(env, memory=memory, use_goals=not args.no_goals)
    if policy == "random":
        return RandomPolicy(env)
    from craftax_agent.jev_policy import JevActionPolicy, JevMacroPolicy, ObjectivePlanner
    planner = ObjectivePlanner(env, ttl=args.ttl, memory=memory) if args.planner else None   # optional System Two, any jev policy
    if policy == "jev_macro":
        return JevMacroPolicy(env, conf_floor=args.conf_floor, planner=planner, chooser=args.chooser,
                              objective_source="sched" if args.objective == "sched" else "llm", seed=args.seed_now)
    return JevActionPolicy(env, mode={"jev_action_map": "map", "jev_action_sim": "sim"}[policy], planner=planner)


def run(policy: str, env: CraftaxTextEnv, seed: int, steps: int, args, tag: str) -> dict:
    obs = env.reset(seed)
    args.seed_now = seed
    pol = make_policy(policy, env, args)
    out = run_dir(tag, new_run_id(args.env, seed), smoke=steps < SMOKE_STEPS)
    out.mkdir(parents=True)
    print(f"   -> {out.relative_to(RUNS.parent.parent)}")
    (out / "system_prompt.txt").write_text(env.system_prompt)
    frames = out / "frames"
    if args.render:
        frames.mkdir(exist_ok=True)
        obs["image"].save(frames / "0000.png")
    log = open(out / "steps.jsonl", "w")
    total_reward, lat, jev_ms, tokens, sources = 0.0, [], [], 0, {}
    unlock_steps: list[int] = []
    llm_in = llm_out = 0
    t_start = time.perf_counter()
    for step in range(steps):
        obs_before = obs["text"]
        action, reply, info = pol.decide(obs) if policy == "llm" else pol.decide()
        obs, reward, done, ex = env.step(action)
        if args.render:
            obs["image"].save(frames / f"{step + 1:04d}.png")
        pol.note_outcome(info["macro"], action, reward, ex["unlocked"])
        total_reward += reward
        lat.append(info.get("latency_s", 0.0)); jev_ms.append(info.get("jev_ms", 0)); tokens += info.get("jev_input_tokens") or 0
        llm_in += info.get("input_tokens") or 0; llm_out += info.get("output_tokens") or 0
        sources[info["source"]] = sources.get(info["source"], 0) + 1
        rec = {"step": step, "action": action, "reward": reward, "done": done, "achievements": env.n_achievements(),
               "score": env.score(), "unlocked": ex["unlocked"], "hp_delta": round(ex["hp_delta"], 2), "reply": reply,
               **{k: v for k, v in info.items() if k != "probabilities"}, "probabilities": info.get("probabilities"),
               "obs_long": obs_before["long_term_context"], "obs_short": obs_before["short_term_context"]}
        log.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n"); log.flush()
        if ex["unlocked"] or step % 100 == 0 or info.get("planner_trigger") or info.get("sched_change"):
            print(f"  [{step:04d}] {action:18} ach={env.n_achievements():2d} conf={info.get('confidence', 0):.2f} "
                  f"src={info.get('source', '-'):20} {'UNLOCKED ' + ', '.join(ex['unlocked']) if ex['unlocked'] else ''}"
                  + (f"  PLAN[{info['planner_trigger']}] -> {info.get('objective')}" if info.get("planner_trigger") else "")
                  + (f"  objective -> {info.get('objective_key')}" if info.get("sched_change") else ""), flush=True)
        unlock_steps += [step] * len(ex["unlocked"])
        if done:
            break
    wall = time.perf_counter() - t_start
    ach_list = [a for a in env.ach_names.values() if a.replace(" ", "_").upper() in {x.name for x in env.Achievement
                if (env.state.achievements[x.value] > 0)}]
    row = {"source": "cli", "run_id": out.name, "variant": args.env, "policy": policy, "name": getattr(pol, "name", policy),
           "chooser": getattr(pol, "chooser", None), "objective_source": getattr(pol, "objective_source", None),
            "planner": bool(getattr(pol, "planner", None)), "seed": seed, "steps": step + 1,
            "achievements": env.n_achievements(),
            "unlocked": ach_list, "reward": round(total_reward, 2), "wall_s": round(wall, 1),
            "mean_step_s": round(sum(lat) / max(1, len(lat)), 3), "mean_jev_ms": round(sum(jev_ms) / max(1, len(jev_ms))),
            "jev_input_tokens": tokens, "sources": sources,
            "escalations": getattr(pol, "n_escalations", 0), "died": bool(done and step + 1 < steps),
            "ach_per_100": [sum(1 for u in unlock_steps if lo <= u < lo + 100) for lo in range(0, steps, 100)],
            "last_unlock_step": max(unlock_steps, default=-1),
            **({"sched": [(s_, k_) for s_, k_ in pol.sched.history]} if getattr(pol, "sched", None) else {}),
            **({"llm_input_tokens": llm_in, "llm_output_tokens": llm_out, "history_n": args.history_n,
                "scratchpad": not args.no_scratchpad, "goals": not args.no_goals, "model": pol.model} if policy == "llm" else {}),
            **({"llm_calls": pol.planner.n_calls, "mean_llm_ms": round(pol.planner.llm_ms_total / max(1, pol.planner.n_calls)),
                "history_n": args.history_n, "scratchpad": not args.no_scratchpad,
                "llm_input_tokens": pol.planner.llm_input_tokens, "llm_output_tokens": pol.planner.llm_output_tokens,
                "triggers": {t: sum(1 for h in pol.planner.history if h["trigger"] == t) for t in ("start", "reached", "board", "expired")},
                "objective_sources": {t: sum(1 for h in pol.planner.history if h["source"] == t) for t in ("llm", "code:objective-default")},
                "objectives": [(h["step"], h["trigger"], h["objective"]) for h in pol.planner.history]}
               if getattr(pol, "planner", None) else {})}
    (out / "meta.json").write_text(json.dumps(row, ensure_ascii=False, indent=1))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=[*POLICIES, "jev_macro_llm"], default="jev_macro")
    ap.add_argument("--planner", action="store_true", help="jev policies: add the LLM objective planner (System Two)")
    ap.add_argument("--env", choices=VARIANTS, default="classic")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--conf-floor", type=float, default=0.35)
    ap.add_argument("--ttl", type=int, default=25, help="--planner: objective expires after this many steps")
    ap.add_argument("--chooser", choices=["jev", "match", "random"], default="jev",
                    help="jev_macro: who picks among the options (match / random = code-only ablations, no jev call)")
    ap.add_argument("--objective", choices=["llm", "sched"], default="llm",
                    help="jev_macro without --planner: 'sched' uses the scripted objective sequence (upper-bound ablation)")
    ap.add_argument("--history-n", type=int, default=5, help="llm / --planner: past LLM turns replayed (0 = memoryless)")
    ap.add_argument("--no-scratchpad", action="store_true", help="llm / --planner: disable the model-written notes")
    ap.add_argument("--no-goals", action="store_true", help="llm: do not inject the achievement board / status notes")
    ap.add_argument("--render", action="store_true", help="save a PNG frame per step (for build_viewer.py)")
    args = ap.parse_args()
    if args.policy == "jev_macro_llm":          # old name, kept as an alias
        args.policy, args.planner = "jev_macro", True
    if args.planner and args.policy not in JEV_POLICIES:
        sys.exit("--planner applies to the jev policies only")
    if args.policy in JEV_POLICIES and args.chooser == "jev" and not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("TYPESAFE_API_KEY missing in .env")
    if (args.policy == "llm" or args.planner) and not (os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_MODEL_NAME")):
        sys.exit("OPENAI_API_KEY / OPENAI_MODEL_NAME missing in .env")
    tag = run_tag(args.policy, args.planner, args.chooser, args.objective)   # runs/<tag>/…, e.g. jev_macro_llm
    env = CraftaxTextEnv(args.env, max_timesteps=10000, render=args.render)
    rows = []
    for seed in args.seeds:
        print(f"== {args.policy} seed={seed} steps={args.steps}")
        rows.append(run(args.policy, env, seed, args.steps, args, tag))
        print(json.dumps(rows[-1], ensure_ascii=False))
    print("\nSUMMARY")
    for r in rows:
        print(f"{r.get('name', POLICIES[r['policy']]):17} seed={r['seed']} ach={r['achievements']:2d} steps={r['steps']:4d} died={r['died']} "
              f"step={r['mean_step_s']:.2f}s jev={r['mean_jev_ms']}ms tokens={r['jev_input_tokens']} esc={r['escalations']} "
              + (f"llm={r['llm_calls']}x{r['mean_llm_ms']}ms {r['triggers']} " if "llm_calls" in r else "")
              + (f"llm_tokens={r['llm_input_tokens']}/{r['llm_output_tokens']} hist={r['history_n']} " if r["policy"] == "llm" else "") + f"{r['unlocked']}")
    # merge into runs/<tag>/summary.json by (variant, seed): a rerun replaces that seed's row, other seeds stay
    if args.steps < SMOKE_STEPS:
        print(f"(summary not written: --steps {args.steps} < {SMOKE_STEPS} is treated as a smoke run, logs under runs/_smoke/)")
        return
    sp = RUNS / tag / "summary.json"
    old = json.loads(sp.read_text()) if sp.exists() else []
    key = lambda r: (r.get("variant", "classic"), r["seed"])
    merged = {key(r): r for r in old} | {key(r): r for r in rows}
    sp.write_text(json.dumps([merged[k] for k in sorted(merged)], ensure_ascii=False, indent=1))
    print(f"summary -> {sp.relative_to(RUNS.parent.parent)}")


if __name__ == "__main__":
    main()
