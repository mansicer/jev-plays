"""Kept for old commands. The memoryless single-step LLM baseline is now
    python -m craftax_agent.run_agent --policy llm --history-n 0 --no-scratchpad --no-goals --seeds <seed> --steps N
Any other flags of run_agent.py apply. `--seed` is accepted here as an alias of `--seeds`.
"""
import sys

from craftax_agent.run_agent import main

if __name__ == "__main__":
    argv = [a if a != "--seed" else "--seeds" for a in sys.argv[1:]]
    if "--policy" not in argv:
        argv = ["--policy", "llm", *argv]
    sys.argv = [sys.argv[0], *argv]
    main()
