"""Build replay.gif and viewer.html for a rendered run directory.

Usage: python craftax_agent/build_viewer.py craftax_agent/runs/llm/20260920-021618-classic-s0
"""
import base64
import json
import sys
from pathlib import Path

from PIL import Image

run_dir = Path(sys.argv[1])
steps = [json.loads(l) for l in (run_dir / "steps.jsonl").open()]
frames = sorted((run_dir / "frames").glob("*.png"))
assert len(frames) == len(steps) + 1, (len(frames), len(steps))

# GIF at 4x nearest-neighbour
imgs = [Image.open(f).convert("RGB") for f in frames]
big = [im.resize((im.width * 4, im.height * 4), Image.NEAREST) for im in imgs]
big[0].save(run_dir / "replay.gif", save_all=True, append_images=big[1:], duration=350, loop=0)

def b64(p):
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()

meta = {
    "run": run_dir.name,
    "model": None,
    "steps": len(steps),
    "total_reward": sum(s["reward"] for s in steps),
    "achievements": steps[-1]["achievements"],
    "frame_w": imgs[0].width, "frame_h": imgs[0].height,
}
# model name is not in the log; read from env file if present
try:
    for line in (run_dir.parent.parent.parent / ".env").read_text().splitlines():
        if line.startswith("OPENAI_MODEL_NAME="):
            meta["model"] = line.split("=", 1)[1].strip()
except Exception:
    pass

data = {
    "meta": meta,
    "frames": [b64(f) for f in frames],
    "steps": [{k: s[k] for k in ("step", "action", "reward", "achievements", "latency_s",
                                  "input_tokens", "output_tokens", "reply", "obs_long", "obs_short")}
              for s in steps],
}
template = (Path(__file__).parent / "viewer_template.html").read_text()
html = template.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
(run_dir / "viewer.html").write_text(html)
print("wrote", run_dir / "replay.gif", "and", run_dir / "viewer.html", f"({len(html)//1024} KB)")
