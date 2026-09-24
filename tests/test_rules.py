"""No API key, ~3 minutes: the harness's legality rules agree with the game's own step function
(craftax_agent/validate_rules.py: for every crafting / placing / sleeping action, legal <=> the state changes)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_legality_audit_classic(tmp_path):
    out = tmp_path / "audit.json"
    subprocess.run([sys.executable, "-m", "craftax_agent.validate_rules", "classic", "1", "200", "20", str(out)],
                   check=True, cwd=Path(__file__).resolve().parent.parent, timeout=900)
    report = json.loads(out.read_text())
    assert report["checks"] > 1000
    assert report["counter_examples"] == 0, report["examples"][:3]
    # coverage, not correctness: rare actions (e.g. Place Plant needs a sapling) may not occur in one short run
    assert len(report["never_legal"]) <= 2, report["never_legal"]
