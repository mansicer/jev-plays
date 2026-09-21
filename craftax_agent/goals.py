"""Code-maintained achievement status for the Craftax LLM agent (Craftax-Classic and Craftax).

Deliberately descriptive, never prescriptive: the harness reports which achievements are done,
which are unlocked (all prerequisite achievements done) and whether their material / station
requirements hold right now, plus factual status notes (numbers and game mechanics) when health,
needs or nearby hostile mobs warrant it. What to do with that is left to the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------- snapshot

@dataclass
class Snap:
    inv: dict
    ach: set                      # achievement names done
    hp: float; food: int; drink: int; energy: int
    mx: dict                      # max stats
    near: dict                    # "crafting_table"/"furnace" -> adjacent (incl. diagonal)
    floor: tuple | None = None    # full only: (on ladder down, monsters killed on this floor, kills needed, floor index)


def snapshot(env) -> Snap:
    state, B = env.state, env.BlockType
    inv = {k: int(getattr(state.inventory, k)) for k in ("wood", "stone", "coal", "iron", "diamond", "sapling")}
    inv["torches"] = 0 if env.classic else int(state.inventory.torches)
    inv["pickaxe"], inv["sword"] = env.tools()
    ach_arr = np.array(state.achievements) > 0
    floor = None
    if not env.classic:
        # DESCEND gate, craftax/craftax/game_logic.py:2422-2443 change_floor(): standing on ItemType.LADDER_DOWN and
        # monsters_killed[level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL (constants.py:339 = 8); the overworld counter starts
        # at 10 (world_gen/world_gen.py:665-667 "First ladder starts open"). No tool or achievement is checked.
        lvl = int(state.player_level)
        r, c = env.pos()
        on_ladder = int(state.item_map[lvl, r, c]) == env.ItemType.LADDER_DOWN.value
        floor = (on_ladder, int(state.monsters_killed[lvl]), int(env.K.MONSTERS_KILLED_TO_CLEAR_LEVEL), lvl)
    return Snap(inv=inv, ach={x.name for x in env.Achievement if ach_arr[x.value]},
                hp=float(state.player_health), food=int(state.player_food),
                drink=int(state.player_drink), energy=int(state.player_energy), mx=env.max_stats(),
                near={"crafting_table": env.near(B.CRAFTING_TABLE), "furnace": env.near(B.FURNACE)}, floor=floor)


# ----------------------------------------------------------------------------- goal table

@dataclass
class Goal:
    key: str
    title: str
    ach: str                      # achievement that marks it done
    requires: tuple = ()          # goal keys done first: the crafting/mining path (exact in classic; in Craftax chest
                                  # loot can also grant the CHEST_LOOT_ACHS below, so there it is a path, not a rule)
    needs: tuple = ()             # ("inv", item, n) | ("station", block) | ("ladder", None) — objective game requirements
    side: bool = False            # off the main tech tree (only affects ordering)


def _need_text(need, s: Snap) -> tuple[str, bool]:
    kind, what, *rest = need
    if kind == "inv":
        n = rest[0]; have = s.inv[what]
        return f"{n} {what} (have {have})", have >= n
    if kind == "ladder":          # full only (the goal is filtered out in classic, see board())
        on, killed, need, lvl = s.floor
        start = " (the overworld ladder starts open)" if lvl == 0 else ""
        return (f"standing on ladder down ({'yes' if on else 'no'}); monsters killed on this floor {killed}/{need}{start}",
                bool(on and killed >= need))
    ok = s.near[what]
    return f"adjacent {'table' if what == 'crafting_table' else what} ({'yes' if ok else 'no'})", ok


T, F = ("station", "crafting_table"), ("station", "furnace")
GOALS: list[Goal] = [
    Goal("collect_wood", "Collect Wood", "COLLECT_WOOD"),
    Goal("place_table", "Place Table", "PLACE_TABLE", ("collect_wood",), (("inv", "wood", 2),)),
    Goal("wood_pickaxe", "Make Wood Pickaxe", "MAKE_WOOD_PICKAXE", ("place_table",), (("inv", "wood", 1), T)),
    Goal("wood_sword", "Make Wood Sword", "MAKE_WOOD_SWORD", ("place_table",), (("inv", "wood", 1), T)),
    Goal("collect_stone", "Collect Stone", "COLLECT_STONE", ("wood_pickaxe",)),
    Goal("place_furnace", "Place Furnace", "PLACE_FURNACE", ("collect_stone",), (("inv", "stone", 1),)),
    Goal("stone_pickaxe", "Make Stone Pickaxe", "MAKE_STONE_PICKAXE", ("collect_stone",), (("inv", "wood", 1), ("inv", "stone", 1), T)),
    Goal("stone_sword", "Make Stone Sword", "MAKE_STONE_SWORD", ("collect_stone",), (("inv", "wood", 1), ("inv", "stone", 1), T)),
    Goal("place_stone", "Place Stone", "PLACE_STONE", ("collect_stone",), (("inv", "stone", 1),)),
    Goal("collect_coal", "Collect Coal", "COLLECT_COAL", ("wood_pickaxe",)),
    Goal("make_torch", "Make Torch", "MAKE_TORCH", ("collect_coal",), (("inv", "wood", 1), ("inv", "coal", 1), T)),
    Goal("place_torch", "Place Torch", "PLACE_TORCH", ("make_torch",), (("inv", "torches", 1),)),
    Goal("make_arrow", "Make Arrow", "MAKE_ARROW", ("collect_stone",), (("inv", "wood", 1), ("inv", "stone", 1), T)),
    Goal("collect_iron", "Collect Iron", "COLLECT_IRON", ("stone_pickaxe",)),
    Goal("iron_pickaxe", "Make Iron Pickaxe", "MAKE_IRON_PICKAXE", ("collect_iron", "place_furnace", "collect_coal"),
         (("inv", "wood", 1), ("inv", "stone", 1), ("inv", "iron", 1), ("inv", "coal", 1), T, F)),
    Goal("iron_sword", "Make Iron Sword", "MAKE_IRON_SWORD", ("collect_iron", "place_furnace", "collect_coal"),
         (("inv", "wood", 1), ("inv", "stone", 1), ("inv", "iron", 1), ("inv", "coal", 1), T, F)),
    Goal("iron_armour", "Make Iron Armour", "MAKE_IRON_ARMOUR", ("collect_iron", "place_furnace"),
         (("inv", "iron", 3), ("inv", "coal", 3), T, F)),
    Goal("collect_diamond", "Collect Diamond", "COLLECT_DIAMOND", ("iron_pickaxe",)),
    Goal("descend", "Enter Dungeon", "ENTER_DUNGEON", (), (("ladder", None),)),
    Goal("open_chest", "Open Chest", "OPEN_CHEST", ("descend",)),
    # off the main tech tree
    Goal("collect_sapling", "Collect Sapling", "COLLECT_SAPLING", side=True),
    Goal("place_plant", "Place Plant", "PLACE_PLANT", ("collect_sapling",), (("inv", "sapling", 1),), side=True),
    Goal("eat_plant", "Eat Plant", "EAT_PLANT", ("place_plant",), side=True),
    Goal("collect_drink", "Collect Drink", "COLLECT_DRINK", side=True),
    Goal("eat_cow", "Eat Cow", "EAT_COW", side=True),
    Goal("wake_up", "Wake Up", "WAKE_UP", side=True),
    Goal("defeat_zombie", "Defeat Zombie", "DEFEAT_ZOMBIE", side=True),
    Goal("defeat_skeleton", "Defeat Skeleton", "DEFEAT_SKELETON", side=True),
]
GOAL_BY_KEY = {g.key: g for g in GOALS}
# Craftax (full) only: add_items_from_chest (craftax/craftax/game_logic.py:25-165) can put coal/iron/diamond, torches,
# arrows or a pickaxe/sword of tier 1-4 in the inventory, and calculate_inventory_achievements (:2845-2965, run every
# step at :3070) then sets these achievements from the inventory alone. Classic has no chests and sets them only from
# the mining/crafting action itself, so there `requires` is exact.
CHEST_LOOT_ACHS = frozenset({"COLLECT_COAL", "COLLECT_IRON", "COLLECT_DIAMOND", "MAKE_TORCH", "MAKE_ARROW",
                             "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE",
                             "MAKE_WOOD_SWORD", "MAKE_STONE_SWORD", "MAKE_IRON_SWORD"})


def _depth(g: Goal) -> int:
    return 0 if not g.requires else 1 + max(_depth(GOAL_BY_KEY[k]) for k in g.requires)


# ----------------------------------------------------------------------------- status notes (facts only)

# Literals inside update_player_intrinsics (classic craftax_classic/game_logic.py:1303-1325, full craftax/game_logic.py
# :1925-1947): necessities = food > 0, drink > 0, (energy > 0 OR is_sleeping); recover +2.0/step asleep, +1.0 awake
# (+1 health when > 25); otherwise -0.5 asleep, -1.0 awake (-1 health when < -15). Not exported by the game.
REGEN = ("Health regenerates 1 per 26 steps awake (1 per 13 asleep) while food and drink are > 0 and energy is > 0 or "
         "you are asleep; otherwise it drops 1 per 16 steps awake (1 per 31 asleep).")
REGEN_BOSS = "On the boss floor health neither drops from starvation nor regenerates while starving."   # full :1936 * not_boss
# Decay thresholds 25/20/30 and the 0.125-per-dexterity coefficient are literals inside update_player_intrinsics
# (full craftax/game_logic.py:1869-1910; classic craftax_classic/game_logic.py:1256-1290, no dexterity, no boss floor).
_NEED_T = {"food": 25, "drink": 20, "energy": 30}


def _boss_floor(env) -> bool:
    """Full: is_fighting_boss (craftax/util/game_logic_utils.py:7-8) = player_level == num_levels - 1."""
    return not env.classic and int(env.state.player_level) == env.static_params.num_levels - 1


def need_rates(env) -> dict[str, str]:
    """Per-need decay period from the live state: floor(T / coeff) + 1 steps, coeff = 1 - 0.125 * (dexterity - 1)
    (classic: coeff 1 -> 26/21/31). Food/drink accrue at half rate asleep; nothing drains on the full boss floor."""
    if _boss_floor(env):
        return {k: "does not drop on the boss floor" for k in _NEED_T}
    coeff = 1.0 if env.classic else 1.0 - 0.125 * (int(env.state.player_dexterity) - 1)
    p = {k: int(t // coeff) + 1 for k, t in _NEED_T.items()}
    return {"food": f"drops 1 per {p['food']} steps awake (half as fast asleep)",
            "drink": f"drops 1 per {p['drink']} steps awake (half as fast asleep)",
            "energy": f"drops 1 per {p['energy']} steps awake"}


def status_notes(env) -> list[str]:
    s = snapshot(env)
    notes = []
    boss = _boss_floor(env)
    if s.hp <= 4:
        notes.append(f"health {s.hp:.0f}/{s.mx['health']}. {REGEN}" + (f" {REGEN_BOSS}" if boss else ""))
    rates = need_rates(env)
    for k in ("food", "drink", "energy"):
        v = getattr(s, k)
        if v <= 3:
            if boss:
                notes.append(f"{k} {v}/{s.mx[k]}: {rates[k]}.")
            elif k == "energy":
                notes.append(f"energy {v}/{s.mx['energy']}: {rates['energy']}; at 0 while awake health stops regenerating "
                             f"and drops 1 per 16 steps (asleep it still regenerates 1 per 13 steps).")
            else:
                notes.append(f"{k} {v}/{s.mx[k]}: {rates[k]}; at 0 health stops regenerating and drops 1 per 16 steps "
                             f"awake (1 per 31 asleep).")
    hostile = [m for m in env.visible_mobs() if m["hostile"] and m["dist"] <= 2]
    if hostile:
        r, c = env.pos()
        for m in hostile:
            notes.append(f"{m['name']} at (x={m['c']}, y={m['r']}), {m['dist']} tile{'s' if m['dist'] != 1 else ''} away, "
                         f"health {m['health']:g}: {m['attack']}")
        dmg = {m["name"]: env.player_damage(m) for m in hostile}          # full: includes that mob's defenses
        if len(set(dmg.values())) == 1:
            notes.append(f"Your melee damage per hit: {next(iter(dmg.values())):g}.")
        else:
            notes.append("Your melee damage per hit: " + ", ".join(f"{n} {d:g}" for n, d in dmg.items()) + ".")
    return notes


# ----------------------------------------------------------------------------- board

def board(env) -> dict:
    s = snapshot(env)
    valid = {a.name for a in env.Achievement}
    goals = sorted((g for g in GOALS if g.ach in valid), key=lambda g: (g.side, _depth(g)))
    done = [g for g in goals if g.ach in s.ach]
    done_keys = {g.key for g in done}
    met, unmet, locked = [], [], []
    for g in goals:
        if g.key in done_keys:
            continue
        missing_prereqs = [GOAL_BY_KEY[k].title for k in g.requires if k not in done_keys]
        if missing_prereqs:
            locked.append({"title": g.title, "after": missing_prereqs, "loot": not env.classic and g.ach in CHEST_LOOT_ACHS})
            continue
        reqs = [{"text": t, "ok": ok} for t, ok in (_need_text(n, s) for n in g.needs)]
        (met if all(r["ok"] for r in reqs) else unmet).append({"title": g.title, "reqs": reqs})
    return {"done": [g.title for g in done], "met": met, "unmet": unmet, "locked": locked,
            "total": len(goals), "notes": status_notes(env), "chest_loot": not env.classic}


def board_text(b: dict) -> str:
    lines = []
    if b["notes"]:
        lines.append("Status notes:")
        lines += [f" - {n}" for n in b["notes"]]
        lines.append("")
    lines.append(f"Achievement status (tracked by the harness; {len(b['done'])}/{b['total']} done):")
    lines.append(" Done: " + (", ".join(b["done"]) or "none"))
    if b["met"]:
        lines.append(" Unlocked, requirements met now:")
        for g in b["met"]:
            req = "; ".join(r["text"] for r in g["reqs"])
            lines.append(f"  - {g['title']}" + (f" — {req}" if req else ""))
    if b["unmet"]:
        lines.append(" Unlocked, requirements not met:")
        for g in b["unmet"]:
            lines.append(f"  - {g['title']} — " + "; ".join(r["text"] for r in g["reqs"]))
    if b["locked"]:
        groups: dict[str, list[str]] = {}
        for g in b["locked"]:
            groups.setdefault(", ".join(g["after"]), []).append(g["title"])
        loot = ("; 'after X' is the crafting/mining path, in dungeons chest loot can also grant ores, torches, arrows, "
                "pickaxes and swords" if b.get("chest_loot") else "")
        lines.append(f" Locked ({len(b['locked'])}{loot}): " + "; ".join(f"{', '.join(t)} (after {after})" for after, t in groups.items()))
    return "\n".join(lines)
