"""jev (TypeSafe AI System One) policies for Craftax. Three variants share one interface
(decide() -> (primitive action, reply text, info); note_outcome(); reset()):

  Jev-Macro       JevMacroPolicy   code builds a macro-action candidate table (targets, stations, crafts,
                                   combat, explore) grounded to primitive actions; jev picks one (Choice).
  Jev-Action-Map  JevActionPolicy  jev picks among the env's primitive actions; "Do" is renamed by what it
                  (mode="map")     acts on (Collect wood / Mine stone / Attack zombie …) and every action whose
                                   simulated outcome changes nothing is dropped (it equals Noop).
  Jev-Action-Sim  JevActionPolicy  jev picks among the env's primitive actions, each described by a code-
                  (mode="sim")     computed fact plus the simulated one-step outcome (env.step on a copy).

Everything shown to jev is descriptive, never prescriptive — the same rule as the LLM harness
(`craftax_text_env.observe`, `goals.board_text`, `goals.status_notes`): game mechanics, numbers and the
current status of each option, no advice. Code owns legality, pathing, memory, safety invariants and the
low-confidence fallback. See reports/04-jev-craftax-agent-design.md.
"""
from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from craftax_agent.craftax_text_env import MOB_NAMES
from craftax_agent.goals import GOAL_BY_KEY, GOALS, board as goal_board, snapshot as goal_snapshot

DIRS = {"Move North": (-1, 0), "Move South": (1, 0), "Move West": (0, -1), "Move East": (0, 1)}
DIR_WORD = {"Move North": "north", "Move South": "south", "Move West": "west", "Move East": "east"}
# passive mobs -> kill achievement; killing ANY passive mob gives food +6 (full: game_logic_utils.attack_mob :104-133,
# achievement from MOB_ACHIEVEMENT_MAP row 0; classic: cows only, game_logic.py:108-137)
FOOD_MOBS = {"cow": "EAT_COW", "bat": "EAT_BAT", "snail": "EAT_SNAIL"}
TIER = ["none", "wood", "stone", "iron", "diamond"]

# block -> (verb, effect of Do, achievement, pickaxe tier needed)   — game facts only.
# Effect strings are format templates: "{food}" / "{drink}" are the current caps from env.max_stats()
# (classic: fixed 9; full: 7 + 2*dexterity, game_logic_utils.py:365-372) -> always effect.format(**mx).
# Full-only blocks (fire_tree .. chest) are skipped in classic, whose BlockType lacks them (game_logic.py:183-439).
RESOURCE = {
    "tree": ("chop", "gives 1 wood; the tree becomes grass", "COLLECT_WOOD", 0),
    "stone": ("mine", "gives 1 stone", "COLLECT_STONE", 1),
    "coal": ("mine", "gives 1 coal", "COLLECT_COAL", 1),
    "iron": ("mine", "gives 1 iron", "COLLECT_IRON", 2),
    "diamond": ("mine", "gives 1 diamond", "COLLECT_DIAMOND", 3),
    "water": ("drink from", "drink +1 per Do (max {drink})", "COLLECT_DRINK", 0),
    "ripe_plant": ("eat", "food +4 (max {food})", "EAT_PLANT", 0),
    # Craftax (full) only
    "fire_tree": ("chop", "gives 1 wood; the tree becomes fire grass", "COLLECT_WOOD", 0),
    "ice_shrub": ("chop", "gives 1 wood; the shrub becomes ice grass", "COLLECT_WOOD", 0),
    "stalagmite": ("mine", "gives 1 stone; becomes path", "COLLECT_STONE", 1),
    "sapphire": ("mine", "gives 1 sapphire", "COLLECT_SAPPHIRE", 4),
    "ruby": ("mine", "gives 1 ruby", "COLLECT_RUBY", 4),
    "fountain": ("drink from", "drink +1 per Do (max {drink})", "COLLECT_DRINK", 0),
    "chest": ("open", "random loot (torches, coal/iron/diamond/sapphire/ruby, arrows, potions, books, bow); "
                      "the chest becomes path", "OPEN_CHEST", 0),
}
# semantic (map-mode) option names for Do on a RESOURCE block; default = "<Verb> <block>"
DO_LABEL = {"tree": "Collect wood", "ripe_plant": "Eat plant", "fountain": "Drink from fountain"}
MATERIAL = {"tree": "wood", "stone": "stone", "coal": "coal", "iron": "iron", "diamond": "diamond",   # block -> inventory key
            "fire_tree": "wood", "ice_shrub": "wood", "stalagmite": "stone", "sapphire": "sapphire", "ruby": "ruby"}
CRAFT_ACH = {  # primitive action -> achievement
    "Place Table": "PLACE_TABLE", "Place Furnace": "PLACE_FURNACE", "Place Stone": "PLACE_STONE",
    "Place Plant": "PLACE_PLANT", "Make Wood Pickaxe": "MAKE_WOOD_PICKAXE", "Make Stone Pickaxe": "MAKE_STONE_PICKAXE",
    "Make Iron Pickaxe": "MAKE_IRON_PICKAXE", "Make Wood Sword": "MAKE_WOOD_SWORD",
    "Make Stone Sword": "MAKE_STONE_SWORD", "Make Iron Sword": "MAKE_IRON_SWORD",
    "Make Diamond Pickaxe": "MAKE_DIAMOND_PICKAXE", "Make Diamond Sword": "MAKE_DIAMOND_SWORD",
    "Make Iron Armour": "MAKE_IRON_ARMOUR", "Make Diamond Armour": "MAKE_DIAMOND_ARMOUR",
    "Make Arrow": "MAKE_ARROW", "Make Torch": "MAKE_TORCH", "Place Torch": "PLACE_TORCH",
}
CRAFT_FACT = {  # mechanics (cost / effect), as in the game manual
    "Place Table": "costs 2 wood; crafting station for tools",
    "Place Furnace": "costs 1 stone; with a table, enables iron tools",
    "Place Stone": "costs 1 stone; places a stone block on the faced tile",
    "Place Plant": "costs 1 sapling; the plant ripens over time and can then be eaten",
    "Make Wood Pickaxe": "costs 1 wood; can mine stone and coal",
    "Make Stone Pickaxe": "costs 1 wood + 1 stone; can mine iron",
    "Make Iron Pickaxe": "costs 1 wood + 1 stone + 1 iron + 1 coal; can mine diamond",
    "Make Wood Sword": "costs 1 wood; attack damage 2 (no sword: 1)",
    "Make Stone Sword": "costs 1 wood + 1 stone; attack damage 3",
    "Make Iron Sword": "costs 1 wood + 1 stone + 1 iron + 1 coal; attack damage 5",
    # Craftax (full) only
    "Make Diamond Pickaxe": "costs 1 wood + 3 diamond; can mine sapphire and ruby",
    "Make Diamond Sword": "costs 1 wood + 2 diamond; attack damage 8",
    "Make Iron Armour": "costs 3 iron + 3 coal per piece",
    "Make Diamond Armour": "costs 3 diamond per piece",
    "Make Arrow": "costs 1 wood + 1 stone; gives 2 arrows",
    "Make Torch": "costs 1 wood + 1 coal; gives 4 torches",
    "Place Torch": "costs 1 torch; lights the tiles around it",
}
GOAL_BY_TITLE = {g.title: g for g in GOALS}   # the board reports goals by title; resolve back to the Goal object


def _need_ok(need, snap) -> bool:
    """Evaluate one structured Goal.needs entry — ("inv", item, n), ("station", block) or ("ladder", None) — against the snapshot."""
    kind, what, *rest = need
    if kind == "ladder":
        on, killed, need_kills, _lvl = snap.floor
        return bool(on and killed >= need_kills)
    return snap.inv[what] >= rest[0] if kind == "inv" else snap.near[what]


def _serves_goal(block: str, goal, snap) -> bool:
    """True if gathering this block is the goal's own achievement or covers one of its unmet material needs."""
    if goal.ach == RESOURCE[block][2]:
        return True
    mat = MATERIAL.get(block)
    return mat is not None and any(n[0] == "inv" and n[1] == mat and not _need_ok(n, snap) for n in goal.needs)


def bucket(d: int) -> str:
    if d <= 1:
        return "adjacent"
    if d <= 4:
        return f"near ({d} steps)"
    return f"far ({d} steps)"


def ach_status(name: str, done: set) -> str:
    """'achievement X: not yet unlocked', or '' once it is (an unlocked achievement is no longer a reason to act)."""
    if name in done:
        return ""
    return f"achievement {name.replace('_', ' ').title()}: not yet unlocked"


_SEMI = re.compile(r"(;\s*)+(?=;)|;\s*$|;\s*(?=\))")


def _clean(desc: str) -> str:
    """Collapse the '; ;' / trailing ';' left by empty ach_status() pieces."""
    return _SEMI.sub("", desc).replace("; )", ")").strip()


_CLASSIC_KILL_ACH = {"zombie": "DEFEAT_ZOMBIE", "cow": "EAT_COW", "skeleton": "DEFEAT_SKELETON"}   # classic game_logic.py:96,129,162


def kill_ach(env, mob: dict) -> str | None:
    """Achievement the game awards for killing this visible_mobs() entry. Full: the game's own
    MOB_ACHIEVEMENT_MAP[cls, type_id] (constants.py:553-590, used by game_logic_utils.py:69-79), so the member
    name always matches env.Achievement (e.g. the game's spelling DEFEAT_ORC_SOLIDER); classic: the hard-coded three."""
    if mob.get("projectile"):
        return None
    if env.classic or mob.get("type_id") is None:
        return _CLASSIC_KILL_ACH.get(mob["name"])
    v = int(np.array(env.K.MOB_ACHIEVEMENT_MAP)[mob["cls"], mob["type_id"]])
    return env.Achievement(v).name if v else None


def sleep_risk_text(env) -> str:
    """Melee damage against a sleeping player. Classic: zombie only, select(is_sleeping, 7, 2) (classic game_logic.py:831-835).
    Full: every melee mob's base damage x(1 + 2.5*is_sleeping) = 3.5x (game_logic.py:1189-1195, MOB_TYPE_DAMAGE_MAPPING
    constants.py:262-273); projectiles are not multiplied but wake the player (:1681-1697)."""
    if env.classic:
        return "a zombie hit does 7 to a sleeping player instead of 2"
    K = env.K
    dmg = np.array(K.MOB_TYPE_DAMAGE_MAPPING)[:, K.MobType.MELEE.value].sum(-1)    # total base damage per melee type
    ex = "; ".join(f"{MOB_NAMES[t]} {dmg[t]:g} -> {dmg[t] * 3.5:g}" for t in range(min(5, len(dmg))))
    return (f"melee hits do 3.5x damage to a sleeping player, before armour ({ex}; a sleeping hit by any of these "
            f"except the zombie exceeds the 9 starting health); projectiles are not multiplied but still wake you")


@dataclass
class Macro:
    key: str
    desc: str
    action: str          # primitive action executed if chosen
    kind: str            # resource | craft | survival | combat | explore | wait
    goal_match: bool = False   # code-side only (fallback): advances the first unlocked tech-tree goal


# ----------------------------------------------------------------------------- geometry

def bfs(walkable: np.ndarray, start: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """Shortest step counts over `walkable` from start; parent map for path reconstruction."""
    H, W = walkable.shape
    dist = np.full((H, W), -1, dtype=int)
    parent: dict = {}
    dist[start] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and walkable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    return dist, parent


def first_step(parent: dict, start, goal) -> str | None:
    cur = goal
    while parent.get(cur) not in (None, start):
        cur = parent[cur]
    if parent.get(cur) != start:
        return None
    dr, dc = cur[0] - start[0], cur[1] - start[1]
    for name, v in DIRS.items():
        if v == (dr, dc):
            return name
    return None


def dir_to(frm, to) -> str | None:
    dr, dc = to[0] - frm[0], to[1] - frm[1]
    for name, v in DIRS.items():
        if v == (dr, dc):
            return name
    return None


# ----------------------------------------------------------------------------- candidate table

class MacroTable:
    """Builds the per-step candidate table from a CraftaxTextEnv (classic or full)."""

    def __init__(self, env):
        self.env = env
        B = env.BlockType
        # Walkable = the game's move_player rule on a block id: not in K.SOLID_BLOCKS (env._solid_ids; classic includes
        # WATER, full includes WALL_MOSS but not DARKNESS/ICE_SHRUB) and not WATER/LAVA (full COLLISION_LAND_CREATURE
        # constants.py:176; classic lets you step onto LAVA but that sets health to 0, so it is kept out on purpose).
        # Mobs are masked per step in build(); env.walkable(r, c) is the same rule on the live map.
        self.walk_ids = {b.value for b in B if b.value not in env._solid_ids
                         and b not in (B.WATER, B.LAVA, B.OUT_OF_BOUNDS, B.INVALID)}
        self.name_of = {b.value: b.name.lower() for b in B}
        self.known: dict[int, np.ndarray] = {}     # level -> remembered map (-1 = never seen); harness memory
        self.pocket = None                          # shelter target, fixed once found until used or invalid
        self.last_ctx = {"hostile_d": None, "enclosed": False}
        self.plant_age = None                       # steps since Place Plant (set by the policy)

    def reset(self):
        self.known = {}
        self.pocket = None
        self.plant_age = None

    def build(self, recent: list[dict], objective=None, notes: str = "", rng=None, rules: list | None = None) -> tuple[list[Macro], dict]:
        """Capabilities only (2026-09-21 evening, report 06 §10): every feasible macro is listed, nothing is filtered,
        ranked or pre-empted by code; the order is shuffled each step (no position prior). `objective` / `notes`
        are the LLM planner's intent and enter the state as facts; code never annotates which option serves them
        (goal_match is kept for the code-only `match` chooser and is not shown to jev)."""
        env = self.env
        s = env.state
        snap = goal_snapshot(env)
        gb = goal_board(env)
        legal = set(env.legal_actions())
        pos = env.pos()
        true_map = env.level_map()
        H, W = true_map.shape
        r0, r1 = max(0, pos[0] - env.view_r), min(H, pos[0] + env.view_r + 1)
        c0, c1 = max(0, pos[1] - env.view_c), min(W, pos[1] + env.view_c + 1)
        known = self.known.setdefault(env.level(), np.full((H, W), -1, dtype=int))
        known[r0:r1, c0:c1] = true_map[r0:r1, c0:c1]
        m = known
        vis = np.zeros((H, W), dtype=bool); vis[r0:r1, c0:c1] = True
        mobs = [(mm["name"], mm["r"], mm["c"], mm) for mm in env.visible_mobs() if not mm.get("projectile")]
        mob_at = {(mr, mc): n for n, mr, mc, _ in mobs}
        pick, sword = env.tools()
        mx = snap.mx
        done_ach = snap.ach
        light = float(s.light_level)
        goal = objective.goal if objective else None
        need_mat, need_by = material_needs(snap, env)

        walkable = np.isin(m, list(self.walk_ids))
        for (mr, mc) in mob_at:
            walkable[mr, mc] = False
        walkable[pos] = True
        dist, parent = bfs(walkable, pos)
        seen_frac = f"{int((known >= 0).sum())} of {H * W} tiles seen"
        facing = env.facing()
        face_name = self.name_of.get(int(m[facing]), "out of bounds") if 0 <= facing[0] < H and 0 <= facing[1] < W else "out of bounds"
        face_mob = mob_at.get(facing)
        hostile = [(n, mr, mc, abs(mr - pos[0]) + abs(mc - pos[1]), m_) for n, mr, mc, m_ in mobs if m_["hostile"]]
        hostile.sort(key=lambda t: t[3])
        bid_of = {n: v for v, n in self.name_of.items()}
        macros: list[Macro] = []

        def nearest_reachable(block_id: int):
            best = None
            rr, cc = np.nonzero(m == block_id)
            for tr, tc in zip(rr, cc):
                for dr, dc in DIRS.values():
                    ar, ac = tr + dr, tc + dc
                    if 0 <= ar < H and 0 <= ac < W and dist[ar, ac] >= 0:
                        d = int(dist[ar, ac])
                        if best is None or d < best[2]:
                            best = ((int(tr), int(tc)), (int(ar), int(ac)), d)
            return best

        def approach(target, adj) -> str | None:
            if pos == adj or (abs(pos[0] - target[0]) + abs(pos[1] - target[1]) == 1):
                if facing == target:
                    return "Do"
                return dir_to(pos, target)
            return first_step(parent, pos, adj)

        def where(target, d) -> str:
            mem = "" if vis[target] else ", remembered from earlier, currently out of view"
            return f"{bucket(d)}{mem}"

        def serves(bname: str) -> bool:      # code-only: used by the `match` chooser, never shown to jev
            return bool(goal and _serves_goal(bname, goal, snap))

        # --- resources / drink / food plants: everything reachable and minable
        for bname, (verb, effect, ach, need) in RESOURCE.items():
            bid = bid_of.get(bname)
            if bid is None or pick < need:
                continue
            hit = nearest_reachable(bid)
            if not hit:
                continue
            target, adj, d = hit
            prim = approach(target, adj)
            if prim is None:
                continue
            mat = MATERIAL.get(bname)
            extra = ""
            if bname in ("water", "fountain"):
                extra = f"; drink is {snap.drink}/{mx['drink']}"
            elif bname == "ripe_plant":
                extra = f"; food is {snap.food}/{mx['food']}"
            elif mat is not None:
                short = need_mat.get(mat, 0) - snap.inv.get(mat, 0)
                extra = (f"; still needed: {short} more {mat} for {', '.join(need_by[mat][:3])} (have {snap.inv.get(mat, 0)})" if short > 0
                         else f"; no remaining achievement needs more {mat} (have {snap.inv.get(mat, 0)})")
            step_desc = ("you are facing it, Do acts on it" if prim == "Do"
                         else "this step turns to face it" if d + 1 <= 1 else "this step moves toward it")
            macros.append(Macro(bname, f"{bname.replace('_', ' ')}: {where(target, d + 1)}; {step_desc}; Do {effect.format(**mx)}; "
                                       f"{ach_status(ach, done_ach)}{extra}", prim, "resource", goal_match=serves(bname)))

        # --- stations: walk back to a remembered table / furnace an unlocked goal needs (materials ready)
        station_d = {}
        for station, label in (("crafting_table", "table"), ("furnace", "furnace")):
            bid = bid_of.get(station)
            hit = nearest_reachable(bid) if bid is not None else None
            station_d[station] = hit[2] if hit else None
            if snap.near[station] or not hit:
                continue
            needs_it = [g.title for g in (GOAL_BY_TITLE[x["title"]] for x in gb["met"] + gb["unmet"])
                        if ("station", station) in g.needs and all(_need_ok(n, snap) for n in g.needs if n[0] == "inv")]
            if not needs_it:
                continue
            target, adj, d = hit
            if pos == adj:
                continue
            prim = first_step(parent, pos, adj)
            if prim:
                macros.append(Macro(label, f"{label}: {where(target, d)}; this step moves toward it; standing next to it "
                                           f"allows: {', '.join(needs_it[:3])} (materials for these are in the inventory)",
                                    prim, "craft", goal_match=bool(goal and goal.title in needs_it)))

        # --- sapling / plant
        want_grass = "COLLECT_SAPLING" not in done_ach or (snap.inv.get("sapling", 0) > 0 and "PLACE_PLANT" not in done_ach)
        if want_grass:
            if face_name == "grass" and "Do" in legal and not face_mob and "COLLECT_SAPLING" not in done_ach:
                macros.append(Macro("grass_sapling", "Do on the faced grass: 1 in 10 chance of 1 sapling per Do; "
                                                     + ach_status("COLLECT_SAPLING", done_ach), "Do", "resource",
                                    goal_match=bool(goal and goal.ach == "COLLECT_SAPLING")))
            elif face_name != "grass" or face_mob:
                hit = nearest_reachable(bid_of["grass"])
                if hit:
                    target, adj, d = hit
                    prim = approach(target, adj)
                    if prim:
                        macros.append(Macro("grass", f"face a grass tile ({bucket(d + 1)}): Do on grass gives a sapling with 1 in 10 chance; "
                                                     f"Place Plant needs a faced grass tile; sapling in inventory: {snap.inv.get('sapling', 0)}",
                                            prim, "resource", goal_match=bool(goal and goal.ach in ("COLLECT_SAPLING", "PLACE_PLANT"))))
        if "EAT_PLANT" not in done_ach and "plant" in bid_of and not any(mm.key == "ripe_plant" for mm in macros):
            hit = nearest_reachable(bid_of["plant"])
            if hit:
                target, adj, d = hit
                prim = approach(target, adj)
                if prim and prim != "Do":
                    age = f"planted {self.plant_age} steps ago" if self.plant_age is not None else "planting step unknown"
                    macros.append(Macro("plant", f"walk to the placed plant {where(target, d + 1)}: {age}; a plant ripens 600 steps after "
                                                 f"planting; eating a ripe plant gives food +4; {ach_status('EAT_PLANT', done_ach)}",
                                        prim, "resource", goal_match=bool(goal and goal.ach == "EAT_PLANT")))

        # --- mobs
        for n, mr, mc, m_ in mobs:
            if n in FOOD_MOBS:
                d = abs(mr - pos[0]) + abs(mc - pos[1])
                adj_tiles = [(mr + dr, mc + dc) for dr, dc in DIRS.values()]
                best = min((t for t in adj_tiles if 0 <= t[0] < H and 0 <= t[1] < W and dist[t] >= 0),
                           key=lambda t: dist[t], default=None)
                if d == 1:
                    prim = "Do" if facing == (mr, mc) else dir_to(pos, (mr, mc))
                elif best is not None:
                    prim = first_step(parent, pos, best)
                else:
                    prim = None
                if prim:
                    macros.append(Macro("cow", f"{n}: {bucket(d)}; it has {m_['health']:g} health, your hit does {env.player_damage(m_):g}; "
                                               f"killing it gives food +6 (max {mx['food']}); {ach_status(kill_ach(env, m_) or FOOD_MOBS[n], done_ach)}; food is {snap.food}/{mx['food']}",
                                        prim, "resource", goal_match=bool(goal and goal.ach == (kill_ach(env, m_) or FOOD_MOBS[n]))))
                break
        if hostile:
            n, mr, mc, d, hm = hostile[0]
            ka = kill_ach(env, hm)
            gm = bool(goal and ka and goal.ach == ka)
            hits = int(np.ceil(hm["health"] / max(env.player_damage(hm), 1e-9)))
            if d == 1:
                prim = "Do" if facing == (mr, mc) else dir_to(pos, (mr, mc))
                macros.append(Macro("attack", f"attack the adjacent {n}: it has {hm['health']:g} health, your hit does {env.player_damage(hm):g} "
                                              f"({TIER[sword]} sword, {hits} hit{'s' if hits != 1 else ''} to kill); it {hm['attack']}; "
                                              f"your health {snap.hp:.0f}/{mx['health']}" + (f"; {ach_status(ka, done_ach)}" if ka else ""),
                                    prim, "combat", goal_match=gm))
            elif not hm.get("projectile") and max(abs(mr - pos[0]), abs(mc - pos[1])) <= 2:
                mid = ((pos[0] + mr) // 2, (pos[1] + mc) // 2)
                if d == 2 and (mr == pos[0] or mc == pos[1]) and walkable[mid]:
                    macros.append(Macro("attack", f"step {DIR_WORD[dir_to(pos, mid)]} next to the {n} (2 tiles away in a straight line): "
                                                  f"you then face it; it has {hm['health']:g} health, your hit does {env.player_damage(hm):g} "
                                                  f"({TIER[sword]} sword, {hits} hits to kill); it {hm['attack']}; your health {snap.hp:.0f}/{mx['health']}"
                                                  + (f"; {ach_status(ka, done_ach)}" if ka else ""), dir_to(pos, mid), "combat", goal_match=gm))
                else:
                    macros.append(Macro("hold", f"Noop: the {n} is {d} tiles away, not in a straight line; mobs hit only from an "
                                                f"orthogonally adjacent tile and it steps toward you on most steps; once adjacent, a move "
                                                f"toward it turns you to face it without moving; it has {hm['health']:g} health, your hit does "
                                                f"{env.player_damage(hm):g}; your health {snap.hp:.0f}/{mx['health']}", "Noop", "combat", goal_match=gm))
            best_mv, best_d = None, d
            for mv, (dr, dc) in DIRS.items():
                nr, nc = pos[0] + dr, pos[1] + dc
                if 0 <= nr < H and 0 <= nc < W and walkable[nr, nc]:
                    nd = abs(nr - mr) + abs(nc - mc)
                    if nd > best_d:
                        best_mv, best_d = mv, nd
            if best_mv:
                chase = "; within 10 tiles it steps toward you on about 3 of 4 steps at your speed" if n == "zombie" else ""
                macros.append(Macro("flee", f"move {DIR_WORD[best_mv]}: distance to the {n} goes from {d} to {best_d}; "
                                            f"it {hm['attack']}{chase}; your health {snap.hp:.0f}/{mx['health']}", best_mv, "combat"))

        # --- crafting / placing: every legal not-done craft; Place Table / Furnace again when an unlocked goal needs the station here
        inv_txt = ", ".join(f"{k} {v}" for k, v in snap.inv.items() if k in ("wood", "stone", "coal", "iron", "diamond", "sapling") and v) or "no materials"
        STATION_OF = {"Place Table": ("crafting_table", "table", "wood", 2), "Place Furnace": ("furnace", "furnace", "stone", 1)}
        station_goals = [GOAL_BY_TITLE[x["title"]] for x in gb["met"] + gb["unmet"]]
        for a in env.action_names:
            if a not in legal or a not in CRAFT_ACH:
                continue
            ach = CRAFT_ACH[a]
            fact = CRAFT_FACT.get(a)
            if ach in done_ach:
                st = STATION_OF.get(a)
                if not st or snap.near[st[0]]:
                    continue
                needs_here = [g for g in station_goals if ("station", st[0]) in g.needs and all(_need_ok(n, snap) for n in g.needs if n[0] == "inv")
                              and snap.inv[st[2]] >= st[3] + sum(n[2] for n in g.needs if n[0] == "inv" and n[1] == st[2])]
                if not needs_here:
                    continue
                dd = station_d.get(st[0])
                near_txt = f"the nearest known {st[1]} is {dd} steps away" if dd is not None else f"no {st[1]} is known on the map"
                macros.append(Macro(f"build_{st[1]}", f"{a} here: {fact}; a table and a furnace both within 1 tile are required for "
                                                       f"{', '.join(g.title for g in needs_here[:3])}; {near_txt}; inventory: {inv_txt}",
                                    a, "craft", goal_match=bool(goal and goal in needs_here)))
                continue
            macros.append(Macro(a.lower().replace(" ", "_"), f"{a}: " + (f"{fact}; " if fact else "") + f"{ach_status(ach, done_ach)}; inventory: {inv_txt}",
                                a, "craft", goal_match=bool(goal and goal.ach == ach)))

        # --- sleep / shelter (facts only: open sides, spawn chance, sleeping damage)
        def mob_open(t, frm) -> bool:
            if not env.in_map(*t) or env.solid_at(*t) or self.name_of.get(int(m[t])) == "lava":
                return False
            return any(env.in_map(t[0] + dr, t[1] + dc) and (t[0] + dr, t[1] + dc) != frm
                       and not env.solid_at(t[0] + dr, t[1] + dc) for dr, dc in DIRS.values())
        open_sides = [DIR_WORD[mv] for mv, (dr, dc) in DIRS.items() if mob_open((pos[0] + dr, pos[1] + dc), pos)]
        enclosed = not open_sides
        self.last_ctx = {"hostile_d": hostile[0][3] if hostile else None, "enclosed": enclosed}
        if snap.energy >= mx["energy"] or enclosed:
            self.pocket = None
        if "Sleep" in legal and snap.energy < mx["energy"]:
            spawn = 0.02 + 0.1 * (1 - light) ** 2 if env.classic else None
            night = (f"about {10 * (mx['energy'] - snap.energy)} steps asleep until full; zombies appear 10-13 tiles away with chance "
                     f"{spawn:.0%} per step at this light, hit only from an orthogonally adjacent tile and cannot enter solid blocks" if env.classic
                     else f"about {10 * (mx['energy'] - snap.energy)} steps asleep until full; mobs cannot enter solid blocks")
            macros.append(Macro("sleep", f"Sleep here: energy {snap.energy}/{mx['energy']}; actions are ignored until energy is full; "
                                         f"{sleep_risk_text(env)}; {night}; light is {light:.2f}; open sides around you: "
                                         f"{', '.join(open_sides) if open_sides else 'none (enclosed)'}; hostile mobs in view: "
                                         f"{', '.join(f'{n} {bucket(d)}' for n, _, _, d, _ in hostile) or 'none'}; " + ach_status("WAKE_UP", done_ach),
                                "Sleep", "survival", goal_match=bool(goal and goal.ach == "WAKE_UP")))
            if not enclosed and pick >= 1:
                if self.pocket and not self.pocket_ok(m, dist, pos, self.pocket):
                    self.pocket = None
                if self.pocket is None:
                    self.pocket = self.find_pocket(m, dist, pos, max_steps=25)
                if self.pocket:
                    O, S1, S2, dO = self.pocket
                    n_mine = sum(1 for t in (S1, S2) if self.name_of.get(int(m[t])) != "path")
                    macros.append(Macro("shelter", f"shelter: stone {where(S1, dO + 1)}; walk {dO} steps, dig a 2-tile pocket into it "
                                                   f"({n_mine} to mine), step back to the pocket mouth, seal it with 1 stone (you have "
                                                   f"{snap.inv.get('stone', 0)}) and sleep with solid blocks on all four sides; about "
                                                   f"{dO + 2 * n_mine + 4} steps until sealed; this step: {self.shelter_step(m, pos, facing, self.pocket, legal)}; "
                                                   f"energy {snap.energy}/{mx['energy']}, health {snap.hp:.0f}/{mx['health']}; {night}; light is {light:.2f}",
                                        self.shelter_step(m, pos, facing, self.pocket, legal), "survival", goal_match=bool(goal and goal.ach == "WAKE_UP")))
        if "Rest" in legal:
            macros.append(Macro("rest", f"Rest: health {snap.hp:.0f}/{mx['health']}; does not speed up healing (health regenerates "
                                        "1 per 26 steps whether resting or not, while food, drink and energy are > 0); it locks "
                                        "you into Noop until health is full, food or drink reaches 0, or a mob or projectile hits "
                                        f"you; food and drink keep draining (food {snap.food}/{mx['food']}, drink {snap.drink}/{mx['drink']})",
                                "Rest", "survival"))

        # --- explore (nearest frontier) and cave (frontier with the most known path tiles around it)
        unknown = m < 0
        frontier = np.zeros((H, W), dtype=bool)
        frontier[:-1, :] |= unknown[1:, :]; frontier[1:, :] |= unknown[:-1, :]
        frontier[:, :-1] |= unknown[:, 1:]; frontier[:, 1:] |= unknown[:, :-1]
        frontier &= (dist > 0)
        fr_, fc_ = np.nonzero(frontier)
        if len(fr_):
            k = int(np.argmin(dist[fr_, fc_]))
            goal_tile = (int(fr_[k]), int(fc_[k]))
            mv = first_step(parent, pos, goal_tile)
            if mv:
                macros.append(Macro("explore", f"move {DIR_WORD[mv]} toward the nearest unseen tiles ({bucket(int(dist[goal_tile]))}); {seen_frac}",
                                    mv, "explore", goal_match=bool(goal) and not any(mm.goal_match for mm in macros)))
            pm = (m == bid_of["path"]) if "path" in bid_of else None
            if pm is not None and pm.any():
                score = np.array([int(pm[max(0, r - 3):r + 4, max(0, c - 3):c + 4].sum()) for r, c in zip(fr_, fc_)])
                k2 = int(np.lexsort((dist[fr_, fc_], -score))[0])
                cave_tile = (int(fr_[k2]), int(fc_[k2]))
                mv2 = first_step(parent, pos, cave_tile)
                if mv2 and cave_tile != goal_tile and score[k2] > 0:
                    macros.append(Macro("cave", f"move {DIR_WORD[mv2]} toward the tunnel area: {int(score[k2])} path tiles known within 3 tiles of "
                                                f"the target ({bucket(int(dist[cave_tile]))}), {int(pm.sum())} path tiles known in all; skeletons appear "
                                                f"only on path tiles at least 10 tiles from you (5% per step), have 3 health and shoot arrows doing 2; "
                                                f"your sword is {TIER[sword]}, health {snap.hp:.0f}/{mx['health']}; {ach_status('DEFEAT_SKELETON', done_ach)}",
                                        mv2, "explore", goal_match=bool(goal and goal.ach == "DEFEAT_SKELETON")))
        # --- walled in: mine out, or stay if a hostile waits outside
        walled = int((dist >= 0).sum()) <= 3 and not len(fr_)
        if walled and hostile and hostile[0][3] <= 2:
            n_, _, _, d_, _ = hostile[0]
            macros.append(Macro("stay", f"Noop: stay enclosed; a {n_} is {d_} tiles away outside and cannot reach you through solid blocks; "
                                        f"health {snap.hp:.0f}/{mx['health']} (regenerates 1 per 26 steps while food, drink and energy are > 0)",
                                "Noop", "wait"))
        if walled and pick >= 1:
            outs = []
            for mv, (dr, dc) in DIRS.items():
                t1 = (pos[0] + dr, pos[1] + dc); t2 = (pos[0] + 2 * dr, pos[1] + 2 * dc)
                if not env.in_map(*t1) or self.name_of.get(int(m[t1])) not in ("stone", "coal", "iron", "diamond"):
                    continue
                beyond = env.in_map(*t2) and (walkable[t2] or m[t2] < 0)
                outs.append((0 if beyond else 1, mv, t1))
            if outs:
                outs.sort()
                _, mv, t1 = outs[0]
                what = self.name_of.get(int(m[t1]))
                macros.append(Macro("dig_out", f"mine the {what} {DIR_WORD[mv]} of you: solid blocks on all four sides, no tile is reachable; "
                                               f"Do on it gives 1 {MATERIAL.get(what, what)} and opens the way {'back out' if outs[0][0] == 0 else DIR_WORD[mv]}",
                                    "Do" if facing == t1 else mv, "explore", goal_match=bool(goal)))
        if not macros:
            macros = [Macro("wait", "Noop: no other action is possible right now", "Noop", "wait")]
        for mm in macros:
            mm.desc = _clean(mm.desc)
        if rng is not None:
            rng.shuffle(macros)

        facts = {
            "achievements": {
                "done": f"{len(gb['done'])}/{gb['total']}",
                "unlocked, requirements met now": [g["title"] for g in gb["met"]],
                "unlocked, requirements not met": [g["title"] + " — " + "; ".join(r["text"] for r in g["reqs"]) for g in gb["unmet"]],
                "locked": len(gb["locked"]),
            },
            "notes": gb["notes"] or "none",
            "player": {"health": f"{snap.hp:.0f}/{mx['health']}", "food": f"{snap.food}/{mx['food']}",
                       "drink": f"{snap.drink}/{mx['drink']}", "energy": f"{snap.energy}/{mx['energy']}",
                       "light": f"{light:.2f} ({'bright' if light > 0.6 else 'dim' if light > 0.3 else 'dark, night'})",
                       "facing": (face_mob or face_name)},
            "inventory": {k: v for k, v in snap.inv.items() if v and k not in ("pickaxe", "sword")}
                         | {"pickaxe": TIER[pick], "sword": TIER[sword]},
            "hostile mobs in view": [f"{n} {bucket(d)}, {hm['health']:g} health, {hm['attack']}" for n, _, _, d, hm in hostile] or "none",
            "recent": [f"{rec['macro']} -> {rec['outcome']}" for rec in recent[-5:]] or "episode just started",
        }
        if objective:
            facts["current objective"] = objective.text
        if rules:
            facts["planner rules (highest priority first)"] = list(rules)
        if notes:
            facts["planner notes"] = notes
        return macros, facts

    def known_summary(self) -> str:
        """Nearest remembered instance of each notable block (planner input): 'water at (x=..,y=..) 7 steps away; ...'."""
        env = self.env
        k = self.known.get(env.level())
        if k is None:
            return "nothing seen yet"
        pos = env.pos()
        out = []
        for name in ("water", "tree", "stone", "coal", "iron", "diamond", "crafting_table", "furnace", "plant", "ripe_plant", "path"):
            bid = next((v for v, n in self.name_of.items() if n == name), None)
            if bid is None:
                continue
            rr, cc = np.nonzero(k == bid)
            if not len(rr):
                continue
            d = np.abs(rr - pos[0]) + np.abs(cc - pos[1])
            i = int(np.argmin(d))
            out.append(f"{name.replace('crafting_table', 'table')} at (x={int(cc[i])}, y={int(rr[i])}) {int(d[i])} tiles away" + (f" ({len(rr)} known)" if len(rr) > 1 else ""))
        return "; ".join(out) or "nothing notable seen yet"

    def known_names(self) -> set:
        """Block names ever seen on the current level (harness memory)."""
        k = self.known.get(self.env.level())
        return set() if k is None else {self.name_of[int(v)] for v in np.unique(k) if v >= 0 and int(v) in self.name_of}

    # ---- shelter: a 2-tile pocket dug into solid rock, sealed from inside ----
    def find_pocket(self, m, dist, pos, max_steps: int):
        """(mouth O, S1, S2, steps to O): O reachable & walkable; S1, S2 stone or already-dug path; S3 and the lateral
        neighbours of S1 / S2 known solid. Nearest by walking distance."""
        H, W = m.shape
        solid = np.isin(m, list(self.env._solid_ids))
        stone = next(v for v, n in self.name_of.items() if n == "stone")
        path = next(v for v, n in self.name_of.items() if n == "path")
        best = None
        rr, cc = np.nonzero(dist >= 0)
        for orow, ocol in zip(rr, cc):
            d0 = int(dist[orow, ocol])
            if d0 > max_steps or (best and d0 >= best[3]):
                continue
            for dr, dc in DIRS.values():
                s1 = (orow + dr, ocol + dc); s2 = (orow + 2 * dr, ocol + 2 * dc); s3 = (orow + 3 * dr, ocol + 3 * dc)
                if not all(0 <= t[0] < H and 0 <= t[1] < W for t in (s1, s2, s3)):
                    continue
                if m[s1] not in (stone, path) or m[s2] not in (stone, path) or not solid[s3]:
                    continue
                lat = [(t[0] + dc, t[1] + dr) for t in (s1, s2)] + [(t[0] - dc, t[1] - dr) for t in (s1, s2)]
                if not all(0 <= t[0] < H and 0 <= t[1] < W and solid[t] for t in lat):
                    continue
                if any(m[t] == path and dist[t] < 0 for t in (s1, s2)):   # dug but unreachable: something stands in it
                    continue
                best = ((int(orow), int(ocol)), (int(s1[0]), int(s1[1])), (int(s2[0]), int(s2[1])), d0)
        return best

    def pocket_ok(self, m, dist, pos, pocket) -> bool:
        """A fixed shelter target is still usable: walls solid, S1/S2 stone or path, and we can reach its mouth or are inside."""
        O, S1, S2, _ = pocket
        H, W = m.shape
        solid = np.isin(m, list(self.env._solid_ids))
        stone = next(v for v, n in self.name_of.items() if n == "stone")
        path = next(v for v, n in self.name_of.items() if n == "path")
        dr, dc = S1[0] - O[0], S1[1] - O[1]
        s3 = (S2[0] + dr, S2[1] + dc)
        lat = [(t[0] + dc, t[1] + dr) for t in (S1, S2)] + [(t[0] - dc, t[1] - dr) for t in (S1, S2)]
        if not all(0 <= t[0] < H and 0 <= t[1] < W and solid[t] for t in lat + [s3]):
            return False
        if m[S1] not in (stone, path) or m[S2] not in (stone, path):
            return False
        return pos in (O, S1, S2) or dist[O] >= 0

    def shelter_step(self, m, pos, facing, pocket, legal) -> str:
        """Next primitive for the shelter plan given the current position / facing / known map."""
        O, S1, S2, _ = pocket
        path = next(v for v, n in self.name_of.items() if n == "path")
        walkable = np.isin(m, list(self.walk_ids)); walkable[pos] = True
        dist, parent = bfs(walkable, pos)
        if pos == S1:
            if m[O] != path and not walkable[O]:            # sealed
                return "Sleep" if "Sleep" in legal else "Noop"
            if m[S2] != path:                               # S2 still stone: face it (blocked move turns) and mine
                return "Do" if facing == S2 else dir_to(pos, S2)
            if facing == O:
                return "Place Stone" if "Place Stone" in legal else "Noop"
            return dir_to(pos, S2)                          # step in, then back out facing the mouth
        if pos == S2:
            return dir_to(pos, S1)
        if pos == O:
            if m[S1] != path:
                return "Do" if facing == S1 else dir_to(pos, S1)
            return dir_to(pos, S1)
        return first_step(parent, pos, O) or "Noop"


# ----------------------------------------------------------------------------- System Two: objective planner (LLM)

def objective_fact(gb: dict, objective) -> str | None:
    """`current objective` as a fact for jev: goal title + the board's requirement text. None if the objective
    is not on the board's unlocked-not-done list (reached or locked meanwhile)."""
    if objective is None:
        return None
    by_title = {g["title"]: g for g in gb["met"] + gb["unmet"]}
    g = by_title.get(objective.title)
    if g is None:
        return None
    return objective.title + (" — " + "; ".join(r["text"] for r in g["reqs"]) if g["reqs"] else "")


OBJECTIVE_HINT = (" `current objective`, `planner rules` and `planner notes` are what the planner (a slower model) selected and "
                  "wrote for the coming steps; the rules are listed highest priority first.")


# ----------------------------------------------------------------------------- objective scheduler (code)

@dataclass
class Objective:
    key: str                 # goal key (goals.GOALS) | need:drink | need:food | need:energy
    title: str
    goal: object = None      # goals.Goal for goal nodes
    text: str = ""           # the `current objective` fact given to jev
    source: str = "code:sched"
    stock: dict = field(default_factory=dict)   # material targets gathered while this node is active (no trips back)


# Classic tech tree in the order the scheduler works through it (report 06 §8.3): the two cheap sapling achievements
# right after the wood sword, both iron tools before the opportunistic mob / plant / diamond nodes.
ORDER = ["collect_wood", "place_table", "wood_pickaxe", "wood_sword", "collect_sapling", "place_plant",
         "collect_stone", "place_furnace", "stone_pickaxe", "stone_sword", "place_stone", "collect_drink",
         "collect_coal", "collect_iron", "iron_pickaxe", "iron_sword", "eat_cow", "defeat_zombie",
         "defeat_skeleton", "collect_diamond", "eat_plant",
         # full-game extras keep their board order after the classic set
         "make_torch", "place_torch", "make_arrow", "iron_armour", "descend", "open_chest", "wake_up"]
PLANT_RIPE_STEPS = 600      # classic game_logic.py:1339
# gather targets per node: wood for every tool up to stone tier + a table next to the quarry; stone for furnace, stone
# tools, Place Stone and a spare; coal / iron for both iron tools plus a table + furnace next to the ore
STOCK = {"collect_wood": {"wood": 8}, "collect_stone": {"stone": 5}, "collect_coal": {"coal": 2},
         "collect_iron": {"iron": 2, "wood": 4, "stone": 3, "coal": 2}}


def material_needs(snap, env) -> tuple[dict, dict]:
    """Material still required by not-done achievements: {item: total}, {item: [titles]}. Classic adds 2 wood + 1 stone
    for a table + furnace next to the iron seam while the iron sword is not done (cheaper than walking back)."""
    valid = {a.name for a in env.Achievement}
    need, by = {}, {}
    for g in GOALS:
        if g.ach not in valid or g.ach in snap.ach:
            continue
        for n in g.needs:
            if n[0] == "inv":
                need[n[1]] = need.get(n[1], 0) + n[2]
                by.setdefault(n[1], []).append(g.title)
    if env.classic and "MAKE_IRON_SWORD" not in snap.ach and "MAKE_IRON_PICKAXE" in valid:
        need["wood"] = need.get("wood", 0) + 2; by.setdefault("wood", []).append("a table next to the iron")
        need["stone"] = need.get("stone", 0) + 1; by.setdefault("stone", []).append("a furnace next to the iron")
    return need, by


def looking_for(obj: Objective, snap) -> str | None:
    """What the objective's explore step is searching for (a fact for the explore option)."""
    if obj.key == "need:drink" or obj.key == "collect_drink":
        return "water"
    if obj.key in ("need:food", "eat_cow"):
        return "cow"
    if obj.key == "need:energy":
        return "stone wall to dig a sleeping pocket into"
    if obj.goal is None:
        return None
    for block, (_v, _e, ach, _n) in RESOURCE.items():
        if ach == obj.goal.ach:
            return block
    for n in obj.goal.needs:
        if n[0] == "inv" and snap.inv.get(n[1], 0) < n[2]:
            return {"wood": "tree", "stone": "stone", "coal": "coal", "iron": "iron", "diamond": "diamond"}.get(n[1], n[1])
    return None


class Scheduler:
    """Deterministic objective sequence over the tech tree (System Two in code). Need nodes (drink / energy / food)
    pre-empt the sequence with hysteresis; opportunistic nodes (cow, zombie, skeleton, ripe plant) are skipped until
    their precondition holds. The LLM planner, when present, only overrides this on a stall (see ObjectivePlanner)."""

    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.step, self.plant_step, self.plant_pos, self.need = 0, None, None, None
        self.node, self.node_since, self.last_unlock_step = None, 0, 0
        self.max_index = -1              # furthest ORDER node ever selected: passed nodes are not re-opened for stock
        self.history: list = []          # (step, key)

    def tick(self):
        self.step += 1

    def note(self, unlocked: list[str]):
        if unlocked:
            self.last_unlock_step = self.step
        if "Place Plant" in unlocked:
            self.plant_step, self.plant_pos = self.step, self.env.facing()

    @property
    def node_age(self) -> int:
        return self.step - self.node_since

    def update(self, gb: dict, snap, ctx: dict) -> Objective | None:
        from craftax_agent.goals import need_rates
        env, mx = self.env, snap.mx
        # need hysteresis: enter low, leave full
        if self.need == "drink" and snap.drink >= mx["drink"]:
            self.need = None
        if self.need == "food" and snap.food >= mx["food"] - 1:
            self.need = None
        if self.need == "energy" and snap.energy >= mx["energy"]:
            self.need = None
        obj = None
        threat = ctx.get("threat")          # (name, distance, health, attack text, your damage) of the nearest hostile <= 2 tiles
        if threat and not ctx.get("enclosed"):
            n_, d_, hp_, atk_, dmg_ = threat
            hits = int(np.ceil(hp_ / max(dmg_, 1e-9)))
            text = (f"{n_} {d_} tile{'s' if d_ != 1 else ''} away: it has {hp_:g} health, your hit does {dmg_:g} ({hits} hit{'s' if hits != 1 else ''} "
                    f"to kill); it {atk_}; it moves at your speed; your health {snap.hp:.0f}/{mx['health']}")
            obj = Objective("need:safety", f"deal with the {n_}", None, text)
            self._commit(obj)
            return obj
        # a critically low need (<= 1) pre-empts the active one when it is worse than what is being restored
        if self.need and not ctx.get("enclosed"):
            cur = getattr(snap, self.need)
            for k in ("drink", "food", "energy"):
                if k != self.need and getattr(snap, k) <= 1 and getattr(snap, k) < cur:
                    self.need = k
                    break
        if self.need is None:
            if snap.drink <= 4:
                self.need = "drink"
            elif snap.energy <= 2 or (ctx["light"] < 0.35 and snap.energy <= 5):
                self.need = "energy"
            elif snap.food <= 4:
                self.need = "food"
        if self.need:
            rates = need_rates(env)
            k = self.need
            v = getattr(snap, k)
            if k == "energy":
                text = (f"energy {v}/{mx['energy']}: {rates['energy']}; at 0 while awake health stops regenerating and drops 1 per 16 steps; "
                        f"sleeping restores 1 energy per 10 steps and ends only when energy is full; {sleep_risk_text(env)}; "
                        f"mobs attack from an orthogonally adjacent tile and cannot enter solid blocks; light {ctx['light']:.2f}")
            else:
                text = f"{k} {v}/{mx[k]}: {rates[k]}; at 0 health stops regenerating and drops 1 per 16 steps awake"
            obj = Objective(f"need:{k}", f"restore {k}", None, text)
        else:
            valid = {a.name for a in env.Achievement}
            done_keys = {g.key for g in GOALS if g.ach in snap.ach}
            blocks = {"wood": "tree", "stone": "stone", "coal": "coal", "iron": "iron"}
            for idx, key in enumerate(ORDER):
                g = GOAL_BY_KEY[key]
                if g.ach not in valid or any(k not in done_keys for k in g.requires):
                    continue
                own = key.replace("collect_", "")
                stock = ({k_: v_ for k_, v_ in STOCK.get(key, {}).items() if snap.inv.get(k_, 0) < v_ and (k_ == own or blocks[k_] in ctx["known"])}
                         if idx >= self.max_index else {})   # side materials only when seen on the map; a passed node is never re-opened
                if key in done_keys and not stock:
                    continue
                self.max_index = max(self.max_index, idx)
                if key == "collect_drink" and "water" not in ctx["known"] and snap.drink > 6:
                    continue
                if key == "eat_cow" and "cow" not in ctx["visible"] and snap.food > 6:
                    continue
                if key == "defeat_zombie" and "zombie" not in ctx["visible"]:
                    continue
                if key == "defeat_skeleton" and (ctx["sword"] < 2 or snap.hp < 7 or snap.food < 5 or snap.drink < 5):
                    continue
                if key == "collect_diamond" and (snap.hp < 6 or snap.food < 5 or snap.drink < 5):
                    continue
                if key == "eat_plant" and (self.plant_step is None or self.step - self.plant_step < PLANT_RIPE_STEPS):
                    continue
                if key == "wake_up" and snap.energy > 2:
                    continue
                text = objective_fact(gb, g) or g.title
                if stock:
                    text += "; gathering " + ", ".join(f"{k_} {snap.inv.get(k_, 0)}/{v_}" for k_, v_ in stock.items())
                obj = Objective(key, g.title, g, text, stock=stock)
                break
        self._commit(obj)
        return obj

    def _commit(self, obj):
        new_key = obj.key if obj else None
        if new_key != self.node:
            self.node, self.node_since = new_key, self.step
            self.history.append((self.step, new_key))


# option vocabulary shown to the planner so its rules use words the reflex model can match literally
OPTION_VOCAB = ("attack (hit the adjacent hostile mob / step next to one in a straight line), hold (stand still until a diagonal "
                "mob steps adjacent), flee (step away from the nearest hostile), tree / stone / coal / iron / diamond (walk to and "
                "mine the nearest one), water (walk to water and drink), cow (walk to and kill the cow), grass_sapling / grass "
                "(Do on grass for a sapling), plant (walk back to the placed plant), ripe_plant (eat it), table / furnace (walk "
                "to the remembered station), build_table / build_furnace (place a new one here), make_* / place_* (craft or place "
                "now), sleep (sleep here), shelter (dig a 2-tile pocket into stone, seal it, sleep), dig_out / stay (when walled "
                "in), explore (nearest unseen tiles), cave (toward path tiles, where skeletons are)")


@dataclass
class ObjectivePlanner:
    """LLM on a fixed cadence: picks ONE unlocked achievement as the `current objective` fed to jev as a fact.

    Cadence — checked at the start of every awake step, before the candidate table is built; any hit = one LLM call:
      start    first decision of the episode
      reached  the objective's achievement is now unlocked
      board    the set of unlocked-not-done goals changed since the objective was set (any new achievement)
      expired  `ttl` steps since the last call (sleeping steps count; the call happens on the first awake step)
    Never triggered by low confidence, stuck detection, danger or hostiles: emergencies stay with code invariants + jev.

    Code guarantees: the objective is always one of the board's unlocked-not-done goals. An invalid / unparsable /
    failed LLM answer -> first unlocked main-line goal in GOALS order (source "code:objective-default").
    Only the title (+ the board's requirement text) reaches jev; the LLM's <reason> is logged, never fed back.

    Memory: the same LLMMemory as the LLM policy — the last `history_n` planner turns (board + status -> answer) are
    replayed, and the LLM keeps a <scratchpad> across calls (what it tried, what failed, remembered locations).
    """
    env: object
    ttl: int = 25
    model: str = ""
    client: object = None
    memory: object = None             # LLMMemory; default LLMMemory() (5 turns + scratchpad)
    mode: str = "cadence"             # cadence (Action policies) | anomaly (Jev-Macro: only when the code scheduler stalls)
    objective: object = None          # goals.Goal
    plan: str = ""                    # the LLM's concrete intent lines for the reflex model (facts in the jev state)
    rules: list = field(default_factory=list)   # standing, unconditional, priority-ordered rules (first: hostile adjacent)
    event_cooldown: int = 5           # an event trigger (hostile within 2 tiles / a need crossing 4) waits this long after any call
    source: str = ""                  # llm | code:objective-default
    reason: str = ""
    trigger_last: str = ""
    step: int = 0                     # env steps so far (ticked by the policy after every env.step, asleep included)
    set_step: int = -1
    last_call_step: int = -1
    eligible_at_set: frozenset = field(default_factory=frozenset)
    n_calls: int = 0
    llm_ms_total: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    history: list = field(default_factory=list)   # [{step, trigger, objective, source, reason, ms}]

    def __post_init__(self):
        from craftax_agent.llm_memory import LLMMemory
        if self.memory is None:
            self.memory = LLMMemory()
        if self.client is None:
            import os
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL") or None,
                                 max_retries=6)   # transient connection errors must not kill a 500-step run
            self.model = self.model or os.environ["OPENAI_MODEL_NAME"]

    def reset(self):
        self.objective, self.source, self.reason, self.trigger_last, self.plan, self.rules = None, "", "", "", "", []
        self.step, self.set_step, self.last_call_step, self.eligible_at_set = 0, -1, -1, frozenset()
        self.n_calls, self.llm_ms_total, self.llm_input_tokens, self.llm_output_tokens, self.history = 0, 0.0, 0, 0, []
        self.memory.clear()

    def tick(self):
        self.step += 1

    def trigger(self, gb: dict, stalled: bool = False, event: str | None = None) -> str | None:
        eligible = frozenset(g["title"] for g in gb["met"] + gb["unmet"])
        if event and self.objective is not None and self.step - self.last_call_step >= self.event_cooldown:
            return "event"
        if self.mode == "anomaly":
            # the code scheduler owns the objective; the LLM is asked only when it has stalled, and its answer
            # overrides the scheduler until reached or for ttl steps
            if self.objective is not None and (self.objective.title in gb["done"] or self.step - self.set_step >= self.ttl):
                self.objective = None
            return "stalled" if stalled and self.objective is None else None
        if self.objective is None:
            return "start"
        if self.objective.title in gb["done"]:
            return "reached"
        if eligible != self.eligible_at_set:
            return "board"
        if self.step - self.last_call_step >= self.ttl:
            return "expired"
        return None

    def update(self, gb: dict, recent: list[dict], stalled: bool = False, stall_text: str = "", map_text: str = "",
               event: str | None = None) -> dict | None:
        """Run the cadence for this step. Returns an info dict when the objective was (re)set, else None."""
        trig = self.trigger(gb, stalled, event)
        if trig is None:
            return None
        eligible = [g["title"] for g in gb["met"] + gb["unmet"]]
        prev = self.objective
        why = {"start": "episode start", "reached": f"reached: {prev.title if prev else ''}",
               "board": "the set of unlocked goals changed", "expired": f"expired after {self.ttl} steps",
               "stalled": stall_text or "the scheduler's objective made no progress",
               "event": f"event: {event}"}[trig]
        title, reason, ms, toks, plan, rules = self._ask(gb, recent, eligible, prev, why, map_text)
        self.plan, self.rules = plan, rules
        goal = GOAL_BY_TITLE.get(title) if title in eligible else None
        source = "llm"
        if goal is None:
            goal = next((GOAL_BY_TITLE[g.title] for g in GOALS if not g.side and g.title in eligible), None) \
                   or (GOAL_BY_TITLE[eligible[0]] if eligible else None)
            source = "code:objective-default"
        self.objective, self.source, self.reason, self.trigger_last = goal, source, reason, trig
        self.set_step = self.last_call_step = self.step
        self.eligible_at_set = frozenset(eligible)
        self.n_calls += 1
        self.llm_ms_total += ms
        self.llm_input_tokens += toks[0]; self.llm_output_tokens += toks[1]
        rec = {"step": self.step, "trigger": trig, "objective": goal.title if goal else None, "source": source,
               "reason": reason, "plan": plan, "rules": rules, "llm_ms": round(ms), "llm_input_tokens": toks[0], "llm_output_tokens": toks[1],
               "llm_answer": title}
        self.history.append(rec)
        return rec

    def annotate(self, info: dict, reply: str, plan_info: dict | None, objective_text: str | None) -> str:
        """Add the objective fields to a policy's info dict and reply text (same for every jev policy)."""
        info["objective"] = objective_text
        info["objective_source"] = self.source
        info["objective_age"] = self.step - self.set_step
        reply += f"\nobjective: {objective_text}  [{self.source}, set {self.step - self.set_step} steps ago]"
        if plan_info:
            info.update({"planner_trigger": plan_info["trigger"], "llm_ms": plan_info["llm_ms"],
                         "input_tokens": plan_info["llm_input_tokens"], "output_tokens": plan_info["llm_output_tokens"],
                         "planner_reason": plan_info["reason"], "planner_answer": plan_info["llm_answer"], "planner_plan": plan_info.get("plan", ""),
                         "planner_rules": plan_info.get("rules", []),
                         "context_turns": min(self.memory.history_n, len(self.memory.turns) - 1),
                         "scratchpad": self.memory.scratchpad})
            reply += f"\nplanner ({plan_info['trigger']}, {plan_info['llm_ms']}ms): {plan_info['reason']}"
        return reply

    def _status(self, recent: list[dict]) -> str:
        env = self.env
        snap = goal_snapshot(env)
        mx = snap.mx
        pick, sword = env.tools()
        hostile = [f"{m['name']} at distance {abs(m['r'] - env.pos()[0]) + abs(m['c'] - env.pos()[1])}, {m['health']:g} health, {m['attack']}"
                   for m in env.visible_mobs() if m.get("hostile")]
        inv = ", ".join(f"{k} {v}" for k, v in snap.inv.items() if v and k not in ("pickaxe", "sword")) or "no materials"
        return "\n".join([
            f"Health {snap.hp:.0f}/{mx['health']}, food {snap.food}/{mx['food']}, drink {snap.drink}/{mx['drink']}, "
            f"energy {snap.energy}/{mx['energy']}, light {float(env.state.light_level):.2f}",
            f"Inventory: {inv}; pickaxe {TIER[pick]}, sword {TIER[sword]}",
            "Hostile mobs in view: " + ("; ".join(hostile) or "none"),
            "Recent steps: " + ("; ".join(f"{r['macro']} -> {r['outcome']}" for r in recent[-5:]) or "episode just started"),
        ])

    NOTE_TOPICS = ("which objectives you set so far and how they ended, what blocked them, remembered locations "
                   "with coordinates (e.g. 'water at (49,21)')")

    def _ask(self, gb, recent, eligible, prev, why, map_text: str = "") -> tuple[str | None, str, float, tuple[int, int], str, list]:
        from craftax_agent.goals import board_text
        # replayable core of this turn (board + status + map + what happened to the previous objective); the eligible list,
        # the notes block and the format instructions are restated every turn and stripped from the replay
        hist_msg = (f"Step {self.step}.\n{board_text(gb)}\n\n{self._status(recent)}\n"
                    + (f"Known map (nearest of each): {map_text}\n" if map_text else "")
                    + f"\nPrevious objective: {prev.title if prev else 'none'} ({why}).")
        msg = (f"{hist_msg}\nEligible objectives (unlocked, not done): {'; '.join(eligible)}\n\n"
               + (self.memory.notes_block() + "\n\n" if self.memory.use_scratchpad else "")
               + f"Choose exactly one eligible objective for roughly the next {self.ttl} steps, and write instructions for the fast "
               "reflex model that picks each step's action. The reflex model sees every feasible option with its facts "
               "(distances, costs, mob health, open sides...), the objective title with its requirement text, your rules and your "
               "plan; it matches text LITERALLY: it cannot evaluate conditions (never write if / unless / avoid / only when), "
               "does no arithmetic, and follows whichever line matches the current facts most strongly. Option names it sees: "
               f"{OPTION_VOCAB}.\n"
               "<rules>: standing rules that stay valid until you are asked again, one per line, highest priority first, each "
               "written as 'SITUATION: OPTION NAME' using the option names above; the situation must be a fact the reflex "
               "model can read off the state (a number, a mob in view, an item held), never a condition on the future. "
               "You decide what the rules cover and which thresholds to use. 3 to 5 rules.\n"
               "<plan>: what to do for the objective, concrete, at most 3 short lines (coordinates from the known map when useful).\n"
               "Reply with <objective>TITLE</objective>, <rules>...</rules>, <plan>...</plan> and <reason>at most 30 words</reason>."
               + self.memory.ask_suffix(self.NOTE_TOPICS))
        t0 = time.perf_counter()
        try:
            resp = self.client.responses.create(model=self.model, instructions=self.env.system_prompt,
                                                input=self.memory.messages(msg))
            text = resp.output_text or ""
            u = resp.usage
            toks = (int(getattr(u, "input_tokens", 0) or 0), int(getattr(u, "output_tokens", 0) or 0))
        except Exception as e:      # the planner must never kill the loop
            return None, f"(planner error: {type(e).__name__}: {e})"[:200], (time.perf_counter() - t0) * 1000, (0, 0), self.plan, self.rules
        ms = (time.perf_counter() - t0) * 1000
        self.memory.record(hist_msg, text, self.step)
        m = re.search(r"<objective>(.*?)</objective>", text, re.S | re.I)
        r = re.search(r"<reason>(.*?)</reason>", text, re.S | re.I)
        pl = re.search(r"<plan>(.*?)</plan>", text, re.S | re.I)
        plan = "; ".join(x.strip(" -*") for x in pl.group(1).strip().splitlines() if x.strip(" -*")) if pl else ""
        ru = re.search(r"<rules>(.*?)</rules>", text, re.S | re.I)
        rules = [x.strip(" -*0123456789.") for x in ru.group(1).strip().splitlines() if x.strip(" -*0123456789.")] if ru else []
        title = m.group(1).strip() if m else None
        if title and title not in eligible:      # tolerate case / punctuation differences
            norm = {t.lower().strip(" ."): t for t in eligible}
            title = norm.get(title.lower().strip(" ."), title)
        return title, (r.group(1).strip() if r else text.strip()[:200]), ms, toks, plan, rules


# ----------------------------------------------------------------------------- policy

@dataclass
class JevMacroPolicy:
    """Jev-Macro: jev picks a macro from the code-built candidate table; confidence gate + deterministic fallback."""
    env: object
    conf_floor: float = 0.35          # below this (or 4 steps without progress): deterministic fallback
    model: str = "jev-latest"
    planner: ObjectivePlanner | None = None   # optional System Two: sets `current objective` on a fixed cadence
    recent: list = field(default_factory=list)
    stuck: int = 0
    n_escalations: int = 0
    _client: object = None
    _table: MacroTable | None = None
    _last_sig: tuple | None = None
    sched: Scheduler | None = None     # code objective scheduler: only with objective_source="sched" (scripted upper bound)
    objective: Objective | None = None
    chooser: str = "jev"               # jev | match (code picks an option serving the objective, no jev) | random
    objective_source: str = "llm"      # llm (planner) | sched | none
    seed: int = 0
    _rng: object = None
    _seen_hostiles: frozenset = frozenset()
    _seen_notes: frozenset = frozenset()

    @property
    def name(self) -> str:
        base = {"jev": "Jev-Macro", "match": "Match-Macro", "random": "Random-Macro"}[self.chooser]
        return base + ("+LLM" if self.planner else "+sched" if self.objective_source == "sched" else "")

    def __post_init__(self):
        from typesafe_sdk import RetryPolicy, TypeSafeClient
        self._client = TypeSafeClient(model=self.model, timeout=20, retry=RetryPolicy(max_retries=3))
        self._table = MacroTable(self.env)
        self.sched = Scheduler(self.env)
        assert self.chooser in ("jev", "match", "random"), self.chooser
        self._rng = np.random.default_rng(self.seed)

    def reset(self):
        self.recent, self.stuck, self._last_sig, self.n_escalations, self.objective = [], 0, None, 0, None
        self._seen_hostiles, self._seen_notes = frozenset(), frozenset()
        self._table.reset()
        self.sched.reset()
        if self.planner:
            self.planner.reset()

    # signature used for progress detection
    def _sig(self):
        s = self.env.state
        inv = s.inventory
        return (self.env.pos(), int(self.env.state.player_direction),
                tuple(int(getattr(inv, k)) for k in ("wood", "stone", "coal", "iron", "diamond", "sapling")),
                self.env.n_achievements(), int(s.player_food), int(s.player_drink), int(s.player_energy))

    def note_outcome(self, macro_key: str, action: str, reward: float, unlocked: list[str]):
        """Call after env.step so the next state carries `recent` with real outcomes."""
        sig = self._sig()
        moved = self._last_sig is not None and sig[0] != self._last_sig[0]
        inv_changed = self._last_sig is not None and sig[2] != self._last_sig[2]
        progress = bool(unlocked) or moved or inv_changed or reward > 0
        if unlocked:
            out = "unlocked " + ", ".join(unlocked)
        elif inv_changed:
            out = "inventory changed"
        elif moved:
            out = f"moved ({action.lower()})"
        elif action.startswith("Move"):
            out = f"blocked, now facing {DIR_WORD[action]}"
        else:
            out = f"{action.lower()}: nothing changed"
        self.recent.append({"macro": macro_key, "action": action, "outcome": out, "progress": progress})
        self.recent = self.recent[-12:]
        self.stuck = 0 if progress else self.stuck + 1
        self._last_sig = sig
        self.sched.tick()
        self.sched.note(unlocked)
        if "Place Plant" in unlocked:
            self._table.plant_age = 0
        elif self._table.plant_age is not None:
            self._table.plant_age += 1
        if self.planner:
            self.planner.tick()

    def questions(self, macros: list[Macro], facts: dict):
        from typesafe_sdk import Choice, Noul
        instr = ("You are playing Crafter (2D survival): unlock achievements while staying alive; the game ends "
                 "when health reaches 0. `achievements` and `notes` are tracked by the game harness; `recent` lists "
                 "what the last options did."
                 + (OBJECTIVE_HINT if "current objective" in facts else "")
                 + " Which option should be executed this step?")
        return {
            "macro": Choice(instructions=instr, criteria={mm.key: mm.desc for mm in macros}),
            "danger": Noul(instructions="Will the player lose health within the next 2 steps? Use `hostile mobs in view`, "
                                        "`player` and `notes`."),
        }

    def state(self, facts: dict) -> dict:
        return dict(facts)

    def decide(self) -> tuple[str, str, dict]:
        """Returns (primitive action, human-readable reply, info) like Session.llm_decide."""
        if self.env.incapacitated():   # asleep or (full) resting: the game replaces every action by NOOP -> no API call, no stuck counting
            why = "asleep" if bool(self.env.state.is_sleeping) else "resting"
            self.stuck = 0
            return "Noop", f"[code:{why}] wait -> Noop (no jev call while {why})", {
                "latency_s": 0.0, "macro": "wait", "source": f"code:{why}", "confidence": 1.0, "danger": 0.0,
                "jev_ms": 0, "jev_input_tokens": 0, "jev_output_tokens": 0, "n_options": 1, "probabilities": {}, "stuck": 0, "goal_top": why,
                "objective": self.objective.title if self.objective else None, "objective_key": self.objective.key if self.objective else None}
        t0 = time.perf_counter()
        env = self.env
        gb, snap = goal_board(env), goal_snapshot(env)
        vm = env.visible_mobs()
        pr, pc = env.pos()
        near = sorted(((abs(mm["r"] - pr) + abs(mm["c"] - pc), mm) for mm in vm if mm["hostile"] and not mm.get("projectile")
                       and abs(mm["r"] - pr) + abs(mm["c"] - pc) <= 2), key=lambda x: x[0])
        near = [dict(mm, dist=d_) for d_, mm in near]
        ctx = {"visible": {mm["name"] for mm in vm}, "known": self._table.known_names(),
               "light": float(env.state.light_level), "sword": env.tools()[1], "enclosed": self._table.last_ctx["enclosed"],
               "threat": (near[0]["name"], near[0]["dist"], near[0]["health"], near[0]["attack"], env.player_damage(near[0])) if near else None}
        prev_key = self.sched.node
        obj, plan_info, notes, rules = None, None, "", []
        if self.planner:                           # System Two (LLM): cadence + events (hostile within 2 tiles, a need crossing 4)
            # events = the harness's own facts changing: a hostile mob newly in view, or a new status note (goals.status_notes)
            hostiles_now = frozenset(mm["name"] for mm in vm if mm["hostile"] and not mm.get("projectile"))
            notes_now = frozenset(n.split(":")[0] for n in gb["notes"])
            event = None
            if hostiles_now - self._seen_hostiles:
                event = "hostile mob in view: " + ", ".join(sorted(hostiles_now - self._seen_hostiles))
            elif notes_now - self._seen_notes:
                event = "status note: " + "; ".join(sorted(notes_now - self._seen_notes))
            self._seen_hostiles, self._seen_notes = hostiles_now, notes_now
            plan_info = self.planner.update(gb, self.recent, map_text=self._table.known_summary(), event=event)
            rules = self.planner.rules
            if self.planner.objective is not None:
                g = self.planner.objective
                obj = Objective(g.key, g.title, g, objective_fact(gb, g) or g.title, source=self.planner.source)
            notes = self.planner.plan
        elif self.objective_source == "sched":     # scripted upper bound only
            obj = self.sched.update(gb, snap, ctx)
        self.objective = obj
        macros, facts = self._table.build(self.recent, objective=obj, notes=notes, rng=self._rng, rules=rules)
        by_key = {mm.key: mm for mm in macros}
        sched_change = (self.sched.node != prev_key) or bool(plan_info)
        base_info = {"objective": obj.title if obj else None, "objective_key": obj.key if obj else None,
                     "objective_source": obj.source if obj else None, "sched_change": sched_change, "n_options": len(macros),
                     "n_serving": sum(1 for mm in macros if mm.goal_match)}
        if len(macros) == 1 or self.chooser != "jev":   # no jev call: single option, or the code-only choosers
            if len(macros) == 1:
                mm, source = macros[0], "code:single"
            elif self.chooser == "match":
                serving = [mm for mm in macros if mm.goal_match]
                mm = serving[int(self._rng.integers(len(serving)))] if serving else macros[int(self._rng.integers(len(macros)))]
                source = "code:match" if serving else "code:match-random"
            else:
                mm, source = macros[int(self._rng.integers(len(macros)))], "code:random"
            met = facts["achievements"]["unlocked, requirements met now"]
            info = {"latency_s": round(time.perf_counter() - t0, 2), "macro": mm.key, "source": source, "confidence": 1.0,
                    "danger": 0.0, "jev_ms": 0, "jev_input_tokens": 0, "jev_output_tokens": 0, "probabilities": {},
                    "stuck": self.stuck, "goal_top": met[0] if met else "", **base_info}
            reply = (f"[{self.name} {source}] {mm.key} -> {mm.action}\noptions ({len(macros)}): " + "; ".join(x.key for x in macros)
                     + (f"\nobjective: {obj.text}" if obj else "") + (f"\nplanner rules: {' | '.join(rules)}" if rules else "")
                     + (f"\nplanner notes: {notes}" if notes else ""))
            if self.planner:
                reply = self.planner.annotate(info, reply, plan_info, facts.get("current objective"))
            return mm.action, reply, info
        t0j = time.perf_counter()
        resp = self._client.system_one(state=self.state(facts), questions=self.questions(macros, facts))
        jev_ms = (time.perf_counter() - t0j) * 1000
        ans = resp.choices["macro"]
        danger = resp.nouls["danger"].noul
        probs = sorted(ans.probabilities.items(), key=lambda kv: -kv[1])
        chosen, conf, source = ans.choice, float(ans.confidence), "jev"

        # code-side safety override: high danger + adjacent hostile + low health -> flee if possible
        hp = float(self.env.state.player_health)
        if danger > 0.7 and "flee" in by_key and hp <= 4 and chosen not in ("flee", "attack"):
            chosen, source = "flee", "code:danger-override"

        # confidence gate
        stuck = self.stuck >= 4
        if source == "jev" and (conf < self.conf_floor or stuck):
            chosen, source = self._fallback(macros, probs), "code:fallback"
        if chosen not in by_key:
            chosen, source = probs[0][0], "code:invalid-fix"
        mm = by_key[chosen]
        top3 = ", ".join(f"{k} {v:.2f}" for k, v in probs[:3])
        reply = (f"[{self.name} {source}] {mm.key} -> {mm.action}   conf={conf:.2f} danger={danger:.2f} jev={jev_ms:.0f}ms\n"
                 f"top: {top3}\noptions ({len(macros)}): " + "; ".join(x.key for x in macros))
        met = facts["achievements"]["unlocked, requirements met now"]
        info = {"latency_s": round(time.perf_counter() - t0, 2), "macro": mm.key, "source": source,
                "confidence": round(conf, 3), "danger": round(danger, 3), "jev_ms": round(jev_ms),
                "jev_input_tokens": resp.usage.input_tokens, "jev_output_tokens": resp.usage.output_tokens,
                "probabilities": {k: round(v, 3) for k, v in probs[:5]}, "stuck": self.stuck,
                "goal_top": met[0] if met else "", "chosen_serves": mm.goal_match, **base_info}
        if obj:
            reply += f"\nobjective: {obj.text}"
        if rules:
            reply += f"\nplanner rules: {' | '.join(rules)}"
        if notes:
            reply += f"\nplanner notes: {notes}"
        if self.planner:
            reply = self.planner.annotate(info, reply, plan_info, facts.get("current objective"))
        return mm.action, reply, info

    def _fallback(self, macros: list[Macro], probs) -> str:
        """Deterministic, no tech-tree knowledge: jev's top option that was not chosen 3+ times in the last 6 steps
        without any change, else a random option (the harness does not know what is 'right')."""
        no_prog: dict[str, int] = {}
        for rec in self.recent[-6:]:
            if not rec.get("progress"):
                no_prog[rec["macro"]] = no_prog.get(rec["macro"], 0) + 1
        for k, _ in probs:
            if no_prog.get(k, 0) < 3:
                return k
        return macros[int(self._rng.integers(len(macros)))].key


# ----------------------------------------------------------------------------- primitive-action policies

@dataclass
class JevActionPolicy:
    """Jev-Action-Sim / Jev-Action-Map: jev picks directly among the env's legal primitive actions (argmax, no gate).

    state = the text observation the LLM agent sees (with the human-oriented "Do target" label neutralised)
    + the achievement board; instructions = the game manual. Each option carries a code-computed fact and the
    simulated one-step outcome. No macro table, no map memory, no pathing.

      mode="sim"  Jev-Action-Sim: primitive action names as-is.
      mode="map"  Jev-Action-Map: "Do" renamed by its target (Collect wood / Mine stone / Attack zombie …) and
                  every action whose simulated outcome changes nothing is dropped (it equals Noop).
    """
    env: object
    mode: str = "sim"                 # "sim" | "map"
    model: str = "jev-latest"
    planner: ObjectivePlanner | None = None   # optional System Two: sets `current objective` on a fixed cadence
    recent: list = field(default_factory=list)
    n_escalations: int = 0
    _client: object = None

    def __post_init__(self):
        from typesafe_sdk import RetryPolicy, TypeSafeClient
        assert self.mode in ("sim", "map"), self.mode
        self._client = TypeSafeClient(model=self.model, timeout=20, retry=RetryPolicy(max_retries=3))

    annotate = True
    simulate = True

    @property
    def semantic(self) -> bool:
        return self.mode == "map"

    @property
    def name(self) -> str:
        return ("Jev-Action-Map" if self.semantic else "Jev-Action-Sim") + ("+LLM" if self.planner else "")

    def reset(self):
        self.recent = []
        if self.planner:
            self.planner.reset()

    def describe(self, action: str) -> str | None:
        """Factual, one-step description of a primitive action: what tile it touches and what happens."""
        env = self.env
        s = env.state
        m = env.level_map(); H, W = m.shape
        pos = env.pos()
        mobs = {(mm["r"], mm["c"]): mm for mm in env.visible_mobs() if not mm.get("projectile")}
        name_of = {b.value: b.name.lower() for b in env.BlockType}
        done = {x.name for x in env.Achievement if s.achievements[x.value] > 0}
        pick, sword = env.tools()
        mx = env.max_stats()

        def tile(r, c):
            if not (0 <= r < H and 0 <= c < W):
                return "out of bounds"
            if (r, c) in mobs:
                return mobs[(r, c)]["name"]
            return name_of.get(int(m[r, c]), "?")

        if action in DIRS:
            dr, dc = DIRS[action]
            r, c = pos[0] + dr, pos[1] + dc
            what = tile(r, c)
            if not env.in_map(r, c):
                return f"the tile {DIR_WORD[action]} is the map edge: you do not move, you turn to face it"
            if (r, c) in mobs:
                return f"the tile {DIR_WORD[action]} is occupied by a {what}: you do not move, you turn to face it"
            if env.classic and what == "lava":   # classic move_player :1380-1383 allows it; update_health :1642-1647 -> 0
                return f"the tile {DIR_WORD[action]} is lava: you walk onto it and die (health set to 0, the episode ends)"
            if env.walkable(r, c):               # game move_player rule (solid set, water/lava, mobs) on the live map
                return f"moves one tile {DIR_WORD[action]} onto {what}; you then face {DIR_WORD[action]}"
            return f"the tile {DIR_WORD[action]} is {what} (solid): you do not move, you turn to face it"
        if action == "Do":
            fr, fc = env.facing()
            what = tile(fr, fc)
            if (fr, fc) in mobs:
                mm = mobs[(fr, fc)]
                ka = kill_ach(env, mm)
                gain = f"; killing it gives food +6 (max {mx['food']})" if mm["name"] in FOOD_MOBS else ""
                return (f"faced tile is a {mm['name']} with {mm['health']:g} health: your hit does {env.player_damage(mm):g}{gain}"
                        + (f"; {ach_status(ka, done)}" if ka else ""))
            if what in RESOURCE:
                verb, effect, ach, need = RESOURCE[what]
                if pick < need:
                    return f"faced tile is {what}: mining it requires a {TIER[need]} pickaxe, you have {TIER[pick]}; nothing happens"
                return f"faced tile is {what}: Do {effect.format(**mx)}; {ach_status(ach, done)}"
            if what == "grass":
                return "faced tile is grass: random chance (about 1 in 10) of 1 sapling; " + ach_status("COLLECT_SAPLING", done)
            if not env.classic and what in ("crafting_table", "furnace"):   # full game_logic.py:236-262; classic do() ignores them
                return f"faced tile is {what}: Do removes it (it becomes path; the materials are not refunded)"
            if not env.classic and what in ("necromancer", "necromancer_vulnerable"):   # full game_logic.py:441-464
                return "faced tile is the Necromancer: Do damages it only while it is vulnerable (necromancer_vulnerable)"
            return f"faced tile is {what}: Do has no effect on it"
        if action in CRAFT_FACT:
            inv = {k: int(getattr(s.inventory, k)) for k in ("wood", "stone", "coal", "iron", "diamond", "sapling")}
            inv_txt = ", ".join(f"{k} {v}" for k, v in inv.items() if v) or "no materials"
            return f"{CRAFT_FACT[action]}; {ach_status(CRAFT_ACH[action], done)}; inventory: {inv_txt}"
        if action == "Sleep":
            hostile = [mm for mm in mobs.values() if mm["hostile"]]
            return (f"energy {int(s.player_energy)}/{mx['energy']}; actions are ignored until energy is full; {sleep_risk_text(env)}; "
                    f"hostile mobs in view: {len(hostile)}; light is {float(s.light_level):.2f}; "
                    + ach_status("WAKE_UP", done))
        if action == "Noop":
            return "does nothing this step"
        return None

    # ---- one-step lookahead: run the (pure, jitted) env step on a copy of the state and diff the result
    _RES = ("tree", "stone", "coal", "iron", "diamond", "water", "ripe_plant", "crafting_table", "furnace",
            "fire_tree", "ice_shrub", "stalagmite", "sapphire", "ruby", "fountain", "chest")
    # inventory scalars reported in the simulated outcome (missing fields read as 0 in the other variant)
    _INV_KEYS = ("wood", "stone", "coal", "iron", "diamond", "sapling", "sapphire", "ruby", "torches", "arrows", "books", "bow",
                 "pickaxe", "sword", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword")
    _noop_cache: tuple | None = None    # ((timestep, rng bytes), simulated Noop state, done) for the current decision

    def _nearest(self, state) -> dict[str, int | None]:
        """Manhattan distance to the nearest visible block of each resource type, from the player's position."""
        env = self.env
        m = np.array(state.map if env.classic else state.map[state.player_level])
        r, c = (int(v) for v in np.array(state.player_position))
        H, W = m.shape
        win = m[max(0, r - env.view_r):r + env.view_r + 1, max(0, c - env.view_c):c + env.view_c + 1]
        r0, c0 = max(0, r - env.view_r), max(0, c - env.view_c)
        out = {}
        for b in env.BlockType:
            name = b.name.lower()
            if name not in self._RES:
                continue
            rr, cc = np.nonzero(win == b.value)
            out[name] = int(np.min(np.abs(rr + r0 - r) + np.abs(cc + c0 - c))) if len(rr) else None
        return out

    def simulate_outcome(self, action: str) -> str:
        return self._simulate(action)[0]

    def simulate_changes(self, action: str) -> bool:
        return self._simulate(action)[1]

    def _sim_key(self):
        """The PRNG key CraftaxTextEnv.step will consume next: `self.rng, k = split(self.rng)` (craftax_text_env.py:210),
        forwarded unchanged to craftax_step by EnvironmentNoAutoReset.step. env.rng itself is NOT advanced here, so the
        simulation reproduces the real step (mob moves/attacks, spawns, sapling draw) as long as env.rng is unchanged
        between _simulate and env.step, which is how jev_agent.py / server/app.py use it."""
        import jax
        _, k = jax.random.split(self.env.rng)
        return k

    def _noop_baseline(self, k):
        """Simulated Noop under the same key, cached per decision (keyed on timestep + rng). craftax_step splits its rng
        unconditionally (classic game_logic.py:1666-1702, full :3020-3068), so an action and Noop stepped with one key
        differ exactly where the action acts: hunger/thirst/fatigue ticks, regen, mob moves and hits are identical."""
        env = self.env
        s = env.state
        tag = (int(s.timestep), np.asarray(env.rng).tobytes())
        if self._noop_cache is None or self._noop_cache[0] != tag:
            _, nb, _, done_b, _ = env._step(k, s, env.action_index["Noop"], env.params)
            self._noop_cache = (tag, nb, bool(done_b))
        return self._noop_cache[1], self._noop_cache[2]

    def _mob_health(self, state) -> dict[str, list]:
        """{mob name: [health per slot]} for the mob arrays of the current level (mask-independent, for diffing)."""
        env = self.env
        if env.classic:
            return {n: np.array(getattr(state, arr).health) for n, arr in (("zombie", "zombies"), ("cow", "cows"), ("skeleton", "skeletons"))}
        lvl = int(state.player_level)
        out = {}
        for arr, offset in (("melee_mobs", 0), ("passive_mobs", 8), ("ranged_mobs", 16)):
            cls = getattr(state, arr)
            out[arr] = (np.array(cls.health[lvl]), np.array(cls.type_id[lvl]), offset)
        return out

    def _simulate(self, action: str) -> tuple[str, bool]:
        """(outcome text, changed): step a copy of the state with the key env.step will use. `changed` is True iff the
        resulting EnvState differs anywhere (every pytree leaf: map, item map, mobs' health, inventory, stats, mana, xp,
        floor, flags ...) from a Noop stepped with the same key, i.e. the action does something Noop would not; the
        text lists the literal one-step deltas (position/facing, inventory, achievements, stats, flags, mob health)."""
        from jax.tree_util import keystr, tree_leaves_with_path
        env = self.env
        s = env.state
        idx = env.action_index[action]
        k = self._sim_key()
        _, ns, reward, done, _ = env._step(k, s, idx, env.params)
        nb, done_b = (ns, bool(done)) if action == "Noop" else self._noop_baseline(k)
        diff = [keystr(p) for (p, a), (_, b) in zip(tree_leaves_with_path(ns), tree_leaves_with_path(nb))
                if not np.array_equal(np.asarray(a), np.asarray(b))]
        changed = bool(diff) or bool(done) != done_b
        parts = []
        p0 = tuple(int(v) for v in np.array(s.player_position)); p1 = tuple(int(v) for v in np.array(ns.player_position))
        turned = int(ns.player_direction) != int(s.player_direction)
        lvl0, lvl1 = (0, 0) if env.classic else (int(s.player_level), int(ns.player_level))
        if lvl1 != lvl0:   # full change_floor :2488-2493
            parts.append(f"you {'descend' if lvl1 > lvl0 else 'ascend'} to floor {lvl1}")
        elif p1 != p0:
            parts.append(f"you move {DIR_WORD[action]}" if action in DIRS else "you move")
        elif action in DIRS:
            parts.append("you do not move" + ("; you turn to face " + DIR_WORD[action] if turned else ""))
        # facing tile after (only when the move or turn changed it)
        m1 = np.array(ns.map if env.classic else ns.map[ns.player_level])
        dr, dc = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[int(ns.player_direction)]
        fr, fc = p1[0] + dr, p1[1] + dc
        name_of = {b.value: b.name.lower() for b in env.BlockType}
        if (p1 != p0 or turned or lvl1 != lvl0) and 0 <= fr < m1.shape[0] and 0 <= fc < m1.shape[1]:
            parts.append(f"facing {name_of.get(int(m1[fr, fc]), '?')}")
        # inventory (scalars + armour/potions arrays) / achievements / stats / full-only flags
        for key in self._INV_KEYS:
            d = int(getattr(ns.inventory, key, 0)) - int(getattr(s.inventory, key, 0))
            if d:
                parts.append(f"{key.replace('_', ' ')} {d:+d}")
        for key, label in (("armour", "armour piece"), ("potions", "potion")):
            if hasattr(ns.inventory, key):
                d = int(np.array(getattr(ns.inventory, key)).sum() - np.array(getattr(s.inventory, key)).sum())
                if d:
                    parts.append(f"{label} {d:+d}")
        a0 = np.array(s.achievements) > 0; a1 = np.array(ns.achievements) > 0
        for i in np.flatnonzero(a1 & ~a0):
            parts.append(f"achievement {env.ach_names[int(i)]} unlocked")
        for key, label in (("player_health", "health"), ("player_food", "food"), ("player_drink", "drink"), ("player_energy", "energy"),
                           ("player_mana", "mana"), ("player_xp", "xp"), ("player_strength", "strength"),
                           ("player_dexterity", "dexterity"), ("player_intelligence", "intelligence")):
            if hasattr(ns, key):
                d = float(getattr(ns, key)) - float(getattr(s, key))
                if d:
                    parts.append(f"{label} {d:+g}")
        if bool(ns.is_sleeping) and not bool(s.is_sleeping):
            parts.append("you fall asleep")
        if bool(getattr(ns, "is_resting", False)) and not bool(getattr(s, "is_resting", False)):
            parts.append("you start resting")
        if hasattr(ns, "learned_spells"):   # full read_book :2703-2724 (0 fireball, 1 iceball)
            for i in np.flatnonzero(np.array(ns.learned_spells) & ~np.array(s.learned_spells)):
                parts.append(f"you learn {('fireball', 'iceball')[int(i)] if int(i) < 2 else 'a spell'}")
        for key, label in (("sword_enchantment", "sword"), ("bow_enchantment", "bow")):   # full enchant :2736-2797
            if hasattr(ns, key) and int(getattr(ns, key)) != int(getattr(s, key)):
                parts.append(f"your {label} is enchanted with {('none', 'fire', 'ice')[int(getattr(ns, key))]}")
        if hasattr(ns, "armour_enchantments") and not np.array_equal(np.array(ns.armour_enchantments), np.array(s.armour_enchantments)):
            parts.append("an armour slot is enchanted")
        if hasattr(ns, "item_map") and not np.array_equal(np.array(ns.item_map[lvl1]), np.array(s.item_map[lvl0])) and lvl1 == lvl0:
            parts.append("torch placed" if action == "Place Torch" else "the item map changes")
        if hasattr(ns, "player_projectiles") and lvl1 == lvl0:
            n0 = int(np.array(s.player_projectiles.mask[lvl0]).sum()); n1 = int(np.array(ns.player_projectiles.mask[lvl1]).sum())
            if n1 > n0:
                parts.append("you fire an arrow" if action == "Shoot Arrow" else "you cast a spell" if action.startswith("Cast") else "a projectile of yours appears")
        # mob health changed by the action (vs the same-key Noop, so incidental changes are not attributed to it)
        if lvl1 == lvl0:
            h1, hb, hs = self._mob_health(ns), self._mob_health(nb), self._mob_health(s)
            for key in h1:
                if env.classic:
                    for i in np.flatnonzero(h1[key] != hb[key]):
                        parts.append(f"{key} health {float(hb[key][i]):g} -> {float(h1[key][i]):g}")
                else:   # names from the pre-step type ids (what visible_mobs() shows; update_mobs may rewrite them)
                    (ha, _, offset), (hbb, _, _), (_, tid, _) = h1[key], hb[key], hs[key]
                    for i in np.flatnonzero(ha != hbb):
                        parts.append(f"{MOB_NAMES.get(offset + int(tid[i]), 'creature')} health {float(hbb[i]):g} -> {float(ha[i]):g}")
        # distances to nearest visible resources, before -> after (only those that change)
        b, a = self._nearest(s), self._nearest(ns)
        for name in self._RES:
            if b.get(name) != a.get(name):
                bb = "none in view" if b.get(name) is None else f"{b[name]}"
                aa = "none in view" if a.get(name) is None else f"{a[name]}"
                parts.append(f"nearest {name.replace('_', ' ')} {bb} -> {aa} steps")
        if bool(done):
            parts.append("the episode ends")
        if changed and not parts:
            parts.append("state changes: " + ", ".join(d.lstrip(".") for d in diff))
        return ("; ".join(parts) if parts else "nothing changes"), changed

    def note_outcome(self, macro_key, action, reward, unlocked):
        if self.planner:
            self.planner.tick()

    def semantic_name(self, action: str) -> str | None:
        """Name for a primitive action given the current state; None = the action would have no effect."""
        if action != "Do":
            return action
        env = self.env
        s = env.state
        fr, fc = env.facing()
        m = env.level_map()
        mobs = {(mm["r"], mm["c"]): mm["name"] for mm in env.visible_mobs() if not mm.get("projectile")}
        if (fr, fc) in mobs:
            return f"Attack {mobs[(fr, fc)]}"
        if not (0 <= fr < m.shape[0] and 0 <= fc < m.shape[1]):
            return None
        what = {b.value: b.name.lower() for b in env.BlockType}.get(int(m[fr, fc]), "?")
        pick, _ = env.tools()
        if what in RESOURCE:     # tree/ores/water/ripe plant + full: fire tree, ice shrub, stalagmite, sapphire, ruby, fountain, chest
            verb, _, _, need = RESOURCE[what]
            return DO_LABEL.get(what, f"{verb.split()[0].capitalize()} {what.replace('_', ' ')}") if pick >= need else None
        if what == "grass":
            return "Collect sapling"
        # sand, path, unripe plant, lava …: Do does nothing. crafting_table / furnace: classic ignores them; in full
        # Do destroys the station (game_logic.py:236-262) — deliberately never offered as a map-mode option.
        return None

    def _criterion(self, action: str) -> str | None:
        parts = []
        if self.annotate:
            parts.append(self.describe(action) or "")
        if self.simulate:
            parts.append("outcome of this step: " + self.simulate_outcome(action))
        return "; ".join(x for x in parts if x) or None

    def decide(self) -> tuple[str, str, dict]:
        from typesafe_sdk import Choice
        from craftax_agent.goals import board as goal_board, board_text
        env = self.env
        if env.incapacitated():   # asleep or (full) resting: the game replaces every action by NOOP (classic :1660, full :3011-3012)
            why = "asleep" if bool(env.state.is_sleeping) else "resting"
            return "Noop", f"[code:{why}]", {"latency_s": 0.0, "macro": "Noop", "source": f"code:{why}", "confidence": 1.0,
                                             "danger": 0.0, "jev_ms": 0, "jev_input_tokens": 0, "jev_output_tokens": 0, "n_options": 1,
                                             "probabilities": {}, "stuck": 0, "plan": "", "goal_top": why}
        obs = env.observe()["text"]
        legal = env.legal_actions()
        # option key -> primitive action (identity unless `semantic`). With `semantic`, any action whose simulated
        # state equals the same-key simulated Noop is dropped too; Noop itself and chance-based Do stay.
        keyed = {}
        for a in legal:
            k = self.semantic_name(a) if self.semantic else a
            if k is None:
                continue
            if self.semantic and self.simulate and a != "Noop" and k != "Collect sapling" \
                    and not self.simulate_changes(a):
                continue
            keyed[k] = a
        if not keyed:
            keyed = {"Noop": "Noop"}
        # The observation is written for humans/LLMs and labels the faced tile "Do target: …". Probing showed that
        # phrase alone lifts P(Do) from ~0.26 to ~0.53 on a fixed state (report §6), so it is neutralised here.
        obs_txt = (obs["long_term_context"] + "\n" + obs["short_term_context"].split("Available actions:")[0].rstrip()
                   ).replace("Do target", "Tile in front")
        gb = goal_board(env)
        state = {"observation": obs_txt, "achievements": board_text(gb)}
        plan_info, objective_text = None, None
        if self.planner:                           # System Two cadence runs before jev is asked
            plan_info = self.planner.update(gb, self.recent)
            objective_text = objective_fact(gb, self.planner.objective)
            if objective_text:
                state["current objective"] = objective_text
        t0 = time.perf_counter()
        resp = self._client.system_one(
            state=state,
            questions={"action": Choice(instructions=env.system_prompt + "\n\n"
                                        + (OBJECTIVE_HINT.strip() + " " if objective_text else "")
                                        + "Which action should be executed this step?",
                                        criteria={k: self._criterion(a) for k, a in keyed.items()})})
        jev_ms = (time.perf_counter() - t0) * 1000
        ans = resp.choices["action"]
        probs = sorted(ans.probabilities.items(), key=lambda kv: -kv[1])
        key = ans.choice if ans.choice in keyed else probs[0][0]
        action = keyed[key]
        reply = (f"[{self.name}] {key} -> {action}   conf={ans.confidence:.2f} jev={jev_ms:.0f}ms\ntop: "
                 + ", ".join(f"{k} {v:.2f}" for k, v in probs[:3]))
        info = {"latency_s": round(time.perf_counter() - t0, 2), "macro": key, "source": "jev",
                "confidence": round(float(ans.confidence), 3), "danger": 0.0, "jev_ms": round(jev_ms),
                "jev_input_tokens": resp.usage.input_tokens, "jev_output_tokens": resp.usage.output_tokens, "n_options": len(keyed),
                "probabilities": {k: round(v, 3) for k, v in probs[:5]}, "stuck": 0, "plan": "", "goal_top": ""}
        if self.planner:
            reply = self.planner.annotate(info, reply, plan_info, objective_text)
        return action, reply, info
