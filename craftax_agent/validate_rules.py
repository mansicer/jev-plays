"""Runtime validation of legal_actions() against the real game step.

Invariant checked at every step, for every Place*/Make*/Sleep/Rest/Descend/Ascend/Shoot/Cast*/Enchant*/Drink*/Level Up*/Read Book:
    action in env.legal_actions()  <=>  env._step(k, state, action) differs from env._step(k, state, Noop)
in at least one of the DIFF_FIELDS below (k = the key env.step() would use, jax.random.split(env.rng)[1]).
Placement extra: for a legal Place X the simulated faced tile must now hold the placed block/item.

usage: validate_rules.py <variant> <n_seeds> <steps_per_seed> [synthetic_per_seed]
"""
import sys, time, random, json, collections
import numpy as np
import jax, jax.numpy as jnp
from jax.tree_util import tree_leaves_with_path, keystr
from craftax_agent.craftax_text_env import CraftaxTextEnv, DIR_VEC

variant = sys.argv[1]
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
N_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
N_SYN = int(sys.argv[4]) if len(sys.argv) > 4 else 40
OUT = sys.argv[5] if len(sys.argv) > 5 else None
SEED0 = int(sys.argv[6]) if len(sys.argv) > 6 else 0

DIFF_FIELDS = ("inventory", "map", "item_map", "achievements", "player_level", "player_position",
               "player_strength", "player_dexterity", "player_intelligence", "player_xp", "player_mana",
               "is_sleeping", "is_resting", "learned_spells", "player_projectiles", "player_projectile_directions",
               "sword_enchantment", "bow_enchantment", "armour_enchantments", "monsters_killed",
               "player_health", "player_food", "player_drink", "player_energy",
               "growing_plants_positions", "growing_plants_mask", "growing_plants_age")
CHECK_PREFIX = ("Place ", "Make ", "Sleep", "Rest", "Descend", "Ascend", "Shoot", "Cast ", "Enchant ", "Drink ",
                "Level Up ", "Read Book")

env = CraftaxTextEnv(variant, max_timesteps=10000, render=False)
B, K = env.BlockType, env.K
classic = env.classic
ACTIONS = [a for a in env.action_names if a.startswith(CHECK_PREFIX)]
NOOP = env.action_index["Noop"]
MOVE_OF = {(0, -1): "Move West", (0, 1): "Move East", (-1, 0): "Move North", (1, 0): "Move South"}
VEC_OF = {v: k for k, v in MOVE_OF.items()}


# ---------------------------------------------------------------- diff machinery
def _flags(a, b):
    return jax.tree_util.tree_map(lambda x, y: jnp.any(x != y), a, b)


flags_jit = jax.jit(_flags)


def diff_fields(ns, nb):
    fl = flags_jit(ns, nb)
    out = []
    for p, v in tree_leaves_with_path(fl):
        if bool(v):
            out.append(keystr(p))
    return out


def in_fields(paths):
    return sorted({p for p in paths if any(p.startswith("." + f) for f in DIFF_FIELDS)})


def sim(state, action_idx, k):
    _, ns, r, done, _ = env._step(k, state, action_idx, env.params)
    return ns, bool(done)


PLACED = {"Place Stone": ("map", B.STONE.value), "Place Table": ("map", B.CRAFTING_TABLE.value),
          "Place Furnace": ("map", B.FURNACE.value), "Place Plant": ("map", B.PLANT.value)}
if not classic:
    PLACED["Place Torch"] = ("item", env.ItemType.TORCH.value)


def faced_after(ns, fr, fc):
    if classic:
        return int(ns.map[fr, fc]), None
    lvl = int(ns.player_level)
    return int(ns.map[lvl, fr, fc]), int(ns.item_map[lvl, fr, fc])


counters = collections.Counter()
examples = []      # counter-examples
caveats = []       # accepted, source-backed same-step effects (reported separately)
legal_hits = collections.Counter()   # action -> times legal & changed (coverage)
illegal_hits = collections.Counter()


def context(tag, seed, step, action, verdict, diff, extra="", nb=None):
    s = env.state
    if nb is not None:
        extra += f" | after Noop: health {float(nb.player_health)} food {int(nb.player_food)} drink {int(nb.player_drink)} max_health {env.max_stats()['health']}"
    inv = {k: (int(v) if np.ndim(v) == 0 else np.array(v).tolist()) for k, v in vars(s.inventory).items()}
    fr, fc = env.facing()
    return {"variant": variant, "tag": tag, "seed": seed, "step": step, "action": action, "our_verdict": verdict,
            "sim_diff": diff, "faced": env.block_at(fr, fc).name, "faced_item": env.item_at(fr, fc),
            "mob_on_faced": env.mob_at(fr, fc), "pos": env.pos(), "level": env.level(),
            "inv": inv, "health": float(s.player_health), "food": int(s.player_food), "drink": int(s.player_drink),
            "energy": int(s.player_energy), "mana": None if classic else int(s.player_mana),
            "xp": None if classic else int(s.player_xp), "sleeping": bool(s.is_sleeping),
            "resting": None if classic else bool(s.is_resting), "extra": extra}


def check_state(tag, seed, step):
    """Run the invariant on env.state for every check action. Returns number of checks."""
    legal = set(env.legal_actions())
    _, k = jax.random.split(env.rng)
    s = env.state
    nb, done_b = sim(s, NOOP, k)
    fr, fc = env.facing()
    n = 0
    for a in ACTIONS:
        n += 1
        ns, done = sim(s, env.action_index[a], k)
        paths = diff_fields(ns, nb)
        rel = in_fields(paths)
        changed = bool(rel)
        ours = a in legal
        if ours != changed:
            # Known caveat (documented in the manual): full game_logic.py:3055 update_mobs runs before :3064
            # update_player_intrinsics, and is_starting_rest (:1850-1852) reads the post-hit health. So a Rest issued at
            # full health starts a rest iff a hit lands in that very step. Legality is defined on the pre-step state, so
            # this is not decidable; accept ONLY the exact signature: action Rest, we say illegal, pre-step health == max,
            # the Noop baseline (same key) lost health this step, and the only relevant difference is is_resting.
            if (a == "Rest" and not ours and float(s.player_health) >= env.max_stats()["health"]
                    and float(nb.player_health) < float(s.player_health) and rel == [".is_resting"]):
                counters["caveat_rest_hit"] += 1
                caveats.append(context(tag, seed, step, a, ours, paths, "Rest at full health, hit landed this step", nb=nb))
                continue
            counters["counter"] += 1
            examples.append(context(tag, seed, step, a, ours, paths, nb=nb))
            continue
        (legal_hits if ours else illegal_hits)[a] += 1
        if ours and a in PLACED:
            kind, val = PLACED[a]
            if not env.in_map(fr, fc):
                examples.append(context(tag, seed, step, a, ours, paths, "legal placement facing off-map"))
                counters["counter"] += 1
                continue
            blk, itm = faced_after(ns, fr, fc)
            got = blk if kind == "map" else itm
            if got != val:
                # full game_logic.py:1662-1676 / classic :1192-1200: a mob projectile that reaches a CRAFTING_TABLE or
                # FURNACE turns it into PATH in the same step. Accept only that exact signature: table/furnace placed
                # (materials consumed vs the Noop baseline) and a projectile field changed, and the tile is now PATH.
                cost_ok = ((a == "Place Table" and int(ns.inventory.wood) == int(nb.inventory.wood) - 2)
                           or (a == "Place Furnace" and int(ns.inventory.stone) == int(nb.inventory.stone) - 1))
                proj_moved = any(p.startswith(".arrows") or p.startswith(".mob_projectiles") for p in paths)
                if cost_ok and proj_moved and got == B.PATH.value:
                    counters["caveat_placed_then_shot"] += 1
                    caveats.append(context(tag, seed, step, a, ours, paths, "placed block destroyed by a projectile this step"))
                    continue
                counters["counter"] += 1
                examples.append(context(tag, seed, step, a, ours, paths,
                                        f"faced tile after sim = {got}, expected {val}"))
    counters["checks"] += n
    return n


# ---------------------------------------------------------------- jev-free heuristic driver
def mob_positions():
    s = env.state
    pts = set()
    if classic:
        for arr in (s.cows, s.zombies, s.skeletons):
            m = np.array(arr.mask); p = np.array(arr.position)
            for i in np.flatnonzero(m):
                pts.add((int(p[i, 0]), int(p[i, 1])))
    else:
        lvl = int(s.player_level)
        for arr in (s.melee_mobs, s.passive_mobs, s.ranged_mobs):
            m = np.array(arr.mask[lvl]); p = np.array(arr.position[lvl])
            for i in np.flatnonzero(m):
                pts.add((int(p[i, 0]), int(p[i, 1])))
    return pts


def food_mob_positions():
    s = env.state
    pts = set()
    if classic:
        m = np.array(s.cows.mask); p = np.array(s.cows.position)
        for i in np.flatnonzero(m):
            pts.add((int(p[i, 0]), int(p[i, 1])))
    else:
        lvl = int(s.player_level)
        m = np.array(s.passive_mobs.mask[lvl]); p = np.array(s.passive_mobs.position[lvl])
        for i in np.flatnonzero(m):
            pts.add((int(p[i, 0]), int(p[i, 1])))
    return pts


def walk_mask():
    m = env.level_map()
    s = env.state
    mm = np.array(s.mob_map if classic else s.mob_map[s.player_level])
    w = ~np.isin(m, list(env._solid_ids)) & ~mm & (m != B.LAVA.value)
    if not classic:
        w &= (m != B.WATER.value)
    return w


def plan_to(target_tiles: set, max_radius=14):
    """BFS to a walkable tile 4-adjacent to any target tile. Returns action name ('Move X') or None."""
    if not target_tiles:
        return None
    w = walk_mask()
    H, W = w.shape
    r0, c0 = env.pos()
    # already adjacent?
    for (dr, dc), name in MOVE_OF.items():
        if (r0 + dr, c0 + dc) in target_tiles:
            return name
    prev = {(r0, c0): None}
    q = collections.deque([(r0, c0)])
    goal = None
    while q:
        r, c = q.popleft()
        if abs(r - r0) + abs(c - c0) > max_radius:
            continue
        for (dr, dc) in MOVE_OF:
            nr, nc = r + dr, c + dc
            if (nr, nc) in target_tiles and (r, c) != (r0, c0):
                goal = (r, c); break
            if 0 <= nr < H and 0 <= nc < W and w[nr, nc] and (nr, nc) not in prev:
                prev[(nr, nc)] = (r, c); q.append((nr, nc))
        if goal:
            break
    if not goal:
        return None
    cur = goal
    while prev[cur] != (r0, c0):
        cur = prev[cur]
    return MOVE_OF[(cur[0] - r0, cur[1] - c0)]


def tiles_of(*blocks):
    m = env.level_map()
    pts = set()
    for b in blocks:
        for r, c in zip(*np.nonzero(m == b.value)):
            pts.add((int(r), int(c)))
    return pts


def facing_block():
    fr, fc = env.facing()
    return env.block_at(fr, fc), (fr, fc)


def heuristic(step_rng: random.Random):
    s, inv = env.state, env.state.inventory
    legal = env.legal_actions()
    if legal == ["Noop"]:
        return "Noop"
    i = int
    pick, sword = env.tools()
    mx = env.max_stats()
    fb, (fr, fc) = facing_block()
    table, furnace = env.near(B.CRAFTING_TABLE), env.near(B.FURNACE)
    r = step_rng.random()
    # random exploration 25 % (Sleep/Rest rarely: they lock the player for many steps)
    if r < 0.25:
        pool = [a for a in legal if a not in ("Sleep", "Rest")] or legal
        if "Sleep" in legal and step_rng.random() < 0.02:
            return "Sleep"
        if "Rest" in legal and step_rng.random() < 0.03:
            return "Rest"
        a = step_rng.choice(pool)
        if a in VEC_OF:   # never walk into lava (instant death)
            dr, dc = VEC_OF[a]; pr, pc = env.pos()
            if env.block_at(pr + dr, pc + dc) == B.LAVA:
                return "Noop"
        return a
    # opportunistic rare actions
    for a in ("Descend", "Ascend", "Enchant Sword", "Enchant Armour", "Enchant Bow", "Read Book",
              "Level Up Strength", "Level Up Dexterity", "Level Up Intelligence", "Shoot Arrow", "Cast Fireball",
              "Cast Iceball", "Drink Potion Red", "Drink Potion Green", "Drink Potion Blue", "Drink Potion Pink",
              "Drink Potion Cyan", "Drink Potion Yellow"):
        if a in legal and step_rng.random() < 0.5:
            return a
    # crafting when next to a table
    craft_order = ["Make Wood Pickaxe", "Make Wood Sword", "Make Stone Pickaxe", "Make Stone Sword",
                   "Make Iron Pickaxe", "Make Iron Sword", "Make Diamond Pickaxe", "Make Diamond Sword",
                   "Make Iron Armour", "Make Diamond Armour", "Make Arrow", "Make Torch"]
    for a in craft_order:
        if a in legal:
            if classic:
                owned = {"Make Wood Pickaxe": pick >= 1, "Make Wood Sword": sword >= 1, "Make Stone Pickaxe": pick >= 2,
                         "Make Stone Sword": sword >= 2, "Make Iron Pickaxe": pick >= 3, "Make Iron Sword": sword >= 3}.get(a, False)
                if owned and step_rng.random() < 0.8:
                    continue
            return a
    # placing
    if "Place Table" in legal and not table:
        return "Place Table"
    if "Place Furnace" in legal and not furnace and i(inv.stone) >= 2 and table:
        return "Place Furnace"
    if "Place Plant" in legal and step_rng.random() < 0.7:
        return "Place Plant"
    if "Place Stone" in legal and i(inv.stone) >= 3 and step_rng.random() < 0.3:
        return "Place Stone"
    if not classic and "Place Torch" in legal and step_rng.random() < 0.5:
        return "Place Torch"
    # needs
    if i(s.player_drink) < 5:
        a = plan_to(tiles_of(B.WATER) | (tiles_of(B.FOUNTAIN) if not classic else set()))
        if a:
            if fb in (B.WATER,) or (not classic and fb == B.FOUNTAIN):
                return "Do"
            return a
    if i(s.player_food) < 5:
        cows = food_mob_positions()
        a = plan_to(cows)
        if a:
            if (fr, fc) in cows:
                return "Do"
            return a
        a = plan_to(tiles_of(B.RIPE_PLANT))
        if a:
            return "Do" if fb == B.RIPE_PLANT else a
    # resources
    want = []
    if i(inv.wood) < (2 if not table else 1) + 3:
        want.append(B.TREE)
    if pick >= 1 and i(inv.stone) < 5:
        want.append(B.STONE)
    if pick >= 1 and i(inv.coal) < 2:
        want.append(B.COAL)
    if pick >= 2 and i(inv.iron) < 2:
        want.append(B.IRON)
    if pick >= 3:
        want.append(B.DIAMOND)
    if not classic and pick >= 3:
        want += [B.RUBY, B.SAPPHIRE]
    if i(inv.wood) < 9 and B.TREE not in want:
        want.append(B.TREE)
    for b in want:
        a = plan_to(tiles_of(b))
        if a:
            return "Do" if fb == b else a
    if not classic and "Rest" in legal and step_rng.random() < 0.05:
        return "Rest"
    if "Sleep" in legal and i(s.player_energy) < 3 and step_rng.random() < 0.3:
        return "Sleep"
    pool = [a for a in legal if a.startswith("Move") or a == "Do"] or legal
    a = step_rng.choice(pool)
    if a in VEC_OF:
        dr, dc = VEC_OF[a]; pr, pc = env.pos()
        if env.block_at(pr + dr, pc + dc) == B.LAVA:
            return "Noop"
    return a


# ---------------------------------------------------------------- synthetic state perturbation
def cast_like(old, v):
    return jnp.asarray(v, dtype=jnp.asarray(old).dtype)


def perturb(rng: random.Random, base):
    """Random, mostly plausible perturbation of inventory / stats / faced tile / ladder / flags."""
    env.state = base            # env.facing()/env.pos() below must read the base state
    s = base
    inv = s.inventory
    kw = {}
    for key in ("wood", "stone", "coal", "iron", "diamond", "sapling"):
        if rng.random() < 0.6:
            kw[key] = cast_like(getattr(inv, key), rng.choice([0, 0, 1, 2, 3, 5, 9]))
    if classic:
        for key in ("wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"):
            if rng.random() < 0.4:
                kw[key] = cast_like(getattr(inv, key), rng.choice([0, 1]))
    else:
        for key in ("arrows", "torches", "books", "ruby", "sapphire"):
            if rng.random() < 0.5:
                kw[key] = cast_like(getattr(inv, key), rng.choice([0, 0, 1, 2, 5, 98, 99]))
        for key in ("pickaxe", "sword", "bow"):
            if rng.random() < 0.5:
                kw[key] = cast_like(getattr(inv, key), rng.choice([0, 1, 2, 3, 4] if key != "bow" else [0, 1]))
        if rng.random() < 0.5:
            kw["armour"] = cast_like(inv.armour, [rng.choice([0, 0, 1, 2]) for _ in range(4)])
        if rng.random() < 0.5:
            kw["potions"] = cast_like(inv.potions, [rng.choice([0, 0, 1, 3]) for _ in range(6)])
    s = s.replace(inventory=inv.replace(**kw))
    skw = {}
    mx = {"health": 9, "food": 9, "drink": 9, "energy": 9}
    if not classic:
        from craftax.craftax.util.game_logic_utils import get_max_health, get_max_food, get_max_drink, get_max_energy, get_max_mana
    for key, mxk in (("player_health", "health"), ("player_food", "food"), ("player_drink", "drink"), ("player_energy", "energy")):
        if rng.random() < 0.4:
            top = mx[mxk] if classic else int({"health": get_max_health, "food": get_max_food, "drink": get_max_drink,
                                                "energy": get_max_energy}[mxk](s))
            skw[key] = cast_like(getattr(s, key), rng.choice([0, 1, 3, top - 1, top]))
    if not classic:
        if rng.random() < 0.6:
            skw["player_mana"] = cast_like(s.player_mana, rng.choice([0, 1, 2, 8, 9, int(get_max_mana(s))]))
        if rng.random() < 0.5:
            skw["player_xp"] = cast_like(s.player_xp, rng.choice([0, 0, 1, 3]))
        if rng.random() < 0.3:
            attr = rng.choice(["player_strength", "player_dexterity", "player_intelligence"])
            skw[attr] = cast_like(getattr(s, attr), rng.choice([1, 2, int(env.params.max_attribute)]))
        if rng.random() < 0.4:
            skw["learned_spells"] = cast_like(s.learned_spells, [rng.random() < 0.5, rng.random() < 0.5])
        if rng.random() < 0.2:
            skw["is_resting"] = cast_like(s.is_resting, True)
    if rng.random() < 0.15:
        skw["is_sleeping"] = cast_like(s.is_sleeping, True)
    s = s.replace(**skw)
    # faced tile
    fr, fc = env.facing()
    H, W = s.map.shape[-2:]
    if 0 <= fr < H and 0 <= fc < W:
        if rng.random() < 0.7:
            choices = [b for b in B if b.name not in ("OUT_OF_BOUNDS", "INVALID")]
            if not classic:
                choices += [B.ENCHANTMENT_TABLE_FIRE, B.ENCHANTMENT_TABLE_ICE, B.WATER, B.GRASS, B.LAVA, B.PATH, B.SAND]
            else:
                choices += [B.WATER, B.GRASS, B.LAVA, B.PATH, B.SAND]
            b = rng.choice(choices)
            if classic:
                s = s.replace(map=s.map.at[fr, fc].set(b.value))
            else:
                lvl = int(s.player_level)
                s = s.replace(map=s.map.at[lvl, fr, fc].set(b.value))
        if rng.random() < 0.2:
            if classic:
                s = s.replace(mob_map=s.mob_map.at[fr, fc].set(True))
            else:
                lvl = int(s.player_level)
                s = s.replace(mob_map=s.mob_map.at[lvl, fr, fc].set(True))
        if not classic and rng.random() < 0.3:
            lvl = int(s.player_level)
            it = rng.choice([env.ItemType.NONE, env.ItemType.TORCH, env.ItemType.LADDER_DOWN])
            s = s.replace(item_map=s.item_map.at[lvl, fr, fc].set(it.value))
    # enchantment table in front (full)
    if not classic and rng.random() < 0.3 and 0 <= fr < H and 0 <= fc < W:
        lvl = int(s.player_level)
        tbl = rng.choice([B.ENCHANTMENT_TABLE_FIRE, B.ENCHANTMENT_TABLE_ICE])
        s = s.replace(map=s.map.at[lvl, fr, fc].set(tbl.value),
                      player_mana=cast_like(s.player_mana, rng.choice([8, 9, 9, int(get_max_mana(s))])),
                      inventory=s.inventory.replace(ruby=cast_like(s.inventory.ruby, rng.choice([0, 1, 2])),
                                                    sapphire=cast_like(s.inventory.sapphire, rng.choice([0, 1, 2]))))
    # iron workshop: table AND furnace within CLOSE_BLOCKS (classic constants.py:76-88 / full :351-363, the 8 neighbours)
    # plus the iron recipe materials (classic game_logic.py:444-459 / full is_crafting_iron_*), so Make Iron X is reachable
    if rng.random() < 0.3:
        pr, pc = env.pos()
        nbrs = [(pr + dr, pc + dc) for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))
                if 0 <= pr + dr < H and 0 <= pc + dc < W and (pr + dr, pc + dc) != (fr, fc)]
        if len(nbrs) >= 2:
            (tr, tc), (ur, uc) = rng.sample(nbrs, 2)
            if classic:
                s = s.replace(map=s.map.at[tr, tc].set(B.CRAFTING_TABLE.value).at[ur, uc].set(B.FURNACE.value))
            else:
                lvl = int(s.player_level)
                s = s.replace(map=s.map.at[lvl, tr, tc].set(B.CRAFTING_TABLE.value).at[lvl, ur, uc].set(B.FURNACE.value))
            inv = s.inventory
            ikw = {k: cast_like(getattr(inv, k), rng.choice([0, 1, 3, 9])) for k in ("wood", "stone", "iron", "coal")}
            if not classic:
                ikw["pickaxe"] = cast_like(inv.pickaxe, rng.choice([0, 1, 2, 2, 3, 4]))
                ikw["sword"] = cast_like(inv.sword, rng.choice([0, 1, 2, 2, 3, 4]))
                ikw["armour"] = cast_like(inv.armour, [rng.choice([0, 0, 1, 2]) for _ in range(4)])
            else:
                ikw["iron_pickaxe"] = cast_like(inv.iron_pickaxe, rng.choice([0, 1]))
                ikw["iron_sword"] = cast_like(inv.iron_sword, rng.choice([0, 1]))
            s = s.replace(inventory=inv.replace(**ikw))
    # Ascend (full): change_floor game_logic.py:2451-2461 needs item LADDER_UP under the player and player_level > 0;
    # move the player to the level-1 up ladder (world_gen.py:663 up_ladders) so the state is a real one
    if not classic and rng.random() < 0.2:
        lvl = 1
        pr, pc = (int(v) for v in np.array(s.up_ladders[lvl]))
        it = rng.choice([env.ItemType.LADDER_UP, env.ItemType.LADDER_UP, env.ItemType.NONE])
        # change_floor game_logic.py:2474-2492 runs EVERY step and grants 1 XP while achievements[LEVEL_ACHIEVEMENT_MAP[level]]
        # is unset; a real descend sets it in the same step, so a level-1 state without it is unreachable -> set it here
        ach_i = int(K.LEVEL_ACHIEVEMENT_MAP[lvl])
        s = s.replace(player_level=cast_like(s.player_level, lvl),
                      player_position=cast_like(s.player_position, [pr, pc]),
                      item_map=s.item_map.at[lvl, pr, pc].set(it.value),
                      achievements=s.achievements.at[ach_i].set(True))
        env.state = s      # env.pos()/facing() below must see the new level and position
        fr, fc = env.facing()
    # ladder under the player (full)
    if not classic and rng.random() < 0.35:
        lvl = int(s.player_level)
        pr, pc = env.pos()
        it = rng.choice([env.ItemType.LADDER_DOWN, env.ItemType.LADDER_UP, env.ItemType.NONE])
        s = s.replace(item_map=s.item_map.at[lvl, pr, pc].set(it.value))
        mk = rng.choice([0, 7, 8, 12])
        s = s.replace(monsters_killed=s.monsters_killed.at[lvl].set(mk))
    return s


# ---------------------------------------------------------------- main loop
t0 = time.time()
per_seed = {}
for seed in range(SEED0, SEED0 + N_SEEDS):
    env.reset(seed)
    rng = random.Random(1000 * seed + 7)
    dead = None
    life_support = 0
    actions_taken = collections.Counter()
    step = 0
    saved_states = []
    for step in range(N_STEPS):
        s = env.state
        if float(s.player_health) <= (7 if bool(s.is_sleeping) else 4):   # keep the trajectory alive: restore stats (validation, not gameplay)
            mx = env.max_stats()
            env.state = s.replace(player_health=cast_like(s.player_health, mx["health"]),
                                  player_food=cast_like(s.player_food, mx["food"]),
                                  player_drink=cast_like(s.player_drink, mx["drink"]))
            life_support += 1
        check_state("drive", seed, step)
        if step % 25 == 0:
            saved_states.append((step, env.state))
        a = heuristic(rng)
        actions_taken[a] += 1
        _, _, done, _ = env.step(a)
        if done:
            dead = step
            break
    # synthetic phase: perturb saved states
    for j in range(N_SYN):
        st, base = saved_states[j % len(saved_states)]
        env.state = perturb(rng, base)
        check_state(f"synthetic(from step {st})", seed, j)
    per_seed[seed] = {"steps": step + 1, "dead_at": dead, "life_support": life_support,
                      "achievements": None,
                      "actions": dict(actions_taken.most_common(12))}
    print(f"[{variant}] seed {seed}: {step + 1} steps, dead_at={dead}, life_support={life_support}, "
          f"checks so far {counters['checks']}, counter-examples {counters['counter']}, {time.time() - t0:.0f}s", flush=True)

report = {"variant": variant, "seeds": per_seed, "checks": counters["checks"], "counter_examples": len(examples),
          "legal_and_changed": dict(legal_hits), "illegal_and_unchanged": dict(illegal_hits),
          "never_legal": [a for a in ACTIONS if legal_hits[a] == 0],
          "caveats": {k: v for k, v in counters.items() if k.startswith("caveat")},
          "examples": examples[:60], "caveat_examples": caveats[:20], "elapsed_s": round(time.time() - t0)}
print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=1, default=str))
for e in examples[:30]:
    print("COUNTER-EXAMPLE:", json.dumps(e, default=str))
for e in caveats[:6]:
    print("CAVEAT:", json.dumps(e, default=str))
if OUT:
    json.dump(report, open(OUT, "w"), indent=1, default=str)
