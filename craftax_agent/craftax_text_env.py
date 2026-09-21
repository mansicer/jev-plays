"""Text-observation wrapper for vanilla Craftax — both variants:

  * "classic": Craftax-Classic-Symbolic-v1 — faithful Crafter reimplementation. 64x64 single map,
               7x9 view, 17 actions, 22 achievements (1 point each), no levels/mana/xp.
  * "full":    Craftax-Symbolic-v1 — 9 floors, 9x11 view, 43 actions, 67 achievements, max score 226.

The text layout (Last action / Position / You see / Facing / status / inventory / Available actions)
follows alem-env's language wrapper so the agent, goal board and UI are shared by both variants.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import numpy as np
from PIL import Image

from craftax.craftax_env import make_craftax_env_from_name

VARIANTS = ("classic", "full")
_MOVE = {"LEFT": "Move West", "RIGHT": "Move East", "UP": "Move North", "DOWN": "Move South"}
DIR_NAME = {1: "west", 2: "east", 3: "north", 4: "south"}
DIR_VEC = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
TOOL_TIER = ["none", "wood", "stone", "iron", "diamond"]
POTION_COLOURS = ["red", "green", "blue", "pink", "cyan", "yellow"]
LEVEL_NAMES = ["Overworld", "Dungeon", "Gnomish Mines", "Sewers", "Vault", "Troll Mines",
               "Fire Realm", "Ice Realm", "Graveyard"]
MOB_NAMES = {0: "zombie", 1: "gnome warrior", 2: "orc soldier", 3: "lizard", 4: "knight", 5: "troll",
             6: "pigman", 7: "frost troll", 8: "cow", 9: "bat", 10: "snail", 16: "skeleton",
             17: "gnome archer", 18: "orc mage", 19: "kobold", 20: "archer", 21: "deep thing",
             22: "fire elemental", 23: "ice elemental"}


def xy(pos):
    return f"(x={int(pos[1])}, y={int(pos[0])})"


def rel(dr, dc):
    parts = []
    if dr:
        parts.append(f"{abs(dr)} step{'s' if abs(dr) != 1 else ''} {'south' if dr > 0 else 'north'}")
    if dc:
        parts.append(f"{abs(dc)} step{'s' if abs(dc) != 1 else ''} {'east' if dc > 0 else 'west'}")
    return " and ".join(parts) if parts else "here"


# ------------------------------------------------------------------ system prompts

def _solid_names(K, extra=()) -> str:
    """Block names in K.SOLID_BLOCKS (the game's is_in_wall / is_in_solid_block set) plus extra movement blockers,
    rendered for the manual (classic constants.py:91-105 is a jnp array, full constants.py:366-390 a list)."""
    names = {b.value: b.name for b in K.BlockType}
    return ", ".join([names[int(v)].lower().replace("_", " ") for v in np.array(K.SOLID_BLOCKS)] + list(extra))


from craftax.craftax_classic import constants as _KC
from craftax.craftax import constants as _KF

# classic: SOLID_BLOCKS already holds WATER and lacks LAVA (move_player lets you step onto lava, game_logic.py:1374-1383).
_SOLID_CLASSIC = _solid_names(_KC)
# full: WATER/LAVA are not "solid" but block the player through COLLISION_LAND_CREATURE (constants.py:176,
# util/game_logic_utils.py:294-334), so they are appended to the movement list.
assert list(_KF.COLLISION_LAND_CREATURE[1:]) == [True, True], _KF.COLLISION_LAND_CREATURE
_SOLID_FULL = _solid_names(_KF, ("water", "lava"))

PROMPT_COMMON = """## How to play
- Each turn choose exactly one action from the "Available actions" list.
- Movement is absolute: Move North/South/East/West. A move attempt always sets your facing direction, even if blocked. Tiles that block movement: {solid_list}, and any tile a mob stands on.{lava_note} Use a blocked move to turn and face a tile.
- Do acts on the tile you face: chop trees (1 wood each), mine stone/coal/iron/diamond (needs the right pickaxe), attack mobs, drink water, eat ripe plants{do_extra}.
- Crafting needs you to stand next to (incl. diagonal) the station; you do not need to face it. Placing puts a block on the tile you face. Table/furnace/stone go on any non-solid tile (grass, sand, path, wood floor, lava{place_extra}); Place Stone also works on water (the water tile becomes stone); Place Plant needs a grass tile. A mob standing on the tile blocks any placement{place_item_extra}.
- Coordinates: x grows east, y grows south. "Position" and "You see" give absolute (x, y).

## Survival
- health, food, drink, energy (max 9{stat_extra}). Food/drink/energy drain over time; at 0 you lose health.
- Eat: kill a cow (3 health: 3 hits bare-handed, 2 with a wood sword, 1 with stone or better) for +6 food, or Do on a ripe plant for +4. Drink: each Do on water gives +1{drink_extra}. Energy: Sleep (you wake on your own at full energy; while asleep your actions are ignored, food/drink drain at half speed and health regenerates twice as fast, but a melee hit on a sleeper does 3.5x its normal damage (zombie: 7) and any melee or projectile hit wakes you early){rest_extra}.
- Night lowers the light level and raises the zombie spawn chance to 0.02 + 0.1 x (1 - light)^2 per step (0.02 by day, up to 0.12 at full dark); zombies do the same damage day or night.{night_extra} Keep a sword; retreat when health <= 4.
- Health regenerates 1 per 26 steps awake (1 per 13 asleep) while food and drink are > 0 and energy is > 0 or you are asleep; otherwise it drops 1 per 16 steps awake (1 per 31 asleep). Food drops 1 per ~26 steps, drink per ~21, energy per ~31 awake (food/drink drain at half rate while asleep){decay_extra}.

## Combat
{combat}

## Resource chain and recipes
- Trees -> wood (no tool). Stone/Coal need a wood pickaxe; Iron needs a stone pickaxe; Diamond needs an iron pickaxe{chain_extra}.
- Place Table: 2 wood (on any non-solid tile, no station needed). Place Furnace: 1 stone (on any non-solid tile, no table needed).
- At a table: Wood Pickaxe/Sword = 1 wood. Stone Pickaxe/Sword = 1 wood + 1 stone.{table_extra}
- At a table AND a furnace (both within 1 tile of you, incl. diagonal): Iron Pickaxe/Sword = 1 wood + 1 stone + 1 iron + 1 coal.{furnace_extra}
"""

PROMPT_CLASSIC = ("You are playing Craftax-Classic (Crafter): a 2D survival game. Unlock as many of the 22 achievements as "
                  "you can while staying alive; the game ends if health reaches 0 or after the step limit. Each achievement "
                  "scores once (1 point, 22 total); the score is kept when you die. The per-step Reward shown in the "
                  "observation = points of achievements unlocked that step + 0.1 x your health change that step.\n\n"
                  + PROMPT_COMMON.format(solid_list=_SOLID_CLASSIC,
                                         lava_note=" LAVA IS NOT A BLOCKER: moving onto a lava tile is allowed and sets health to 0, ending the episode immediately (Place Stone on it turns it into stone).",
                                         do_extra="; Do on a table or furnace has no effect",
                                         place_extra="", place_item_extra="",
                                         stat_extra="", drink_extra="", rest_extra="",
                                         night_extra=" Skeletons spawn at a flat 0.05 per step at any time of day, only on path (cave/tunnel) tiles.",
                                         decay_extra="", chain_extra="", table_extra="", furnace_extra="",
                                         combat="- Your melee damage per hit: 1 bare-handed, 2 wood sword, 3 stone sword, 5 iron sword.\n"
                                                "- Zombie: 5 health; deals 2 per hit (7 if you are asleep), then waits 5 steps; moves at your speed.\n"
                                                "- Skeleton: 3 health; keeps 4-5 tiles (row offset + column offset) away and shoots an arrow at that range (or when cornered), then cannot shoot for 4 steps. Arrows fly 1 tile per step in a straight line along the axis of greatest offset, only hit you if you are in that line, cross water, stop at other solid blocks or mobs, deal 2 damage and wake you; an arrow that reaches a table or furnace destroys it (the tile becomes path). Cow: 3 health.")
                  + """
## Progression
1. Wood (5+; the table costs 2 and each tool 1, 8 wood for the whole tree; inventory caps at 9) -> table -> wood pickaxe + wood sword. 2. Stone -> furnace (no table needed to place it; iron crafting needs table and furnace both within 1 tile of you), stone pickaxe/sword, Place Stone. 3. Coal + iron -> iron pickaxe/sword. 4. Diamond with the iron pickaxe.
Side achievements: Collect Sapling (Do on grass: 10% chance per try), Place Plant (sapling on grass), Eat Plant (a placed sapling ripens about 600 steps later; nothing damages it and mobs cannot walk onto it; Do on the ripe plant gives +4 food and it regrows from a sapling), Collect Drink, Eat Cow, Wake Up (Sleep then wake), Defeat Zombie, Defeat Skeleton (skeletons live in dark cave areas).

## Achievements (1 point each, 22 total)
Collect Wood, Place Table, Eat Cow, Collect Sapling, Collect Drink, Make Wood Pickaxe, Make Wood Sword, Place Plant, Defeat Zombie, Collect Stone, Place Stone, Eat Plant, Defeat Skeleton, Make Stone Pickaxe, Make Stone Sword, Wake Up, Place Furnace, Collect Coal, Collect Iron, Collect Diamond, Make Iron Pickaxe, Make Iron Sword.

Be decisive and efficient: do not repeat an action that changed nothing; if a move is blocked, you are now facing that tile — Do on it if useful, otherwise go around.""")

PROMPT_FULL = ("You are playing Craftax, a survival and dungeon-crawling game. Maximize achievements (score) while staying alive; "
               "the game ends if health reaches 0 or after the step limit. Each achievement scores once (1, 3, 5 or 8 points "
               "by tier, 226 total); the score is kept when you die. The per-step Reward shown in the observation = points "
               "of achievements unlocked that step + 0.1 x your health change that step.\n\n"
               + PROMPT_COMMON.format(solid_list=_SOLID_FULL,
                                      lava_note=" Water and lava block movement without damage.",
                                      do_extra=", open chests, gather wood from fire trees and ice shrubs, mine stalagmites for stone (wood pickaxe). Do on a crafting table or furnace REMOVES it (the tile becomes path, nothing is refunded)",
                                      place_extra=", water, gravel, fire/ice grass, darkness — in Craftax water is not solid, so table/furnace can go on water too",
                                      place_item_extra="; so does an item (torch/ladder) already on the tile. Place Torch needs grass/sand/path/fire grass/ice grass with no item on it",
                                      stat_extra=" at start; mana too", drink_extra=" (fountains too)",
                                      rest_extra=". Rest: only works while health is below max AND food and drink are both > 0 (with food or drink at 0 the rest ends in the same step and nothing happens); it locks you into auto-idle (all your actions are ignored) until health is full, food or drink hits 0, or a mob or projectile hits you; it does not heal faster than standing still (1 per 26 steps) and food/drink drain at the normal rate. Mobs act before your Rest is evaluated, so if a hit lands on the same step you start resting even from full health",
                                      night_extra=" Skeletons spawn at a flat 0.05 per step at any time of day on any grass or path tile of the overworld. Underground floors have no day/night effect: melee mobs spawn at 0.06 and ranged mobs at 0.05 per step, tripled until you have killed 8 monsters on that floor.",
                                      decay_extra="; every dexterity point above 1 slows all three drains by 12.5% (dex 5: ~51/41/61 steps); on the final boss floor food, drink, energy and starvation health loss stop entirely",
                                      chain_extra="; Sapphire/Ruby need a diamond pickaxe",
                                      table_extra=" Arrow = 1 wood + 1 stone (gives 2). Torch = 1 wood + 1 coal (gives 4). Diamond Pickaxe = 1 wood + 3 diamond; Diamond Sword = 1 wood + 2 diamond; Diamond Armour = 3 diamond per piece. Tools only upgrade (you cannot craft a tier you already have or a lower one).",
                                      furnace_extra=" Iron Armour = 3 iron + 3 coal per piece.",
                                      combat="- Your melee damage per hit: 1/2/3/5/8 for no/wood/stone/iron/diamond sword, x(1 + 0.25 x (strength - 1)); a fire/ice-enchanted sword adds 50% of the base sword damage as fire/ice damage (x(1 + 0.05 x (intelligence - 1))). Armour (4 slots): each iron piece blocks 10% and each diamond piece 20% of physical damage (4 diamond = 80%); a piece enchanted at a fire table (ruby) also blocks 20% of fire damage, at an ice table (sapphire) 20% of ice damage; enchantments do not reduce physical damage.\n"
                                             "- Overworld: zombie 5 health / 2 damage per hit (5-step cooldown), skeleton 3 health / arrows 2 damage (4-step cooldown). Melee hits do 3.5x damage while you sleep (zombie: 7); projectiles are not multiplied. Ranged mobs fire at row+column distance 4-5 (or when cornered) in a straight line along the axis of greatest offset; projectiles cross water, stop at other solid blocks or mobs, wake you and interrupt Rest, and destroy any table or furnace they hit. Deeper floors have stronger mobs; the health of nearby hostile mobs and the damage they do to you (after your armour) are listed under Status notes.\n"
                                             "- Resistances: Vault knights/archers block 50% of physical damage; Troll Mines trolls 20%; Fire Realm mobs (pigmen, fire elementals) block 90% of physical and all fire damage; Ice Realm mobs (frost trolls, ice elementals) 90% of physical and all ice damage. On the Graveyard (final floor) all damage you take is x1.5 (x5.25 melee if asleep).")
               + """
## Progression
1. Wood -> table -> wood pickaxe + wood sword. 2. Stone -> furnace, stone pickaxe/sword. 3. Coal + iron -> iron tools, iron armour. 4. Kill 8 hostile monsters (melee or ranged mobs, with any weapon; cows/bats/snails do not count) on a floor to unlock its ladder_down (the overworld ladder starts unlocked); stand on it and Descend. The first visit to each floor gives 1 XP; spend it with Level Up (1 XP each, max 5 per attribute): strength = +1 max health, +25% melee damage; dexterity = +2 max food/drink/energy, needs drain 12.5% slower, arrows +20% damage; intelligence = +3 max mana, +25% mana regen, +50% fireball/iceball damage, +5% enchant damage.
5. In dungeons open chests (torches, ores/gems, potions, arrows, sometimes a pickaxe/sword; the first chest you open on the Dungeon floor holds the bow; the first chest opened on the Sewers floor and the first on the Vault floor each hold a book, later chests on those floors do not). Read Book learns fireball/iceball; Cast costs 2 mana. Potions: the 6 colours map randomly each episode to +8 or -3 health, mana or energy; drink one to learn its effect. Enchanting: you must FACE an enchantment table (being adjacent is not enough) and spend 9 mana plus 1 gem: a fire table takes 1 ruby, an ice table 1 sapphire. Enchant Sword needs a sword, Enchant Bow the bow, Enchant Armour at least one armour piece (it enchants one random not-yet-enchanted armour slot, which may be an empty slot).
6. Diamond gear, then the deeper realms and the final boss.

## Achievements (score points: 1 basic, 3 intermediate, 5 advanced, 8 very advanced; max 226)
Collect Wood, Place Table, Eat Cow, Collect Sapling, Collect Drink, Make Wood Pickaxe, Make Wood Sword, Place Plant, Defeat Zombie, Collect Stone, Place Stone, Eat Plant, Defeat Skeleton, Make Stone Pickaxe, Make Stone Sword, Wake Up, Place Furnace, Collect Coal, Collect Iron, Collect Diamond, Make Iron Pickaxe, Make Iron Sword, Make Arrow, Make Torch, Place Torch, Collect Sapphire, Collect Ruby, Make Diamond Pickaxe, Make Diamond Sword, Make Iron Armour, Make Diamond Armour, Enter Gnomish Mines, Enter Dungeon, Enter Sewers, Enter Vault, Enter Troll Mines, Enter Fire Realm, Enter Ice Realm, Enter Graveyard, Defeat Gnome Warrior/Archer, Orc Soldier/Mage, Lizard, Kobold, Troll, Deep Thing, Pigman, Fire/Ice Elemental, Frost Troll, Knight, Archer, Eat Bat, Eat Snail, Find Bow, Fire Bow, Learn/Cast Fireball, Learn/Cast Iceball, Open Chest, Drink Potion, Enchant Sword, Enchant Armour, Damage/Defeat Necromancer.

Be decisive and efficient: do not repeat an action that changed nothing; if a move is blocked, you are now facing that tile — Do on it if useful (a table or furnace would be removed), otherwise go around.""")


# ------------------------------------------------------------------ env

class CraftaxTextEnv:
    def __init__(self, variant: str = "classic", max_timesteps: int = 10000, render: bool = True, pixel_size: int = 16):
        assert variant in VARIANTS, variant
        self.variant = variant
        self.classic = variant == "classic"
        if self.classic:
            from craftax.craftax_classic import constants as K
            from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
            env_name = "Craftax-Classic-Symbolic-v1"
        else:
            from craftax.craftax import constants as K
            from craftax.craftax.renderer import make_craftax_pixel_renderer
            env_name = "Craftax-Symbolic-v1"
        self.K = K
        self.Action, self.Achievement, self.BlockType = K.Action, K.Achievement, K.BlockType
        self.ItemType = None if self.classic else K.ItemType
        self.action_names = [_MOVE.get(a.name, a.name.replace("_", " ").title()) for a in K.Action]
        self.action_index = {n: i for i, n in enumerate(self.action_names)}
        self.ach_names = {a.value: a.name.replace("_", " ").title() for a in K.Achievement}
        self.ach_index = {n: i for i, n in self.ach_names.items()}
        self.ach_points = np.ones(len(K.Achievement), dtype=int) if self.classic else np.array(K.ACHIEVEMENT_REWARD_MAP)
        self.max_score = int(self.ach_points.sum())
        self.n_ach_total = len(K.Achievement)
        self.view_r, self.view_c = K.OBS_DIM[0] // 2, K.OBS_DIM[1] // 2
        B = K.BlockType
        self.ignore_blocks = {B.INVALID, B.OUT_OF_BOUNDS, B.GRASS, B.PATH, B.SAND}
        if not self.classic:
            self.ignore_blocks |= {B.DARKNESS, B.WALL, B.WALL_MOSS, B.FIRE_GRASS, B.ICE_GRASS, B.GRAVEL}

        self.env = make_craftax_env_from_name(env_name, auto_reset=False)
        self.params = self.env.default_params.replace(max_timesteps=max_timesteps)
        self._reset = jax.jit(self.env.reset)
        self._step = jax.jit(self.env.step)
        self._render = jax.jit(make_craftax_pixel_renderer(pixel_size)) if render else None
        # Game tables / functions reused by the rule helpers below (no re-implemented conditions).
        self.static_params = getattr(self.env, "static_env_params", None)
        self._solid_ids = frozenset(int(v) for v in np.array(K.SOLID_BLOCKS))   # classic: jnp array (incl. WATER); full: list
        self._item_none = 0 if self.classic else K.ItemType.NONE.value
        if not self.classic:
            from craftax.craftax.util.game_logic_utils import (get_damage_done_to_player, get_player_damage_vector,
                                                                get_damage)
            self._can_place_item = np.array(K.CAN_PLACE_ITEM_MAPPING)                     # constants.py:397-406
            self._proj_of_ranged = np.array(K.RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING)  # constants.py:324-335
            self._defense = np.array(K.MOB_TYPE_DEFENSE_MAPPING)                          # constants.py:293-322
            sp = self.static_params
            # jitted once: the game's own damage functions (un-jitted jnp calls per visible mob would be slow)
            self._dmg_to_player = jax.jit(lambda state, vec: get_damage_done_to_player(state, sp, vec))
            self._dmg_by_player = jax.jit(lambda state, defense: get_damage(get_player_damage_vector(state), defense))
        self.system_prompt = PROMPT_CLASSIC if self.classic else PROMPT_FULL
        self.state = None
        self.rng = None
        self.last_action = "none"
        self.last_reward = 0.0
        self.last_unlocked: list[str] = []
        self.last_hp_delta = 0.0
        self.total_reward = 0.0

    # ---- lifecycle ----
    def reset(self, seed: int) -> dict:
        self.rng = jax.random.PRNGKey(seed)
        self.rng, k = jax.random.split(self.rng)
        _, self.state = self._reset(k, self.params)
        self.last_action, self.last_reward, self.last_unlocked = "none", 0.0, []
        self.last_hp_delta, self.total_reward = 0.0, 0.0
        return self.observe()

    def step(self, action: str):
        idx = self.action_index.get(action, 0)
        ach_before = np.array(self.state.achievements) > 0
        hp_before = float(self.state.player_health)
        self.rng, k = jax.random.split(self.rng)
        _, self.state, reward, done, _ = self._step(k, self.state, idx, self.params)
        ach_after = np.array(self.state.achievements) > 0
        unlocked = [self.ach_names[int(i)] for i in np.flatnonzero(ach_after & ~ach_before)]
        reward = float(reward)
        self.last_action = action if action in self.action_index else f"{action} (unparsable -> Noop)"
        self.last_reward, self.last_unlocked = reward, unlocked
        self.last_hp_delta = float(self.state.player_health) - hp_before
        self.total_reward += reward
        return self.observe(), reward, bool(done), {"unlocked": unlocked, "hp_delta": self.last_hp_delta}

    # ---- derived facts ----
    def score(self) -> int:
        return int((np.array(self.state.achievements) > 0).astype(int) @ self.ach_points)

    def n_achievements(self) -> int:
        return int((np.array(self.state.achievements) > 0).sum())

    def level(self) -> int:
        return 0 if self.classic else int(self.state.player_level)

    def level_map(self):
        return np.array(self.state.map if self.classic else self.state.map[self.state.player_level])

    def pos(self):
        return tuple(int(v) for v in np.array(self.state.player_position))

    def near(self, block) -> bool:
        r, c = self.pos()
        m = self.level_map()
        return bool((m[max(0, r - 1):r + 2, max(0, c - 1):c + 2] == block.value).any())

    def facing(self):
        dr, dc = DIR_VEC[int(self.state.player_direction)]
        r, c = self.pos()
        return r + dr, c + dc

    # ---- game-rule helpers (mirror craftax game_logic for both variants; reusable by other modules) ----
    def in_map(self, r: int, c: int) -> bool:
        """Game in_bounds(): (r, c) lies on the current level map. Check before indexing (JAX wraps negative indices)."""
        H, W = self.state.map.shape[-2:]
        return 0 <= r < H and 0 <= c < W

    def block_at(self, r: int, c: int):
        """BlockType at (r, c) on the current level; OUT_OF_BOUNDS off-map."""
        if not self.in_map(r, c):
            return self.BlockType.OUT_OF_BOUNDS
        s = self.state
        return self.BlockType(int(s.map[r, c] if self.classic else s.map[s.player_level, r, c]))

    def item_at(self, r: int, c: int) -> int:
        """ItemType value at (r, c) (full). Classic has no item map and always returns self._item_none."""
        if self.classic or not self.in_map(r, c):
            return self._item_none
        s = self.state
        return int(s.item_map[s.player_level, r, c])

    def mob_at(self, r: int, c: int) -> bool:
        """Game is_in_mob(): a mob (mob_map) or the player stands on (r, c)."""
        if not self.in_map(r, c):
            return False
        s = self.state
        occupied = s.mob_map[r, c] if self.classic else s.mob_map[s.player_level, r, c]
        return bool(occupied) or (r, c) == self.pos()

    def solid_at(self, r: int, c: int) -> bool:
        """Game is_in_wall() (classic; WATER is solid there) / is_in_solid_block() (full) via K.SOLID_BLOCKS."""
        return self.in_map(r, c) and self.block_at(r, c).value in self._solid_ids

    def walkable(self, r: int, c: int) -> bool:
        """Game move_player() rule: in bounds, not solid, no mob. Full (COLLISION_LAND_CREATURE) also blocks WATER
        and LAVA; classic lets you step onto LAVA (classic game_logic.py:1374-1383)."""
        if not self.in_map(r, c) or self.solid_at(r, c) or self.mob_at(r, c):
            return False
        return self.classic or self.block_at(r, c) not in (self.BlockType.WATER, self.BlockType.LAVA)

    def can_place(self, allow_water: bool = False) -> bool:
        """Game place_block() gate on the faced tile (Place Table/Furnace/Stone): in bounds, no mob, not solid,
        full: no item on it (classic game_logic.py:586-720, full :828-1047). allow_water=True adds Place Stone's
        'faced tile == WATER' exception (classic :641-650, full :893-906)."""
        fr, fc = self.facing()
        if not self.in_map(fr, fc) or self.mob_at(fr, fc):
            return False
        if allow_water and self.block_at(fr, fc) == self.BlockType.WATER:
            return True
        return not self.solid_at(fr, fc) and self.item_at(fr, fc) == self._item_none

    def can_place_item(self) -> bool:
        """Game Place Torch gate (full game_logic.py:925-942): faced tile in bounds, no mob, block in
        CAN_PLACE_ITEM_BLOCKS (grass/sand/path/fire_grass/ice_grass) and no item already on it."""
        if self.classic:
            return False
        fr, fc = self.facing()
        if not self.in_map(fr, fc) or self.mob_at(fr, fc):
            return False
        return bool(self._can_place_item[self.block_at(fr, fc).value]) and self.item_at(fr, fc) == self._item_none

    def incapacitated(self) -> bool:
        """craftax_step replaces every action with NOOP while sleeping (classic :1660, full :3011) or resting (full :3012)."""
        s = self.state
        return bool(s.is_sleeping) or bool(getattr(s, "is_resting", False))

    def projectile_slot_free(self) -> bool:
        """Full: fewer than static_params.max_player_projectiles (3) of your projectiles are in flight on this floor
        (shoot_projectile :2498-2510, cast_spell :2539-2559). Classic has no projectiles -> False."""
        if self.classic:
            return False
        s = self.state
        return int(np.array(s.player_projectiles.mask[s.player_level]).sum()) < self.static_params.max_player_projectiles

    def tools(self) -> tuple[int, int]:
        """(pickaxe tier, sword tier) 0..4 for both variants."""
        inv = self.state.inventory
        if self.classic:
            p = 3 if int(inv.iron_pickaxe) else 2 if int(inv.stone_pickaxe) else 1 if int(inv.wood_pickaxe) else 0
            s = 3 if int(inv.iron_sword) else 2 if int(inv.stone_sword) else 1 if int(inv.wood_sword) else 0
            return p, s
        return int(inv.pickaxe), int(inv.sword)

    def max_stats(self) -> dict:
        s = self.state
        if self.classic:
            return {"health": 9, "food": 9, "drink": 9, "energy": 9}
        from craftax.craftax.util.game_logic_utils import (get_max_health, get_max_food, get_max_drink,
                                                            get_max_energy, get_max_mana)
        return {"health": int(get_max_health(s)), "food": int(get_max_food(s)), "drink": int(get_max_drink(s)),
                "energy": int(get_max_energy(s)), "mana": int(get_max_mana(s))}

    def player_damage(self, mob: dict | None = None) -> float:
        """Damage one Do hit deals. Full: get_damage(get_player_damage_vector(state), MOB_TYPE_DEFENSE_MAPPING[type_id, cls])
        (util/game_logic_utils.py:32-55, 252-277) for a visible_mobs() entry, so sword enchantments and the mob's
        floor defenses are included; with no mob (or a classic/projectile entry) the no-defense figure."""
        s = self.state
        if self.classic:
            from craftax.craftax_classic.game_logic import get_player_attack_damage
            return float(get_player_attack_damage(s))
        defense = np.zeros(3, dtype=np.float32)
        if mob and mob.get("type_id") is not None:
            defense = self._defense[mob["type_id"], mob["cls"]]
        return round(float(self._dmg_by_player(s, defense)), 3)   # float32 -> clean decimal

    def visible_mobs(self) -> list[dict]:
        """Mobs in view: name, position, Chebyshev distance, health, hostility and a factual attack line."""
        s = self.state
        r, c = self.pos()
        out = []

        def add(name, mr, mc, health, hostile, attack, cls=None, type_id=None, projectile=False):
            if abs(mr - r) <= self.view_r and abs(mc - c) <= self.view_c:
                out.append({"name": name, "r": mr, "c": mc, "dist": max(abs(mr - r), abs(mc - c)),
                            "health": float(health), "hostile": hostile, "attack": attack,
                            "cls": cls, "type_id": type_id, "projectile": projectile})

        if self.classic:
            groups = ((s.zombies, "zombie", True, "deals 2 per hit (7 while you are asleep), then waits 5 steps; moves at your speed"),
                      (s.cows, "cow", False, ""), (s.skeletons, "skeleton", True, "shoots arrows for 2 damage, 4-step cooldown"),
                      (s.arrows, "arrow", False, "incoming; hits for 2 on contact, moves 1 tile per step"))
            for cls, name, hostile, attack in groups:
                pos = np.array(cls.position); mask = np.array(cls.mask); hp = np.array(cls.health)
                for i in np.flatnonzero(mask):
                    add(name, int(pos[i][0]), int(pos[i][1]), hp[i], hostile, attack, projectile=name == "arrow")
            return out
        lvl = s.player_level
        light = np.array(s.light_map[lvl])
        dmg = np.array(self.K.MOB_TYPE_DAMAGE_MAPPING)
        MT, PT = self.K.MobType, self.K.ProjectileType

        def elements(vec):   # names the non-physical parts of a damage vector (before armour)
            e = [n for n, v in zip(("fire", "ice"), vec[1:]) if v]
            return f" (includes {'/'.join(e)} damage)" if e else ""

        def proj_name(t):
            try:
                return PT(t).name.lower().rstrip("2")
            except ValueError:
                return "projectile"

        for cls, offset, kind in ((s.melee_mobs, 0, MT.MELEE), (s.passive_mobs, 8, MT.PASSIVE), (s.ranged_mobs, 16, MT.RANGED)):
            pos = np.array(cls.position[lvl]); mask = np.array(cls.mask[lvl]); tid = np.array(cls.type_id[lvl])
            hp = np.array(cls.health[lvl])
            for i in np.flatnonzero(mask):
                mr, mc = int(pos[i][0]), int(pos[i][1])
                if light[mr, mc] <= 0.05:   # renderer.py:122-125 hides tiles AND mobs in darkness
                    continue
                t = int(tid[i])
                if kind == MT.MELEE:   # game_logic.py:1189-1201: base * (1 + 2.5*is_sleeping) -> get_damage_done_to_player
                    vec = dmg[t][MT.MELEE.value]
                    awake = float(self._dmg_to_player(s, vec)); asleep = float(self._dmg_to_player(s, vec * 3.5))
                    attack = (f"deals {awake:g} per hit ({asleep:g} while you are asleep){elements(vec)}, "
                              "then waits 5 steps; moves at your speed")
                elif kind == MT.RANGED:   # game_logic.py:1505-1516 projectile type, :1678-1685 damage by projectile type
                    p = int(self._proj_of_ranged[t]); vec = dmg[p][MT.PROJECTILE.value]
                    hit = float(self._dmg_to_player(s, vec))
                    attack = f"shoots {proj_name(p)}s for {hit:g} damage{elements(vec)}, 4-step cooldown"
                else:
                    attack = ""
                add(MOB_NAMES.get(offset + t, "creature"), mr, mc, hp[i], kind != MT.PASSIVE, attack, cls=kind.value, type_id=t)
        for cls, who in ((s.mob_projectiles, "incoming"), (s.player_projectiles, "yours")):
            pos = np.array(cls.position[lvl]); mask = np.array(cls.mask[lvl]); tid = np.array(cls.type_id[lvl])
            for i in np.flatnonzero(mask):
                mr, mc = int(pos[i][0]), int(pos[i][1])
                if light[mr, mc] <= 0.05:
                    continue
                t = int(tid[i])
                if who == "incoming":   # game_logic.py:1610-1711 _move_mob_projectile
                    vec = dmg[t][MT.PROJECTILE.value] if 0 <= t < len(dmg) else np.zeros(3)
                    attack = f"incoming; hits for {float(self._dmg_to_player(s, vec)):g} on contact{elements(vec)}, moves 1 tile per step"
                else:
                    attack = "your projectile; moves 1 tile per step"
                add(proj_name(t), mr, mc, 0, False, attack, projectile=True)
        return out

    def legal_actions(self) -> list[str]:
        s, inv, B = self.state, self.state.inventory, self.BlockType
        if self.incapacitated():   # every action is replaced by NOOP while sleeping/resting
            return ["Noop"]
        i = lambda v: int(v)
        table, furnace = self.near(B.CRAFTING_TABLE), self.near(B.FURNACE)
        fr, fc = self.facing()
        face_block = self.block_at(fr, fc)
        pick, sword = self.tools()
        mx = self.max_stats()
        ok = {
            "Noop": True, "Move West": True, "Move East": True, "Move North": True, "Move South": True, "Do": True,
            "Sleep": i(s.player_energy) < mx["energy"],
            "Place Stone": i(inv.stone) > 0 and self.can_place(allow_water=True),
            "Place Table": i(inv.wood) >= 2 and self.can_place(),
            "Place Furnace": i(inv.stone) > 0 and self.can_place(),   # no table needed in either variant
            "Place Plant": i(inv.sapling) > 0 and face_block == B.GRASS and not self.mob_at(fr, fc)
                           and self.item_at(fr, fc) == self._item_none,   # classic :671-711, full :996-1041
        }
        if self.classic:   # Crafter lets you re-craft a tool you already own
            ok.update({
                "Make Wood Pickaxe": table and i(inv.wood) >= 1,
                "Make Stone Pickaxe": table and i(inv.wood) >= 1 and i(inv.stone) >= 1,
                "Make Iron Pickaxe": table and furnace and i(inv.wood) >= 1 and i(inv.stone) >= 1 and i(inv.iron) >= 1 and i(inv.coal) >= 1,
                "Make Wood Sword": table and i(inv.wood) >= 1,
                "Make Stone Sword": table and i(inv.wood) >= 1 and i(inv.stone) >= 1,
                "Make Iron Sword": table and furnace and i(inv.wood) >= 1 and i(inv.stone) >= 1 and i(inv.iron) >= 1 and i(inv.coal) >= 1,
            })
            return [a for a in self.action_names if ok.get(a, False)]

        I = self.ItemType
        here_item = int(s.item_map[s.player_level, s.player_position[0], s.player_position[1]])
        armour = np.array(inv.armour); potions = np.array(inv.potions); spells = np.array(s.learned_spells)
        ench = face_block in (B.ENCHANTMENT_TABLE_FIRE, B.ENCHANTMENT_TABLE_ICE)
        gem = i(inv.ruby) if face_block == B.ENCHANTMENT_TABLE_FIRE else i(inv.sapphire)   # enchant() :2740-2744
        can_ench = ench and i(s.player_mana) >= 9 and gem >= 1                              # could_enchant :2746-2749
        proj_free = self.projectile_slot_free()
        ok.update({
            # rest starts at :1850-1852 (health < max) but :1857-1863 ends it in the SAME step when food <= 0 or drink <= 0
            # (decay at :1874-1890 runs after, so the pre-step values decide); a hit landing this step can start a rest even at
            # full health (update_mobs :3055 runs before update_player_intrinsics :3064) -- not decidable here, documented in the manual
            "Rest": float(s.player_health) < mx["health"] and i(s.player_food) > 0 and i(s.player_drink) > 0,
            "Place Torch": i(inv.torches) > 0 and self.can_place_item(),
            "Make Wood Pickaxe": table and i(inv.wood) >= 1 and pick < 1,
            "Make Stone Pickaxe": table and i(inv.wood) >= 1 and i(inv.stone) >= 1 and pick < 2,
            "Make Iron Pickaxe": table and furnace and i(inv.wood) >= 1 and i(inv.stone) >= 1 and i(inv.iron) >= 1 and i(inv.coal) >= 1 and pick < 3,
            "Make Diamond Pickaxe": table and i(inv.wood) >= 1 and i(inv.diamond) >= 3 and pick < 4,
            "Make Wood Sword": table and i(inv.wood) >= 1 and sword < 1,
            "Make Stone Sword": table and i(inv.wood) >= 1 and i(inv.stone) >= 1 and sword < 2,
            "Make Iron Sword": table and furnace and i(inv.wood) >= 1 and i(inv.stone) >= 1 and i(inv.iron) >= 1 and i(inv.coal) >= 1 and sword < 3,
            "Make Diamond Sword": table and i(inv.wood) >= 1 and i(inv.diamond) >= 2 and sword < 4,
            "Make Iron Armour": table and furnace and i(inv.iron) >= 3 and i(inv.coal) >= 3 and (armour < 1).any(),
            "Make Diamond Armour": table and i(inv.diamond) >= 3 and (armour < 2).any(),
            "Make Arrow": table and i(inv.wood) >= 1 and i(inv.stone) >= 1 and i(inv.arrows) < 99,     # literal 99 in game_logic.py:764
            "Make Torch": table and i(inv.wood) >= 1 and i(inv.coal) >= 1 and i(inv.torches) < 99,     # game_logic.py:778
            "Shoot Arrow": i(inv.bow) >= 1 and i(inv.arrows) >= 1 and proj_free,
            "Cast Fireball": bool(spells[0]) and i(s.player_mana) >= 2 and proj_free,
            "Cast Iceball": bool(spells[1]) and i(s.player_mana) >= 2 and proj_free,
            "Read Book": i(inv.books) > 0,
            "Enchant Sword": can_ench and sword > 0,                 # :2756-2761
            "Enchant Armour": can_ench and int(armour.sum()) > 0,    # :2763-2768
            "Enchant Bow": can_ench and i(inv.bow) > 0,              # :2751-2754
            "Descend": here_item == I.LADDER_DOWN.value
                       and i(s.monsters_killed[s.player_level]) >= self.K.MONSTERS_KILLED_TO_CLEAR_LEVEL
                       and i(s.player_level) < self.static_params.num_levels - 1,
            "Ascend": here_item == I.LADDER_UP.value and i(s.player_level) > 0,
        })
        for j, colour in enumerate(POTION_COLOURS):
            ok[f"Drink Potion {colour.title()}"] = int(potions[j]) > 0
        for attr in ("Dexterity", "Strength", "Intelligence"):
            ok[f"Level Up {attr}"] = i(s.player_xp) >= 1 and i(getattr(s, f"player_{attr.lower()}")) < self.params.max_attribute
        return [a for a in self.action_names if ok.get(a, False)]

    # ---- text ----
    def observe(self) -> dict:
        s, B = self.state, self.BlockType
        r, c = self.pos()
        lvl = self.level()
        m = self.level_map(); H, W = m.shape
        items = None if self.classic else np.array(s.item_map[lvl])
        light = None if self.classic else np.array(s.light_map[lvl])
        r0, r1 = max(0, r - self.view_r), min(H, r + self.view_r + 1)
        c0, c1 = max(0, c - self.view_c), min(W, c + self.view_c + 1)

        seen: dict[str, tuple[int, int, int]] = {}   # name -> (dist, row, col)
        counts: dict[str, int] = {}
        for rr in range(r0, r1):
            for cc in range(c0, c1):
                if light is not None and light[rr, cc] <= 0.05:
                    continue
                names = []
                b = B(int(m[rr, cc]))
                if b not in self.ignore_blocks:
                    names.append(b.name.lower())
                if items is not None and int(items[rr, cc]) != self.ItemType.NONE.value:
                    names.append(self.ItemType(int(items[rr, cc])).name.lower())
                for n in names:
                    d = abs(rr - r) + abs(cc - c)
                    counts[n] = counts.get(n, 0) + 1
                    if n not in seen or d < seen[n][0]:
                        seen[n] = (d, rr, cc)
        see_lines = [f"- {n} {rel(rr - r, cc - c)} {xy((rr, cc))}" + (f" (x{counts[n]} in view)" if counts[n] > 1 else "")
                     for n, (d, rr, cc) in sorted(seen.items(), key=lambda kv: kv[1][0])]
        mobs = self.visible_mobs()
        mob_lines = [f"- {m['name']} {rel(m['r'] - r, m['c'] - c)} {xy((m['r'], m['c']))}, "
                     + (m["attack"] if m["projectile"] else f"health {m['health']:g}")
                     for m in sorted(mobs, key=lambda m: m["dist"])]

        fr, fc = self.facing()
        face_desc = "out of bounds"
        if 0 <= fr < H and 0 <= fc < W:
            face_desc = B(int(m[fr, fc])).name.lower()
            if items is not None and int(items[fr, fc]) != self.ItemType.NONE.value:
                face_desc = f"{self.ItemType(int(items[fr, fc])).name.lower()} on {face_desc}"
            for m in mobs:
                if (m["r"], m["c"]) == (fr, fc):
                    face_desc = f"{m['name']} (on {face_desc})"
        light_level = float(s.light_level)
        if light is not None and lvl > 0:   # underground: day/night (light_level) is unused; light_map governs visibility
            lit = int((light[r0:r1, c0:c1] > 0.05).sum()); total = (r1 - r0) * (c1 - c0)
            light_line = (f"Light: underground, day/night does not apply here; tile light at your position {float(light[r, c]):.2f}; "
                          f"{lit}/{total} tiles in view are lit (tiles and mobs with light <= 0.05 are hidden; "
                          "a placed torch lights tiles within about 4 steps; lava and the up-ladder area also glow)")
        else:
            light_line = f"Light: {'bright' if light_level > 0.6 else 'dim' if light_level > 0.3 else 'dark (night)'} ({light_level:.2f})"
        # reward = achievement points unlocked this step + 0.1 x health change (classic game_logic.py:1695-1700, full :3074-3079)
        why = [f"unlocked {n} +{int(self.ach_points[self.ach_index[n]])}" for n in self.last_unlocked]
        if self.last_hp_delta:
            why.append(f"health {self.last_hp_delta:+g} -> {self.last_hp_delta * 0.1:+.1f}")
        reward_line = f"Reward: {self.last_reward:+.1f}" + (f" ({'; '.join(why)})" if why else "") + f"; total this game {self.total_reward:+.1f}"
        if self.classic:
            where = "Location: Overworld (single map, 64x64)"
        else:
            killed = int(s.monsters_killed[lvl]); need = self.K.MONSTERS_KILLED_TO_CLEAR_LEVEL
            where = (f"Location: {LEVEL_NAMES[lvl]} (floor {lvl + 1}/9) — monsters killed on this floor: {killed}/{need}, "
                     + ("ladder down unlocked" if killed >= need else f"ladder down locked until {need} monsters are killed here"))
        long_ctx = "\n".join([
            f"Last action: {self.last_action}", reward_line, "",
            f"Step: {int(s.timestep)}/{self.params.max_timesteps} (ends early if you die)",
            f"Position: {xy((r, c))}", where,
            f"Achievements: {self.n_achievements()}/{self.n_ach_total}" + ("" if self.classic else f" — score {self.score()}/{self.max_score}"),
            light_line,
            "", "You see:" if see_lines else "You see: only open ground.", *see_lines,
            "", "Mobs in view:" if mob_lines else "Mobs in view: none.", *mob_lines,
            "", f"Facing: {DIR_NAME[int(s.player_direction)]}. Do target: {face_desc} {xy((fr, fc))}.",
        ])

        inv = s.inventory
        mx = self.max_stats()
        pick, sword = self.tools()
        inv_lines = [f"- {k}: {int(getattr(inv, k))}" for k in ("wood", "stone", "coal", "iron", "diamond", "sapling")
                     if int(getattr(inv, k)) > 0]
        if not self.classic:
            inv_lines += [f"- {k}: {int(getattr(inv, k))}" for k in ("sapphire", "ruby", "torches", "arrows", "books")
                          if int(getattr(inv, k)) > 0]
        inv_lines.append(f"- pickaxe: {TOOL_TIER[pick]}")
        inv_lines.append(f"- sword: {TOOL_TIER[sword]}")
        status = ["Your status:",
                  f"- health: {float(s.player_health):.0f}/{mx['health']}",
                  f"- food: {int(s.player_food)}/{mx['food']}",
                  f"- drink: {int(s.player_drink)}/{mx['drink']}",
                  f"- energy: {int(s.player_energy)}/{mx['energy']}"]
        if bool(s.is_sleeping):
            status.append("- sleeping (actions are ignored until you wake: energy back at max, or a hit wakes you)")
        extra = []
        if not self.classic:
            armour = [int(a) for a in np.array(inv.armour)]; potions = [int(p) for p in np.array(inv.potions)]
            spells = [bool(x) for x in np.array(s.learned_spells)]
            if int(inv.bow) > 0:
                inv_lines.append("- bow: yes")
            if any(armour):
                inv_lines.append("- armour (helmet/chest/legs/boots): " + "/".join(TOOL_TIER[a] if a else "none" for a in armour))
            if any(potions):
                inv_lines.append("- potions: " + ", ".join(f"{n} {col}" for n, col in zip(potions, POTION_COLOURS) if n))
            if any(spells):
                inv_lines.append("- spells: " + ", ".join(n for n, k in zip(("fireball", "iceball"), spells) if k))
            status += [f"- mana: {int(s.player_mana)}/{mx['mana']}",
                       f"- xp: {int(s.player_xp)}" + (" (spend it with Level Up)" if int(s.player_xp) > 0 else "")]
            if bool(s.is_resting):
                status.append("- resting (actions are ignored until health is full, food or drink hits 0, or you are hit)")
            extra = ["", "Attributes:", f"- dexterity: {int(s.player_dexterity)}", f"- strength: {int(s.player_strength)}",
                     f"- intelligence: {int(s.player_intelligence)}"]
        short_ctx = "\n".join([*status, *extra, "", "Your inventory:", *inv_lines,
                               "", "Available actions:", *(f" - {a}" for a in self.legal_actions())])
        img = None
        if self._render is not None:
            img = Image.fromarray(np.array(self._render(s)).astype(np.uint8))
        return {"text": {"long_term_context": long_ctx, "short_term_context": short_ctx}, "image": img}
