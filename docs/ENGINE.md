# The engine

`src/engine/` is the part of the game that is not a language model: character
sheets, the d20 rolls, and (from PR-16) combat. It is pure Python, imports
nothing from `src/agents`, `src/graph`, or LangChain, and every function that
rolls takes an optional random source so a whole fight can be replayed exactly
under test.

The reason it exists is the rule the whole project runs on: **the model never
produces a number that matters.** The Dungeon Master calls a tool, the engine
resolves it, and the model narrates a result it was handed. This is PR-05's
dice lesson (a formal language is read by a parser, not a model) applied to
the rest of the rules.

## Modules

| Module | Holds | Since |
|---|---|---|
| `character.py` | `Character`, `Abilities`, `Weapon`, the `Ability` / `Skill` / `Proficiency` / `Condition` enums, the skill→ability table | PR-14 |
| `checks.py` | `ability_check`, `saving_throw`, `attack_roll`, `roll_damage`; `RollMode` and `resolve_mode`; the frozen result types | PR-14 |
| `pregens.py` | Six level-1 SRD characters as data, `pregen()` to hand one out, the weapon table | PR-14 |
| `combatant.py` | `Combatant` (a creature in a fight) with `from_monster`, `Attack`, `Damage` | PR-15 |
| `combat.py` | Initiative, turns, damage, conditions, death saves, rests | PR-16 |

## `Character`

A frozen pydantic model. It **round-trips through JSON unchanged** — that is
its contract, because it lives inside the checkpointed `GameState` and, later,
in whatever the Discord bot persists. Change a sheet with
`model_copy(update={...})`, never in place.

Stored: identity (`player_id`, `name`), race, class, level, the six ability
scores, skill proficiencies (`proficient` or `expertise`), save proficiencies,
AC, max/current/temp HP, hit die size and dice remaining, speed, conditions,
inventory, weapons, notes.

Derived, never stored, so a sheet cannot contradict itself:

| | From |
|---|---|
| `proficiency_bonus` | level: +2 at 1–4, +3 at 5–8, … +6 at 17–20 |
| `ability_modifier(a)` | `(score − 10) // 2`; floor division makes 9 → −1 |
| `skill_modifier(s)` | the skill's ability, plus the bonus once for `proficient`, twice for `expertise` |
| `save_modifier(a)` | the ability, plus the bonus if the save is proficient |
| `passive_perception` | 10 + Perception modifier |
| `attack_ability(w)` | ranged → DEX; finesse → the higher of STR/DEX; else STR (thrown without finesse is still STR) |
| `attack_bonus(w)` | that ability's modifier plus proficiency |
| `damage_bonus(w)` | that ability's modifier |

A fresh sheet is at full HP with every hit die unspent (`current_hp` and
`hit_dice_remaining` default from `max_hp` and `level`). `current_hp > max_hp`
is rejected at construction.

## `checks.py`

Every roll returns a frozen dataclass — `CheckResult`, or `AttackResult` with a
`DamageRoll` on a hit — that records the dice rolled, the die kept, the
modifier, the total, the DC or AC, and the verdict. `describe()` renders the
whole arithmetic; that string is what the DM is given to narrate from:

```
Dexterity (Stealth) check: d20 [3, 10]→10 + 7 = 17 (advantage) vs DC 15 — success
Rapier attack: d20 12 + 5 = 17 vs AC 14 — hit; damage 1d8 [6] + 3 = 9 piercing
```

Rules applied, each pinned by a test in `tests/test_engine_checks.py`:

- **Advantage / disadvantage.** Two d20, keep the higher / lower. Any
  advantage together with any disadvantage cancels to a normal roll, however
  many of each (`resolve_mode`).
- **Checks and saves succeed on total ≥ DC.** A natural 20 or 1 is *not*
  special on a check or a save.
- **Attacks.** Natural 20 always hits and is a critical; natural 1 always
  misses; otherwise total ≥ AC.
- **Critical damage** rolls every damage die twice and adds the modifier once.
  `1d8+1d6` becomes `2d8+2d6`.
- **Versatile weapons** use their bigger die when `two_handed=True`.
- **Damage never goes below 0.**
- A named skill decides the ability: `ability_check(c, STR, Skill.STEALTH)` is
  still a Dexterity (Stealth) check.

## Monsters: `src/srd/` and `Combatant`

The vendored `corpus/srd/*.json` is read as **data** by `src/srd/`, not only
as retrieval text: `monster("goblin")`, `spell("fireball")`,
`equipment("longsword")`, `condition("prone")`, `magic_item("bag of
holding")`. Lookup is exact by slug or name (case-insensitive), then fuzzy
within two edits — `gobln` is the goblin, `ogr` is the ogre and not the orc
(ties go to the more similar name) — and anything further away raises
`UnknownEntry` with suggestions. A silently wrong monster is worse than a
question.

`src/srd/bestiary.py` turns a stat block into an engine `Combatant`:

```python
summon("goblin")                    # AC 15, 7 HP, Scimitar +4 for 1d6+2 slashing
summon_group("goblin", 3)           # ids goblin-1, goblin-2, goblin-3
summon("ogre", roll_hp=True, rng=…) # 7d10 + 7 × CON instead of the listed 59
```

`Combatant` is the monster-side counterpart of `Character`: frozen, JSON
round-trippable, with the numbers a fight needs (AC, HP, abilities, listed
save and skill totals, attacks with typed damage) and the text the DM narrates
from (traits, multiattack, legendary actions, senses, languages). Every one of
the 334 SRD monsters builds; a test proves it. Three shapes in the data that
the loader handles on purpose:

- **damage is a list** — an adult black dragon's bite is `2d10+6 piercing`
  plus `1d8 acid`, two `Damage` components. A rider that needs a saving throw
  (a giant spider's poison) is not a damage entry; it stays in `desc`;
- **flat damage** — a rat's bite is `"1"`, no dice;
- **a choice of damage types** — a djinni's scimitar; the first option is taken.

## Pregens

Six SRD-only level-1 characters built from the standard array plus racial
bonuses, so every number is checkable against the rulebook:

| key | | AC | HP | Weapons |
|---|---|---|---|---|
| `fighter` | Dorn Ironfist, human | 18 | 13 | longsword, handaxe |
| `rogue` | Kara Swiftfoot, halfling | 14 | 9 | rapier, shortbow, dagger |
| `cleric` | Brother Aldric, hill dwarf | 16 | 12 | mace |
| `wizard` | Elowen Nightbreeze, high elf | 13 | 7 | quarterstaff, dagger |
| `ranger` | Thessaly Vane, wood elf | 14 | 11 | longbow, shortsword |
| `barbarian` | Grask Bonecrusher, half-orc | 14 | 15 | greataxe, javelin |

`pregen("rogue", player_id="discord:123", name="Pip")` returns a deep copy
owned by that player. The templates are never handed out by reference.

## What is deliberately not modelled

- Multiclassing, feats, and per-weapon proficiency (every pregen is proficient
  with what it carries).
- Spells and spell slots — the cleric and wizard list theirs in `notes`, and
  the DM narrates them until PR-24.
- Class features with mechanics (Sneak Attack, Rage, Second Wind) are described
  in `notes`; PR-16 or PR-24 give the ones that matter in combat real tools.
- The mechanical effects of conditions on rolls (poisoned → disadvantage, and
  so on). The `Condition` enum exists; the effect table is PR-16's.
