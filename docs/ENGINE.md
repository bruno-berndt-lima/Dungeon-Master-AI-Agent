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
| `combat.py` | `Encounter`, initiative and turns, attacks, damage and healing, conditions and their effects, death saves, rests; every function returns `Event`s | PR-16 |

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

## Combat: `combat.py`

Every function is pure: it takes frozen models and returns new ones plus a
list of `Event`s — `kind`, a `text` the DM narrates from, and the numbers in
`data`. Illegal actions raise `RulesError` with a sentence for the player
("It is goblin-1's turn, not Kara's"); the engine never guesses.

An `Encounter` holds every participant as a `Combatant` — party members via
`Combatant.from_character`, monsters from the bestiary — with the initiative
order, the round, and whose turn it is. Between fights the character *sheet*
is the source of truth; `sync_party` writes HP, temp HP, conditions and death
back when a fight ends.

```python
enc, events = start_encounter([dorn, kara], summon_group("goblin", 2), rng=rng)
enc, events = attack(enc, "Kara", "goblin-1", attack_name="rapier", rng=rng)
enc, events = end_turn(enc, rng=rng)          # skips the dead; rolls death saves for the downed
party = sync_party(party, enc)                # when enc.active is False
```

`enc.sheet()` is the scene sheet the DM will read (PR-19):

```
Round 2 — Kara's turn.
Order: Kara (18), goblin-1 (18), Dorn (12), goblin-2 (12)
  Kara: AC 14, 9/9 HP
  ✝ goblin-1 (Goblin): AC 15, 0/7 HP, dead
  Dorn: AC 18, 8/13 HP
  goblin-2 (Goblin): AC 15, 7/7 HP, prone
```

Rules applied, each pinned by a test in `tests/test_engine_combat.py`:

- **Initiative** is d20 + DEX, highest first; ties by DEX modifier, then name.
- **Temporary HP** absorb damage first and do not stack (the higher value stands).
- **Resistance** halves damage rounded down, **immunity** zeroes it,
  **vulnerability** doubles it, matched by damage type. The "from nonmagical
  attacks" qualifier is not modelled: every attack here counts as nonmagical.
- **A monster at 0 HP is dead.** A character drops unconscious and starts
  death saves — unless the damage left over past 0 is at least their maximum
  HP, which kills outright.
- **Death saves:** 10+ succeeds, a 20 restores 1 HP, a 1 is two failures;
  three successes stabilise, three failures kill. Damage while at 0 HP is a
  failed save, two on a critical. A downed character's save is rolled
  automatically as their turn comes round, and their turn is skipped.
- **Conditions:** incapacitated / paralyzed / petrified / stunned /
  unconscious cannot act. Blinded, poisoned, prone, restrained and frightened
  attackers have disadvantage. Attacks against a blinded, restrained,
  paralyzed, stunned, unconscious or petrified target have advantage; a melee
  hit on a paralyzed or unconscious target is a critical. A prone target gives
  melee advantage and ranged disadvantage. Invisible attackers have
  advantage; invisible targets impose disadvantage. Poisoned and frightened
  give disadvantage on checks; restrained gives disadvantage on DEX saves;
  paralyzed, stunned, unconscious and petrified fail STR and DEX saves
  outright (`save_mode` returns `None`). Condition immunities are honoured.
- **Attacks** happen on the attacker's turn (`enforce_turn=False` for a
  reaction or a DM override), against a living target, by a creature that
  can act; a natural 1 misses, a natural 20 crits, otherwise total ≥ AC.
  Every damage component is rolled and applied in turn, dice doubled on a crit.
- **`move_first`** puts the combatant who opened the fight at the top of the
  order — a declared attack on an unsuspecting creature lands first.
- **The fight ends** in `victory` when every monster is dead, `defeat` when
  every character is dead or at 0 HP, or by `end_encounter(enc, "fled")`.
- **A short rest** spends hit dice for d(hit die) + CON each (never below 0);
  **a long rest** restores all HP, half the hit dice (at least one), and
  clears temporary HP.

A scripted goblin fight (fighter and rogue against two goblins) runs to
completion under `random.Random(7)` and replays identically; twenty seeds are
checked to end lawfully.

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
- Distance and movement. "Melee" and "ranged" come from the weapon; a prone
  target is assumed within reach of a melee attacker.
- Reactions, opportunity attacks, bonus actions, and monsters' multiattack
  as a rule (the text is kept; the DM calls `attack` once per attack).
- Exhaustion, and the "nonmagical" qualifier on resistances.
