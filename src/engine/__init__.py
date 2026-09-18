"""The 5e engine. Pure Python, no model, no I/O.

Everything mechanical lives here — character sheets, ability checks, saving
throws, attack rolls — so that the Dungeon Master never has to produce a
number. The model narrates and picks tools; this package decides outcomes.

Nothing in this package may import from `src.agents`, `src.graph`, or
`langchain`. `tests/test_engine_*.py` pins that.
"""
