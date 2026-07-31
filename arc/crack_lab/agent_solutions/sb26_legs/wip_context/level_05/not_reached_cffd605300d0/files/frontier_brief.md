# Unverified frontier brief: sb26 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and begin with the indexed API surface, then reproduce any inherited observations before probing level 5. I’ll keep clone experiments compact and bounded, and put all actual mechanics into reusable legs.
- The baseline is clean: levels 1–4 complete in 54 moves, so level 5 is the only missing dispatch. I’m now narrowing the existing diagram leg’s assumptions and the harness’s clone surface before taking any exploratory actions.
- Level 5 is a different configuration, not a direct fit for the existing diagram leg: it shows nine top symbols, two central structures with eight marked positions total, and eight selectable bottom choices. A bottom click changes only the selection outline; I’m testing whether this is a single-choice inference puzzle before considering any broader search.
- Painting is confirmed: each bottom token can be placed into any of the eight central dots, so the dense progress measure is “filled slots / 8”; submission remains the sparse completion check. The two central outlines act as a 5-slot and 3-slot diagram. I’m comparing freshly reproduced level 1–4 layouts to identify the ordering rule that level 5 extends, rather than treating 8! arrangements as opaque guesses.
- The reproduced earlier solutions reveal the stable mechanic: the top row defines a traversal result, while the central dots are filled from a bottom multiset; level 5 is the first case with repeated symbols and an extra traversal step. I’m adding a separate general leg for repeated palette tokens, with duplicate-aware prefix search capped at the exact 10,080 distinct arrangements and slot filling as its dense progress invariant.

## Preserved local probes

- `probe_l5.py` (3146 bytes)
- `probe_level2.py` (4310 bytes)
- `probe_progression.py` (757 bytes)
- `probe_rule.py` (2739 bytes)
- `search_level2.py` (1722 bytes)

Run or inspect the smallest relevant probe before writing another one.
