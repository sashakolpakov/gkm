# Unverified frontier brief: sk48 level 4

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- I’ll stay inside the workspace and begin with the indexed interfaces, then reproduce any inherited observations before probing level 3. I’ll keep experiments compact and ensure `play_level_3` is composition-only.
- The baseline is clean: levels 1–2 complete in 98 moves, so the missing dispatch is the only immediate blocker. I’m now reading only the indexed player/leg bodies and the harness-facing setup needed to obtain a level-3 clone.
- The first probe identifies the avatar experimentally: action 1 moves the left-hand 6×6 object up one lane, action 2 is blocked at the lower boundary, and actions 3/4 retract/extend its horizontal tether; action 6 is contextually inert here. The level’s dense progress signal is the ordered train attached to that tether: field tokens are vertically arranged `14,9,8,12`, while the bottom request reads `8,12,9,14`. I’m probing token pickup and lane transitions on clones now.
- Contact alone only overlaps the token and lights a small request indicator; retracting immediately reverses that state, so it is not yet a pickup. That makes action 6 the likely context-sensitive interaction. I’m testing it specifically while the tether crosses the requested token, rather than treating its inert starting-state result as definitive.
- The level introduces a constraint absent from earlier layouts: an identically aligned tether tip does not attach to the apparent first bottom-request token. I’m checking all four lane tokens under the same controlled reach/retract experiment; whichever one actually moves will disambiguate which display is the goal ordering and avoid planning from a visual assumption.
- The controlled reach test is symmetric across all four tokens, so the ordering assumption is not the issue. The remaining documented action surface includes coordinate form `step(6, x, y)`; a bare 6 was an incomplete probe. I’m now clicking only observed object centers (field tokens, request tokens, avatar), then checking exact frame deltas on clones.
- A key structural fact emerged: the lower band is not a request display—it contains a second 6×6 agent behind an uncrossable wall, plus its own horizontal token row. Its changes appear on a periodic turn cadence while the upper avatar acts. I’m isolating that agent with top-avatar blocked moves to learn its autonomous cycle and determine how the two reachable regions cooperate.
- Simple reach/retract lengths through the right wall still do not move a token, so I’m switching from hand-enumeration to a small observation-keyed clone search. The goal predicate is deliberately dense and physical—“the field 8’s bounding box moved,” not “pixels changed”—with the lower display/timer excluded from state keys.
- The dense probe confirms the core new affordance: threading the tether through a token and then moving vertically pushes that token—and any token immediately below it—by one lane. That is a reusable staging operation, not a pickup. I’m now running a bounded search over the physical upper region for the first real reward, using only actions 1–4 and deduplicating away the timer display.

## Preserved local probes

- `probe_level3.py` (5753 bytes)

Run or inspect the smallest relevant probe before writing another one.
