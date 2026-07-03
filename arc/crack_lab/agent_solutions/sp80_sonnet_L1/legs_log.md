# Leg-library debrief log

Recurring composition patterns and repeated novelty.

---

## sp80 L1 debrief — sense_align_trigger

**Pattern observed in `play_level_1`:**

```
sense_fn(frame) → target
align_fn(env, target)
trigger_fn(env)
```

Read the world once, move the avatar/tool to a computed position, fire an
irreversible action.  The three legs are independent concerns and can be
swapped per-level without touching the others.

**Candidate higher-order leg (now in `legs.py`):**

```python
def sense_align_trigger(env, sense_fn, align_fn, trigger_fn):
    align_fn(env, sense_fn(env.frame()))
    trigger_fn(env)
```

**When it applies:** any level where the mechanic is "read the frame to find
the right configuration, reposition the avatar/tool, fire."  For sp80 this is
deflect-and-pour; it would also host shoot, drop, or activate mechanics that
follow the same sense → position → act spine.

**Contrast with prior-game patterns:**
- ls20: `play_sequence(env, fixed_list)` — pure replay, no sensing.
- wa30: `sokoban_deliver(env)` — one monolithic leg, sensing and navigation
  buried inside; BFS replaces an explicit align step.
- sp80: sensing and alignment are separated legs, making each individually
  testable and swappable.

**Upgrade path:** if a future level needs repeated pours (re-sense after each
pour), the spine becomes `while not done: sense_align_trigger(...)`, which
wraps the same three legs without rewriting them.
