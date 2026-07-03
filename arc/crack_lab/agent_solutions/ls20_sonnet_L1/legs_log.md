# Leg-library debrief log

Recurring composition patterns and repeated novelty.

---

## 2026-07-03 — debrief after clearing ls20 level 1

### Pattern observed: sequence-replay player

Every level player so far follows an identical two-line shape:

```python
_LK = [1, 2, ...]          # hardcoded action sequence
def play_level_K(env):
    play_sequence(env, _LK)
```

`play_sequence` (in legs.py) already captures the replay primitive.  
The *player* itself is nothing but a sequence constant bound to a function.

### Candidate higher-order leg: `sequence_player`

```python
def sequence_player(actions):
    """Return a play_level_K function that replays a fixed action sequence."""
    def _play(env):
        play_sequence(env, actions)
    return _play
```

Usage would collapse each level player to a single assignment:

```python
play_level_1 = sequence_player([1,1,1,1,1,1,2,1,2,1,...])
```

**When to promote:** once a second level player with a hardcoded sequence appears, add `sequence_player` to legs.py and replace both player definitions. Until then, the current thin `play_sequence` call is sufficient — no premature abstraction.
