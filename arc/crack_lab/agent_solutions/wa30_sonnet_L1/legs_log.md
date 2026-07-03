# Leg-library debrief log

Recurring composition patterns and repeated novelty.

---

## Debrief: wa30 level 1 (sokoban_deliver)

### Legs added / refined
- `make_step_budget(env, max_real)` — extracted from the body of `sokoban_deliver`.  
  Returns `(step_fn, budget_ok_fn)`.  `step_fn` guards terminal and counts; `budget_ok_fn`
  checks remaining budget.  Any future leg that needs a step cap should call this instead of
  inlining the counter.
- `sokoban_deliver(env, ...)` — the primary level-1 leg.  Navigate to carrier, attach, BFS-carry
  to slot, release; retry loop with stuck counter.

### Recurring composition pattern observed

Every level player so far is:

```
def play_level_K(env):
    leg_A(env)          # one or a few legs from legs.py
```

Inside each leg the same skeleton recurs:

```
step, budget_ok = make_step_budget(env, max_real)
stuck = 0
while not env.terminal() and budget_ok():
    f = env.frame()
    # 1. scan world (find objects, slots, walls)
    # 2. plan (pick target pair, BFS paths)
    # 3. execute (walk, act, walk, act)
    # 4. update stuck counter; break or jiggle if stuck > threshold
```

### Candidate higher-order leg

`loop_with_budget(env, max_real, plan_fn, exec_fn, stuck_limit=3)`

Would run the scan→plan→execute→stuck-detect cycle parameterised by caller-supplied
`plan_fn(frame, av, world_state) → plan | None` and `exec_fn(plan, step_fn) → None`.
Reduces each new level leg to providing only the domain-specific planner and executor,
with the retry harness shared once.
