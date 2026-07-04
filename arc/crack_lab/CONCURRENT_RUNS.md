# CONCURRENT RUNS ON THIS MACHINE — DO NOT KILL EACH OTHER'S PROCESSES

The user is deliberately running TWO independent cracking sessions at the same time.
Both are legitimate. If you see the other one's processes, LEAVE THEM ALONE.

1. **wa30** — proposer: `codex exec --model gpt-5.4` (+ its `python gkm_legs.py --game=wa30`
   orchestrator), workspace `.../e3e00be1-*/scratchpad/gkm_legs_ws_wa30`. Owned by the
   session in the other terminal.
2. **sp80** — proposer: `claude -p ... sp80 ... --model sonnet` (+ its
   `python -u sp80_launcher.py` orchestrator), workspace
   `.../537f0152-*/scratchpad/gkm_legs_ws_sp80`. Owned by the sp80 session.

Rules for BOTH sessions and their subagents:
- NEVER `pkill`/`killall` by pattern (`python`, `claude`, `codex`, `gkm`...). Kill only
  exact PIDs of processes you yourself spawned.
- Write only inside your OWN workspace/scratchpad. `crack_lab` is shared read-mostly.
- A foreign-looking proposer process is EXPECTED — it is the other run, not a stray.
