# GKM legs cracking — quick recovery notes

Runs die to transient **API drops** ("Connection closed mid-response"), **session
limits** ("You've hit your session limit · resets …"), and **credit-outs**. All are
recoverable: state is checkpointed per level, so just relaunch — it resumes.

## What to run (per game)
Orchestrator: `arc/crack_lab/gkm_legs.py` (run from `arc/crack_lab/`).
```
python gkm_legs.py --game=<GAME> --model=<MODEL> --proposer=claude --max-level=<N> --minutes=45
```
- GAME ∈ `ls20 wa30 sp80` (also g50t, tr87). MODEL ∈ `sonnet` | `claude-fable-5` | `opus`.
- Resumes automatically from checkpoint + workspace files — safe to re-run after any crash.
- `--max-level=N` runs L1..LN sequentially; stops cleanly if a level isn't reached.
- Background it and let the completion notification fire; don't foreground-sleep.

## Key paths
- Workspaces (persist across runs; the "leg library"):
  `/private/tmp/claude-501/-Users-sasha-gkm/e3e00be1-.../scratchpad/gkm_legs_ws_<GAME>/`
  (hardcoded `SCRATCH` in gkm_legs.py — NOT this session's scratchpad.)
  Files: `legs.py` (shared skills), `players.py` (`play_level_K`), `solve.py`,
  `checkpoint.json` (levels reached + marginal-C), `legs_log.md` (debrief),
  `proposer_last.log` (last proposer stdout — READ THIS to see WHY it stopped).
- Run logs (this session cwd = `arc/crack_lab/`): `run_<GAME>*.log`.
- Priors (multi-avatar) live in `gkm_arena.py::PRECONCEPTIONS`; leg task prompt in
  `gkm_legs.py::_propose_task`. Neutral raw-substrate context: `gkm_solve_agent.discovered_context`.
- Prior solutions parked out of reach at `arc/agent_solutions_HIDDEN_during_run/`
  (moved aside for clean-scratch integrity; restore to `arc/crack_lab/agent_solutions/` when done).

## Diagnose a stop (which failure was it?)
```
cat run_<GAME>*.log                                   # orchestrator verdict
tail -20 <WS>/gkm_legs_ws_<GAME>/proposer_last.log    # API/session/credit marker
claude -p "reply OK" --model sonnet --output-format text   # is the API back?
TZ="Europe/Zurich" date                               # vs "resets 6:20am" session window
```
Credit/session markers that abort the run (gkm_legs `_CREDIT_OUT_MARKERS`):
`out of usage credits`, `credit balance`, `session limit`, `rate limit`, `insufficient`, `quota`.
API drop = `Connection closed mid-response` (transient — just relaunch).

## Check progress mid-run
```
WS=/private/tmp/claude-501/-Users-sasha-gkm/e3e00be1-.../scratchpad/gkm_legs_ws_<GAME>
python -c "import json;print(json.load(open('$WS/checkpoint.json'))['reached'])"   # level reached
grep -cE 'def play_level' $WS/players.py ; grep -cE '^def ' $WS/legs.py            # size
find $WS -maxdepth 1 -type f -mmin -5 -printf '%TH:%TM %f\n'                        # heartbeat
tail -20 $WS/legs_log.md                                                            # discovery notes
```
Which claude proposer is alive + which game:
```
for pid in $(pgrep -f "claude -p"); do echo $pid $(lsof -a -p $pid -d cwd -Fn 2>/dev/null|grep ^n|sed 's/.*gkm_legs_ws_//'); done
```

## Stop one game (child first, then orchestrator)
```
kill <claude_pid>            # the claude -p proposer for that ws
kill <orch_pids>             # the two gkm_legs.py --game=<GAME> pids (zsh wrapper + python)
# + TaskStop the background task id if launched via run_in_background
```

## Clean-scratch (grow legs anew)
```
mv arc/crack_lab/agent_solutions arc/agent_solutions_HIDDEN_during_run   # hide prior solutions
rm -rf $SCRATCH/gkm_legs_ws_<GAME>                                        # wipe the game's library
# ALSO delete stale sibling snapshots (agent can Bash-read them): keep only active ws
ls -d $SCRATCH/gkm_legs_ws_* | grep -vE 'gkm_legs_ws_(ls20|wa30|sp80)$' | xargs rm -rf
```

## Status (2026-07-04)
- ls20: L1–L4 ALL validated (marginal_C 70→2→2→0). DONE on Sonnet.
- sp80: L1 validated (single-avatar bar). L2 in progress under multi-avatar priors.
- wa30: L1 validated; parked. Resume with `--model=opus` (Sonnet stalled on L2 delivery).
- Multi-avatar priors are now the default (replaced single-avatar "one object is YOU").
- Model capability is per-game: Sonnet ok for ls20; harder games need Fable/Opus.
