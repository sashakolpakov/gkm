# Reuse accounting without a prescribed form of glue

The accounting is separate from proposal and admission. It accepts the solver's
source snapshots as written. It does not add an admission threshold, refactor a
candidate, or require calls to have a particular spelling. No LLM is invoked by
either accounting component.

`executable_reuse.py` supplies two complementary measurements:

- `analyse_reuse(previous, current, entrypoints)` parses Python and computes a
  finite points-to fixed point. It follows new wrappers, import aliases, local
  aliases, callback parameters, closures, returned functions, dispatch tables and
  methods. Each retained definition has a normalized-AST hash and a possible call
  path. Literal definitions can also be matched after moving into a new scope;
  their preceding binding is recorded. Shared mutation or other unresolved bindings are labelled explicitly;
  actual replay observations resolve which target was selected.
- `ReplayCalls` observes actual Python call events inside an existing independent
  replay. A table lookup or callback needs no source-level guess: the interpreter
  has already selected the function. The observer matches its code object against
  the supplied source, including exception-handling bytecode, and records its
  identity and edges. It also records generated functions in the solver's module
  globals. Compiled body matches are reported separately for anonymous functions
  and other bodies already expressible by the preceding source. Solver arguments and private game/framework code are not recorded.

The static domain contains only source-defined symbols, literal values, allocation
sites and finite external-member abstractions. Unknown computed bindings remain
explicit. They are not classified as non-reuse. Code-only replay observation can
resolve them without removing the freedom that made the glue useful.

## Source accounting

```python
from executable_reuse import analyse_reuse

account = analyse_reuse(
    previous_sources,                 # {relative_filename: bytes}
    current_sources,
    ["players.py:play_level_3"],
)
```

All supplied retained Python definitions are eligible by default, including helpers
outside `legs.py`. Matching uses normalized-AST content, so an unchanged nested helper rebound
inside new glue is also recognized. The result binds both source bundles by SHA-256 and distinguishes
retained entrypoints, direct targets, indirect targets, and unresolved sites.
Duplicate definitions obey their actual binding order; a captured older alias can
still refer to the old definition after a later rebinding.

`audit_marginal_literal_reuse.py` uses this engine for GKM source pairs, OPINE's
released winning planners, and baseline1's available Python winning commands.
The old direct-call fields remain for comparison with previously published tables.
`generated/reuse_accounting.{json,md,tex}` reports coverage explicitly. Cached
comparator rows are not assigned new call paths without their raw source, and a
memory-only artifact is not treated as an executable entrypoint.

## Actual calls during an independent replay

Place the observer in the process that executes the solver, around the existing
replay operation. The verifier still applies its ordinary replay and success checks.

```python
from executable_reuse import ReplayCalls, attach_replay

with ReplayCalls(
    previous_sources,
    current_sources,
    runtime_paths=actual_compiler_filenames,
    entrypoints=["players.py:play_level_3"],
    entrypoint_objects=[actual_player],  # useful for a decorated/aliased entry
) as observer:
    existing_verified_replay()

account = attach_replay(account, observer.report())
```

This is instrumentation around the replay, not a change to the solver source.
Runtime filenames and code fingerprints bind functions to the declared snapshot.
Inherited compiler flags are recorded. An unseen entrypoint, mismatched source,
or collection error prevents attaching an empty trace as evidence of non-reuse;
it does not reject or alter the candidate.

Root scope follows the executing entrypoint, copied asynchronous contexts and
`threading.Thread` descendants started from that scope. Unrelated threads are
excluded. Existing profile hooks are preserved. On exit the observer is disabled;
surviving workers restore their prior hook on their next profile event. Call events
include generator/coroutine resumptions, so they are labelled events rather than
claimed as counts of function-object construction. A subprocess runs its own
observer; a parent trace is not credited with a child's unseen calls.

A command-line wrapper is also available:

```bash
python -B arc/record_replay_calls.py \
  --previous /path/to/previous-source \
  --current /path/to/current-source \
  --entrypoint players.py:play_level_3 \
  --output /tmp/replay-calls.json \
  /path/to/existing-trusted-replay-driver.py
```

`--runtime-map` supplies a JSON map when the verifier uses virtual compiler filenames.
The wrapper does not create a replay certificate: game success remains the existing
driver's responsibility. Source-bound observer reports can be imported using
`audit_marginal_literal_reuse.py --replay-observations INDEX.json`, whose observations
name the system, game, completed level, report path and SHA-256. Source/entrypoint
bindings are checked before attachment.

The frozen GPT-5.6 campaign's retrospective totals remain source measurements.
The new observer has not retroactively turned those histories into runtime traces.
Its unit and command-line tests execute ordinary dispatch, callbacks, closures,
generated glue, branches and threads using small independent example programs.

## Relation to marginal complexity

The conditional AST novelty `M` still measures newly described source. New glue is
charged normally. A reference to an unchanged helper can be direct or can pass
through several new routines. Source-growth counters, static call paths and observed
call events are separate fields; none changes the first-success/replay admission rule.
