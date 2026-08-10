# Phase 0 verification report

**Status:** complete
**Verified:** 2026-07-30
**Scope:** standalone protocol, calibration shell, provenance, source
projection, exact replay, dependency prohibition, and write isolation.

Phase 0 is an executable contract proof. It is not evidence that physical arm
kinematics, collision, contact, grasp, object dynamics, scored levels, or the
browser test room are complete.

## Normative corrections applied

The user’s repository-specific direction has priority over stale references:

- “ARC-style” means the observable experimental shape only. Runtime code does
  not import, install, wrap, subclass, register with, or emulate an ARC API.
- The standalone action set is exactly `(1, 2, 3, 4, 5, 6)`. `ACTION6` closes
  the gripper. Any earlier `ACTION7` wording is superseded.
- `RoboArmEnv` is the project-owned environment facade. No ARC adapter, loader,
  metadata tree, or external game base class was created.

These are enforced by executable dependency, import, action, and factory tests.

## Delivered evidence

| Requirement | Evidence |
|---|---|
| Isolated runtime | `.venv`, `pyproject.toml`, `requirements.lock`, and `references/runtime_manifest.json` |
| Standalone contract | `src/roboarm_game/protocol.py`, `interface.py`, and `environment.py` |
| Calibration shell | Direct deterministic `64×64` `uint8` rendering in `render.py`; command state only |
| Six documented actions | Exact tuple `(1, 2, 3, 4, 5, 6)` and visible transitions/rejection telemetry |
| Reset and frame ownership | Fresh-reset equivalence and defensive-copy tests |
| Exact cloning/replay | Identical clone sequences, independent divergence, fresh replay, and fixed frame digests |
| No ARC binding | AST import scan, declared-dependency audit, and dynamic-import prohibition |
| Public/private boundary | Exact three-file allowlist plus traversal, private-file, destination-escape, and symlink attacks |
| Hardware provenance | Two pinned official revisions, preserved Xacro, SHA-256, and exact XML-to-JSON joint extraction |
| Write boundary | Audited test runner rejects mutation outside `roboarm`; all 73 observed final-run writes were under `artifacts/` |
| Buildability | Pure-Python wheel built with the package README and public/runtime modules present |

## Reproducible commands and observed results

### Runtime

```bash
python3 -m venv .venv
mkdir -p artifacts/tmp
PIP_CACHE_DIR="$PWD/artifacts/pip-cache" TMPDIR="$PWD/artifacts/tmp" \
  .venv/bin/python -m pip install -e '.[test]'
```

Observed local runtime:

```text
CPython 3.14.6
NumPy 2.4.4
pytest 9.0.3
Darwin 25.4.0 arm64
```

### Full Phase-0 suite

```bash
.venv/bin/python tools/run_phase0.py
```

Observed:

```text
........................................
40 passed in 0.69s
```

The runner starts pytest through `tools/audited_pytest.py`, with bytecode,
pytest cache, `TMPDIR`, and XDG cache redirected below `artifacts/`.
`artifacts/write-audit.json` reported:

```text
pytest_exit_code: 0
observed_write_events: 73
events_by_top_level: {"artifacts": 73}
outside_writes_blocked: 0
```

The auditor has an adversarial unit test that submits a parent `arc/` path and
proves it is rejected without creating that path. The source guard separately
tests traversal and symlink attacks.

### Calibration golden replay

For seed `0` and actions `(2, 1, 4, 3, 6, 5)`, the reset frame followed by six
step frames has SHA-256 digests:

```text
0d20eb1fed93e9cfb5890c87f8b4b49545d670a8cefce92ec6581a0c6e2224a3
5950a7372341ba32b6ce9c8d8cfbe2b616ca0ab9e9b0f86878ba22ed19155bb7
2fb6885db674dc3d29f7b0247c7c475b5f53b10ba39d44f770332e0ab1e2fca4
3e1889601220bdcfc277b7a0aa85a8552c76de63bfcbeca421a156d7db5d8e0c
cd8ea48cc6ed0d34db4a443cf42a4f1d79b914442065cf6dfbb5cc3927429ca4
cb1a13d5f7c5f5387c9519c7bee4236671ee7642ea794afb66f48a3827841745
224a863d5faf14b3954420dc990deb009b1f9ca84bef14974ab5a2102e2b5d51
```

The corresponding test performs a fresh replay and checks all bytes.

### Public solver view

```bash
.venv/bin/python -c \
  'from pathlib import Path; from roboarm_game.source_guard import materialize_public_sources; r=Path.cwd(); print(*(p.relative_to(r) for p in materialize_public_sources(r/"artifacts/public-source-projection", write_root=r/"artifacts")), sep="\n")'
```

Observed exact projection:

```text
artifacts/public-source-projection/README.md
artifacts/public-source-projection/interface.py
artifacts/public-source-projection/protocol.py
```

`environment.py`, `render.py`, `state.py`, fixtures, tests, and hidden future
mechanics are absent. This is the Phase-0 projection proof; a separate-process
campaign boundary remains a later release gate.

### Hardware source verification

```bash
curl -fsSL \
  https://raw.githubusercontent.com/waveshareteam/roarm_ws/40dbd84b553695212fab713e8465f817ba95454d/src/roarm_main/roarm_description/urdf/roarm_m2/roarm_m2.xacro \
  | diff -u references/hardware/roarm_m2.xacro -
```

Observed exit status: `0`, with no diff.

```bash
shasum -a 256 references/hardware/roarm_m2.xacro
```

Observed:

```text
4144e6f20919554755c7dd515f7a236a9aa128fc5507bede3b66cba4ee751c2c
```

The test suite parses every joint and compares its type, parent, child, origin,
axis, and limits lexically against `references/hardware/transforms.json`.

### Package build

```bash
PIP_CACHE_DIR="$PWD/artifacts/pip-cache" TMPDIR="$PWD/artifacts/tmp" \
  .venv/bin/python -m pip wheel --no-deps --wheel-dir artifacts/wheels .
```

Observed final wheel:

```text
gkm_roboarm-0.0.1-py3-none-any.whl
SHA-256 3ac440f77c27f2997d5c96604bd0afce8d6c8ac764b9fcde8c3f0e290586b623
```

Archive inspection confirmed that `roboarm_game/README.md`, the public
contract, the environment facade, renderer, state, and source guard are
packaged.

## Write-scope statement

All project source and persistent evidence created for this phase is below
`/Users/sasha/gkm/roboarm`. The parent repository already had a very large,
concurrently changing dirty state, so an undifferentiated root `git status`
cannot attribute writes. Instead, the evidence is:

1. every applied source patch targeted `roboarm/`;
2. configured cache, build, wheel, and test destinations are under
   `roboarm/artifacts/`;
3. the final test process enforced and logged its write boundary;
4. tests reject escaping paths and non-venv symlinks; and
5. no implementation file, registration entry, or generated project artifact
   was added under `arc/` or another sibling.

The standard `.venv/bin` interpreter symlinks point to the system Python; those
are references, not writes through the links.

## Explicitly deferred

Phase 0 stops before:

- exact forward/inverse kinematics;
- swept collision and reachability;
- contact, pushing, grasp, attachment, gravity, and support;
- scored Levels 1–10 and the oracle;
- production process isolation for a solver;
- realistic browser camera rendering and the interactive live test room; and
- any physical-arm controller.

The newly added browser-live acceptance gate must be driven by those operational
mechanics. A canned animation or photorealistic sprite sequence will not satisfy
it.
