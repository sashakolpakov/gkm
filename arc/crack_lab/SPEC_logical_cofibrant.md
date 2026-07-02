# Pre-registration: logical-cell tracking + cofibrant object identity + LLM-supplied, data-verified move-rule

**Architect step (pre-registered before implementation).** Reconcile log entry to
follow in FINDINGS.md. Substrates are new SIBLINGS in `crack_lab/`, not rewrites.

## Problem (from reconnaissance, verified 2026-06-19)

`dynamics_learn.py` learns wa30's transition rule by a pixel min-corner anchor +
exact-cell shape match and tops out at **49% held-out fidelity**. Probes
(`probe_logical{,2,3}.py`) show why:

- wa30 is a **16×16 logical grid** rendered 4× into 64×64 px (pitch 4, phase (0,3)).
- The avatar (colour 14, 12 px) **is not a box and rotates** with facing: 4×3 for
  up/down, 3×4 for left/right (`rigid_shape=False`). Its centroid and min-corner
  therefore jitter by sub-cell amounts that a pixel model reads as motion noise.
- Object-id is the whole lever. Mapping a component to the logical cell holding the
  **majority of its pixels** (majblock), at the detected phase, gives:
  pixel 49% → centroid 54% → **majblock 83% joint (avatar 95%, box 86%)**.

## Claims to test (with falsification criteria)

- **C1 (logical cells + cofibrant object-id beat pixels).** majblock object-id on a
  logical grid yields strictly higher held-out fidelity than the pixel baseline on
  wa30. *Falsified if* majblock ≤ pixel, or if the gain is an artifact (see controls).
- **C2 (cofibrant avatar).** The avatar can be identified and tracked by its logical
  cell + action-response ROLE, invariant to its pixel shape, recovering a single
  action→unit-vector map. *Falsified if* no consistent per-action unit vector exists
  at logical resolution, or tracking needs the pixel shape.
- **C3 (structured rule → near-exact).** Adding wall-blocking + push semantics on top
  of the action→vector map raises avatar fidelity toward ~100% and joint ≥ 90% on
  wa30. *Falsified if* the structured rule does not beat the raw most-common-delta
  model, or joint stays < 90%.
- **C4 (LLM supplies a VERIFIED model).** A local LLM (ollama), given only the
  symbolic displacement summary + a few logical transitions, proposes the move-rule
  (action vectors + push/blocking); the fidelity gate verifies it on held-out
  logical transitions and accepts iff it clears threshold and beats the data-only
  baseline. *Falsified if* the LLM rule cannot be verified, OR (honest negative) if
  it never beats the baseline — report either way.
- **C5 (generality, not wa30-overfit).** The same logical-cell + cofibrant-avatar
  read-off runs on ls20/g50t/tr87 and recovers each game's (possibly different,
  non-standard) action→vector map at logical resolution. Report per-game fidelity;
  honest null where a game isn't grid-navigational.

## Controls / ablations (must run, report even if unflattering)

1. **Pixel baseline** (current 49%) vs logical — already the comparison axis.
2. **Object-id ablation**: centroid vs minsnap vs majblock at the same pitch/phase.
3. **Constant predictor floor**: "everything stays" fidelity — guards against the
   box-rarely-moves inflation making any model look good.
4. **Train/test split is by time** (70/30 on a random walk), no leakage; report n.
5. **Data-only rule vs LLM rule**: the LLM never replaces the gate; if it loses to
   the data-only most-common-delta model, say so.
6. **Avatar-only vs box-only vs joint** breakdown, so we see where error lives.

## "Cofibrant definition" — what it means here (categorical framing)

The renderer is a map `render : LogicalState → Pixels` that is not injective on
object *shape* (the avatar's 12 px realise as 4×3 or 3×4 for the same logical cell).
A **cofibrant object definition** is one that **lifts along `render`**: given the
noisy pixels, it recovers the logical object up to the renderer's transformations,
because it is defined by data invariant under those transformations — the object's
logical cell + colour-set + action-response role, NOT its pixel shape. Operationally
that is the majblock cell + the verified action→vector role. "I don't care how it
transforms" = the definition quotients out the rendering fibre. The verified move-
rule is then a model on the cofibrant (logical) objects, where dynamics is clean.

## Deliverables (siblings)

- `logical_grid.py` — pitch/phase detection, majblock object-id, logical grid view.
- `cofibrant.py` — cofibrant avatar/object identity + tracking by role/continuity.
- `dynamics_model.py` — structured logical move-rule (vectors + blocking + push) +
  `fidelity()` gate + data-only learner.
- `llm_dynamics.py` — local-LLM rule proposer (reuses `llm_binder.ollama_json`),
  verified by the gate.
- `run_logical_dynamics.py` — runner: fidelity table, ASCII logical grid + avatar
  trajectory, controls; honest report.
- FINDINGS.md reconcile entry + `experiments/` report; memory update.
