# Bongard Concept-Induction Experiments

This subject studies reusable abstraction in Bongard-style concept induction. The
full subject guide is [`BONGARD.md`](BONGARD.md); this page also documents the newer
typed semantic-cone track.

The controlled symbolic experiments and the rendered Bongard-LOGO adapter have
different evidential scope. The former measure whether a shared predicate macro is
selected over duplicated rule bodies when primitive atoms are supplied. They do not
show discovery of those atoms from pixels. The semantic-cone track below adds typed
factorization and explicit witness checks; unrestricted predicate search remains a
separate control path.

The unrestricted `predicates.py` crack loop remains the control path: a
proposer may write arbitrary `p_*(panel)` measurements and the existing
verifier/MDL selector composes and admits them. The semantic-pure path is
stricter: a proposer supplies typed semantic cones, and the harness
type-checks, validates semantic witness coverage, verifies, prices, and
archives them separately.

## Architecture

```text
panels
  -> proposer
  -> typed cone proposals
  -> leg registry
  -> mechanical compiler
  -> semantic admissibility gate
  -> support / LOO / naturality / contrast diagnostics
  -> conditional MDL selection
  -> artifact
```

The proposer is "cofibered" in the operational sense: for each problem object
`p`, it proposes candidates in the fiber of semantic cones over `p`. In
semantic-pure mode it does not write final image-processing code. New reusable
legs are requested as typed missing arrows and are admitted only after
separate implementation, contract checks, replay, and pricing.

## Current Entry Points

- `crack_lab/run_semantic_cone.py` - semantic-cone experiment runner.
- `crack_lab/cofibered_proposer.py` - LLM-backed cone proposal interface.
- `crack_lab/semantic_ir.py` - JSON-serializable typed IR.
- `crack_lab/semantic_compiler.py` - factorization-enforcing compiler.
- `crack_lab/semantic_verifier.py` - support and leave-one-out verifier.
- `crack_lab/semantic_legs.py` - initial typed visual leg registry.

Unit tests may use static fixture proposals. Experiments should use the
cofibered LLM proposer and should report `NO_PROPOSALS` or `MISSING_LEG` rather
than silently falling back to unrestricted predicate search. Unrestricted
results remain valid, but they must be labeled as `UNRESTRICTED`.

## Minimal Smoke

```bash
python3 -m py_compile bongard/crack_lab/*.py
python3 -m pytest bongard/crack_lab
```

An LLM run requires `ANTHROPIC_API_KEY.env.local` or `ANTHROPIC_API_KEY`:

```bash
python3 -u bongard/crack_lab/run_semantic_cone.py \
  --dataset-dir downloads/Bongard-LOGO \
  --source both --limit 5 --proposer anthropic --model sonnet
```
