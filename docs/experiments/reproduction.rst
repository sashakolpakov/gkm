Reproduction and Integrity Checks
=================================

Safe local verification
-----------------------

Create a local environment and install both runtime and documentation
dependencies:

.. code-block:: bash

   python3 -m venv .venv
   .venv/bin/python -m pip install -r bongard/requirements.txt
   .venv/bin/python -m pip install -r docs/requirements.txt

Run the repository tests without launching a campaign:

.. code-block:: bash

   .venv/bin/python -m pytest -q bongard/tests

The focused current-authority suite is:

.. code-block:: bash

   .venv/bin/python -m pytest -q \
       bongard/tests/test_panel_action_count_skeleton_graph_dev_command.py \
       bongard/tests/test_panel_action_count_skeleton_graph_passed_fit_protocol.py \
       bongard/tests/test_panel_action_count_skeleton_graph_inference_custody.py \
       bongard/tests/test_panel_action_count_skeleton_graph_calibration_prereg.py \
       bongard/tests/test_panel_action_count_skeleton_graph_calibration_runner.py \
       bongard/tests/test_panel_action_count_skeleton_graph_custody_incident.py \
       bongard/tests/test_panel_action_count_skeleton_graph_custody_incident_persistence.py \
       bongard/tests/test_panel_action_count_skeleton_graph_custody_gap.py \
       bongard/tests/test_panel_retired_pipeline_archive.py \
       bongard/tests/test_phase4_stale_code_retirement.py \
       bongard/tests/test_pipeline_registry.py

Run the synthetic representation experiment separately.  It constructs all
PNG bytes in memory, has no corpus/file input, and does not authorize an
official campaign:

.. code-block:: bash

   .venv/bin/python -m pytest -q \
       bongard/tests/test_panel_action_count_synthetic_identifiability.py \
       bongard/tests/test_panel_action_count_synthetic_pooled_control.py \
       bongard/tests/test_panel_action_count_ordered_path_inversion.py \
       bongard/tests/test_panel_action_count_synthetic_benchmark.py \
       bongard/tests/test_panel_action_count_connected_synthetic.py \
       bongard/tests/test_panel_action_count_connected_synthesizer.py \
       bongard/tests/test_panel_action_count_connected_benchmark.py

This paired test is intentionally slower than the record-level checks because
it renders the frozen carrier grid, extracts the 112-feature control, and runs
the ordered observer rather than replaying a cached score.

The frozen grid contains all 54 count pairs in five carrier families under
four D4 nuisances at fixed stroke width 2 and scale 1000: 648 training rows
(641 unique PNGs) and 432 held-out rows (425 unique PNGs).  The current paired result is 432/432
for the ordered observer and 189/432 for the pooled control; the two descriptive
family comparisons are 216/216 versus 115/216 on ``radial`` and 216/216 versus
74/216 on ``staggered``.  The target is partial and PNG-only: a panel is
resolved only when every connected ink component exactly matches one finite
line or annular sector. Candidate pairs are independently fitted; a post-fit
target-resolvability gate makes unresolved cases produce a set or GAP. This
safety behavior is therefore policy-enforced rather than independent evidence.

The current source-bound phase-0 replay record is
``sha256:4a2221b9b39a22ee0b60b2b3dd0ac5859c0b15de92e16a6c238cb6a5aaf774f3``.
The prior pre-issuer-bridge record is
``sha256:3e48a026d3b3bc3126c7a3ee8d424c52b3e1ad837043d1b5d5da61d39ff90bb0``;
the generated panels, predictions, and metrics are identical.

The connected phase is a separate 1,060-PNG fixed-catalog experiment.  Three
whole carrier families provide 636 training rows, while ``radial`` and
``staggered`` provide 424 held-out rows with zero complete-D4-orbit overlap.
Every carrier/nuisance cell contains all 54 single-shape count pairs and 52
two-shape layouts.  Its input gate checks that the exact PNG byte string was
previously issued by the process-local synthetic renderer.  This
SHA-256-keyed fixture-provenance check is not a digital signature,
external-file authentication, or official release authority.  The raw
observer is 424/424 target-set correct and exact-reconstructing, versus
106/424 for the fixed-32 pooled control.  Layout
comparisons are 216/216 versus 43/216 and 208/208 versus 63/208; family
comparisons are 212/212 versus 45/212 and 212/212 versus 61/212.  Its 212-pair
pooled-feature assignment uses every held-out occurrence exactly once and has
both endpoints correct in 212/212 raw pairs versus 17/212 control pairs.

Held-out exact-cover targets are constructed only after raw predictions, and
the raw API is exercised while the target oracle raises.  Raw observer and
target still share the complete primitive catalog, including held-family
geometry, so this reproduces fixed-catalog inversion rather than unseen-
grammar induction.  A target-free ablation removing held-family-only catalog
masks leaves 0/424 held-out PNGs with an exact cover.  All held-out targets are
singleton sets; the ambiguity
gate therefore has no ambiguous held-out case.  The connected record digest is
``sha256:0e5f711a6e686cfb9c2b1ff2cde1559a06f15542846794ce44cde57e6a368aff``;
it does not replace the earlier phase-0 digest.

Build the Sphinx documentation with warnings as errors:

.. code-block:: bash

   .venv/bin/python -m sphinx -W -b html docs docs/_build/html

The safe semantic/custody tests use only generated PNGs and temporary stores:

.. code-block:: bash

   python3 -m pytest -q \
     bongard/tests/test_panel_program_semantics.py \
     bongard/tests/test_panel_program_official_task.py \
     bongard/tests/test_object_bongard_release_gate.py

They prove complete 399-by-12 support evaluation, whole-formula ambiguity
semantics, typed and cause-specific zero-survivor records, freeze/commit
custody, and counterfeit rejection.  They are
not a recipe or authority for opening the official corpus.

Lean is not a prerequisite.  Python is the canonical evaluator, persistence,
and replay implementation.

The lifecycle registry is a descriptive report and an opt-in guard, not a
universal interception layer for direct modules.  A retained source file is
not evidence that new execution is authorized.

No live reproduction recipe
---------------------------

Do not invoke a corpus-facing command as a smoke test.  Exposure is consuming,
and a failed or abandoned attempt is not rerolled.  The previous coverage-drill
and campaign command lines have been removed from this page because their
cohorts and stores are historical evidence, not reusable launch parameters.

The latest skeleton calibration is also not runnable.  Its action-program
authority was exposed before the prediction barrier, the incident tombstone
was durably published, and the branch ended in a typed custody GAP.  A receipt,
record digest, old command, or fresh output directory cannot re-authorize it.

Official-test PNGs and labels remain unopened under that campaign.  The HD
action-program authority does not: all 4,400 HD programs are conservatively
pre-exposed.  Any verification statement must preserve this distinction.

What cold verification proves
-----------------------------

The current tests exercise four separate claims:

* Development replay binds the fixed observer configuration, feature bank,
  model, predictions, metrics, and source identities.
* Inference-custody replay joins archived canonical raw outputs with zero pixel
  or model calls.  Fresh verification, tested separately, re-extracts and
  re-infers rather than trusting serialized outputs.
* Incident persistence verifies the exact predecessor, one direct successor,
  tombstone claim, and persistence receipt through a fresh process-local
  capability.
* Custody-GAP replay verifies metadata only.  It cannot create a support
  matrix, semantic version-space result, predicate, rank, or query decision.

A green replay therefore establishes integrity and chronology for its stated
inputs.  It does not prove that an empirical observer interpreted a PNG
correctly, and it does not turn a custody failure into semantic evidence.

The synthetic paired experiment is not a replay or a re-entry mechanism.  Its
green result establishes only an easy, disconnected, exact-component
mechanistic contrast.  It neither solves general connected action boundaries
nor supplies a confidence or generalization claim.  It cannot change the
terminal custody state or authorize official pixels, programs, labels,
calibration, target inference, queries, or benchmark promotion.

Likewise, the connected phase establishes only exact inversion under its
known fixed catalog.  It does not establish official transfer, unseen-catalog
induction, or broad visual generalization, and it grants no campaign authority.

Historical artifacts
--------------------

The loop/relational pilot, five-task TRAIN campaign, panel-soft zero-coverage
drill, and atomic attempts remain immutable historical records.  Their old
command lines and source identities are audit evidence only.  Physically
retired source survives as authenticated inert bytes; the archive is neither
an import path nor an execution fallback.

The current status and claim boundaries are summarized in :doc:`bongard`.
