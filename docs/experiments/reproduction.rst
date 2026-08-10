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

Build the Sphinx documentation with warnings as errors:

.. code-block:: bash

   .venv/bin/python -m sphinx -W -b html docs docs/_build/html

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

Historical artifacts
--------------------

The loop/relational pilot, five-task TRAIN campaign, panel-soft zero-coverage
drill, and atomic attempts remain immutable historical records.  Their old
command lines and source identities are audit evidence only.  Physically
retired source survives as authenticated inert bytes; the archive is neither
an import path nor an execution fallback.

The current status and claim boundaries are summarized in :doc:`bongard`.
