# Shared-witness calibration v1: failure analysis

This note analyzes the sealed live calibration at
`downloads/ShapeBongard_V2_full/object_shared_witness_calibration_20260808_v1`.
It does not authorize a benchmark launch or reuse any query or official-test
pixels.

## Custody result

- Authorization: `sha256:eb20a5c44763442250766964d6caca2d14f4b6c1eadf80a5f92eaa0ca35e7b85`
- Blind batch: `sha256:a2761bacf38ced8414498930cbbc94b0e2355f2da29a123d7803320c471a4092`
- Assessment: `sha256:99442963d5b1dbc576055893a85410a71cd5ce0358d5ecd4357532ebc4368d16`
- Cold replay: `sha256:f555430d5df964e10693088517de1a5be2615dbd8a445232450c9ba9362d33a6`
- Result: `sha256:516d010af9fcf46b19744eaad15e38c7b4819dfecad42b3f4dcec306661a2578`
- Calls: 48 fresh, 0 reused, 0 during assessment/replay.
- Query, broad-cohort, and official-test pixels opened: no.
- Verdict: rejected; no candidate rank selected.

The custody machinery worked. The visual semantics did not.

## Exact failure measurements

Rank 0 used anchor `patterned loop network`, axis `junction organization`,
and endpoints `shared hub` versus `distributed junction`.

- Pass 0: 4 present, 6 certified absent, 2 indeterminate.
- Pass 1: 4 present, 4 certified absent, 4 indeterminate.
- Five of twelve panel dispositions changed between the two identical passes.
- One target panel flipped all the way from certified absent to present.
- Nine of 24 rank/pass observations had more than one clear anchor instance.
- One of 33 clear-anchor entity observations supported both endpoints.

Rank 1 used anchor `decorated contour network`, axis `contour termination`,
and endpoints `closed circuit` versus `free ended`.

- Pass 0: 3 present, 2 certified absent, 7 indeterminate.
- Pass 1: 3 present, 3 certified absent, 6 indeterminate.
- Three of twelve panel dispositions changed between passes.
- Ten of 24 rank/pass observations had more than one clear anchor instance.
- Ten of 34 clear-anchor entity observations supported both supposedly
  alternative endpoints.
- Two observations changed the inventory itself from two entities to three.

Both ranks had confident contradictions on target and foil support panels.
Neither failure can be repaired by negation, orientation reversal, threshold
tuning, or treating abstention as false.

## Missing theory

The Python aggregation was exact, but its input atoms were not operationally
defined. `shared hub` was a content-addressed phrase, not a fixed visual test.
The vision model therefore changed its interpretation between calls.

The entity inventory was also only nominally candidate-independent. Every
candidate prompt regenerated boxes, entity count, and descriptions from the
pixels. Candidate wording could change the scene decomposition before Python
saw it.

The shared anchors were not selective enough. On roughly two fifths of the
rank/pass observations, multiple figures were clear anchor matches. Similar
distractors then supplied the opposite endpoint, producing contradictions or
indeterminacy. The current gate records this evidence but has no separate
anchor coverage/selectivity test.

Finally, rank 1's endpoints were not empirically mutually exclusive. A closed
network can also contain free-ended branches. Positive wording alone does not
make two properties alternatives of one axis.

Lean cannot repair any of these failures. It can prove that a Boolean or
four-state formula was applied as written; it cannot make `shared hub` acquire
a stable visual meaning. Python is sufficient for the exact layer.

## Replacement boundary

The next admissible pipeline is:

1. Derive candidate-independent object proposals and zoomed crops from the
   exact panel bytes, with stable IDs and provenance.
2. Make one neutral vision call that freezes an entity inventory and typed
   interval observables before any candidate or group role exists.
3. Allow Codex to propose a closed Python predicate over registered positive
   atoms and explicit entity/pair/panel quantifiers. Open-vocabulary tags such
   as `bird-like` remain soft atoms with entity IDs, boxes, visible evidence,
   and four-state uncertainty.
4. Gate anchor coverage, anchor selectivity, endpoint exclusivity, and repeat
   stability separately from final support accuracy.
5. Freeze and reload the complete predicate and support evidence before any
   query pixel callback can run.

Until that replacement passes two independent historical calibration passes,
the fixed unused benchmark cohort remains sealed.
