# Bongard experiment history

This file keeps the experimental record separate from the current design.
None of the runs below is a successful Bongard benchmark, and none authorizes
official-test model access.

## Corpus and ledger snapshots

The pinned ShapeBongard V2 release has 12,000 tasks and 168,000 panels:
3,600 `ff`, 4,000 `bd`, and 4,400 `hd`. Its primary split is 9,300 train, 900
validation, and 1,800 test.

Three exact-unused counts appear in artifacts and are correct for their time:

- **10,047** was the train/validation count after Stage-A A3;
- **10,044** was the pre-coverage-pilot baseline after three later atomic
  records; and
- **10,020** is the post-pilot count after 24 further exposures.

At the second snapshot, 156 task IDs had been exposed. At the third, 180 had
been exposed. The pre-pilot 10,044 remaining
IDs comprise 2,998 `ff`, 3,431 `bd`, and 3,615 `hd`, or 9,156 train and 888
validation. Exact-image-unseen does not imply semantic independence: most of
this pool reuses generator concepts already represented in the ledger.

Strict reusable DRILL capacity fell to zero. A separate strict DEV reserve had
16 tasks, all `bd`, before the pilot and has 15 afterward. Although the pilot
excluded exact DEV task IDs, it did not exclude engineering tasks sharing a
DEV semantic disclosure key; one selected task therefore disqualified one DEV
unit. The official test split has remained sealed from model use; hashing its
bytes for release authentication is not exposure.

## Stage-A calibration sequence

### A1: transport failure

A1 attempted the earlier soft-description calibration protocol. Proposer calls
completed, but scorer transport failed. No usable scores, fitted calibration,
semantic accuracy, negation comparison, Stage-B authorization, or benchmark
result was produced. Its selected tasks remain consumed.

### A2: invalidated source mutation

A2 was a new repaired-protocol experiment, not an A1 reroll. A concurrent edit
changed executable Python after the protocol/cohort freeze. The run was
invalidated and produced no terminal scientific artifact. Partial process
counts and lost outputs cannot support semantic inference. Its cohort remains
consumed.

This incident led to complete Python source-boundary receipts and source-drift
checks around exposure, model calls, persistence, and replay.

### A3: underpowered calibration bin

A3 completed its proposer/scorer transport but failed the preregistered
calibration fit: the upper bin contained six clusters, below the fixed minimum
of eight. Stage B did not run. The intended orientation was better than its
complement on those calibration observations, so this was not evidence that
negation “won.” It was also not a benchmark score.

The historical exact-unused count immediately after A3 was 10,047. Exact
replay later showed zero strict reusable DRILL capacity and 16 strict `bd` DEV
units at the pre-coverage-pilot boundary.

## Atomic one-task sequence

The atomic sequence used repeated-generator, exact-image-unseen training tasks
to debug transport and synthesis. It was never an independent evaluation.
Each selected task was consumed before model access and was not rerolled.

### Attempt 1: terminal-wrapper failure

Exposure and part of the run state persisted, but failed terminal construction
could not serialize a frozen precommit object. No recoverable prediction or
score exists. The failure motivated a durable recovery path for the
post-run/pre-terminal window.

### Attempt 2: prompt/parser mismatch

Twelve neutral support descriptions and one proposal receipt persisted. The
prompt required observer questions ending in `?`, while the shared parser
forbade that character. The run stopped before support scoring, formula
selection, query access, prediction, label reveal, or score. The parser contract
was repaired prospectively; the task was not rerun.

### Attempt 3: live soft predicate did not separate support

The one-shot proposal was:

> Is a small triangle attached to a tilted quadrilateral?

The journal closed 25 validated calls: twelve descriptions, one proposal, and
twelve support observations. All six positive supports were `present`. The
negative supports were three operational nonmatches, two `present`, and one
`indeterminate`. Python raised `NoExactSeparatorError` before formula freeze,
query access, prediction, query-label materialization, or score. Cold replay of
the prefix passed.

This exposed the representation failure: one fluent sentence bundled object
shape, relative size, orientation, and attachment, while the scorer did not
preserve object roles consistently.

## Correction of the A3 panel mapping

A later support-only forensic used deterministic loop geometry and reported a
perfect 6/6-positive versus 6/6-negative shape/ratio separator. That result is
correct on the actual support panels. A subsequent 5+1-versus-5+1 report was
wrong because it mapped held-out positive source index 4 and held-out negative
source index 5 into the support set and omitted two resolved supports.

With the archived source indices restored:

- the base triangle/quadrilateral area-ratio-at-most-1/8 predicate has six
  positive support `present` results and six negative support
  `certified_absent` results;
- exhaustive search of the historical 2,520-member contact-inclusive
  diagnostic relational superlanguage finds four exact support separators:
  ratio threshold 1/12 or 1/8, each with no obliqueness clause or a 5-degree
  role-1 obliqueness clause, and all within its 1,260-member contact-disabled
  proposer-reachable half;
- the held-out positive and negative are both `indeterminate` under the 1/8
  base query, while the 1/12 variants certify the negative held-out absent but
  leave the positive held-out `indeterminate`; and
- no registered candidate separates all fourteen panels exactly, while point
  contact remains unresolved on the thick-stroke attachment.

The 6/6 support fit is post-hoc resubstitution, not prediction or
generalization. The mapping error is why every diagnostic must bind source
indices, exact PNG digests, and support/held-out roles rather than reporting
unbound panel counts.

## Current line

The successor is Python-first: exact PNGs become candidate-independent typed
loop packets and optional neutral calibrated tags; a closed positive predicate
uses one explicit object binding; the formula freezes before query release;
and Python replays the result without a model. Lean is optional and removable,
never semantic authority.

The typed packet, relational evaluator, contact uncertainty, and experimental
vision-tag schema are fixture-tested. The 24-task exact-unused train/validation
coverage pilot completed with report digest
`sha256:f78626c51b0af34cb0ccd96ed56041a51bcaeb453d3f26b10ea1ed1377542ae0`:
336/336 panels extracted, yielding 17,876 loops (10,354 substantive), 4,516
present versus 13,360 indeterminate polygon/obliqueness observations, and
267,197 pairs with 46 present contacts, 116,520 certified separations, and
150,631 indeterminate contacts. The subsequent v3-library ablation found 0/24
exact seven-per-side separators, 0/168 exact 6+6 folds, zero held-out
generalizers, and a best forward profile of 8/14. No proposer participated.

The hardened v4 runner and v4 campaign orchestrator are implemented and
fixture-tested, including the explicit fixed five-task exact-unused TRAIN
semantics-reused engineering mode. At this pre-run snapshot no real plan has
been frozen or published and no model campaign has been executed. There is no
benchmark number to infer yet.

The next adversarial audit found that the proposed 15-task strict DEV run was
invalid for a second, independent reason beyond its leaking schedule
commitment: none of its intended concepts is fully expressible in the v3
two-closed-loop polygon/ratio language. No DEV pixels or proposer calls were
made. The missing deterministic topology, curvature, and symmetry observables
were already implemented elsewhere but had not been glued into the runner.
The implemented repair is a closed Python union with a pre-frozen support-only
expressibility oracle. Its corrected v2 65,678-member proposer-reachable A3
gate found exactly four separators, all among 1,260 contact-disabled
relational predicates, and zero from the direct-count and symmetry branches. The
strict 15-task DEV cohort remains stopped at 0/15 intended-concept
expressibility. Any exact-unused TRAIN run before a new DEV freeze is reported
only as semantics-reused representation engineering; none has run at this
snapshot. Python remains canonical and Lean optional and removable.
