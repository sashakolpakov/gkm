# Mechanical versus Codex glue: paired A/B protocol

The first pilot did not isolate the need for LLM glue: its two proposers had
different grammars. This follow-up fixes the admissible object identically in
both arms before running either one.

- Frozen common library: a copy machine acquired by mechanical enumeration,
  using two registers. The same library is reset for each paired condition.
  Neither arm receives the other's candidate or feedback.
- Common glue: one fresh state, one TOKEN transition, 1–8 instructions, then
  the same retained copy entry. The same graph pushout, exact replay,
  retention, and necessary-reuse checks apply to both arms.
- Tasks: duplicate first; swap the first pair; rotate the first three left;
  reverse the first three. All have variable remaining suffixes.
- Fixed seeds: 1, 11, 23. Each condition has 12 training and 12 validation
  examples of lengths 3–6, and 48 hidden examples of lengths 3–14. Token pools
  are disjoint. Task names and hidden/validation examples are not model input.
- Mechanical arm: breadth-first instruction search, at most 8,192 evaluated
  candidates and 120 seconds. Reject irreparable output-prefix errors; merge
  equal configurations on all training examples. This equivalence is only
  relative to the training sample, not a theorem about unseen inputs.
- Codex arm: one `gpt-5.6-sol` proposal, medium reasoning, at most 120 seconds,
  existing Codex login. No tools or execution feedback. The output schema and
  host parser enforce the same grammar as the mechanical arm.
- Order alternates between arms. Every choice is frozen before any hidden
  evaluation. Report exact accuracy, successful conditions, instruction
  counts, elapsed time, mechanical evaluations, and model token usage.

No loss-complexity formula decides between replay-valid attachments in this
A/B test. Mechanical enumeration prefers shorter instruction words by order;
Codex is asked for compact glue. The model's pretrained knowledge and token
cost do not become equivalent to a mechanical evaluation just because the
wall-time ceiling is the same.

If mechanical search succeeds as often as Codex, this is evidence that LLM
glue is unnecessary **for this grammar and task family**. A mechanical failure
at the cap is evidence about that search budget, not a proof that LLMs are
necessary. Neither result settles return-capable or recursive attachment.

Run from the repository root (12 model calls, one per paired condition):

```bash
python3 transduction/glue_ab.py --budget 8192 \
  --output output/transduction_cofibration/paired-ab-new
```

## Completed results: 3 October 2026

Artifacts: `output/transduction_cofibration/20261003-paired-ab/`. The protocol
and source hashes were saved before the first proposal. A separate Python
process subsequently checked those hashes, rebuilt all 24 pushouts, replayed
training/validation/hidden executions, checked unchanged old copy behavior,
and matched every Codex candidate to its original completed response. It also
reran the deterministic mechanical search and recovered every saved candidate.

| Measure | Mechanical search | Codex `gpt-5.6-sol` |
|---|---:|---:|
| Admitted task/seed conditions | 12/12 | 12/12 |
| Correct hidden examples | 576/576 | 576/576 |
| Median proposal time | 0.064 s | 10.90 s |
| Total proposal time | 0.99 s | 125.75 s |
| Model calls | 0 | 12 |

Per task, identical across the three seeds:

| Task | Mechanical candidates evaluated | Glue instructions, both arms |
|---|---:|---:|
| Duplicate first | 2 | 1 |
| Swap first pair | 96 | 5 |
| Rotate first three left | 443 | 7 |
| Reverse first three | 864 | 8 |

The arms produced identical instruction words in only 3 of the 12 pairs; the
remaining pairs used different but successfully replayed implementations.
Both preserve the old machine and pass the test that removing its transitions
reduces new-task training accuracy. None of these results comes from adding
another independent solver while leaving the old machine unused.

Mechanical search used 4,215 candidate evaluations in total and peaked at
865 remembered training configurations. It is substantially stronger than the
pilot's blind enumeration: because output cannot be retracted, it can reject
wrong output prefixes immediately. Equal training configurations need not be
explored again with a longer word. These are generic search operations, not
task-specific solution templates.

Codex timings include CLI startup and inference. Its completed responses
reported 102,178 input tokens (47,488 cached), 2,297 output tokens, and a
separate reasoning-output field of 1,169 tokens. These are usage counts, not
an API-dollar estimate. The observed proposer process-group peak was
162.1 MiB, below the enforced 768 MiB ceiling. No model tool events occurred.

### Interpretation

**LLM glue is unnecessary for this tested instruction-word grammar.**
Mechanical search matched its accuracy and compactness, with much lower
elapsed time and no model usage. This supports using mechanical search first
for this class of attachment.

The scope matters: these are four deliberately related transformations with
a shared copy suffix, not twelve unrelated task families. Each proposer may
add a straight instruction sequence and enter an old machine; neither may
introduce conditional control, new loops, recursive calls, or return interfaces.
Three data seeds do not establish performance across those richer programs.
The next discriminating A/B test must enlarge the **same grammar for both
arms**, and include tasks that actually need those additional structures.
