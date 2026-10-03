# Fixed basis, broader composition search

This experiment tests the next question after the cumulative growth benchmark:
with the elementary instructions and acquired starting library held fixed,
how far can mechanical search go, and when does an LLM find useful glue faster?

The earlier benchmark and all of its execution code remain unchanged. New
experiments use `composition_search.py` and `composition_benchmark.py`.

## Results: 2026-10-03

Completed artifacts: `output/transduction_composition/20261003-fixed-basis-ab`.
With the same frozen basis and shared composition grammar, mechanical search
solved all ten tasks. Codex solved seven; its other three calls exceeded the
120-second condition budget without delivering an accepted complete proposal.
These are timeouts, not evidence that the functions are inexpressible.

| Measure | Mechanical enumeration | Codex `gpt-5.6-sol` |
| --- | ---: | ---: |
| Admitted tasks | 10/10 | 7/10 |
| Hidden cases answered correctly | 480/480 | 336/480 |
| Stress cases answered correctly | 120/120 | 84/120 |
| Total proposal-stage time | 14.549 s | 674.793 s |

All seven accepted Codex programs were exact on all their held-out cases:
420/420. The remaining 180 cases had no admitted program, rather than an
incorrect returned output. Aggregate coverage counts those unanswered cases
as failures. Both arms' accepted programs also passed the exhaustive short-input
audit described below. Proposal-stage times exclude admission checks and final
audits, but include Codex request/response latency and its timeouts.

### Observed difficulty, not just generating-program size

| Generating calls | Family | Mechanical solution calls | Candidates tested | Mechanical seconds | Codex result, seconds |
| ---: | --- | ---: | ---: | ---: | --- |
| 2 | Pipeline | 2 | 30 | 0.003 | Exact, 53.532 |
| 2 | Mixed | 2 | 65 | 0.002 | Exact, 14.691 |
| 3 | Pipeline | 3 | 300 | 0.006 | Exact, 27.623 |
| 3 | Mixed | 3 | 844 | 0.009 | Exact, 23.782 |
| 4 | Pipeline | 2 | 27 | 0.002 | Exact, 27.287 |
| 4 | Mixed | 4 | 4,540 | 0.034 | Exact, 72.671 |
| 5 | Pipeline | 3 | 238 | 0.006 | Exact, 94.982 |
| 5 | Mixed | 4 | 16,952 | 0.052 | Timeout, 120.154 |
| 6 | Pipeline | 6 | 1,248,911 | 3.370 | Timeout, 120.045 |
| 6 | Mixed | 6 | 4,053,996 | 11.066 | Timeout, 120.027 |

The four- and five-call generating programs sometimes had shorter solutions.
Those shorter solutions survived the long-input and exhaustive short-input
checks; they were not merely counted as successful training fits. This is why
the experiment reports both generating size and discovered size.

For both six-call tasks, enumeration exhausted every normalized expression with
fewer than six calls before finding a training-exact solution. Six is therefore
the minimum training-consistent call count in this bounded grammar, not a claim
about absolute Kolmogorov complexity or a different language. The hardest case
reached its solution after 4,053,996 candidates while making only 4,997 actual
retained-procedure executions, with 14,108,777 exact cache hits. The cache key
is the procedure plus its exact input, including intermediate pipeline inputs.
There is no unsound merging of programs by training fingerprints.

Search tested 5,325,903 candidates overall. The largest recorded harness peak
was 64.30 MiB, below its 512 MiB stop. The largest recorded successful Codex
process-group peak was 161 MiB, below its 768 MiB stop. The seven completed-call
receipts report 78,203 input tokens and 11,212 output tokens, including 10,291
reported reasoning output tokens. The three timed-out calls have no complete
usage receipt, so those figures are not total billed or consumed usage. No
accepted call used tools.

### Attachment, replay and interpretation

Independent replay reproduced all ten mechanical selections and candidate
counts; checked model requests, replies and training feedback; and rebuilt
**32 pushouts** across the 17 admitted proposer conditions. The same replay
checked retention, selected executions, and unshared controls. Source hashes
matched the frozen protocol. All old basis programs remained intact.

The private generating witnesses for **all ten tasks** compiled and passed
training, validation, hidden and stress execution under the same primitive
basis and resource limits. This includes the three model timeouts. These private
witnesses establish available solutions; they are not counted as proposer wins.

The exhaustive audit tested 5,296 equality patterns per task through length
eight. All passed. The seven accepted model programs matched the mechanical
programs after normalization, so the audit reused identical program/target
evidence: 52,960 unique executions covering 17 admitted conditions. Longer
inputs remain sampled, up to length 512; this is not a proof for all lengths.

The hardest mixed solution itself reaches twelve stored procedures, including
its entry and mechanically lifted helpers. Its reachable shared representation
uses 77 term units versus 139 when each call subtree is copied separately.
Those are the prior benchmark's instruction/constructor/interface units, not
byte counts. No new primitive operation or hard free-energy admission rule
was introduced.

**Conclusion:** mechanical search is the stronger default on this tested
fixed-basis family. Its cost rose from tens to millions of candidates, but
remained below twelve seconds per condition. This experiment did not find a
crossover in favor of direct LLM proposals. It does not compare every mechanical
method, an LLM with execution tools, or an LLM-guided mechanical search. Ten
seeded tasks and a six-call bound also do not establish the scaling boundary
for larger grammars. A further ladder should distinguish genuinely necessary
larger compositions from longer descriptions of short functions.

All experiment processes finished. The full transduction suite passes 60 tests
plus seven subtests. Earlier benchmark sources and results are unchanged.

## What is fixed, and what is broadened

The basis is reconstructed from the certified initial acquisitions of
`output/transduction_growth/20261003-returning-ab`: five learned instruction
fragments and the first six learned whole-input procedures. Their bodies,
interfaces, hashes, and primitive interpreter remain unchanged. The six whole
procedures implement copying, token duplication, pair swapping, triple
reversal, alternating-position selection, and run collapse. The proposers see
opaque IDs and executable bodies, not these descriptive task labels. No later
composition solutions from the earlier benchmark are imported.

Both arms have the identical recursive grammar:

```text
Program = Call(one of the six retained whole-input procedures)
        | Pipe(Program, Program)
        | IfNextEqual(Program, Program)
```

`Pipe` passes its first result as the second program's input. `IfNextEqual`
selects its first branch when the first two input tokens exist and match;
otherwise it selects its second branch. A branch receives the unchanged input.
The experiment broadens pipeline and conditional compositions, not all the
other control constructs at once. Loops and local fragment calls already live
inside the retained procedures. No new elementary or control operation is
introduced. Recursive grammar means finite nested syntax, not recursive calls
to an unfinished procedure.

The difficulty ladder permits two, three, four, five, or six Call leaves.
Before exact normalization, the numbers of syntax trees with exactly one
through six leaves are 6; 72; 1,728; 51,840; 1,741,824; and 62,705,664.
Different trees can compute the same function, so these are syntax counts,
not counts of distinct behaviors or proofs of task difficulty.

## Mechanical search

Search enumerates programs by increasing leaf count, then operator, left/right
size split, and retained procedure order. This replaces the earlier ten fixed
templates with systematic enumeration of bounded expression trees.

Both arms use the same exact normalization: pipelines are right-associated;
nested branches testing the same unchanged input lose unreachable alternatives;
and identical branches collapse. No sample-based equivalence is assumed.
In particular, agreement on original training inputs does not justify merging
two programs when a pipeline may later feed them different intermediate inputs.

The search evaluator caches only exact `(retained procedure, input)` results,
with a 10,000-entry least-recently-used limit. Each cache miss executes the
unchanged graph interpreter. Compositions use their pure functional semantics.
Tests compare this evaluator with compiled quotient graphs. Every selected
program is freshly compiled and executed before admission. Training examples
are ordered by decreasing input length for both arms, without task-specific
ranking. Candidates stop at the first failing training example.

Per condition, mechanical search has ten million candidates, 120 seconds, and a
512 MiB process peak-memory stop. It stores completed smaller syntax layers
and streams the active layer. If it finds a training-exact candidate, search
stops; fixed validation and retention gates decide admission. Failure at those
later gates does not cause another search in this protocol.

The candidate cap exceeds the complete normalized grammar through six leaves;
the wall-time or memory limit can therefore stop search, but a small arbitrary
candidate cap cannot create an apparent LLM advantage on this ladder.
The normalization leaves 9,040,320 syntax trees in total through six leaves.

## LLM arm and matched conditions

The LLM is the user-requested `gpt-5.6-sol`, using existing Codex access and the
data-only transport. It receives the same starting library, grammar bound,
training examples, and semantics. Its answer is a schema-checked postorder
syntax tree, not executable source. Node indices must refer backward; unused
nodes, hidden shared subtrees, calls outside the basis, and new operations are
rejected. The tree is then normalized by the same rules as mechanical search.
The [official structured-output interface](https://developers.openai.com/codex/noninteractive#create-structured-outputs-with-a-schema)
is used for the final proposal, followed by independent host validation.

The LLM gets at most two proposals within **one 120-second total proposal-stage
budget**, not 120 seconds per retry. Only training failures are returned on a
retry. The Codex process group is stopped above 768 MiB, as in the earlier
transport. No tools, target program, validation data, or hidden data are given
to the model. Equal wall budgets are not a claim of equal compute, memory,
energy, or monetary cost.

## Tasks, freezing, and genuine attachment

Ten tasks are generated before either arm runs: one pipeline and one mixed
conditional/pipeline task at each leaf bound, using fixed seeds 7 and 19.
Generation rejects identity padding, a normalization that reduces the requested
size, unsafe worst-case buffer growth, agreement with a single basis call on
all training inputs, and leaves having no observable training effect when
replaced by identity. No task is chosen using an A/B outcome. Generating-program
size is not a proved minimum solution size; a smaller successful solution is
reported as such.

Every condition starts from the same frozen library. Solving an earlier
condition does not provide a shortcut on a later one. This isolates composition
search difficulty; it is not another cumulative curriculum.

Nested pipeline operands are mechanically named as local helper procedures
because the existing interpreter's pipeline interface expects named whole-input
procedures. Each helper contains exactly the proposed operand, without any
host-chosen behavior. The helper attachments and the final attachment form a
joint candidate. Their graph pushouts are certified; the entire candidate is
accepted or rejected together. The fixed library bodies are neither edited nor
flattened into the proposal.

The 48 training examples per task have length at most 12. Forty-eight separate
validation examples gate admission. All choices are frozen before 48 hidden
examples through length 97 and twelve stress examples at lengths 128, 257 and
512 are evaluated. Token pools are disjoint between splits. Executions are
limited to 200,000 interpreter steps and 4,096 output/intermediate tokens.

After the freeze, the private generating programs are also compiled and replayed
under the identical basis, grammar, and resource envelope. This establishes
whether even a task missed by a proposer has a valid available solution; those
witnesses are not proposer successes and are never fed back to either arm.
Unshared execution controls and retention checks remain in place. Independent
replay reconstructs task generation, model requests and feedback, attachments,
and recorded executions. Mechanical selections can also be reproduced.

`audit_composition.py` additionally checks every token-equality pattern through
length eight after the freeze: 5,296 patterns per admitted task. Identical
program/target pairs share audit work, with hashes recorded. Longer inputs
remain sampled; this finite audit is not a proof for all lengths.

## Run

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -u transduction/composition_benchmark.py \
  --output output/transduction_composition/new-run
PYTHONDONTWRITEBYTECODE=1 python3 -u transduction/composition_benchmark.py \
  --output output/transduction_composition/new-run --replay --reproduce-search
PYTHONDONTWRITEBYTECODE=1 python3 -u transduction/audit_composition.py \
  --output output/transduction_composition/new-run
```

The default basis path refers to the previous recorded acquisition. A new basis
can be produced by the existing growth runner and passed with `--basis`.
Use a new output directory; completed artifacts are not overwritten. Results
remain conditional on the supplied pure-functional grammar and the ten seeded
tasks, not a general comparison of all program-synthesis methods.
