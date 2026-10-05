# Does a longer feedback loop improve reusable discovery?

## Fixed follow-up protocol

The [frontier pilot](CONTROL_FRONTIER.md) stopped the failed fresh-data LLM trial
after two replies, despite having time left. This follow-up tests a larger search
allowance and repeated independent trials. It does not change the target task or
the proposer prompt to make the observed failure easier.

This protocol is written before the first measured follow-up call.

- Use only the `framed` acquisition/transfer pair, the family whose first success
  failed to repeat. This is a targeted reliability study, not a rerun of all four
  frontier families. Their earlier failures remain in the record.
- Fix data seeds **2, 3 and 4**. Seed 2 revisits the earlier failure; 3 and 4 are
  new. Run **two independent LLM trials per seed**, each from the original
  library. The deterministic mechanical search runs **once per seed**.
- Allow **600 seconds per acquisition or transfer stage**, including the common
  reuse probe. Allow **six model replies per stage**, versus two previously.
  Stop at the first training-exact candidate, then apply private validation and
  old-task preservation as before. No private failure is returned to a proposer.
- Raise the mechanical candidate ceiling from 2,000,000 to **8,000,000**, so the
  old ceiling does not prematurely consume the larger time allowance. Keep its
  grammar, ordering, sketch pool and eviction policy unchanged.
- Preserve `gpt-5.6-sol`, the existing managed Codex transport, reasoning setting,
  public examples, instruction text, response schema and feedback policy. Each
  reply sees the full current training set, the previous proposal and its first
  two failing public examples. This is explicit host-carried feedback, not a
  claim that the CLI session preserves hidden reasoning between calls.
- Use the same immutable cells, typed language, exact graph attachment verifier,
  complete-tree retention and shared reuse probe. Size remains diagnostic.
  No new primitive, desired helper interface or solution sketch is supplied.
- Freeze a manifest before each batch, recording every scheduled trial and the
  source hashes. Never carry a discovered helper between independent trials.
  Keep failures, including a solved acquisition with unsuccessful transfer.
- Run at most one mechanical worker and one model client at a time, in separate
  process groups with sampled **640 MiB** limits and termination on overflow.
  Their batches may overlap; no heavy audits or test suite run during the search.
  These are elapsed-time comparisons, not equal-hardware or equal-compute tests.

The primary outcome is a successful acquisition followed by genuine use of an
unchanged acquired helper on the transfer task. Solving two tasks by separate
specialized helpers does not count. Transfer must pass hidden execution, retain
the helper hash, visit it dynamically and lose correctness when its body is
disabled. All admitted attachments receive fresh replay and the same finite
shape/continuation audit as the pilot.

Supplementary diagnostic, specified after launch when inspecting the first
acquisition but before inspecting any transfer result: distinguish reuse of an
acquired helper's recursive control from calls that use only its leaf branch.
The original elision metric can count either. Inspect fresh execution traces
for reentry to an acquired helper while its earlier invocation is still active.
Report this separately, without changing admission, prompts or the original
transfer metric. A leaf-only use is not evidence that the learned traversal
transferred.

A second, explicitly post hoc diagnostic was added after the LLM batch: for a
replacement helper, mechanically check whether fixing its additional arguments
recovers an acquired helper's exact AST, including recursive calls. Fixed
arguments must remain the same pure literals on every recursive call; argument
evaluation order is preserved. The conservative check rejects unsupported
lexical bindings rather than guessing equivalence. This is evidence of an
interface extension, not execution of the old cell and not an additional
categorical pushout certificate. Passing the extra fixed arguments costs
interpreter steps before specialization; this check does not claim equal step
cost or identical acceptance at every finite step limit. It cannot improve the
original reuse count.

For every trial, report the public accuracy after each reply and the reply/time
of first success. Mark successes that fit the old two-reply/180-second limits
separately from those requiring extra work. This is a descriptive prefix
comparison, not a separate randomized budget arm. The model does not receive
the configured budget in its prompt. Model sampling is not seed-controlled;
repeating the same data seed helps distinguish that variation from changes in
the data, but six trials are still a small sample.

Do not compare an LLM's best-of-two result with a mechanical single-trial result
without stating the extra cost. Report individual trials and counts. The fixed
batch is not extended after looking at outcomes. A failure of this baseline is
not a proof that mechanical synthesis cannot solve the task: its search remains
heuristic and can evict a sketch before reaching a useful binding.

The worst-case allowance is six LLM acquisition/transfer trials and three
mechanical acquisition/transfer trials. A transfer normally needs only the
shared reuse probe; if it does need proposals, its additional time and calls are
reported. No API key or new billing route is introduced. Usage comes from
completed managed-Codex receipts; incomplete call usage is not guessed.

## Commands and artifacts

```sh
python transduction/frontier_benchmark.py --output output/transduction_frontier/20261004-retries-mechanical --families framed --arms mechanical --seeds 2 3 4 --repeats 1 --seconds 600 --replies 6 --candidate-limit 8000000
python transduction/frontier_benchmark.py --output output/transduction_frontier/20261004-retries-codex --families framed --arms codex --seeds 2 3 4 --repeats 2 --seconds 600 --replies 6 --candidate-limit 8000000
python transduction/frontier_benchmark.py --output output/transduction_frontier/20261004-retries-mechanical --replay
python transduction/frontier_benchmark.py --output output/transduction_frontier/20261004-retries-codex --replay
```

Each batch has a `trials.json` manifest, per-trial frozen sources and selections,
and an aggregate `batch-summary.json`. Replay checks that no planned trial or
failure was omitted and reconstructs intermediate public scores and feedback,
as well as successful attachments. Earlier experiments retain their original
`sources.tar.gz`; replay them with that archived source, not the changed harness.

The [official Codex command documentation](https://learn.chatgpt.com/docs/developer-commands#codex-exec)
was checked before editing the retry policy. The existing structured-output
transport was retained; this experiment changes the harness's search allowance,
not the selected model or API.

## Results

**Cleanly complete:** all six LLM trials, three mechanical trials and final
audits finished. No experiment process remains running. The final suite passes
104 tests and 15 subtests. Changes remain local and uncommitted.

The LLM solved both tasks in all six trials and reused its acquired helper
unchanged in four. Mechanical search found no acquisition in three ten-minute
trials. This is a repeated practical advantage on this task family and against
this bounded baseline, not a general impossibility result for mechanical search.

### Mechanical trials

| Data seed | Candidates tested | Search time | Outcome |
| --- | ---: | ---: | --- |
| 2 | 3,720,042 | 600.34 s | Time limit; acquisition failed; transfer blocked |
| 3 | 3,871,213 | 600.44 s | Time limit; acquisition failed; transfer blocked |
| 4 | 4,182,430 | 600.26 s | Time limit; acquisition failed; transfer blocked |

All three stopped at the time limit, not the expanded candidate ceiling or
memory cap. The largest sampled worker RSS was 44.4 MiB. The private certificates
still construct valid solutions inside the mechanical grammar on every seed.
This baseline's ordering and sketch eviction remain limitations; a better
synthesis algorithm may close the gap.

### Independent LLM trials

Every acquisition and transfer task passed admission and all 40 hidden plus 13
stress cases: **12/12 tasks and 636/636 held-out cases**. This does not mean every
trial achieved reusable transfer. **Four of six** used the acquired helper
unchanged. The other two solved the second task by adding new helpers and never
called their first acquired helper.

| Data seed | Trial | Acquisition public accuracy by reply | Acquisition time | Second task | Unchanged-helper transfer |
| --- | ---: | --- | ---: | --- | --- |
| 2 | 1 | 24/26 → 26/26 | 174.77 s | New proposal, 43.50 s | No |
| 2 | 2 | 26/26 | 80.37 s | Existing helper binding, 0.28 s | Yes |
| 3 | 1 | 26/26 | 84.38 s | Existing helper binding, 0.28 s | Yes |
| 3 | 2 | 4/26 → 26/26 | 183.33 s | Existing helper binding, 0.28 s | Yes |
| 4 | 1 | 26/26 | 135.94 s | Existing helper binding, 0.28 s | Yes |
| 4 | 2 | 26/26 | 122.45 s | New proposal, 59.49 s | No |

All four successful transfers visited the acquired helper on all 40 hidden
examples, kept its exact hash and fell to 0/40 when its body was disabled. The
host selected the new bindings with no model call. The two unused helpers were
visited zero times; disabling them left 40/40 correct. We do not count those
trials as strict reuse successes. Freshly replayed traces additionally show
recursive reentry to the retained helper in all 160 hidden cases across the
four strict transfers: actual learned traversal is used, not only a leaf branch.

The first request on seed 2 was byte-for-byte identical to that in the previous
failed seed-2 trial. No successful helper or solution hint was imported. The
different first-reply outcomes on the same public data demonstrate substantial
variation between model calls.

### What the larger allowance did and did not establish

**No successful stage required a third reply.** All acquisition replies stopped
after one or two proposals, and the two second-task model calls each succeeded
on their first proposal. Five acquisitions fit the old two-reply/180-second
limits. The sixth took 183.33 seconds, only about three seconds beyond the old
limit; one should not build a strong causal claim around that small margin.

The new evidence therefore comes mainly from more independent trials. It does
not show that six conversational turns were necessary, or that the old failed
trajectory would have recovered on its third turn. We did not continue that
particular saved trajectory. The data support a repeatable practical discovery
capability on this family, with a remaining failure mode in choosing a reusable
interface early enough, not universal LLM superiority.

### Two kinds of library growth

The four strict transfers reuse five-position recursive control with different
operation bindings. One helper needs only two procedure arguments; another uses
a Boolean and two procedures; two use three procedures. Those signatures were
chosen by the model. The second task adds only 6–8 expression nodes for a new
root calling the retained helper, versus 35–45 nodes for acquisition.

In seed 2 trial 1, the first helper `P5(mode)` fixes its two operations internally.
The second task introduces `P6(mode, a, b)`, making those operations arguments.
In seed 4 trial 2, `L0(leaf)` accepts one operation at entry but fixes the
recursive calls' operations internally. Its replacement `L1(leaf, odd, even)`
exposes those internal choices too. These transfers add 45 and 47 nodes rather
than a small new binding. Both old helpers remain intact for their original
tasks; neither is called on the new task.

The post hoc specialization check **passed in both cases**:

```text
P6(s, swap_pair, copy_pair)     specializes exactly to P5(s)
L1(leaf, swap_pair, copy_pair) specializes exactly to L0(leaf)
```

Here `swap_pair` is `F004` and `copy_pair` is `C01`. The checker verifies exact
source-tree identity after substituting those fixed arguments, including every
recursive call; it does not merely retest a few examples. Its records bind both
helper hashes and retain the recovered body. Extra argument-passing step costs
are not asserted equal. The host did not alter either proposed helper.

Thus these were verifiable interface extensions, not unrelated replacements.
But the old cells were still not executed on the second task, so the original
strict reuse score stays **4/6**. This distinction matters for a future design:
an ARC-like campaign can allow a useful interface to emerge after encountering
another task, instead of treating immediate reusable-interface discovery as the
only form of learning. That is a possible follow-up, not another experiment
launched here.

### Effort and verification

Ten completed model receipts record 146,973 input tokens, including 33,920 cached,
and 34,416 output tokens, including 32,499 reasoning tokens. There were no timed
out or unreceipted model calls. Total stage search time was 885.36 seconds across
six independent acquisition/transfer trials. These are all-trial costs, not just
the cost of whichever run succeeded fastest. The largest sampled model process
group was 161.4 MiB, below the 640 MiB cap. Managed Codex access was unchanged;
no dollar cost is inferred.

All **21 attachment certificates** passed fresh reconstruction and replay.
Both batch manifests replayed every scheduled outcome, including the three
mechanical failures, and verified intermediate public scores and feedback.
The post-selection audit passed **30,048/30,048** shape cases and **972/972**
concatenated-region cases across all 12 admitted task roots. The mechanical arm
has no admitted program to score; zero certificates are not presented as a
behavioral success.

The specialization audit was added only after inspecting the fixed batch's
replacements, so it is explicitly exploratory. Its regression test rejects
changed recursive bindings and attempts to erase effectful argument evaluation.
It neither changes any promotion nor upgrades the original reuse metric.

The practical finding is stronger than the single pilot success, but remains
bounded: one targeted family, three data seeds, and six model fresh starts.
No result for the harder `feedback` family is implied. All comparisons use the
same exact attachment semantics; flexibility is in generating the helper and
its interface, not in a weaker verifier.

No further experiment or model trial starts automatically. Completion is
followed by a summary and a wait for the user's reply.
