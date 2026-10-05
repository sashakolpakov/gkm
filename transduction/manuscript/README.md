# Transduction manuscript

Build the existing paper from the repository root:

```sh
make -C transduction/manuscript
```

The output is `transduction.pdf`. LaTeX build products are ignored by Git.
`transduction.tex` contains the original primitive-sufficiency study and the
discussion. It includes `machine_growth.tex`, which reports the machine-growth
experiments completed on 3–4 October 2026. The editorial audit preserves the
reported results while correcting their interpretation and removing repetition.
The subsequent reproduction described below regenerated the missing register
run and assembled the full evidence package without new LLM calls.

## Selection rules are not interchangeable

- Original register-transducer benchmark: training loss plus solver size,
  four lambda runs, then the smallest member of their validation Pareto set
  within an absolute loss tolerance of 0.075 from the best validation loss.
- Later matched attachment studies: the first training-exact proposal faces
  validation and preservation gates; the complete admitted proposal is retained.
  Reuse and small-glue preferences influence proposal construction and order.
  Description size is measured afterwards, not used as a hard energy selector.

The expression-node ledger of the interface studies is not the original
rules-plus-actions metric, the earlier modular language's term count, or ARC's
LOC/literal metric.

## Evidence map

Report links are relative to this directory. Each report identifies its fixed
protocol, implementation, artifact location, replay command and limitations.

| Manuscript material | Supporting report | Critical qualification |
| --- | --- | --- |
| Original primitive results | [Register benchmark](../register_transducer_benchmark.md) | All ten rows reproduced with full artifacts; six hidden examples per condition and output-only correctness. |
| Tail attachment pilot | [Pilot](../COFIBRATION_EXPERIMENT_20261003.md) | Model and mechanical glue grammars differed. |
| Matched instruction glue | [Paired A/B](../GLUE_AB_PROTOCOL.md) | Both 12/12; mechanical search faster. |
| Returning calls | [Growth](../GROWTH_BENCHMARK.md) | Both 24/24 within supplied control forms. |
| Nested compositions | [Composition](../COMPOSITION_BENCHMARK.md) | Mechanical 10/10, model 7/10; three model timeouts. |
| Recursive control | [Recursive](../RECURSIVE_BENCHMARK.md) | Initial pilot failed; successful follow-up also changed the solver. |
| Chosen interfaces | [Interfaces](../INTERFACE_BENCHMARK.md) | Both 6/6; host selected all transfer bindings. Discarded unequal-objective trial is excluded. |
| Harder layout pilot | [Frontier](../CONTROL_FRONTIER.md) | Initial model success failed to repeat on fresh data. |
| Repeated trials | [Retries](../FRONTIER_RETRIES.md) | Model 6/6 acquisition/next-task pairs, but only 4/6 unchanged transfers; host binder used. |
| Independent cumulative arms | [Direct A/B](../COFIBRATION_AB.md) | Model authored every attachment; 8 admitted, 2 failed, 2 blocked versus 4, 2, 6 mechanically. |
| Harder task, longer allowance | [Hard A/B](../HARD_COFIBRATION_AB.md) | Model 4/4; mechanical 2 failed acquisitions and 2 blocked transfers. One direct reuse, one verified interface extension. |
| Future value of extensions | [Retention assay](../TRANSFER_VALUE.md) | Not another proposer A/B. Mechanical adapters used frozen model-created libraries. |

The new numerical tables were checked against the local `analysis.json` files
for the two direct A/B runs and the retention assay. The hard trial's extension
claim is additionally supported by `interface-extension.json`: exact recovered
AST, preserved recursive binding, and 564/564 behavioral comparisons. It does
not assert unchanged runtime use of the old helper or equal interpreter cost.

## Reproducibility limits

The [unified reproduction package](../REPRODUCIBILITY.md) includes the detailed
artifacts from the ignored `output/` directories, matching historical sources,
and offline replay commands. It is not yet a public archival deposit. Distribute
the package with the manuscript; the paper alone does not contain the evidence.

Both proposers use the same exact graph attachment verifier. The final search
baseline is a bounded typed template/edit enumeration, not every mechanical
synthesis method. The task seeds share mechanics; behavioral case counts are
not independent discovery trials. Public feedback may guide a retry, but
private validation failures end the sequence and hidden evaluation occurs
only after choices freeze. These distinctions are retained in the paper.

## Editorial and evidence audit: 4 October 2026

The audit was repeated after the manual interruption. It covers both LaTeX
sources, the bibliography, implementation checks and the three recent frozen
experiments. Experimental code, saved proposals and results were not changed;
no model calls or new discovery runs were made.

Substantive corrections:

- **Primitive sets and cost.** `bidirectional` is a separate branch, not a
  strict extension of the register sets. In `pattern_fsa.py`,
  `genome_complexity` sums rules and instructions, including unused rules.
  Available registers, unused states and the vocabulary have no separate cost.
  Removed claims that the objective prices every resource or proves minimality.
- **Selection.** `validation_elbow` uses the 0.075 tolerance rule described
  above, not a geometric knee. The later interface studies instead retain the
  whole first proposal passing training, validation and preservation checks.
  Their expression-node diagnostic excludes parameter declarations. A drop
  means a smaller addition, not a shrinking cumulative library.
- **Older success metric.** `evaluate_genome` compares output without requiring
  a halt. The reproduction script uses a 32-instruction cap and only six hidden
  cases per condition. The paper now separates that metric from the later
  requirement for normal return and complete input consumption. The original
  matrix is retained as reported; no matching raw run archive was located and
  the evolutionary search was not rerun.
- **What was actually compared.** Early interface transfers used a host binder
  in both arms. The direct A/B studies made the model propose its own bindings.
  The retention experiment is a separate before/after comparison, not another
  model-versus-search discovery trial. Search coverage and equal elapsed-time
  budgets do not establish equal compute or exhaustive enumeration. Mechanical
  candidates also face a 12,000-step preliminary screen before the common
  60,000-step execution check.
- **Reuse and extension.** The paper distinguishes unchanged recursive use
  from exact recovery of an old program by fixing a new argument. Neither a
  valid pushout alone nor a size drop establishes useful transfer. Claims of
  universal generalisation, hard energy selection in the later studies, and
  causal effects of reuse instructions were removed or qualified.

Verification completed after the interruption:

- **125 tests and 23 subtests passed** with
  `PYTHONDONTWRITEBYTECODE=1 python -m pytest transduction -q -p no:cacheprovider`.
- Fresh replay of both direct comparisons reproduced their saved summaries
  and private grammar-coverage records exactly: 12 accepted programs in the
  first comparison and four in the harder one. The former has 18 accepted
  attachment certificates and two further certificates belonging to a rejected
  proposal; the latter has eight accepted certificates.
- Fresh replay of the retention experiment reproduced all 117 outcomes,
  99 accepted programs and 109 certificates. Its source check correctly rejected
  the current `audit_frontier.py`, which had changed after the run. Replaying
  from its own `sources.tar.gz` passed without bypassing that check. All replay
  results were compared in memory with the existing files, not overwritten.
- All seven claimed direct recursive transfers were checked again for actual
  recursive reentry: **280/280 hidden executions**. The hard seed-41 extension
  also reproduced the exact syntax identity and **564/564 behavioural
  specialisations**, without claiming equal interpreter step costs.
- The bibliography was checked against primary records. The loss-complexity
  reference now cites the [published 2026 article](https://arxiv.org/abs/2507.13543),
  rather than only its earlier preprint. Unused references were removed.

Repeated explanations and unsupported claims were cut, reducing the two source
files by about one fifth. A clean LaTeX/BibTeX build produces ten pages; the final
logs have no undefined references, missing citations or box warnings.

## Evidence regeneration and unified package

After that audit, the missing register experiment was rerun using the original
runner, substrate, seeds, budgets and selection rule. All ten rows matched on
hidden accuracy, loss and complexity. The new run is
`output/transduction_register/20261004-reproduction-v1`; its 40 lambda winners,
generation histories, ten selected programs, full examples and executions are
recorded. The original raw run remains unavailable, but the numerical result
now has a complete, dated reproduction rather than only a summary report.

The full package covers 21 run or batch records across the manuscript, including
failed and blocked cases. It retains historical files without editing them and
supplies isolated source versions for replay. Two early frontier audits lacked
their exact auxiliary source copies; these were regenerated from the saved
programs, checking every existing result field, with separate new audit receipts.
The earliest tail pilot has execution evidence but no original source manifest;
its matching replay implementation is newly bound by package checksums.

See [reproduction instructions](../REPRODUCIBILITY.md) for the package layout,
verification command, original-search rerun command and environment requirements.
