# Recorded proposer tokens and API-equivalent cost

The reporting cutoff is execution 454 (`gkm_legs_ws_lf52_arc_agi3_n125_long_coherence_reset_fm780fkp`). It is retrospective; subsequent plateau attempts remain in the accounting.

| Scope | Turns | Input (includes cached) | Cached input | Output (includes reasoning) | Reasoning | Standard USD | All-long-context USD | Missing usage turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| through_cutoff | 454 | 1,797,570,821 | 1,758,867,584 | 22,886,441 | 16,049,437 | 482.16 | 849.88 | 3 |
| after_cutoff | 48 | 298,214,453 | 293,016,448 | 3,739,957 | 2,662,526 | 77.10 | 135.49 | 0 |
| all_recorded | 502 | 2,095,785,274 | 2,051,884,032 | 26,626,398 | 18,711,963 | 559.25 | 985.38 | 3 |

Each codex_exec records cumulative usage for one proposer turn, potentially spanning many API requests. Count that execution once, including failed, discarded and infrastructure work; corrections/outcomes are not extra executions. Cached input is a subset of input and reasoning is a subset of output; neither is added twice. Host orchestration, manuscript editing, local compute and later API action replay are outside proposer-token accounting.

API-equivalent valuation of reported tokens, not an invoice. Per-request context sizes are not in the CLI capture; both standard and all-long-context scenarios are shown. Cache writes use the reported counter. Unknown usage is not valued at zero.

3 historical captures have no terminal usage report. Their counters are unknown, not zero-cost executions. The scenarios price only reported usage; they are not bounds on the missing tokens.

[Official GPT-6.1 Sol prices](https://developers.openai.com/api/docs/models/gpt-6.1-sol) were checked on 2026-11-25. Rates per million tokens: input $2, cached input $0.10, 30-minute cache writes $2.50, output $10. Reasoning is already included in output. No Batch/Flex/Fast discount or premium is assumed.
