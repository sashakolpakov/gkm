# sc25 WIP artifact

No replay-validated level was promoted before the bounded attempt reached the
API usage limit. The `wip_context/` tree preserves the clean level-1 probes,
transcript, candidate solver, and credit-out record.

This WIP is restart context, not solver evidence. A future continuation should
restore its non-promoted probes, then independently replay and pass the taint
guard before promoting any level.
