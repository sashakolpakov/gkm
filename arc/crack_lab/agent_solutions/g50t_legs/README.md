# g50t WIP artifact

No replay-validated level has been promoted for `g50t` yet. The
`wip_context/` tree preserves the bounded level-1 attempt, including probes,
transcripts, blocked commands, and the unsuccessful candidate solver.

A normal continuation restores the non-promoted probe context from these
snapshots. Candidate `legs.py`, `players.py`, and checkpoints remain untrusted
until a fresh replay reaches the claimed level and promotion passes the taint
guard.
