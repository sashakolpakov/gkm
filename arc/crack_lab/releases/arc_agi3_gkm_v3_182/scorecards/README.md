# Official scorecards and client receipts

The admitted paths are fixed at 182/183 levels and 6,685 stored actions. Official
scores use the service's scoring rule and must not be inferred from raw coverage.
The full game inventory remains 183 levels even though lf52 stops at level 9.

## ONLINE shakedown

- Public card: https://arcprize.org/scorecards/67eba075-9415-421b-83e4-f881cebf399c
- Client runtime: `5061a897520f0e6adf88a71bae8e98cf88e51dd0`.
- All 25 game endpoints matched; the closed public card records 182 levels, 24
  complete games, 6,685 actions and a score of 99.27272727272728.
- `online-run.json` is the unmodified original collector receipt. Its FAIL status
  came from applying the complete-183 postflight contract to a partial-182 card.
  The service also appends an `agent` tag and does not expose the client source
  URL or opaque value in the public record.
- `online-public.json` is the closed public record. The separate
  `online-public-verification.json` checks it with the partial-release contract;
  it does not rewrite the original collector receipt or open a second card.

The partial validator was published before Competition replay. It checks the
receipt's full inventory, each admitted endpoint, provider run states and actions,
zero actions beyond the unclaimed frontier, total completed levels/games, operation
mode and tags. The complete-183 validator remains unchanged. A missing source URL
in the public API is not represented as a server attestation: source binding comes
from the client command, runtime hashes, frozen tree and release receipt.

## Competition replay

The Competition replay uses the same immutable action paths, with no model calls
or local clone lookahead. Its client runtime is
`a5b301d328d04d4b771bd36dd299b79939a93c36`.
The closed [Competition scorecard](https://arcprize.org/scorecards/fb027490-5351-4a77-85f2-41959e3e6b64) records **99.27272727%**,
182/183 levels, 24 complete games and 6,685 actions. `competition-run.json` is the
unmodified client receipt. Its FAIL status records rejection of the SDK close payload
(nested-run accounting, ar25), despite all 25 endpoints matching and a confirmed close.
`competition-public.json` is the separately fetched closed public record, and
`competition-public-verification.json` validates that record independently.
