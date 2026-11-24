import copy
import json
from pathlib import Path

import pytest

from generate_marginal_profiles import build_profiles, direction_reversals


def inputs():
    root = Path(__file__).resolve().parents[1]
    return (json.loads((root.parent / "audit_results/marginal-literal-reuse.json").read_text()),
            json.loads((root / "generated/marginal_complexity_by_level.json").read_text()))


def test_marginals_preserve_dc22_dips_instead_of_substituting_source_growth():
    result = build_profiles(*inputs())
    dc22 = next(g for g in result["games"] if g["game"] == "dc22")
    assert dc22["M_bytes"] == [1058, 604, 716, 885, 828, 1379]
    assert dc22["direction_reversals"] == 3
    assert dc22["sharp_drop_with_static_call_levels"] == []
    assert result["summary"]["marginal_decreases"] == 64


def test_unknown_predecessors_and_unclaimed_levels_stay_missing():
    result = build_profiles(*inputs())
    profiles = {g["game"]: g for g in result["games"]}
    assert profiles["ft09"]["M_bytes"][1:3] == [None, None]
    assert profiles["lf52"]["M_bytes"][-2:] == [None, None]
    assert profiles["tr87"]["M_bytes"] == [None] * profiles["tr87"]["total_levels"]
    assert direction_reversals([1, 4, None, 3, 1]) == 0
    assert direction_reversals([1, 4, 4, 2, 3]) == 2


def test_bridge_over_missing_winning_source_is_rejected():
    audit, table = inputs()
    row = next(r for r in audit["rows"] if r["system"] == "GKM" and r["game"] == "ft09" and r["completed_level"] == 3)
    row["marginal_ast_zlib_bytes"] = 123
    with pytest.raises(ValueError, match="bridges missing source"):
        build_profiles(audit, table)


def test_duplicate_or_unauthenticated_source_is_rejected():
    audit, table = inputs()
    row = next(r for r in audit["rows"] if r["system"] == "GKM")
    audit["rows"].append(copy.deepcopy(row))
    with pytest.raises(ValueError, match="duplicate winning boundary"):
        build_profiles(audit, table)
    audit["rows"].pop()
    row["source_checkpoint_exact"] = False
    with pytest.raises(ValueError, match="not exact"):
        build_profiles(audit, table)
