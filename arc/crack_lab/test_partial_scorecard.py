import copy
import json
from pathlib import Path
import pytest
import replay_scorecard as R


@pytest.fixture
def case():
    root = Path(__file__).parent / "releases/arc_agi3_gkm_v3_182"
    raw = json.loads((root / "scorecards/online-public.json").read_text())
    receipt = json.loads(next((root / "receipts").glob("*.json")).read_text())
    games = sorted(receipt["inventory"])
    plan = {g: json.loads((root / "artifacts" / (g+"_legs") / "checkpoint.json").read_text()) for g in games}
    args = dict(mode="online", card_id=raw["card_id"], source_url="https://github.com/sashakolpakov/gkm/tree/5061a897520f0e6adf88a71bae8e98cf88e51dd0", games=games, plan=plan, scorecard_tags=["gkm","gpt61","zero-seed","v3-182","cutoff-n125"], scorecard_opaque=None, inventory=receipt["inventory"])
    return R.normalize_partial_public_scorecard(raw), args


def test_real_partial_public_card_keeps_full_inventory(case):
    card, args = case
    assert R.validate_partial_closed_scorecard(card, **args) is card
    assert card["total_levels"] == 183 and card["total_levels_completed"] == 182
    assert card["total_environments_completed"] == 24
    assert card["source_url"] is None  # The public API does not echo this field.


@pytest.mark.parametrize("change", ["mode", "source", "tags", "total", "unclaimed", "extra_action", "missing_game"])
def test_partial_rejects_inconsistent_public_evidence(case, change):
    card, args = case
    if change == "mode": args["mode"] = "competition"
    elif change == "source": card["source_url"] = "https://example.com/wrong-source"
    elif change == "tags": card["tags"].append("unexpected")
    elif change == "total": card["total_levels"] = 182
    elif change == "missing_game": card["environments"].pop()
    elif change == "extra_action": card["environments"][0]["actions"] += 1
    elif change == "unclaimed":
        row = next(r for r in card["environments"] if r["id"].startswith("lf52"))
        row["runs"][-1]["level_actions"][-1] = 1
    with pytest.raises(ValueError): R.validate_partial_closed_scorecard(card, **args)


def test_complete_contract_still_requires_terminal_win(case):
    card, args = case
    row = next(r for r in card["environments"] if r["id"].startswith("lf52"))
    with pytest.raises(ValueError): R._validate_complete_environment_score(row, game="lf52", target=10)
