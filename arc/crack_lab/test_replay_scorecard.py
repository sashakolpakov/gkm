import pytest

from replay_scorecard import decode_action


def test_decode_action_supports_keys_and_coordinate_tokens():
    assert decode_action(4) == (4, None)
    assert decode_action([6, 38, 17]) == (6, {"x": 38, "y": 17})
    assert decode_action((6, 0, 63)) == (6, {"x": 0, "y": 63})


@pytest.mark.parametrize("action", ([5, 1, 2], [6, 1], 0, 8))
def test_decode_action_rejects_invalid_tokens(action):
    with pytest.raises(ValueError):
        decode_action(action)
