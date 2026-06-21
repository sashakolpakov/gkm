"""Hermetic tests for the consolidated GKM cracker (gkm_crack.py).

Offline, no network/LLM (use_llm=False). The sequential L1->L2 crack is proven by
replaying the discovered action path on a fresh env via the public interface.
"""
import time
import gkm_crack as G


# The from-scratch L1->L2 path discovered by `crack(use_llm=False)` (deterministic).
L1L2_PATH = [1, 1, 5, 1, 1, 5, 4, 1, 1, 4, 5, 3, 3, 3, 3, 2, 5, 3, 5, 4, 5, 1, 3, 2, 5,
             4, 2, 1, 1, 4, 4, 4, 2, 5, 2, 2, 2, 4, 4, 4, 4, 4, 4, 5, 2, 3, 3, 3, 3, 3,
             3, 3, 2, 5, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             2, 5, 2, 4, 1, 5, 2, 2, 2, 4, 4, 1, 5]


def test_connector_built_from_scratch():
    """Connector is built by anchor + container percept + interaction probe, with
    NO game constants typed in. On wa30 it grounds the pick-up-and-carry mechanic."""
    c = G.CarryConnector.build("wa30", use_llm=False, verbose=False)
    assert c.B.avatar == 14            # anchor (controllable object)
    assert c.B.carrier == 9            # movable carrier colour
    assert c.B.region == 2             # target interior colour
    assert c.B.mechanic == "pick_up_and_carry"
    assert c.B.carried_border != c.B.rest_border    # learned border states differ


def test_l1_l2_path_validates():
    """Replaying the discovered path on a fresh env reaches level 2 (the crack)."""
    c = G.CarryConnector.build("wa30", use_llm=False, verbose=False)
    assert c.validate(L1L2_PATH, 2) is True


def test_sequential_crack_reaches_level_2():
    """End-to-end: the abstract cone, driven only by the connector, cracks L1->L2
    from scratch and replay-validates. (Runs the search; ~minutes.)"""
    out = G.crack("wa30", max_level=2, use_llm=False, verbose=False)
    assert out["reached"] >= 2
    assert out["validated"] is True


def test_discovery_phase_grounds_move_and_carry():
    """The discovery phase learns the action semantics by interaction (no LLM):
    move(1-4) and pick_up_and_carry(5) on wa30 L1."""
    import discovery
    verbs, effects, w = discovery.discover("wa30", use_llm=False, verbose=False)
    assert set(verbs) == {"move", "pick_up_and_carry"}
    assert verbs["move"]["actions"] == [1, 2, 3, 4]
    assert verbs["pick_up_and_carry"]["actions"] == [5]
