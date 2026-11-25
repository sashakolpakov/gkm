# Per-level players only compose reusable legs.
from legs import solve_peg_jumps, solve_peg_ferry, move_socket_team, jump_supported_route


def play_level_1(env):
    solve_peg_jumps(env)


def play_level_2(env):
    solve_peg_ferry(env)


def play_level_3(env):
    move_socket_team(env, (3,), expected_centers=((38, 14), (32, 50)))
    solve_peg_jumps(
        env, extra_sockets=((38, 14),),
        target_pegs=((38, 14), (62, 14), (14, 50), (26, 50)))
    move_socket_team(env, (4, 4, 4), expected_centers=((32, 14), (20, 50)))
    solve_peg_jumps(
        env, target_pegs=((44, 14), (50, 20), (56, 20), (50, 32),
                          (56, 32), (44, 44), (2, 50), (50, 50)))
    move_socket_team(
        env, (1, 1, 4, 4, 2, 2, 4),
        expected_centers=((32, 14), (38, 50)))
    solve_peg_jumps(
        env, extra_sockets=((38, 50),), target_pegs=((2, 50), (38, 50)))
    move_socket_team(
        env, (3, 1, 1, 3, 3, 2, 2, 3, 3),
        expected_centers=((38, 14), (32, 50)))
    solve_peg_jumps(env)


def play_level_4(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_arc_agi3_n0_fresh_frontier_fty5uh_g/candidate_level4_verified.json
    for action in [[6, 14, 26], [6, 26, 26], [6, 26, 26], [6, 38, 26], 3, [6, 38, 26], [6, 50, 26], 4, 4, 4, 4, [6, 20, 26], [6, 32, 26], [6, 32, 26], [6, 32, 38], [6, 32, 38], [6, 32, 50], [6, 50, 26], [6, 50, 38], [6, 50, 38], [6, 50, 50], [6, 50, 50], [6, 38, 50], [6, 38, 50], [6, 26, 50], 3, 3, 3, 3, 2, 2, 4, 4, 4, [6, 41, 20], [6, 41, 32], 3, 3, [6, 29, 32], [6, 29, 20], [6, 29, 20], [6, 17, 20], 3, 3, [6, 17, 20], [6, 17, 32], [6, 17, 32], [6, 17, 44], [6, 17, 44], [6, 29, 44]]:
        env.step(action)
