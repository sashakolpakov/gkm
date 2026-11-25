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


def play_level_5(env):
    move_socket_team(
        env, (3,) * 4 + (4,) * 3 + (2,) * 3 + (3,) + (2,) * 3,
        expected_centers=((32, 26), (32, 44)))
    jump_supported_route(
        env, (((14, 26), (26, 26)), ((26, 26), (38, 26))),
        expected_pegs=((38, 26), (44, 26)))
    move_socket_team(env, (1,) * 3, expected_centers=((32, 8), (32, 26)))
    jump_supported_route(
        env, (((44, 26), (32, 26)),), expected_pegs=((32, 26),))
    move_socket_team(
        env, (2,) * 3 + (3,) * 2 + (2,) * 4 + (1,) * 4
        + (4,) * 5 + (1,) * 4 + (4,) * 10 + (2,),
        expected_centers=((14, 8), (32, 26)))
    jump_supported_route(
        env, (((32, 26), (44, 26)),),
        expected_pegs=((53, 8), (11, 26), (47, 32), (35, 56)))
    move_socket_team(
        env, (1,) + (3,) * 7 + (2,) * 2,
        expected_centers=((17, 26),))
    jump_supported_route(
        env, (((11, 26), (23, 26)), ((23, 26), (35, 26))),
        expected_pegs=((53, 8), (35, 26), (47, 32), (35, 56)))
    move_socket_team(
        env, (1,) * 2 + (4,) * 4 + (2,) * 2,
        expected_centers=((3, 26), (41, 26)))
    jump_supported_route(
        env, (((35, 26), (47, 26)), ((47, 32), (47, 20))),
        expected_pegs=((53, 8), (47, 20), (35, 56)))
    move_socket_team(
        env, (1,) * 2 + (4,), expected_centers=((3, 20), (47, 14)))
    jump_supported_route(
        env, (((47, 20), (47, 8)), ((53, 8), (41, 8)),
              ((41, 8), (29, 8))),
        expected_pegs=((29, 8), (35, 56)))
    move_socket_team(env, (3,) * 3, expected_centers=((29, 14),))
    jump_supported_route(
        env, (((29, 8), (29, 20)), ((29, 20), (29, 32))),
        expected_pegs=((29, 32), (35, 56)))
    move_socket_team(
        env, (3,) * 2 + (2,) * 4 + (4,) * 2,
        expected_centers=((29, 38),))
    jump_supported_route(
        env, (((29, 32), (29, 44)), ((29, 44), (29, 56)),
              ((35, 56), (23, 56))))


def play_level_6(env):
    jump_supported_route(
        env, (((20, 20), (20, 32)), ((20, 26), (20, 38)),
              ((20, 32), (20, 44)), ((20, 38), (20, 50)),
              ((14, 50), (26, 50)), ((26, 56), (26, 44)),
              ((20, 44), (32, 44)), ((26, 44), (38, 44)),
              ((32, 44), (44, 44)), ((38, 44), (50, 44)),
              ((24, 44), (36, 44))),
        expected_pegs=((30, 44),), expected_blue=((36, 44),))
    move_socket_team(
        env, (4,) * 10 + (1,) * 2 + (3,) * 2 + (1,) * 2,
        expected_centers=((30, 32), (48, 32)))
    jump_supported_route(
        env, (((30, 32), (30, 20)), ((30, 20), (42, 20)),
              ((48, 32), (48, 20)), ((42, 20), (54, 20))),
        expected_pegs=((10, 20), (58, 26), (34, 38), (22, 50)),
        expected_blue=((4, 20),))
    move_socket_team(
        env, (2,) * 2 + (4,) * 2 + (1,) * 4,
        expected_centers=((4, 32), (52, 20), (3, 44)))
    jump_supported_route(
        env, (((4, 20), (16, 20)), ((10, 20), (22, 20)),
              ((16, 20), (28, 20)), ((22, 20), (34, 20)),
              ((28, 20), (40, 20)), ((34, 20), (46, 20)),
              ((40, 20), (52, 20)), ((46, 20), (58, 20)),
              ((58, 20), (58, 32))),
        expected_pegs=((58, 32), (34, 38), (22, 50)),
        expected_blue=((52, 20),))
    move_socket_team(
        env, (2,) * 2, expected_centers=((3, 44), (52, 32)))
    jump_supported_route(
        env, (((58, 32), (46, 32)),),
        expected_pegs=((46, 32), (34, 38), (22, 50)),
        expected_blue=((52, 32),))
    move_socket_team(
        env, (2,) * 2 + (3,) * 2 + (1,) * 2,
        expected_centers=((40, 32),))
    jump_supported_route(
        env, (((46, 32), (34, 32)), ((34, 32), (34, 44))),
        expected_pegs=((34, 44), (22, 50)), expected_blue=((40, 32),))
    move_socket_team(env, (2,) * 2, expected_centers=((40, 44),))
    jump_supported_route(
        env, (((40, 44), (28, 44)), ((34, 44), (22, 44)),
              ((22, 44), (22, 56))))
