# Per-level players only compose reusable legs.
from legs import solve_peg_jumps, solve_peg_ferry, move_socket_team, jump_supported_route, jump_clipped_helper


def play_level_1(env):
    solve_peg_jumps(env)


def play_level_2(env):
    solve_peg_ferry(env)


def play_level_3(env):
    jump_supported_route(env, (((14, 14), (14, 26)), ((14, 26), (26, 26)), ((26, 26), (26, 14))), expected_pegs=((26, 14), (32, 14), (62, 14), (14, 50), (26, 50)), expected_blue=())
    move_socket_team(env, (3,), expected_centers=((38, 14), (32, 50)))
    jump_supported_route(env, (((26, 14), (38, 14)),), expected_pegs=((38, 14), (62, 14), (14, 50), (26, 50)), expected_blue=())
    move_socket_team(env, (4, 4, 1, 1, 4), expected_centers=((32, 14), (26, 38)))
    jump_supported_route(env, (((56, 20), (44, 20)), ((56, 32), (44, 32)), ((32, 14), (44, 14)), ((44, 14), (44, 26)), ((44, 26), (44, 38)), ((44, 38), (44, 50))), expected_pegs=((2, 50), (44, 50), (50, 50)), expected_blue=())
    move_socket_team(env, (4, 2, 2, 4), expected_centers=((32, 14), (38, 50)))
    jump_supported_route(env, (((50, 50), (38, 50)),), expected_pegs=((2, 50), (38, 50)), expected_blue=())
    move_socket_team(env, (3, 1, 1, 3, 3, 2, 2, 3, 3), expected_centers=((38, 14), (32, 50)))
    jump_supported_route(env, (((32, 50), (20, 50)), ((20, 50), (8, 50))))


def play_level_4(env):
    # Recovered from verified proposer path artifact: checkpoint.json+/Users/sasha/gkm/output/arc_agi3_gpt61_zero_seed_20261116T144404Z/source/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_arc_agi3_n0_fresh_frontier_fty5uh_g/candidate_level4_verified.json
    for action in [[6, 14, 26], [6, 26, 26], [6, 26, 26], [6, 38, 26], 3, [6, 38, 26], [6, 50, 26], 4, 4, 4, 4, [6, 20, 26], [6, 32, 26], [6, 32, 26], [6, 32, 38], [6, 32, 38], [6, 32, 50], [6, 50, 26], [6, 50, 38], [6, 50, 38], [6, 50, 50], [6, 50, 50], [6, 38, 50], [6, 38, 50], [6, 26, 50], 3, 3, 3, 3, 2, 2, 4, 4, 4, [6, 41, 20], [6, 41, 32], 3, 3, [6, 29, 32], [6, 29, 20], [6, 29, 20], [6, 17, 20], 3, 3, [6, 17, 20], [6, 17, 32], [6, 17, 32], [6, 17, 44], [6, 17, 44], [6, 29, 44]]:
        env.step(action)


def play_level_5(env):
    move_socket_team(env, (4, 4, 4, 3, 2, 2, 2, 3, 3, 2, 2, 2, 4, 4, 2, 4, 1), expected_centers=((32, 26), (50, 38)))
    jump_supported_route(env, (((14, 26), (26, 26)), ((26, 26), (38, 26))), expected_pegs=((38, 26), (44, 26)), helper_colors=(8,))
    move_socket_team(env, (3, 3, 3, 3, 3, 1, 1), expected_centers=((32, 14), (50, 26)))
    jump_supported_route(env, (((38, 26), (50, 26)),), expected_pegs=((32, 26), (56, 26)), helper_colors=(8,))
    move_socket_team(env, (1, 4, 4, 4, 2), expected_centers=((14, 8), (32, 26), (50, 26)))
    jump_supported_route(env, (((32, 26), (44, 26)), ((11, 26), (23, 26))), expected_pegs=((53, 8), (23, 26), (47, 32), (35, 56)), helper_colors=(8,))
    move_socket_team(env, (1, 1, 4, 4, 4, 4, 2, 2), expected_centers=((3, 26), (41, 26)))
    jump_supported_route(env, (((23, 26), (35, 26)), ((35, 26), (47, 26)), ((47, 32), (47, 20))), expected_pegs=((53, 8), (47, 20), (35, 56)), helper_colors=(8,))
    move_socket_team(env, (1, 1, 4), expected_centers=((47, 14), (3, 20)))
    jump_supported_route(env, (((47, 20), (47, 8)), ((53, 8), (41, 8))), expected_pegs=((41, 8), (35, 56)), helper_colors=(8,))
    move_socket_team(env, (3, 3, 3), expected_centers=((29, 14),))
    jump_supported_route(env, (((41, 8), (29, 8)), ((29, 8), (29, 20))), expected_pegs=((29, 20), (35, 56)), helper_colors=(8,))
    move_socket_team(env, (3, 3, 2, 2, 2, 2, 4, 4), expected_centers=((29, 38),))
    jump_supported_route(env, (((29, 20), (29, 32)), ((29, 32), (29, 44)), ((29, 44), (29, 56)), ((35, 56), (23, 56))), helper_colors=(8,))


def play_level_6(env):
    move_socket_team(env, (2, 2), expected_centers=((50, 44),))
    jump_supported_route(env, (((20, 20), (20, 32)), ((20, 26), (20, 38)), ((20, 32), (20, 44)), ((20, 38), (20, 50)), ((14, 50), (26, 50)), ((26, 56), (26, 44)), ((20, 44), (32, 44)), ((26, 44), (38, 44)), ((32, 44), (44, 44)), ((38, 44), (50, 44)), ((24, 44), (36, 44))), expected_pegs=((30, 44),), helper_colors=(8,))
    move_socket_team(env, (4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 1, 1, 1), expected_centers=((30, 32), (48, 32)))
    jump_supported_route(env, (((30, 32), (30, 20)), ((30, 20), (42, 20)), ((48, 32), (48, 20)), ((42, 20), (54, 20)), ((4, 20), (16, 20)), ((10, 20), (22, 20)), ((16, 20), (28, 20)), ((22, 20), (34, 20)), ((28, 20), (40, 20)), ((34, 20), (46, 20)), ((40, 20), (52, 20)), ((46, 20), (58, 20)), ((58, 20), (58, 32))), expected_pegs=((58, 32), (34, 38), (22, 50)), helper_colors=(8,))
    move_socket_team(env, (2, 2), expected_centers=((52, 32), (4, 44)))
    jump_supported_route(env, (((58, 32), (46, 32)),), expected_pegs=((46, 32), (34, 38), (22, 50)), helper_colors=(8,))
    move_socket_team(env, (2, 2, 3, 3, 1, 1), expected_centers=((40, 32),))
    jump_supported_route(env, (((46, 32), (34, 32)), ((34, 32), (34, 44))), expected_pegs=((34, 44), (22, 50)), helper_colors=(8,))
    move_socket_team(env, (2, 2), expected_centers=((40, 44),))
    jump_supported_route(env, (((40, 44), (28, 44)), ((34, 44), (22, 44)), ((22, 44), (22, 56))), helper_colors=(8,))


def play_level_7(env):
    move_socket_team(env, (3, 3, 1, 1, 4, 4, 4), expected_centers=((44, 26),))
    jump_supported_route(env, (((44, 14), (44, 26)),), expected_pegs=((8, 14),), helper_colors=(8,))
    move_socket_team(env, (3, 3, 3, 2, 2, 3, 3, 2), expected_centers=((62, 32), (14, 44)))
    jump_supported_route(env, (((14, 44), (14, 56)), ((14, 56), (26, 56)), ((26, 56), (38, 56))), expected_pegs=((8, 14),), helper_colors=(8,))
    move_socket_team(env, (2, 1, 4, 2, 4, 1, 1, 3, 3, 3), expected_centers=((8, 26), (56, 38)))
    jump_supported_route(env, (((8, 14), (8, 26)),), expected_pegs=((8, 26),), helper_colors=(8,))
    move_socket_team(env, (2, 4, 4, 4, 2, 2, 4, 4, 4, 2), expected_centers=((44, 44),))
    jump_supported_route(env, (((44, 44), (44, 56)), ((38, 56), (50, 56)), ((44, 56), (56, 56)), ((6, 56), (18, 56)), ((12, 56), (24, 56)), ((18, 56), (30, 56))), expected_pegs=((24, 56),), helper_colors=(8,))
    move_socket_team(env, (1, 3, 3, 1, 1, 3, 4, 1, 1, 4, 4, 4), expected_centers=((36, 26), (3, 38)))
    jump_supported_route(env, (((36, 26), (48, 26)),), expected_pegs=((48, 26),), helper_colors=(8,))
    move_socket_team(env, (3, 3, 3, 2, 2, 4, 4, 2), expected_centers=((30, 44),))
    jump_supported_route(env, (((30, 56), (30, 44)),), expected_pegs=((48, 26),), helper_colors=(8,))
    move_socket_team(env, (3, 1, 3, 3, 1, 1, 4, 4, 1, 1, 4, 4, 2, 4, 2), expected_centers=((48, 20), (3, 44)))
    jump_supported_route(env, (((48, 20), (48, 32)), ((48, 26), (48, 38)), ((48, 38), (60, 38)), ((20, 38), (32, 38)), ((32, 38), (32, 26))), expected_pegs=((32, 26), (56, 44)), helper_colors=(8,))
    move_socket_team(env, (1, 4, 4, 4, 4, 2), expected_centers=((8, 20), (32, 32), (44, 32)))
    jump_supported_route(env, (((56, 26), (56, 38)),), expected_pegs=((56, 38), (56, 44)), helper_colors=(8,))
    move_socket_team(env, (1, 3, 1, 3, 3, 2, 4, 2, 2), expected_centers=((20, 26), (32, 32), (56, 32), (44, 38)))
    jump_supported_route(env, (((56, 44), (56, 32)),), helper_colors=(8,))


def play_level_8(env):
    move_socket_team(
        env, (3, 3, 1, 1, 1, 1, 4), expected_centers=((14, 38),))
    jump_supported_route(
        env, (((14, 26), (26, 26)), ((32, 50), (32, 38)),
              ((32, 44), (32, 32)), ((32, 38), (32, 26)),
              ((26, 26), (38, 26)), ((32, 26), (32, 38)),
              ((38, 26), (50, 26)), ((50, 26), (50, 38)),
              ((50, 38), (38, 38)), ((38, 38), (26, 38)),
              ((26, 38), (14, 38))),
        helper_colors=(9,), expected_helpers=((32, 32), (32, 38)),
        expected_pegs=((14, 38), (56, 50)))
    move_socket_team(
        env, (3, 2, 2, 2, 2, 4, 4, 4, 2, 3, 3, 3),
        expected_centers=((26, 38), (50, 38), (20, 50)))
    jump_supported_route(
        env, (((56, 44), (44, 44)), ((50, 44), (38, 44)),
              ((44, 44), (32, 44)), ((38, 44), (26, 44)),
              ((26, 38), (26, 50))),
        helper_colors=(9,), expected_pegs=((26, 50), (56, 20)))
    jump_supported_route(
        env, (((26, 44), (38, 44)), ((32, 44), (44, 44)),
              ((38, 44), (50, 44))),
        helper_colors=(9,), expected_pegs=((26, 50), (56, 20)))
    move_socket_team(
        env, (4, 4, 4, 4),
        expected_centers=((26, 38), (50, 38), (44, 50)))
    jump_supported_route(
        env, (((50, 50), (50, 38)),),
        helper_colors=(9,), expected_pegs=((50, 38), (56, 20)))
    move_socket_team(
        env, (1, 4, 1), expected_centers=((56, 38), (26, 44)))
    jump_supported_route(
        env, (((56, 38), (56, 26)),), helper_colors=(9,))


def play_level_9(env):
    jump_supported_route(env, (((20, 44), (32, 44)), ((26, 50), (26, 38)), ((26, 44), (26, 32)), ((26, 38), (26, 26)), ((26, 32), (26, 20)), ((26, 20), (38, 20)), ((38, 20), (38, 32)), ((38, 26), (38, 38)), ((38, 32), (38, 44)), ((38, 38), (38, 50)), ((44, 50), (32, 50)), ((32, 50), (32, 38)), ((38, 50), (38, 38)), ((32, 38), (44, 38))), helper_colors=(9,), expected_pegs=((54, 14), (24, 38)))
    move_socket_team(env, (4, 4, 4, 4, 4, 4, 4, 4), expected_centers=((24, 38),))
    jump_clipped_helper(env, (63, 20), (54, 20), expected_helpers=((54, 20), (60, 20)))
    jump_supported_route(env, (((60, 20), (48, 20)), ((54, 20), (42, 20)), ((48, 20), (36, 20)), ((42, 20), (30, 20)), ((36, 20), (24, 20)), ((30, 20), (18, 20)), ((24, 20), (12, 20)), ((18, 20), (6, 20)), ((6, 14), (6, 26)), ((6, 26), (18, 26)), ((6, 20), (18, 20)), ((12, 20), (24, 20)), ((18, 26), (18, 14)), ((18, 14), (30, 14)), ((30, 14), (42, 14)), ((18, 20), (30, 20)), ((24, 20), (36, 20)), ((30, 20), (42, 20)), ((36, 20), (48, 20)), ((42, 14), (42, 26)), ((42, 26), (54, 26))), helper_colors=(9,), expected_pegs=((54, 26), (24, 38)))
    move_socket_team(env, (4, 4, 4), expected_centers=((24, 38),))
    jump_supported_route(env, (((36, 26), (48, 26)), ((24, 20), (36, 20)), ((30, 20), (42, 20)), ((36, 20), (48, 20)), ((48, 20), (48, 32)), ((48, 26), (48, 38))), helper_colors=(9,), expected_pegs=((24, 38), (48, 38)))
    move_socket_team(env, (4, 4, 4), expected_centers=((24, 38),))
    jump_supported_route(env, (((24, 38), (36, 38)),), helper_colors=(9,), expected_pegs=((32, 6), (44, 60)))
