# Per-level players only compose reusable legs.
from legs import solve_peg_jumps, solve_peg_ferry, move_socket_team


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
