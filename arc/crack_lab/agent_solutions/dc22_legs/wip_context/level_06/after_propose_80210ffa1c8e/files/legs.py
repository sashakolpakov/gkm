# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.


def walk_segments(env, segments):
    """Walk a sequence of ``(direction, distance)`` path segments."""
    for direction, distance in segments:
        for _ in range(distance):
            env.step(direction)


def toggle_control(env, point):
    """Click a coordinate control at an ``(x, y)`` point."""
    env.step(6, point[0], point[1])


def traverse_two_stage_bridge_chain(
        env, lower_control, upper_control,
        entry_segments, pivot_segments, exit_segments):
    """Cross two reconfigurable bridges without stranding the avatar.

    The lower bridge is first moved to connect the starting island.  Once the
    avatar reaches the far pivot, the upper bridge is raised and the lower one
    is restored to connect the exit island.
    """
    toggle_control(env, lower_control)
    walk_segments(env, entry_segments)
    walk_segments(env, pivot_segments)
    toggle_control(env, upper_control)
    toggle_control(env, lower_control)
    walk_segments(env, exit_segments)


def traverse_revealed_three_stage_bridge_chain(
        env, first_control, second_control, revealed_control,
        first_pivot_segments, reveal_segments, retreat_segments,
        return_to_first_segments, final_pivot_segments, exit_segments):
    """Reveal and cross a third bridge beyond two reconfigurable bridges.

    The first two bridges are restored during the return trip so the avatar
    can reach the final bridge while it is still in its traversable state.
    """
    toggle_control(env, first_control)
    walk_segments(env, first_pivot_segments)
    toggle_control(env, second_control)
    walk_segments(env, reveal_segments)
    walk_segments(env, retreat_segments)
    toggle_control(env, second_control)
    walk_segments(env, return_to_first_segments)
    toggle_control(env, first_control)
    walk_segments(env, final_pivot_segments)
    toggle_control(env, revealed_control)
    walk_segments(env, exit_segments)


def traverse_teleport_revealed_exit_chain(
        env, bridge_control, connector_control, teleport_control,
        revealed_exit_control, initial_approach_segments,
        first_crossing_segments, connector_entry_segments,
        teleport_approach_segments, endpoint_segments,
        activation_segments, exit_staging_segments, exit_segments):
    """Cross reconfigurable bridges, teleport, and use a revealed exit.

    The connector is restored before entering the teleporter.  Activating the
    object on the remote island reveals the final control; the original bridge
    is then reoriented once more to connect that control's exit path.
    """
    walk_segments(env, initial_approach_segments)
    toggle_control(env, bridge_control)
    toggle_control(env, connector_control)
    walk_segments(env, first_crossing_segments)
    toggle_control(env, bridge_control)
    walk_segments(env, connector_entry_segments)
    toggle_control(env, connector_control)
    walk_segments(env, teleport_approach_segments)
    toggle_control(env, connector_control)
    walk_segments(env, endpoint_segments)
    toggle_control(env, teleport_control)
    walk_segments(env, activation_segments)
    walk_segments(env, exit_staging_segments)
    toggle_control(env, bridge_control)
    toggle_control(env, revealed_exit_control)
    walk_segments(env, exit_segments)


def traverse_synchronized_builder_teleport_chain(
        env, bridge_control, shuttle_control, teleport_control,
        initial_approach_segments, upper_build_direction, upper_build_phases,
        upper_crossing_segments, bridge_restore_phases,
        shuttle_completion_phases, shuttle_crossing_segments,
        teleport_entry_segments, shuttle_reset_phases,
        lower_approach_segments, lower_build_direction, lower_build_phases,
        exit_segments):
    """Traverse paired incremental bridges joined by a remote teleporter.

    Each builder phase must be followed immediately by an avatar step so the
    emerging bridge remains occupied while it advances.  The synchronized
    shuttle is completed to reach the teleporter, reset after teleporting, and
    then advanced beneath the avatar to connect the final bridge.
    """
    walk_segments(env, initial_approach_segments)
    for _ in range(upper_build_phases):
        toggle_control(env, bridge_control)
        walk_segments(env, ((upper_build_direction, 1),))
    walk_segments(env, upper_crossing_segments)
    for _ in range(bridge_restore_phases):
        toggle_control(env, bridge_control)
    for _ in range(shuttle_completion_phases):
        toggle_control(env, shuttle_control)
    walk_segments(env, shuttle_crossing_segments)
    walk_segments(env, teleport_entry_segments)
    toggle_control(env, teleport_control)
    for _ in range(shuttle_reset_phases):
        toggle_control(env, shuttle_control)
    walk_segments(env, lower_approach_segments)
    for _ in range(lower_build_phases):
        toggle_control(env, shuttle_control)
        walk_segments(env, ((lower_build_direction, 1),))
    walk_segments(env, exit_segments)


def clear_level_5_platform_transfer(
        env, builder_control=(46, 22), crossing_control=(56, 22),
        bridge_control=(52, 42), teleport_control=(52, 46),
        platform_up_control=(50, 28), platform_right_control=(56, 28),
        platform_rotate_control=(51, 35), assembly_left_control=(44, 30),
        assembly_down_control=(60, 30), assembly_right_control=(54, 30),
        assembly_up_control=(50, 30), lower_rotate_control=(52, 14)):
    """Route the movable assembly through both docks and the lower rotator."""
    for _ in range(3):
        toggle_control(env, platform_up_control)
    for _ in range(3):
        toggle_control(env, platform_right_control)
    toggle_control(env, platform_rotate_control)
    for _ in range(3):
        toggle_control(env, assembly_left_control)
    for _ in range(3):
        toggle_control(env, assembly_down_control)
    for _ in range(3):
        toggle_control(env, assembly_right_control)

    walk_segments(env, ((3, 1),))
    toggle_control(env, bridge_control)
    walk_segments(env, ((3, 1), (1, 2), (3, 2), (1, 4), (4, 1), (1, 1)))

    for _ in range(3):
        toggle_control(env, assembly_left_control)
    for _ in range(3):
        toggle_control(env, assembly_up_control)
    for _ in range(3):
        toggle_control(env, assembly_right_control)
    walk_segments(env, ((3, 1), (1, 7)))
    toggle_control(env, teleport_control)

    for _ in range(3):
        toggle_control(env, assembly_left_control)
    for _ in range(3):
        toggle_control(env, assembly_down_control)
    walk_segments(env, ((1, 8),))
    for _ in range(2):
        toggle_control(env, assembly_up_control)
    walk_segments(env, ((1, 2),))
    toggle_control(env, assembly_up_control)
    walk_segments(env, ((1, 1),))
    toggle_control(env, lower_rotate_control)

    walk_segments(env, ((2, 1),))
    toggle_control(env, assembly_down_control)
    walk_segments(env, ((2, 2),))
    for _ in range(2):
        toggle_control(env, assembly_down_control)
    walk_segments(env, ((2, 8), (2, 1), (4, 1)))
    toggle_control(env, bridge_control)
    walk_segments(env, ((2, 4),))

    for _ in range(4):
        toggle_control(env, crossing_control)
        walk_segments(env, ((4, 1),))
    walk_segments(env, ((4, 4), (2, 4), (3, 1)))
    for _ in range(4):
        toggle_control(env, builder_control)
    walk_segments(env, ((2, 1), (3, 2)))
    toggle_control(env, lower_rotate_control)
    walk_segments(env, ((1, 2), (3, 6)))
