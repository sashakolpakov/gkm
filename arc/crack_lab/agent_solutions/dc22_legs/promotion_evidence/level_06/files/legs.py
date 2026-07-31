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


def route_ring_with_dpad(
        env, directions,
        up_control=(50, 32), down_control=(50, 40),
        left_control=(46, 36), right_control=(54, 36)):
    """Move a remote ring while returning the avatar to the D-pad center.

    Each direction is a synchronized three-action phase: enter the matching
    selector, click its coordinate control, then return to the center.  The
    right selector is reached through its paired portal, but obeys the same
    approach/control/return contract.
    """
    phases = {
        1: (1, up_control, 2),
        2: (2, down_control, 1),
        3: (3, left_control, 4),
        4: (4, right_control, 3),
    }
    for direction in directions:
        approach, control, retreat = phases[direction]
        walk_segments(env, ((approach, 1),))
        toggle_control(env, control)
        walk_segments(env, ((retreat, 1),))


def traverse_multidock_ring_transfer_chain(
        env,
        shuttle_control=(55, 7),
        network_control=(51, 25),
        hub_cycle_control=(50, 48),
        ring_up_control=(50, 32),
        ring_down_control=(50, 40),
        ring_left_control=(46, 36),
        ring_right_control=(54, 36),
        dock_transform_control=(50, 18),
        initial_ring_route=(3, 3, 3, 3),
        loaded_ring_route=(4, 4, 4, 1, 4, 4, 1, 1, 3, 3, 1, 1, 1)):
    """Reveal, load, and route a ring through two cooperating dock regions.

    The network control doubles as the checker-pad teleporter.  The shuttle
    first exposes the hub cycler, which retunes the central checker to reach
    otherwise isolated controller regions.  The ring is routed to the bridge
    pivot, transformed while the avatar occupies that dock, then routed to
    the upper dock for the final crossing.
    """
    # Visit the paired 6/7 island and reveal the remote ring indicator.
    walk_segments(env, ((1, 2), (3, 5)))
    toggle_control(env, network_control)
    walk_segments(env, ((1, 2), (4, 1), (3, 1), (2, 2)))
    toggle_control(env, network_control)
    walk_segments(env, ((4, 5), (2, 2)))

    # Carry the shuttle from the start platform to the lower-left region.
    for _ in range(5):
        toggle_control(env, shuttle_control)
    walk_segments(
        env,
        ((3, 4), (2, 1), (3, 1), (2, 1),
         (3, 2), (2, 1), (3, 2), (1, 1)),
    )
    toggle_control(env, shuttle_control)
    walk_segments(env, ((1, 6),))

    # Raise the network bridge and reveal the checker hub control.
    toggle_control(env, network_control)
    walk_segments(env, ((1, 13), (3, 1)))
    for _ in range(2):
        toggle_control(env, hub_cycle_control)

    # Return through the shuttle, then visit the 11/12 and 9/10 pads.
    walk_segments(env, ((4, 1), (2, 19)))
    for _ in range(3):
        toggle_control(env, shuttle_control)
    walk_segments(env, ((4, 1),))
    toggle_control(env, shuttle_control)
    walk_segments(env, ((4, 1), (1, 1), (4, 2), (1, 3)))
    toggle_control(env, network_control)
    toggle_control(env, network_control)
    walk_segments(env, ((4, 1),))
    toggle_control(env, hub_cycle_control)
    walk_segments(env, ((3, 1),))
    toggle_control(env, network_control)

    # Route the empty ring to the bridge pivot.
    route_ring_with_dpad(
        env,
        initial_ring_route,
        up_control=ring_up_control,
        down_control=ring_down_control,
        left_control=ring_left_control,
        right_control=ring_right_control,
    )
    toggle_control(env, network_control)

    # Re-enter the bridge pivot and load its center into the docked ring.
    walk_segments(env, ((2, 3), (3, 2), (2, 1), (3, 1), (4, 1)))
    toggle_control(env, shuttle_control)
    walk_segments(env, ((2, 1), (3, 1)))
    toggle_control(env, shuttle_control)
    walk_segments(env, ((1, 1), (3, 1), (1, 11)))
    toggle_control(env, dock_transform_control)

    # Return to the D-pad with the loaded assembly.
    walk_segments(env, ((2, 11),))
    for _ in range(3):
        toggle_control(env, shuttle_control)
    walk_segments(env, ((4, 1),))
    toggle_control(env, shuttle_control)
    walk_segments(env, ((4, 1), (1, 1), (4, 2), (1, 3)))
    toggle_control(env, network_control)

    # Route the loaded assembly to the upper dock.
    route_ring_with_dpad(
        env,
        loaded_ring_route,
        up_control=ring_up_control,
        down_control=ring_down_control,
        left_control=ring_left_control,
        right_control=ring_right_control,
    )

    # Retune the hub to 11/12, teleport to the upper island, and exit.
    toggle_control(env, network_control)
    walk_segments(env, ((4, 1),))
    for _ in range(3):
        toggle_control(env, hub_cycle_control)
    walk_segments(env, ((3, 1),))
    toggle_control(env, network_control)
    walk_segments(env, ((2, 2), (4, 14), (1, 1), (4, 7)))
