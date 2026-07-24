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
