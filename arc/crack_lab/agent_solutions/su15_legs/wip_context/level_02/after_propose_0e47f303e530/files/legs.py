# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
from perception import connected_components


def follow_diagonal_lattice_to_ring(env, step=6, max_moves=12):
    """Move the playfield avatar diagonally toward the ringed target."""
    start_level = env.levels_completed
    for _ in range(max_moves):
        if env.terminal() or env.levels_completed != start_level:
            return

        frame = env.frame()
        avatars = [
            blob for blob in connected_components(frame, colors=(15,), min_area=9)
            if blob.area == 9 and blob.centroid[0] >= 10
        ]
        targets = connected_components(frame, colors=(3,), min_area=9)
        if not avatars or not targets:
            return

        avatar = max(avatars, key=lambda blob: blob.centroid[0])
        target = min(
            targets,
            key=lambda blob: (
                (blob.centroid[0] - avatar.centroid[0]) ** 2
                + (blob.centroid[1] - avatar.centroid[1]) ** 2
            ),
        )
        ar, ac = avatar.centroid
        tr, tc = target.centroid
        dr = 0 if tr == ar else (step if tr > ar else -step)
        dc = 0 if tc == ac else (step if tc > ac else -step)
        env.step(6, int(round(ac + dc)), int(round(ar + dr)))
