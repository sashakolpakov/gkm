"""Reusable skills inferred from public frames and verified clone actions."""
from perception import arr, safe_step


def match_miniature(env, x, y, spacing=8, tile_size=6):
    """Match a 3x3 click board to its center's miniature, then commit."""
    frame = arr(env.frame())
    miniature = frame[y + spacing:y + spacing + tile_size:tile_size // 3,
                      x + spacing:x + spacing + tile_size:tile_size // 3]
    palette = {0: 8, 2: 9}
    path = []
    for row in range(3):
        for col in range(3):
            if (row, col) == (1, 1):
                continue
            target = palette[int(miniature[row, col])]
            cx = x + col * spacing + tile_size // 2
            cy = y + row * spacing + tile_size // 2
            if int(frame[cy, cx]) != target:
                path.append((6, cx, cy))

    trial = env.clone()
    for action in path:
        safe_step(trial, action)
    if trial.levels_completed <= env.levels_completed:
        raise RuntimeError('Miniature matching did not complete the board on a clone')
    for action in path:
        safe_step(env, action)
