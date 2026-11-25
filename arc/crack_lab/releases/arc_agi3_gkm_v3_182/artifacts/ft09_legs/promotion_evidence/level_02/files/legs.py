"""Reusable skills inferred from public frames and verified clone actions."""
from perception import arr, safe_step


def match_miniature(env, x, y, spacing=8, tile_size=6, additional_boards=()):
    """Match click boards to their center miniatures, merging shared tiles."""
    frame = arr(env.frame())
    targets = {}
    for bx, by in ((x, y), *additional_boards):
        miniature = frame[by + spacing:by + spacing + tile_size:tile_size // 3,
                          bx + spacing:bx + spacing + tile_size:tile_size // 3]
        palette = {0: int(miniature[1, 1]), 2: 9}
        for row in range(3):
            for col in range(3):
                if (row, col) == (1, 1):
                    continue
                target = palette[int(miniature[row, col])]
                cx = bx + col * spacing + tile_size // 2
                cy = by + row * spacing + tile_size // 2
                if (cx, cy) in targets and targets[cx, cy] != target:
                    raise RuntimeError('Overlapping miniatures disagree')
                targets[cx, cy] = target
    path = [(6, cx, cy) for (cx, cy), target in targets.items()
            if int(frame[cy, cx]) != target]

    trial = env.clone()
    for action in path:
        safe_step(trial, action)
    if trial.levels_completed <= env.levels_completed:
        raise RuntimeError('Miniature matching did not complete the board on a clone')
    for action in path:
        safe_step(env, action)
