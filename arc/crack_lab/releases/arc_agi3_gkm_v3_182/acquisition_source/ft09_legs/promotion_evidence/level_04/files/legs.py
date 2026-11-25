"""Reusable skills inferred from public frames and verified clone actions."""
from perception import arr, safe_step


def match_miniature(env, x, y, spacing=8, tile_size=6, additional_boards=(),
                    tile_colors=None):
    """Satisfy miniature same/different constraints on shared cycling tiles."""
    frame = arr(env.frame())
    targets = {}
    for bx, by in ((x, y), *additional_boards):
        miniature = frame[by + spacing:by + spacing + tile_size:tile_size // 3,
                          bx + spacing:bx + spacing + tile_size:tile_size // 3]
        center = int(miniature[1, 1])
        colors = set(tile_colors) if tile_colors is not None else {center, 9}
        for row in range(3):
            for col in range(3):
                if (row, col) == (1, 1):
                    continue
                choices = ({center} if int(miniature[row, col]) == 0
                           else colors - {center})
                cx = bx + col * spacing + tile_size // 2
                cy = by + row * spacing + tile_size // 2
                key = (cx, cy)
                targets[key] = targets.get(key, colors) & choices
                if not targets[key]:
                    raise RuntimeError('Overlapping miniature constraints disagree')

    trial = env.clone()
    path = []
    cycle_bound = len(tile_colors) if tile_colors is not None else 2
    for (cx, cy), choices in targets.items():
        for _ in range(cycle_bound):
            if int(arr(trial.frame())[cy, cx]) in choices:
                break
            action = (6, cx, cy)
            safe_step(trial, action)
            path.append(action)
            if trial.levels_completed > env.levels_completed:
                break
        if trial.levels_completed > env.levels_completed:
            break
        if int(arr(trial.frame())[cy, cx]) not in choices:
            raise RuntimeError('Tile cycle cannot satisfy miniature constraint')
    if trial.levels_completed <= env.levels_completed:
        raise RuntimeError('Miniature matching did not complete the board on a clone')
    for action in path:
        safe_step(env, action)
