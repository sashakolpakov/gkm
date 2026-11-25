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


def match_coupled_miniatures(env, miniatures, tile_colors, spacing=8,
                            tile_size=6, ignored_marker=3):
    """Match binary miniature clues using clone-measured coupled toggles."""
    base_level = env.levels_completed
    frame = arr(env.frame()).copy()
    targets = {}
    for x, y in miniatures:
        miniature = frame[y:y + tile_size:tile_size // 3,
                          x:x + tile_size:tile_size // 3]
        center = int(miniature[1, 1])
        other, = set(tile_colors) - {center}
        for row in range(3):
            for col in range(3):
                marker = int(miniature[row, col])
                if (row, col) == (1, 1) or marker == ignored_marker:
                    continue
                position = (x + (col - 1) * spacing + tile_size // 2,
                            y + (row - 1) * spacing + tile_size // 2)
                desired = center if marker == 0 else other
                if position in targets and targets[position] != desired:
                    raise RuntimeError('Miniature constraints disagree')
                targets[position] = desired

    positions = list(targets)
    actions = [(6, x, y) for x, y in positions]
    basis = {}
    for index, action in enumerate(actions):
        trial = env.clone()
        safe_step(trial, action)
        if trial.levels_completed > env.levels_completed:
            safe_step(env, action)
            return
        changed = arr(trial.frame())
        column = sum((int(changed[y, x]) != int(frame[y, x])) << i
                     for i, (x, y) in enumerate(positions))
        combination = 1 << index
        while column:
            pivot = column.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (column, combination)
                break
            column ^= basis[pivot][0]
            combination ^= basis[pivot][1]

    remaining = sum((int(frame[y, x]) != targets[x, y]) << i
                    for i, (x, y) in enumerate(positions))
    solution = 0
    while remaining:
        pivot = remaining.bit_length() - 1
        if pivot not in basis:
            raise RuntimeError('Observed toggles cannot satisfy the clues')
        remaining ^= basis[pivot][0]
        solution ^= basis[pivot][1]
    path = [action for i, action in enumerate(actions) if solution >> i & 1]
    trial = env.clone()
    for action in path:
        safe_step(trial, action)
        if trial.levels_completed > env.levels_completed:
            break
    if trial.levels_completed <= env.levels_completed:
        raise RuntimeError('Coupled miniature plan did not complete on a clone')
    for action in path:
        safe_step(env, action)
        if env.levels_completed > base_level:
            break
