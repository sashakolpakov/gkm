"""Reusable skills inferred from public frames and clone experiments."""
from perception import arr, connected_components, safe_step


def collect_ordered_chain(env):
    """Extend a row-mounted gripper to collect the colors shown in its order strip.

    Move the retracted chain between rows, then extend until the next order
    marker fills. Plan and verify on a clone before replaying on the real env.
    """
    trial = env.clone()
    base_level = trial.levels_completed
    blobs = connected_components(trial.frame(), min_area=12)
    heads = [b for b in blobs if b.color == 6 and b.size == (6, 6)]
    head = min(heads, key=lambda b: b.bbox[0])
    strip_head = max(heads, key=lambda b: b.bbox[0])
    strip_y = strip_head.centroid[0]
    markers = sorted(
        [b for b in blobs if b.area == 16 and b.bbox[0] <= strip_y <= b.bbox[2]
         and b.bbox[1] > strip_head.bbox[3]],
        key=lambda b: b.bbox[1],
    )
    colors = [b.color for b in markers]
    path = []

    def step(action):
        if len(path) >= 100:
            raise RuntimeError('Ordered chain planning exceeded its move bound')
        safe_step(trial, action)
        path.append(action)

    def board_blocks(color):
        return [b for b in connected_components(trial.frame(), colors=[color], min_area=12)
                if b.bbox[2] < strip_head.bbox[0]]

    for marker in markers:
        target = board_blocks(marker.color)[0]
        while head.bbox[0] != target.bbox[0] - 1:
            step(1 if head.bbox[0] > target.bbox[0] - 1 else 2)
            head = min(
                [b for b in connected_components(trial.frame(), colors=[6], min_area=12)
                 if b.size == (6, 6)], key=lambda b: b.bbox[0])
        my, mx = (int(v) for v in marker.centroid)
        while int(arr(trial.frame())[my, mx]) != 0:
            step(4)
            if trial.levels_completed > base_level:
                break
        if trial.levels_completed > base_level:
            break
        # The first collected block must be adjacent to the gripper before
        # changing rows; one more retraction would release a chain member.
        while board_blocks(colors[0])[0].bbox[1] > head.bbox[3] + 2:
            step(3)
    if trial.levels_completed <= base_level:
        raise RuntimeError('Ordered chain plan did not complete the level')
    for action in path:
        if env.terminal() or env.levels_completed > base_level:
            break
        safe_step(env, action)
