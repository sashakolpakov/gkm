"""Render every reachable ring placement from documented level-6 frames."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np
from PIL import Image, ImageDraw

import perception
import solve
from probe_l6_right import avatar_position, enter_right


PALETTE = np.asarray([
    (0, 0, 0), (0, 80, 220), (220, 35, 35), (35, 180, 55),
    (245, 205, 35), (130, 130, 130), (220, 45, 190),
    (245, 135, 25), (50, 180, 235), (125, 25, 55),
    (145, 220, 70), (40, 220, 210), (130, 75, 45),
    (245, 120, 180), (70, 70, 70), (245, 245, 245),
], dtype=np.uint8)

CENTER = (58, 34)
MOVES = (
    (1, (6, 50, 34)), (2, (6, 50, 40)),
    (3, (6, 46, 36)), (4, (6, 54, 36)),
)
TO_CENTER = {
    (56, 34): 2, (60, 34): 1, (58, 32): 4, (58, 36): 3,
}


def ring_key(env):
    return perception.arr(env.frame())[6:42, 6:34].tobytes()


def observe(env):
    solve.solve(env)
    root = enter_right(env, 3)
    queue = deque([root.clone()])
    seen = {ring_key(root)}
    frames = [perception.arr(root.frame()).copy()]
    while queue:
        node = queue.popleft()
        position = avatar_position(node)
        for movement, control in MOVES:
            child = node.clone()
            if position != CENTER:
                child.step(TO_CENTER[position])
            child.step(movement)
            child.step(*control)
            key = ring_key(child)
            if key in seen:
                continue
            seen.add(key)
            frames.append(perception.arr(child.frame()).copy())
            queue.append(child)

    tile_w, tile_h, label_h, scale = 40, 42, 10, 3
    montage = Image.new(
        "RGB", (5 * tile_w * scale, 4 * (tile_h * scale + label_h)),
        (80, 80, 80),
    )
    draw = ImageDraw.Draw(montage)
    for index, frame in enumerate(frames):
        row, col = divmod(index, 5)
        pixels = PALETTE[frame[:tile_h, :tile_w]]
        tile = Image.fromarray(pixels).resize(
            (tile_w * scale, tile_h * scale), Image.Resampling.NEAREST
        )
        x, y = col * tile_w * scale, row * (tile_h * scale + label_h)
        montage.paste(tile, (x, y + label_h))
        draw.text((x + 2, y), str(index), fill=(255, 255, 255))
    montage.save("level6_ring_montage.png")


arena.run_program("dc22", observe)
