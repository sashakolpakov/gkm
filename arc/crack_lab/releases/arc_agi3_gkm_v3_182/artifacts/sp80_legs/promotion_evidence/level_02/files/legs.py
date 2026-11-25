"""Reusable skills based only on rendered objects and public actions."""
from perception import connected_components, safe_step


def align_horizontal(env, moving_color, reference_color, center_offset=0, max_moves=16):
    """Align a controlled object's center with an offset from a reference.

    Actions 3 and 4 translate horizontally. Remaining center distance is
    the progress measure; stop if a move does not reduce it.
    """
    def center(color):
        blobs = connected_components(env.frame(), colors=[color])
        blob = max(blobs, key=lambda b: b.area)
        return (blob.bbox[1] + blob.bbox[3]) / 2

    for _ in range(max_moves):
        distance = center(reference_color) + center_offset - center(moving_color)
        if abs(distance) < 1:
            return
        if env.terminal():
            return
        safe_step(env, 4 if distance > 0 else 3)
        remaining = center(reference_color) + center_offset - center(moving_color)
        if abs(remaining) >= abs(distance):
            return
    raise RuntimeError('Horizontal alignment exceeded its move bound')


def activate(env, action=5):
    """Apply a public interaction once at the staged alignment."""
    if not env.terminal():
        safe_step(env, action)


def select_object(env, color, index=0):
    """Select a rendered component by clicking its center, in reading order."""
    blobs = connected_components(env.frame(), colors=[color])
    blob = blobs[index]
    r0, c0, r1, c1 = blob.bbox
    safe_step(env, (6, (c0 + c1) // 2, (r0 + r1) // 2))
