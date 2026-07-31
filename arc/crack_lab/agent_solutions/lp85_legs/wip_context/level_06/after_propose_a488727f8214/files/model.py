"""Offline permutation model of lp85 level-2 tracks (validated against env)."""
ROWS = [17,20,23,26,29,32,35,38,41,44]
COLS = ROWS

RING = ([(0,j) for j in range(2,7)]
        + [(i,6) for i in range(1,10)]
        + [(9,j) for j in (5,4,3,2)]
        + [(i,2) for i in range(8,0,-1)])
LINE3 = [(3,j) for j in range(10)]
LINE6 = [(6,j) for j in range(10)]
TRACKS = {"R": RING, "L3": LINE3, "L6": LINE6}

BUTTONS = {  # name -> (x=col_px, y=row_px), track, direction
    "A_L": ((20,17), "R", -1),
    "A_R": ((39,17), "R", +1),
    "B_L": ((14,26), "L3", -1),
    "B_R": ((48,26), "L3", +1),
    "C_L": ((14,35), "L6", -1),
    "C_R": ((48,35), "L6", +1),
}
ORDER = ["A_L","A_R","B_L","B_R","C_L","C_R"]

CELLS = sorted(set(RING) | set(LINE3) | set(LINE6))

def move_cell(cell, button):
    """Where a token at `cell` ends up after pressing `button`."""
    _, tname, d = BUTTONS[button]
    track = TRACKS[tname]
    if cell not in track:
        return cell
    k = track.index(cell)
    return track[(k + d) % len(track)]

def apply_state(state, button):
    """state: dict cell->color. Returns new dict."""
    out = dict(state)
    _, tname, d = BUTTONS[button]
    track = TRACKS[tname]
    n = len(track)
    for k, cell in enumerate(track):
        out[track[(k + d) % n]] = state[cell]
    return out

def px(button):
    return BUTTONS[button][0]
