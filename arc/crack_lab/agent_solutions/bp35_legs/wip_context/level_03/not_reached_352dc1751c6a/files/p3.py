"""L3 probe harness: gets a real env to the start of level 3 via the validated
checkpoint path, then runs whatever probe function is named on the CLI.

Only uses the documented surface: A.run_program, env.step/frame/clone/terminal/
levels_completed/actions.
"""
import json
import sys

sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A  # noqa: E402

from legs import (AVATAR_ROW, COL_ANCHORS, GRID_COLS, GRID_ROWS, ROW_ANCHORS,
                  avatar_column, band_grid, band_shift, click_action,
                  moves_used)  # noqa: E402

CK = json.load(open('checkpoint.json'))['final_path']


def tup(a):
    return tuple(a) if isinstance(a, (list, tuple)) else (a,)


def to_level3(env):
    for a in CK:
        env.step(*tup(a))
    assert env.levels_completed == 2, env.levels_completed
    return env


def show(env, label=''):
    f = env.frame()
    g = band_grid(f)
    print(f'--- {label} lvl={env.levels_completed} term={env.terminal()} '
          f'moves={moves_used(f)} col={avatar_column(f)}')
    for i, row in enumerate(g):
        mark = '<' if i == AVATAR_ROW else ' '
        print(f'  {i} |{"".join(row)}|{mark}')


def colours(env):
    from collections import Counter
    f = env.frame()
    c = Counter(int(v) for row in f for v in row)
    return dict(sorted(c.items()))


def probe(fn):
    return fn


def main():
    import probes3
    name = sys.argv[1]
    args = sys.argv[2:]
    fn = getattr(probes3, name)

    def prog(env):
        to_level3(env)
        fn(env, *args)

    print('run_program ->', A.run_program('bp35', prog))


if __name__ == '__main__':
    main()
