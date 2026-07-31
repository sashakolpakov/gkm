import json, sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, connected_components, color_counts

PATH = json.load(open('checkpoint.json'))['final_path']


def at_level5():
    env = A.make_env('re86') if hasattr(A, 'make_env') else None
    return env


def main():
    holder = {}

    def prog(env):
        for a in PATH:
            env.step(a)
        holder['env'] = env
        holder['lvl'] = env.levels_completed
        f = arr(env.frame())
        holder['frame'] = f.copy()
        print('levels', env.levels_completed, 'actions', getattr(env, 'actions', None))
        print('colors', color_counts(f))
        for b in connected_components(f, min_area=1):
            if b.area < 300:
                print(f"c={b.color:2d} bbox={b.bbox} sz={b.size} area={b.area}")

    A.run_program('re86', prog)


main()
