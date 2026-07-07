import importlib.util, json, sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
spec = importlib.util.spec_from_file_location("solve", "solve.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ck = json.load(open("checkpoint.json"))
def resumed(env):
    for a in ck["final_path"]:
        env.step(a)
    m.solve(env)
levels, path, err = A.run_program('wa30', resumed)
ok = A.validate('wa30', path, levels)
print("levels", levels, "moves", len(path), "ok", ok)
if levels >= 4 and ok:
    ck2 = {"game": "wa30", "reached": 4,
           "records": ck.get("records", []) + [{"level": 4, "reached": True}],
           "final_path": path, "validated": True}
    json.dump(ck2, open("checkpoint.json", "w"))
    print("checkpoint updated")
