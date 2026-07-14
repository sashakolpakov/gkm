import importlib.util, json, os, sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_legs as G
import gkm_arena as A
taint_reason = G._workspace_taint_reason(os.getcwd())
if taint_reason:
    raise SystemExit(f"TAINTED WORKSPACE: {taint_reason}")
spec = importlib.util.spec_from_file_location("solve", "solve.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def resumed_solve(env):
    ck = None
    if os.path.exists("checkpoint.json"):
        with open("checkpoint.json") as f:
            ck = json.load(f)
    if ck and ck.get("game") == 'ft09' and ck.get("validated") and ck.get("final_path"):
        for act in ck["final_path"]:
            env.step(act)
    m.solve(env)
levels, path, err = A.run_program('ft09', resumed_solve)
ok = A.validate('ft09', path, levels) if path else False
print(f"RESULT levels={levels} moves={len(path)} replay_ok={ok} err={err}")
