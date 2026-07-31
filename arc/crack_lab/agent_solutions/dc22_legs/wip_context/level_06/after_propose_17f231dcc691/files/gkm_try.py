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
    # Prefix optimization must replay the edited solver from level 1 without
    # mutating the supervisor-owned campaign checkpoint.
    if os.environ.get("GKM_FRESH_REPLAY") != "1" and os.path.exists("checkpoint.json"):
        with open("checkpoint.json") as f:
            ck = json.load(f)
    if ck and ck.get("game") == 'dc22' and ck.get("validated") and ck.get("final_path"):
        for act in ck["final_path"]:
            env.step(act)
    m.solve(env)
levels, path, err = A.run_program('dc22', resumed_solve)
ok = A.validate('dc22', path, levels) if path else False
print(f"RESULT levels={levels} moves={len(path)} replay_ok={ok} err={err}")
