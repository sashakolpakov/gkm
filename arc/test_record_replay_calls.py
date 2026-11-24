import json
from pathlib import Path
import subprocess
import sys


def test_replay_cli_observes_dispatch_and_preserves_solver_sources(tmp_path):
    previous, current = tmp_path / 'previous', tmp_path / 'current'
    previous.mkdir();current.mkdir()
    old = b'def retained(values):\n    values.append(1)\n'
    source = old + b"def play(values):\n    routes = {'picked': retained}\n    routes['picked'](values)\n"
    (previous/'solver.py').write_bytes(old);(current/'solver.py').write_bytes(source)
    driver = tmp_path/'replay.py'
    driver.write_text('import sys\nsys.path.insert(0,'+repr(str(current))+')\nfrom solver import play\nvalues=[]\nplay(values)\nassert values==[1]\nraise SystemExit(0)\n')
    output = tmp_path/'observed.json'
    tool = Path(__file__).resolve().parent/'record_replay_calls.py'
    subprocess.run([sys.executable, '-B', str(tool), '--previous', str(previous), '--current', str(current),
                    '--entrypoint', 'solver.py:play', '--output', str(output), str(driver)], check=True)
    report=json.loads(output.read_text())
    assert report['observed_retained_call_events']=={'solver.py:retained':1}
    assert report['errors']==[]
    assert (current/'solver.py').read_bytes()==source
    assert not list(current.rglob('__pycache__'))
