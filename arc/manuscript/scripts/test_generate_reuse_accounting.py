import json
from pathlib import Path

from generate_reuse_accounting import build_report


def test_all_source_pairs_and_cached_comparators_have_explicit_coverage():
    path = Path(__file__).resolve().parents[2]/'audit_results/marginal-literal-reuse.json'
    report=build_report(json.loads(path.read_text()))
    gkm=report['summaries']['GKM']
    assert gkm['exact_pairs_analysed']==144
    assert gkm['pairs_with_retained_call_path']>=54
    assert gkm['sharp_drops_with_retained_call_path']>=13
    assert gkm['source_bound_replay_observations']==0
    assert report['summaries']['OPINE']['exact_pairs_analysed']==0
    assert report['summaries']['OPINE']['cached_or_unavailable_pairs']==121
