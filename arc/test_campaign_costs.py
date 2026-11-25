import json
import pytest
from campaign_costs import execution_rows, build_report, capture_usage


def row(identity="a", **updates):
    return dict(event="codex_exec", thread_id=identity, model="gpt-6.1-sol", game="lf52",
                workspace="ws_" + identity, transcript="turn.jsonl", started_at="2026-11-24T00:00:00Z",
                input_tokens=100, cached_input_tokens=80, output_tokens=20,
                reasoning_output_tokens=15, usage_reported=True,
                protected_transcript_sha256='a'*64, **updates)


def report(rows, cut="a"):
    raw=b"\n".join(json.dumps(r).encode() for r in rows)
    evidence=[dict(thread_id=r['thread_id'],sha256=r['protected_transcript_sha256'],usage={k:r[k] for k in
              ['input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens']})
              for r in rows if r['event']=='codex_exec']
    return build_report(raw,cut,evidence)


def test_subsets_not_double_counted_and_tail_separate():
    q=report([row(),row("b")])
    assert q['summary']['through_cutoff']['input_plus_output_tokens']==120
    assert q['summary']['through_cutoff']['usd_standard_reported_cache']=='0.000248'
    assert q['summary']['after_cutoff']['turns']==1
    assert q['summary']['all_recorded']['input_plus_output_tokens']==240


def test_classification_events_do_not_add_charges():
    q=report([row(),dict(event='codex_exec_classification_correction',thread_id='a')])
    assert q['summary']['all_recorded']['turns']==1


def test_unknown_usage_remains_unknown():
    r=row();r.update(usage_reported=False,input_tokens=0,cached_input_tokens=0,output_tokens=0,reasoning_output_tokens=0)
    q=report([r]);assert q['summary']['all_recorded']['turns_without_reported_usage']==1
    assert q['summary']['all_recorded']['unpriced_turn_ids']==['a']


def test_duplicate_or_inconsistent_tokens_rejected():
    with pytest.raises(ValueError):execution_rows((json.dumps(row())+'\n'+json.dumps(row())).encode())
    r=row();r['cached_input_tokens']=101
    with pytest.raises(ValueError):execution_rows(json.dumps(r).encode())


def test_capture_hash_and_terminal_usage_are_required(tmp_path):
    r=row();p=tmp_path/r['workspace'];p.mkdir()
    (p/r['transcript']).write_text(json.dumps(dict(type='turn.completed',usage={})))
    r['protected_transcript_sha256']='0'*64
    with pytest.raises(ValueError,match='hash mismatch'):capture_usage([r],tmp_path)
