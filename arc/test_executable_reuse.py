from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from executable_reuse import ReplayCalls, analyse_reuse, attach_replay
import pytest


OLD = {"legs.py": b"def walk(env):\n    env.step(1)\n"}


def check(source, *, legs=None):
    files = {**OLD, "players.py": source.encode()}
    if legs is not None:files["legs.py"] = legs.encode()
    return analyse_reuse(OLD, files, ["players.py:play"], target_files={"legs.py"})


def test_new_wrapper_reaches_unchanged_helper():
    result = check("from legs import walk\ndef play(env):\n    route(env)\ndef route(env):\n    walk(env)\n")
    assert result["paths"]["legs.py:walk"] == ["players.py:play", "players.py:route", "legs.py:walk"]
    assert result["indirect_targets"] == ["legs.py:walk"]


def test_alias_and_qualified_imports_are_resolved():
    assert check("import legs as l\ndef play(env):\n    fn = l.walk\n    fn(env)\n")["direct_targets"] == ["legs.py:walk"]
    assert check("from legs import walk as go\ndef play(env):\n    go(env)\n")["direct_targets"] == ["legs.py:walk"]


def test_callback_parameter_and_closure_are_resolved():
    result = check("from legs import walk\ndef glue(fn, env):\n    fn(env)\ndef play(env):\n    glue(walk, env)\n")
    assert result["paths"]["legs.py:walk"] == ["players.py:play", "players.py:glue", "legs.py:walk"]
    closure = check("from legs import walk\ndef play(env):\n    def glue():\n        walk(env)\n    glue()\n")
    assert closure["indirect_targets"] == ["legs.py:walk"]


def test_literal_dispatch_and_getattr_are_resolved():
    assert check("from legs import walk\ntable = {'go': walk}\ndef play(env):\n    table['go'](env)\n")["direct_targets"] == ["legs.py:walk"]
    assert check("import legs\ndef play(env):\n    getattr(legs, 'walk')(env)\n")["direct_targets"] == ["legs.py:walk"]


def test_unknown_dispatch_is_reported_without_becoming_nonreuse():
    result = check("from legs import walk\ntable = {'go': walk}\ndef play(env):\n    table[env.mode](env)\n")
    assert "legs.py:walk" in result["paths"]
    assert result["status"] == "partially_resolved_source_graph"
    assert result["unresolved_calls"]


def test_returned_callable_is_resolved():
    result = check("from legs import walk\ndef factory():\n    return walk\ndef play(env):\n    factory()(env)\n")
    assert result["direct_targets"] == ["legs.py:walk"]


def test_last_definition_and_local_shadowing_are_honored():
    result = check("from legs import walk\ndef play(env):\n    walk(env)\ndef play(env):\n    for action in [1, 2]:\n        env.step(action)\n")
    assert result["paths"] == {}
    assert result["status"] == "resolved_source_graph"
    shadowed = check("from legs import walk\ndef play(env):\n    walk = env.callback\n    walk(env)\n")
    assert shadowed["paths"] == {}


def test_uncalled_nested_definition_does_not_create_an_edge():
    result = check("from legs import walk\ndef play(env):\n    def unused():\n        walk(env)\n    env.step(2)\n")
    assert result["paths"] == {}


def test_changed_definition_is_not_counted_as_unchanged():
    result = check("from legs import walk\ndef play(env):\n    walk(env)\n", legs="def walk(env):\n    env.step(2)\n")
    assert result["paths"] == {}


def test_instance_method_dispatch_is_resolved():
    result = check("from legs import walk\nclass Glue:\n    def go(self, env):\n        walk(env)\ndef play(env):\n    obj = Glue()\n    obj.go(env)\n")
    assert result["paths"]["legs.py:walk"] == ["players.py:play", "players.py:Glue.go", "legs.py:walk"]


def test_recursion_reaches_a_finite_fixed_point():
    result = check("from legs import walk\ndef route(env, flag):\n    if flag:\n        route(env, False)\n    else:\n        walk(env)\ndef play(env):\n    route(env, True)\n")
    assert result["indirect_targets"] == ["legs.py:walk"]


def test_captured_old_definition_survives_later_rebinding():
    current = {"legs.py": OLD["legs.py"] + b"saved = walk\ndef walk(env):\n    env.step(2)\n",
               "players.py": b"from legs import saved\ndef play(env):\n    saved(env)\n"}
    result = analyse_reuse(OLD, current, ["players.py:play"], target_files={"legs.py"})
    assert list(result["paths"]) == ["legs.py:walk@1"]


def test_entrypoint_alias_is_resolved_without_requiring_a_new_function():
    result = check("from legs import walk\nplay = walk\n")
    assert result["resolved_entrypoints"] == ["legs.py:walk"]
    assert result["retained_entrypoint_targets"] == ["legs.py:walk"]


def test_missing_entrypoint_is_unavailable_not_a_nonreuse_result():
    result = check("from legs import walk\n")
    assert result["status"] == "missing_entrypoint"
    assert result["unresolved_calls"]


def test_duplicate_classes_do_not_keep_discarded_methods():
    result = check("from legs import walk\nclass Glue:\n    def old_method(self, env):\n        walk(env)\nclass Glue:\n    def new_method(self, env):\n        env.step(2)\ndef play(env):\n    Glue().new_method(env)\n")
    assert result["paths"] == {}


def test_replay_records_runtime_dispatch_without_changing_code():
    previous = {"solver.py": b"def old(env):\n    env.append(1)\n"}
    current = {"solver.py": previous["solver.py"] + b"\ndef play(env, key):\n    table = {'chosen': old}\n    table[key](env)\n"}
    namespace = {};exec(compile(current["solver.py"], "solver.py", "exec"), namespace)
    values = []
    with ReplayCalls(previous, current, entrypoints=["solver.py:play"]) as trace:
        namespace["play"](values, "chosen")
    report = trace.report()
    assert values == [1]
    assert report["observed_retained_call_events"] == {"solver.py:old": 1}
    assert report["errors"] == []


def test_replay_matches_normal_import_compilation_flags():
    previous = {"solver.py": b"def old(env):\n    env.append(1)\n"}
    current = {"solver.py": previous["solver.py"] + b"\ndef play(env):\n    old(env)\n"}
    namespace = {};exec(compile(current["solver.py"], "solver.py", "exec", dont_inherit=True), namespace)
    with ReplayCalls(previous, current, entrypoints=["solver.py:play"]) as trace:
        namespace["play"]([])
    assert trace.report()["observed_retained_call_events"] == {"solver.py:old": 1}
    assert trace.report()["errors"] == []


def test_replay_records_actual_branch_not_uncalled_static_branch():
    previous = {"solver.py": b"def old(env):\n    env.append(1)\n"}
    current = {"solver.py": previous["solver.py"] + b"\ndef play(env):\n    if False:\n        old(env)\n    env.append(2)\n"}
    namespace = {};exec(compile(current["solver.py"], "solver.py", "exec"), namespace)
    with ReplayCalls(previous, current, entrypoints=["solver.py:play"]) as trace:
        namespace["play"]([])
    assert trace.report()["observed_retained_call_events"] == {}


def test_replay_restores_existing_profiler_and_handles_generated_glue():
    previous = {"solver.py": b"def old(env):\n    env.append(1)\n"}
    current = {"solver.py": previous["solver.py"] + b"\ndef play(env):\n    exec('def glue(env):\\n    old(env)', globals())\n    glue(env)\n"}
    namespace = {};exec(compile(current["solver.py"], "solver.py", "exec"), namespace)
    events = []
    def prior(frame, event, arg):
        if frame.f_code.co_filename == "solver.py" and event == "call":events.append(frame.f_code.co_name)
    sys.setprofile(prior)
    try:
        with ReplayCalls(previous, current, entrypoints=["solver.py:play"]) as trace:
            namespace["play"]([])
        assert sys.getprofile() is prior
    finally:
        sys.setprofile(None)
    report = trace.report()
    assert report["observed_retained_call_events"] == {"solver.py:old": 1}
    assert any(key.startswith("<generated>:glue:") for key in report["call_events"])
    assert "old" in events


def test_replay_follows_aliased_entrypoint_and_binds_source_pair():
    previous = {"solver.py": b"def old(env):\n    env.append(1)\n"}
    current = {"solver.py": previous["solver.py"] + b"play = old\n"}
    namespace = {};exec(compile(current["solver.py"], "solver.py", "exec"), namespace)
    with ReplayCalls(previous, current, entrypoints=["solver.py:play"]) as trace:
        namespace["play"]([])
    analysis = analyse_reuse(previous, current, ["solver.py:play"])
    report = attach_replay(analysis, trace.report())
    assert report["observed_replay_calls"]["observed_retained_call_events"] == {"solver.py:old": 1}
    with pytest.raises(ValueError, match="source binding"):
        attach_replay({**analysis, "previous_source_sha256": {}}, trace.report())


def test_replay_observes_partial_and_lambda_callbacks():
    previous = {"solver.py": b"def old(env):\n    env.append(1)\n"}
    current = {"solver.py": previous["solver.py"] + b"from functools import partial\ndef play(env):\n    fn = partial(old, env)\n    invoke = lambda: fn()\n    invoke()\n"}
    namespace = {};exec(compile(current["solver.py"], "solver.py", "exec"), namespace)
    with ReplayCalls(previous, current, entrypoints=["solver.py:play"]) as trace:
        namespace["play"]([])
    assert trace.report()["observed_retained_call_events"] == {"solver.py:old": 1}


def test_replay_records_new_thread_callback_without_restricting_the_program():
    previous = {"solver.py": b"def old(env):\n    env.append(1)\n"}
    current = {"solver.py": previous["solver.py"] + b"from threading import Thread\ndef play(env):\n    worker = Thread(target=old, args=(env,))\n    worker.start()\n    worker.join()\n"}
    namespace = {};exec(compile(current["solver.py"], "solver.py", "exec"), namespace)
    with ReplayCalls(previous, current, entrypoints=["solver.py:play"]) as trace:
        namespace["play"]([])
    assert trace.report()["observed_retained_call_events"] == {"solver.py:old": 1}


def test_unpacking_does_not_invent_a_retained_target():
    result = check("from legs import walk\ndef fresh(env):\n    env.step(2)\ndef play(env):\n    unused, chosen = walk, fresh\n    chosen(env)\n")
    assert result["paths"] == {}


def test_shadowed_getattr_uses_the_user_function():
    result = check("from legs import walk\ndef fresh(env):\n    env.step(2)\ndef getattr(obj, name):\n    return fresh\ndef play(env):\n    getattr(None, 'walk')(env)\n")
    assert result["paths"] == {}


def test_overwritten_dispatch_table_entry_is_not_retained():
    result = check("from legs import walk\ndef fresh(env):\n    env.step(2)\ndef play(env):\n    table = {'go': walk}\n    table['go'] = fresh\n    table['go'](env)\n")
    assert result["paths"] == {}


def test_instance_method_overwrite_does_not_invent_a_retained_call():
    result = check("from legs import walk\nclass Glue:\n    def go(self, env):\n        walk(env)\ndef fresh(env):\n    env.step(2)\ndef play(env):\n    obj = Glue()\n    obj.go = fresh\n    obj.go(env)\n")
    assert result["paths"] == {}


def test_recursive_external_attribute_domain_is_finite():
    result = check("def play(env):\n    play(env.child)\n")
    assert result["paths"] == {}


def test_replay_resolves_lambda_bound_entrypoint():
    previous = {"solver.py": b"def old(env):\n    env.append(1)\n"}
    current = {"solver.py": previous["solver.py"] + b"play = lambda env: old(env)\n"}
    namespace = {};exec(compile(current["solver.py"], "solver.py", "exec"), namespace)
    with ReplayCalls(previous, current, entrypoints=["solver.py:play"]) as trace:
        namespace["play"]([])
    report = trace.report()
    assert report["observed_retained_call_events"] == {"solver.py:old": 1}
    assert report["errors"] == []


def test_unseen_replay_entrypoint_cannot_be_attached_as_empty_reuse():
    previous = {"solver.py": b"def old(env):\n    env.append(1)\n"}
    current = {"solver.py": previous["solver.py"] + b"def play(env):\n    old(env)\n"}
    with ReplayCalls(previous, current, entrypoints=["solver.py:play"]) as trace:
        pass
    with pytest.raises(ValueError, match="collection errors"):
        attach_replay(analyse_reuse(previous, current, ["solver.py:play"]), trace.report())


def test_code_fingerprint_binds_exception_handling():
    from executable_reuse import code_fingerprint
    namespace = {};exec("def fn():\n    try:\n        return 1 / 0\n    except ZeroDivisionError:\n        return 2\n", namespace)
    code = namespace['fn'].__code__
    if hasattr(code, 'co_exceptiontable'):
        assert code_fingerprint(code) != code_fingerprint(code.replace(co_exceptiontable=b''))


def test_unrelated_thread_is_not_charged_to_an_active_entrypoint():
    from threading import Thread, Event
    previous = {"solver.py": b"def old(env):\n    env.append(1)\n"}
    current = {"solver.py": previous["solver.py"] + b"def play(ready, finish):\n    ready.set()\n    finish.wait()\n"}
    namespace = {};exec(compile(current["solver.py"], "solver.py", "exec"), namespace)
    ready, finish = Event(), Event()
    with ReplayCalls(previous, current, entrypoints=["solver.py:play"]) as trace:
        main = Thread(target=namespace['play'], args=(ready, finish));main.start();ready.wait()
        unrelated = Thread(target=namespace['old'], args=([],));unrelated.start();unrelated.join()
        finish.set();main.join()
    assert trace.report()["observed_retained_call_events"] == {}


def test_worker_outliving_observation_detaches_before_its_next_call():
    from threading import Thread, Event
    previous = {"solver.py": b"def old(env):\n    env.append(1)\n"}
    current = {"solver.py": previous["solver.py"] + b"def play(env):\n    old(env)\n"}
    namespace = {};exec(compile(current["solver.py"], "solver.py", "exec"), namespace)
    go, done = Event(), Event();profiles = []
    def worker():
        go.wait();profiles.append(sys.getprofile());done.set()
    with ReplayCalls(previous, current, entrypoints=["solver.py:play"]) as trace:
        thread = Thread(target=worker);thread.start();namespace['play']([])
    go.set();done.wait();thread.join()
    assert profiles == [None]


def test_replay_tracks_async_children_without_charging_unrelated_coroutines():
    import asyncio
    previous = {"solver.py": b"def old(env):\n    env.append(1)\n"}
    current = {"solver.py": previous["solver.py"] + b"import asyncio\nasync def child(env):\n    old(env)\nasync def play(env, ready, finish):\n    ready.set()\n    await finish.wait()\n    await asyncio.create_task(child(env))\n"}
    namespace = {};exec(compile(current["solver.py"], "solver.py", "exec"), namespace)
    async def run():
        ready, finish = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(namespace['play']([], ready, finish))
        await ready.wait()
        namespace['old']([])
        finish.set()
        await task
    with ReplayCalls(previous, current, entrypoints=["solver.py:play"]) as trace:
        asyncio.run(run())
    assert trace.report()["observed_retained_call_events"] == {"solver.py:old": 1}
    assert trace.report()["errors"] == []


def test_replay_accepts_actual_decorated_entrypoint_object():
    previous = {"solver.py": b"def old(env):\n    env.append(1)\n"}
    current = {"solver.py": previous["solver.py"] + b"def decorate(fn):\n    def wrapper(env):\n        return fn(env)\n    return wrapper\n@decorate\ndef play(env):\n    old(env)\n"}
    namespace = {};exec(compile(current["solver.py"], "solver.py", "exec"), namespace)
    with ReplayCalls(previous, current, entrypoints=["solver.py:play"], entrypoint_objects=[namespace['play']]) as trace:
        namespace['play']([])
    assert trace.report()["observed_retained_call_events"] == {"solver.py:old": 1}
    assert trace.report()["errors"] == []


def test_replay_accepts_a_callable_object_entrypoint():
    previous = {"solver.py": b"def old(env):\n    env.append(1)\n"}
    current = {"solver.py": previous["solver.py"] + b"class Glue:\n    def __call__(self, env):\n        old(env)\nplay = Glue()\n"}
    namespace = {};exec(compile(current["solver.py"], "solver.py", "exec"), namespace)
    with ReplayCalls(previous, current, entrypoints=["solver.py:play"]) as trace:
        namespace['play']([])
    assert trace.report()["observed_retained_call_events"] == {"solver.py:old": 1}


def test_default_callback_uses_definition_time_binding():
    result = check("from legs import walk\ndef fresh(env):\n    env.step(2)\nhandler = fresh\ndef route(env, cb=handler):\n    cb(env)\nhandler = walk\ndef play(env):\n    route(env)\n")
    assert result["paths"] == {}


def test_shared_table_mutation_is_explicit_and_independent_of_function_name():
    template = "from legs import walk\ndef fresh(env):\n    env.step(2)\ntable = {'go': walk}\ndef NAME():\n    table['go'] = fresh\ndef play(env):\n    NAME()\n    table['go'](env)\n"
    results = [check(template.replace('NAME', name)) for name in ['change', 'zchange']]
    assert all(result['status']=='partially_resolved_source_graph' for result in results)
    assert all(any('shared table' in s['reason'] or 'multiple possible' in s['reason'] for s in result['unresolved_calls']) for result in results)
    assert [bool(result['paths']) for result in results] in ([True, True], [False, False])


def test_runtime_index_disambiguates_identical_bodies_with_different_defaults():
    previous = {"solver.py": b"def old(env, unused=1):\n    env.append(1)\n"}
    current = {"solver.py": previous['solver.py'] + b"def old(env, unused=2):\n    env.append(1)\ndef play(env):\n    old(env)\n"}
    namespace = {};exec(compile(current['solver.py'],'solver.py','exec'),namespace)
    with ReplayCalls(previous,current,entrypoints=['solver.py:play']) as trace:
        namespace['play']([])
    assert trace.report()['observed_retained_call_events']=={}
    assert trace.report()['call_events']['solver.py:old']==1
    assert trace.report()['errors']==[]


def test_import_initialization_calls_are_connected_to_the_entrypoint():
    current={**OLD,'util.py':b'from legs import walk\nwalk(None)\n',
             'players.py':b'def play(env):\n    import util\n'}
    result=analyse_reuse(OLD,current,['players.py:play'],target_files={'legs.py'})
    assert result['paths']['legs.py:walk']==['players.py:play','util.py:<module>','legs.py:walk']


def test_replay_disambiguates_a_shared_table_update_without_restricting_glue():
    previous={'solver.py':b'def old(env):\n    env.append(1)\n'}
    current={'solver.py':previous['solver.py']+b"def fresh(env):\n    env.append(2)\ntable={'go':old}\ndef zchange():\n    table['go']=fresh\ndef play(env):\n    zchange()\n    table['go'](env)\n"}
    namespace={};exec(compile(current['solver.py'],'solver.py','exec'),namespace)
    values=[]
    with ReplayCalls(previous,current,entrypoints=['solver.py:play']) as trace:
        namespace['play'](values)
    assert values==[2]
    assert trace.report()['observed_retained_call_events']=={}
    assert trace.report()['errors']==[]


def test_literal_definition_reuse_can_move_between_modules():
    current={'moved.py':OLD['legs.py'], 'players.py':b'from moved import walk\ndef play(env):\n    walk(env)\n'}
    result=analyse_reuse(OLD,current,['players.py:play'])
    assert result['paths']['moved.py:walk']==['players.py:play','moved.py:walk']
    assert result['preceding_definitions']['moved.py:walk']=='legs.py:walk'


def test_python_equal_numeric_table_keys_share_the_last_binding():
    result=check("from legs import walk\ndef fresh(env):\n    env.step(2)\ndef play(env):\n    routes={1:walk,True:fresh,1.0:fresh}\n    routes[1](env)\n")
    assert result['paths']=={}


def test_computed_table_keys_remain_explicitly_unresolved():
    result=check("from legs import walk\ndef fresh(env):\n    env.step(2)\ndef play(env):\n    routes={'go':walk,env.mode:fresh}\n    routes['go'](env)\n")
    assert result['status']=='partially_resolved_source_graph'
    assert result['unresolved_calls']


def test_replay_accounts_for_retained_lambda_code_without_requiring_def_syntax():
    previous={'solver.py':b'old = lambda env: env.append(1)\n'}
    current={'solver.py':previous['solver.py']+b'def play(env):\n    old(env)\n'}
    namespace={};exec(compile(current['solver.py'],'solver.py','exec'),namespace)
    with ReplayCalls(previous,current,entrypoints=['solver.py:play']) as trace:
        namespace['play']([])
    matches=trace.report()['observed_prior_code_body_matches']
    assert any('<lambda>' in name for name in matches)


def test_variadic_callback_bindings_do_not_fall_back_to_module_globals():
    result=check("from legs import walk\ncallbacks=[walk]\ndef fresh(env):\n    env.step(2)\ndef glue(env,*callbacks):\n    callbacks[0](env)\ndef play(env):\n    glue(env,fresh)\n")
    assert result['paths']=={}
    assert result['unresolved_calls']
