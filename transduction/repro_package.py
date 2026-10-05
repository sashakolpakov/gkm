"""Build and verify a portable, offline evidence package for the whole paper.

No replay command can call a model. Historical files are copied without edits;
each study gets its own matching sources. The sole relocation adapter redirects
the composition study's absolute basis path to the identical packaged library.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

BASE = Path(__file__).resolve().parent.parent


def study(name, directory, module, kind="regular", audit=None):
    return dict(id=name, directory="output/" + directory, module=module, kind=kind, audit=audit)


STUDIES = [
    study("register", "transduction_register/20261004-reproduction-v1", "register_artifacts", "register"),
    study("tail-seed7", "transduction_cofibration/20261003-seed7-b", "run_cofibration_experiment", "tail"),
    *[study(f"tail-{s}-{b}", f"transduction_cofibration/20261003-seed{s}-budget{b}",
            "run_cofibration_experiment", "tail") for s in (1, 11, 23) for b in (512, 8192)],
    study("instruction", "transduction_cofibration/20261003-paired-ab", "glue_ab", "instruction"),
    study("returning", "transduction_growth/20261003-returning-ab", "growth_benchmark", audit="audit_growth"),
    study("composition", "transduction_composition/20261003-fixed-basis-ab", "composition_benchmark",
          "composition", "audit_composition"),
    study("recursive-pilot", "transduction_recursive/20261003-paired-ab", "recursive_benchmark"),
    study("recursive", "transduction_recursive/20261003-recursive-followup", "recursive_benchmark", audit="audit_recursive"),
    study("interfaces", "transduction_interface/20261003-growth-v2", "interface_benchmark", audit="audit_interface"),
    study("frontier-pilot", "transduction_frontier/20261003-control-v1", "frontier_benchmark", audit="audit_frontier"),
    study("frontier-repeat", "transduction_frontier/20261004-framed-seed2", "frontier_benchmark", audit="audit_frontier"),
    study("retries-mechanical", "transduction_frontier/20261004-retries-mechanical", "frontier_benchmark", "batch", "audit_frontier"),
    study("retries-model", "transduction_frontier/20261004-retries-codex", "frontier_benchmark", "batch", "audit_frontier"),
    study("direct", "transduction_cofibration_ab/20261004-v1", "cofibration_ab", "direct"),
    study("hard", "transduction_cofibration_ab/20261004-hard-v1", "hard_cofibration_ab", "hard"),
    study("retention", "transduction_transfer_value/20261004-v1", "transfer_value"),
]


def read(path):
    return json.loads(path.read_text())


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def payload_files(folder):
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink excluded from evidence package: {path}")
        if path.is_file():
            if any(part in (".git", "__pycache__", ".pytest_cache") for part in path.parts):
                continue
            if path.name.startswith(".env") or path.name in ("env.local", "auth.json", "credentials"):
                raise ValueError(f"credential file excluded: {path.name}")
            yield path


SECRET = re.compile(rb"(?:sk-(?:or-v1-|proj-)?[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")


def scan(path):
    if path.suffix in (".json", ".jsonl", ".log", ".py", ".md", ".tex", ".txt", ".bib"):
        if SECRET.search(path.read_bytes()):
            raise ValueError(f"possible secret; inspect locally before packaging: {path}")
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            for member in archive:
                if member.isfile() and SECRET.search(archive.extractfile(member).read()):
                    raise ValueError(f"possible secret in source archive: {path.name}/{member.name}")


def source_pool(root):
    pool = {}
    for path in (root / "transduction").glob("*.py"):
        body = path.read_bytes()
        pool[(path.name, hashlib.sha256(body).hexdigest())] = (body, "current repository")
    for item in STUDIES:
        for path in (root / item["directory"]).rglob("*.tar.gz"):
            with tarfile.open(path) as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".py"):
                        continue
                    body = archive.extractfile(member).read()
                    name = Path(member.name).name
                    pool[(name, hashlib.sha256(body).hexdigest())] = (body, str(path.relative_to(root)))
    return pool


def expected_sources(folder, item):
    for name in ("plan.json", "trials.json", "protocol.json"):
        if (folder / name).exists():
            document = read(folder / name)
            document = document.get("plan", document)
            result = dict(document.get("source_hashes", {}))
            break
    else:
        result = {}
    if item["audit"]:
        name = "equality-audit-8.json" if item["audit"] in ("audit_growth", "audit_composition") else "shape-audit.json"
        audits = sorted(folder.rglob(name))
        if audits:
            # All batch trials use the same audit source; do not pick a seed.
            values = {read(p).get("auditor_sha256", read(p).get("audit_source_sha256", read(p).get("audit_source_hash")))
                      for p in audits if p.parent == folder or p.parent.parent == folder}
            values.discard(None)
            if len(values) != 1:
                raise ValueError(f"missing or inconsistent auditor sources: {folder}")
            result[item["audit"] + ".py"] = values.pop()
    if item["kind"] == "hard":
        result["audit_frontier.py"] = read(folder / "interface-extension.json")["auditor_sha256"]
    return result


def seal(folder):
    entries = {str(p.relative_to(folder)): sha(p) for p in payload_files(folder) if p != folder / "SHA256SUMS"}
    path = folder / "SHA256SUMS"
    if path.exists():
        raise ValueError("already sealed")
    path.write_text("".join(f"{value}  {name}\n" for name, value in sorted(entries.items())))
    return entries


def checksums(folder):
    expected = {}
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        value, name = line.split("  ", 1)
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or name in expected:
            raise ValueError("unsafe or duplicate manifest path")
        expected[name] = value
    actual = {str(p.relative_to(folder)): sha(p) for p in payload_files(folder) if p != folder / "SHA256SUMS"}
    if actual != expected:
        raise ValueError("package content differs from SHA256SUMS")
    return len(actual)


def build(root, destination):
    if destination.exists():
        raise ValueError("use a new package directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pool = source_pool(root)
    with tempfile.TemporaryDirectory(prefix=".transduction-package-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        current = root / "transduction"
        for path in payload_files(current):
            if path.suffix not in (".py", ".md", ".txt", ".tex", ".bib", ".pdf") and path.name != "Makefile":
                continue
            scan(path)
            target = staging / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        items = []
        for item in STUDIES:
            origin = root / item["directory"]
            if not ((origin / "summary.json").is_file() or (origin / "batch-summary.json").is_file()):
                raise ValueError(f"incomplete study: {item['id']}")
            for path in payload_files(origin):
                scan(path)
                target = staging / path.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            code = staging / "frozen" / item["id"]
            code.mkdir(parents=True)
            provenance = {}
            for path in current.glob("*.py"):
                if path.name.startswith("test_") or path.name == "repro_package.py":
                    continue
                shutil.copyfile(path, code / path.name)
                provenance[path.name] = dict(sha256=sha(path), origin="supplemental repository dependency")
            hashes = expected_sources(origin, item)
            regenerated_audit = False
            for name, value in hashes.items():
                if Path(name).name != name:
                    raise ValueError(f"missing matching source: {item['id']}/{name}/{value}")
                if (name, value) not in pool:
                    if name != (item["audit"] or "") + ".py":
                        raise ValueError(f"missing matching source: {item['id']}/{name}/{value}")
                    # Only an auxiliary auditor may be regenerated. Never
                    # substitute the discovery, compiler or execution sources.
                    provenance[name].update(origin="regenerated post hoc auditor",
                                            historical_sha256=value)
                    regenerated_audit = True
                    continue
                body, source = pool[(name, value)]
                (code / name).write_bytes(body)
                provenance[name] = dict(sha256=value, origin=source, matched_recorded_hash=True)
            write(code / "source-provenance.json", provenance)
            items.append(dict(**item, code=str(code.relative_to(staging)), recorded_sources=len(hashes),
                regenerated_audit=regenerated_audit,
                source_note=("recorded source hashes verified" if hashes else
                             "earliest pilot did not record source hashes; packaged replay implementation is newly bound")))
        versions = {name: importlib.metadata.version(name) for name in ("z3-solver", "pytest")}
        write(staging / "manifest.json", dict(version=1, created_utc=datetime.now(timezone.utc).isoformat(),
            python=sys.version, platform=platform.platform(), dependencies=versions, studies=items,
            source_checkout=str(root.resolve()),
            discarded_trials="excluded; intentionally invalid trials are not regenerated",
            replay="offline; no credentials; source and artifact files are not rewritten",
            relocation="composition basis path remapped to packaged growth artifact; original hashes checked"))
        shutil.copyfile(current / "REPRODUCIBILITY.md", staging / "README.md")
        seal(staging)
        staging.rename(destination)
    return dict(directory=str(destination), studies=len(items), files=checksums(destination))


def instruction_replay(module, folder):
    from codex_glue import extract
    from cofibration import verify_certificate
    rows = read(folder / "summary.json")["rows"]
    protocol = read(folder / "protocol.json")
    expected = {(seed, task, arm) for seed in protocol["seeds"] for task in protocol["tasks"] for arm in ("mechanical", "codex")}
    if {(r["seed"], r["task"], r["arm"]) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError("instruction trial missing or duplicated")
    for row in rows:
        part = folder / f"s{row['seed']}-{row['task']}-{row['arm']}"
        record = read(part / "selection.json")
        if any(record[k] != v for k, v in row.items() if k != "test"):
            raise ValueError("instruction selection differs from summary")
        square = verify_certificate(record["certificate"])
        rebuilt, entry = module.check_common_grammar(record["candidate"], square.left)
        if rebuilt.graph != square.graph or entry != record["entry"] or square.left.digest != row["library_hash"]:
            raise ValueError("instruction derivation differs")
        task = module.task_for(row["task"], row["seed"])
        for key, pairs in (("training", task.train_pairs), ("validation", task.val_pairs), ("test", task.test_pairs)):
            execution = module.evaluate(square.graph, entry, pairs, square.left.states)
            saved = read(part / "hidden-replay.json") if key == "test" else record[key + "_replay"]
            if normalized(execution) != saved or module.summary(execution) != row[key]:
                raise ValueError("instruction execution differs")
        reuse = module.necessary_reuse(square.graph, entry, task.train_pairs, square.left.states)
        if row["necessary_reuse"] != reuse or row["admitted"] != (row["training"]["exact"] == row["validation"]["exact"] == 1 and reuse):
            raise ValueError("instruction admission differs")
        if row["arm"] == "codex":
            raw, _ = extract([json.loads(line) for line in (part / "proposal/events.jsonl").read_text().splitlines() if line])
            if raw != record["candidate"] or read(part / "proposal/request.json") != normalized(module.prompt_for(square.left, task.train_pairs)):
                raise ValueError("instruction model provenance differs")
        else:
            raw, effort = module.mechanical(square.left, task.train_pairs, protocol["mechanical_max_evaluations"])
            if raw != record["candidate"] or effort["evaluated"] != row["effort"]["evaluated"]:
                raise ValueError("instruction mechanical selection differs")
    return dict(conditions=len(rows), verified_pushouts=len(rows), exact_replay=True)


def worker(package, name):
    manifest = read(package / "manifest.json")
    item = next(s for s in manifest["studies"] if s["id"] == name)
    original_checkout = Path(manifest["source_checkout"])
    def peak_mib():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)
    def memory_watchdog():
        import os
        while True:
            if peak_mib() > 768:
                print("replay worker exceeded 768 MiB", file=sys.stderr, flush=True)
                os._exit(73)
            time.sleep(.1)
    threading.Thread(target=memory_watchdog, daemon=True).start()
    sys.path.insert(0, str(package / item["code"]))
    sys.setrecursionlimit(10000)
    folder = package / item["directory"]
    module = importlib.import_module(item["module"])
    audit = importlib.import_module(item["audit"]) if item["audit"] else None
    writes = []

    def compare_only(path, value):
        if not path.is_file() or read(path) != normalized(value):
            raise ValueError(f"replay differs from saved artifact: {path}")
        writes.append(str(path.relative_to(package)))

    for loaded in list(sys.modules.values()):
        if getattr(loaded, "__file__", None) and Path(loaded.__file__).parent == package / item["code"]:
            if hasattr(loaded, "save"):
                loaded.save = compare_only

    def offline(event, args):
        if event in ("subprocess.Popen", "os.system", "socket.connect", "socket.getaddrinfo"):
            raise RuntimeError("offline replay prohibits processes and network access")
        if event == "open" and len(args) > 2 and isinstance(args[2], int):
            import os
            if args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                raise RuntimeError("offline replay prohibits artifact writes")
            if isinstance(args[0], (str, bytes)):
                path = Path(os.fsdecode(args[0])).resolve()
                if path.is_relative_to(original_checkout) and not path.is_relative_to(package):
                    raise RuntimeError("offline replay prohibits reads from the original checkout")
    sys.addaudithook(offline)
    kind = item["kind"]
    if kind == "register":
        result = module.replay_all(folder)
        compare_only(folder / "summary.json", result)
    elif kind == "tail":
        result = module.replay_run(folder)
    elif kind == "instruction":
        result = instruction_replay(module, folder)
    elif kind == "composition":
        load = module.load_basis
        original = read(folder / "protocol.json")["basis_folder"]
        def relocated(path):
            if str(path) != original:
                raise ValueError("unexpected basis path")
            return load(package / "output/transduction_growth/20261003-returning-ab")
        module.load_basis = relocated
        result = module.replay(folder, reproduce_search=read(folder / "verification.json")["mechanical_search_reproduced"])
    elif kind == "hard":
        with module.configured_engine() as engine:
            result = engine.replay(folder)
        from audit_frontier import exact_extension
        from interface_machine import InterfaceLibrary, Machine
        import itertools
        record = read(folder / "seed-41/codex/task-2/selection.json")
        library = InterfaceLibrary.read(record["library"])
        proof = exact_extension("walk6", library.cells["walk6"], "walk6f", library.cells["walk6f"], library.signatures())
        saved = read(folder / "interface-extension.json")
        if proof is None or any(saved[k] != v for k, v in proof.items()):
            raise ValueError("interface extension differs")
        machine, count = Machine(library), 0
        for flags in itertools.product((False, True), repeat=2):
            for split in ("train", "validation", "hidden", "stress"):
                for source, _ in module.pairs_for("framed_return", 1, split, 41):
                    old = machine.run("walk6", source, flags)
                    new = machine.run("walk6f", source, (*flags, "C01"))
                    if not old["ok"] or not new["ok"] or any(old[k] != new[k] for k in ("output", "cursor", "result")):
                        raise ValueError("interface specialization differs")
                    count += 1
        if count != saved["behavioral_specializations_correct"] or count != saved["behavioral_specializations_total"]:
            raise ValueError("incomplete extension audit")
    elif kind == "batch":
        result = module.replay_batch(folder)
    else:
        result = module.replay(folder)
    if audit:
        if item.get("regenerated_audit"):
            def compare_regenerated(path, value):
                prior = read(path)
                compare_report(prior, normalized(value))
                print(json.dumps(dict(regenerated_audit=str(path.relative_to(package)),
                    report=value, historical_report_sha256=sha(path),
                    all_previous_result_fields_match=True)), flush=True)
            audit.save = compare_regenerated
        folders = ([folder / t["directory"] for t in read(folder / "trials.json")["plan"]["trials"]]
                   if kind == "batch" else [folder])
        for target in folders:
            if item["audit"] in ("audit_growth", "audit_composition"):
                saved = target / "equality-audit-8.json"
                compare_only(saved, audit.audit(target, read(saved)["maximum_length"]))
            elif (target / "shape-audit.json").exists():
                saved = read(target / "shape-audit.json")
                audit.audit(target, saved["maximum_leaves"], saved["assignments_per_shape"])
    # Do not repeat huge row bodies in the operational log.
    compact = {k: v for k, v in result.items() if k not in ("rows", "historical_comparison", "trials")} if isinstance(result, dict) else {}
    print(json.dumps(dict(study=name, ok=True, results=compact, compared_files=len(writes),
                          peak_rss_mib=peak_mib())), flush=True)


def verify(package, results, only=None):
    if results.resolve().is_relative_to(package.resolve()):
        raise ValueError("verification results must be outside the immutable package")
    count = checksums(package)
    payload_manifest = sha(package / "SHA256SUMS")
    selected = [s for s in read(package / "manifest.json")["studies"] if only is None or s["id"] in only]
    if only is not None and {s["id"] for s in selected} != set(only):
        raise ValueError("unknown study")
    results.mkdir(parents=True, exist_ok=False)
    records = []
    for item in selected:
        start = time.monotonic()
        args = [sys.executable, "-B", str(package / "transduction/repro_package.py"),
                "--package", str(package), "--worker", item["id"]]
        proc = subprocess.run(args, text=True, capture_output=True, cwd=results, timeout=1200)
        (results / (item["id"] + ".log")).write_text(proc.stdout + proc.stderr)
        for line in proc.stdout.splitlines():
            if line.startswith('{"regenerated_audit":'):
                write(results / (item["id"] + "-regenerated-audit.json"), json.loads(line))
        row = dict(study=item["id"], ok=proc.returncode == 0, seconds=time.monotonic() - start)
        records.append(row)
        print(json.dumps(row), flush=True)
        if proc.returncode:
            raise RuntimeError(f"replay failed; inspect {results / (item['id'] + '.log')}")
    if checksums(package) != count:
        raise ValueError("replay modified package")
    report = dict(verified_files=count, payload_manifest_sha256=payload_manifest,
                  studies=records, offline=True, payload_unchanged=True,
                  complete=len(selected) == len(read(package / "manifest.json")["studies"]))
    write(results / "verification.json", report)
    return report


def compare_report(old, new):
    """A newly bound auditor must reproduce every old result, not its old hash."""
    if isinstance(old, dict):
        if not isinstance(new, dict):
            raise ValueError("audit structure changed")
        for key, value in old.items():
            if key in ("auditor_sha256", "audit_source_sha256", "audit_source_hash"):
                continue
            if key not in new:
                raise ValueError(f"audit omitted previous field: {key}")
            compare_report(value, new[key])
    elif isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            raise ValueError("audit omitted or added an old trial")
        for a, b in zip(old, new):
            compare_report(a, b)
    elif old != new:
        raise ValueError("regenerated audit changed a previous result")


def certify(package, results):
    count = checksums(package)
    report = read(results / "verification.json")
    expected = [s["id"] for s in read(package / "manifest.json")["studies"]]
    if (not report["complete"] or not report["payload_unchanged"] or report["verified_files"] != count
            or report["payload_manifest_sha256"] != sha(package / "SHA256SUMS")
            or [s["study"] for s in report["studies"]] != expected or not all(s["ok"] for s in report["studies"])):
        raise ValueError("complete matching verification required")
    if (package / "verification").exists():
        raise ValueError("already certified")
    shutil.copytree(results, package / "verification")
    (package / "SHA256SUMS").rename(package / "PAYLOAD_SHA256SUMS")
    seal(package)
    return dict(studies=len(expected), verified_files=checksums(package))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--only", nargs="+")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--checksums", action="store_true")
    mode.add_argument("--certify", action="store_true")
    mode.add_argument("--worker")
    args = parser.parse_args()
    if args.build:
        print(json.dumps(build(BASE, args.output.resolve())))
    elif args.worker:
        worker(args.package.resolve(), args.worker)
    elif args.checksums:
        print(json.dumps(dict(verified_files=checksums(args.package.resolve()))))
    elif args.certify:
        print(json.dumps(certify(args.package.resolve(), args.results.resolve())))
    else:
        verify(args.package.resolve(), args.results.resolve(), args.only)


if __name__ == "__main__":
    main()
