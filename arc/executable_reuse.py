"""Post-hoc source and execution accounting for unrestricted Python solver glue.

The static analysis is a finite, context-insensitive points-to fixed point. It
reports named call paths and unresolved sites, never a grammar for candidates.
The replay observer independently records actual Python calls in supplied solver
sources. Neither component executes a solver or decides whether it is admitted.
"""

from __future__ import annotations

import ast
import builtins
import __future__
from collections import defaultdict, deque
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import PurePosixPath
import sys
import threading
from types import CodeType, FunctionType, MethodType
import weakref


UNKNOWN = ("unknown", "")
EMPTY = frozenset()


@dataclass
class Definition:
    key: str
    filename: str
    name: str
    parent: str
    node: ast.AST
    digest: str


def source_hashes(files: dict[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(files.items())}


class Program:
    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.defs: dict[str, Definition] = {}
        self.modules: dict[str, str] = {}
        self.trees: dict[str, ast.Module] = {}
        self.errors = []
        self.children = defaultdict(dict)
        for filename, raw in sorted(files.items()):
            if not filename.endswith(".py"):
                continue
            try:
                tree = ast.parse(raw, filename=filename)
            except (SyntaxError, ValueError) as exc:
                self.errors.append({"filename": filename, "reason": type(exc).__name__})
                continue
            self.trees[filename] = tree
            module = filename.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
            self.modules[module] = filename
        basenames = defaultdict(list)
        for module, filename in self.modules.items():
            basenames[module.rsplit(".", 1)[-1]].append(filename)
        self.modules.update({name: paths[0] for name, paths in basenames.items() if len(paths) == 1})
        records = []
        def collect(filename, parent, prefix, body):
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    record = {"filename": filename, "parent": parent, "qualname": prefix + node.name, "node": node}
                    records.append(record)
                    collect(filename, record, prefix + node.name + ".", node.body)
                else:
                    for field in ("body", "orelse", "finalbody"):
                        children = getattr(node, field, None)
                        if isinstance(children, list):collect(filename, parent, prefix, children)
                    for handler in getattr(node, "handlers", []):collect(filename, parent, prefix, handler.body)
        for filename, tree in self.trees.items():collect(filename, None, "", tree.body)
        latest = {f"{r['filename']}:{r['qualname']}": r for r in records}
        self.node_keys = {}
        self.locations = {}
        for record in records:
            node = record["node"];base = f"{record['filename']}:{record['qualname']}"
            key = base if latest[base] is record else base + "@" + str(node.lineno)
            record["key"] = key
            self.node_keys[id(node)] = key
        for record in records:
            node = record["node"];filename = record["filename"];key = record["key"]
            parent = record["parent"]["key"] if record["parent"] else filename + ":<module>"
            digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            self.defs[key] = Definition(key, filename, node.name, parent, node, digest)
            self.children[parent][node.name] = key
            first_line = min([node.lineno] + [d.lineno for d in node.decorator_list])
            self.locations[(filename, record["qualname"], first_line)] = key

    def preceding_match(self, definition):
        base = definition.key.rsplit("@", 1)[0]
        matches = [old.key for old in self.defs.values() if old.digest == definition.digest]
        return next((key for key in matches if key.rsplit("@", 1)[0] == base), min(matches) if matches else None)


class SourceAnalysis:
    def __init__(self, previous, current, entrypoints, target_files=None):
        self.old, self.program = Program(previous), Program(current)
        self.roots = list(entrypoints)
        self.target_files = target_files
        self.env = defaultdict(dict)
        self.parameters = defaultdict(dict)
        self.returns = defaultdict(frozenset)
        self.active = set()
        self.edges = set()
        self.tables = {}
        self.table_initials = {}
        self.shared_table_writes = defaultdict(dict)
        self.default_values = {}
        self.sequences = {}
        self.attributes = {}
        self.module_allocations = set()
        self.in_module_initialization = False
        self.active_modules = {root.split(":", 1)[0] for root in entrypoints}
        self.unresolved = {}
        self.changed = False

    def union(self, table, key, values):
        new = table.get(key, EMPTY) | frozenset(values)
        if table.get(key, EMPTY) != new:
            table[key] = new
            self.changed = True

    def module_scope(self, scope):
        return scope.split(":", 1)[0] + ":<module>"

    def lookup(self, scope, name):
        if name in self.env[scope]:
            return self.env[scope][name]
        definition = self.program.defs.get(scope)
        if definition:
            parent = definition.parent
            # Methods do not use their class namespace as a lexical scope.
            if parent in self.program.defs and isinstance(self.program.defs[parent].node, ast.ClassDef):
                parent = self.program.defs[parent].parent
            return self.lookup(parent, name)
        if hasattr(builtins, name):
            return frozenset({("builtin", name)})
        return frozenset({UNKNOWN})

    def attr(self, values, name):
        out = set()
        for kind, value in values:
            if ((kind, value), name) in self.attributes:
                out.update(self.attributes[((kind, value), name)])
                continue
            if kind == "module":
                out.update(self.env[f"{value}:<module>"].get(name, {UNKNOWN}))
            elif kind in {"instance", "class"}:
                owner = value.split("|instance|", 1)[0]
                target = self.program.children[owner].get(name)
                if target:
                    decorators = self.program.defs[target].node.decorator_list
                    tags = {d.id for d in decorators if isinstance(d, ast.Name)}
                    if any(not isinstance(d, ast.Name) or d.id not in {"staticmethod", "classmethod"} for d in decorators):
                        out.add(UNKNOWN)
                    elif "staticmethod" in tags or kind == "class" and "classmethod" not in tags:
                        out.add(("function", target))
                    else:
                        receiver = ("class", owner) if "classmethod" in tags else (kind, value)
                        out.add(("bound", json.dumps([target, receiver])))
                else:
                    out.update(self.env[owner].get(name, {UNKNOWN}))
            elif kind == "external":
                # Attribute spelling comes from a finite AST, not a recursively
                # growing path such as arena.child.child.child.
                out.add(("external", "member:" + name))
            else:
                out.add(UNKNOWN)
        return frozenset(out)

    def expr(self, node, scope):
        if node is None:
            return EMPTY
        if isinstance(node, ast.Name):
            return self.lookup(scope, node.id)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool):value = int(value)
            if isinstance(value, float) and math.isfinite(value) and value.is_integer():value = int(value)
            return frozenset({("literal", json.dumps(value))}) if isinstance(value, (str, int, float, type(None))) else frozenset({UNKNOWN})
        if isinstance(node, ast.Attribute):
            result = self.attr(self.expr(node.value, scope), node.attr)
            if UNKNOWN in result:
                self.unresolved[(scope, node.lineno, node.col_offset)] = "attribute or descriptor binding requires replay resolution"
            return result
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            key = f"{scope}|{node.lineno}|{node.col_offset}"
            self.sequences[key] = tuple(self.expr(x, scope) for x in node.elts)
            return frozenset({("sequence", key)})
        if isinstance(node, ast.Dict):
            key = f"{scope}|{node.lineno}|{node.col_offset}"
            fields = {}
            for k, v in zip(node.keys, node.values):
                for kind, literal in self.expr(k, scope) if k is not None else {UNKNOWN}:
                    fields[literal if kind == "literal" else "*"] = self.expr(v, scope)
            self.tables[key] = fields
            self.table_initials[key] = dict(fields)
            if self.in_module_initialization:self.module_allocations.add(key)
            return frozenset({("table", key)})
        if isinstance(node, ast.Subscript):
            values = self.expr(node.value, scope)
            keys = self.expr(node.slice, scope)
            out = set()
            for kind, value in values:
                if kind == "scope":
                    for k, literal in keys:
                        out.update(self.lookup(value, json.loads(literal)) if k == "literal" and isinstance(json.loads(literal), str) else {UNKNOWN})
                elif kind == "table":
                    fields = dict(self.tables.get(value, {}))
                    if "*" in fields:
                        out.update(fields["*"]);out.add(UNKNOWN)
                    for writer, writes in self.shared_table_writes.items():
                        if writer == scope:continue
                        for (table_id, field), replacements in writes.items():
                            if table_id == value:
                                fields[field] = fields.get(field, EMPTY) | replacements
                                self.unresolved[(scope, node.lineno, node.col_offset)] = "shared table is mutated by a called function; replay resolves ordering"
                    unknown_key = not keys or any(k != "literal" for k, _ in keys)
                    for k, literal in keys:
                        if k == "literal":out.update(fields.get(literal, {UNKNOWN}))
                    if unknown_key:
                        out.update(frozenset().union(*fields.values()))
                        out.add(UNKNOWN)
                elif kind == "sequence":
                    elements = self.sequences.get(value, ())
                    for k, literal in keys:
                        index = json.loads(literal) if k == "literal" else None
                        if type(index) is int and -len(elements) <= index < len(elements):out.update(elements[index])
                        else:out.update(frozenset().union(*elements));out.add(UNKNOWN)
                else:
                    out.add(UNKNOWN)
            return frozenset(out or {UNKNOWN})
        if isinstance(node, ast.IfExp):
            if isinstance(node.test, ast.Constant):
                return self.expr(node.body if node.test.value else node.orelse, scope)
            return self.expr(node.body, scope) | self.expr(node.orelse, scope)
        if isinstance(node, ast.Lambda):
            self.unresolved[(scope, node.lineno, node.col_offset)] = "lambda requires replay identification"
            return frozenset({UNKNOWN})
        if isinstance(node, ast.Call):
            args = [self.expr(x, scope) for x in node.args]
            kwargs = {k.arg: self.expr(k.value, scope) for k in node.keywords}
            return self.call(self.expr(node.func, scope), args, kwargs, scope, node)
        # Visit calls inside arithmetic/comparisons without guessing their result.
        for child in ast.iter_child_nodes(node):
            self.expr(child, scope)
        if isinstance(node, ast.expr):
            self.unresolved[(scope, node.lineno, node.col_offset)] = "runtime expression or implicit protocol call is recorded during replay"
        return frozenset({UNKNOWN})

    def call(self, targets, args, kwargs, scope, site):
        result = set()
        if not targets:
            self.unresolved[(scope, site.lineno, site.col_offset)] = "call target has no resolved source binding"
        if len(targets) > 1:
            self.unresolved[(scope, site.lineno, site.col_offset)] = "multiple possible call targets"
        for kind, target in targets:
            if kind in {"function", "bound", "class"}:
                if kind == "class":
                    instance = ("instance", target + f"|instance|{scope}|{site.lineno}|{site.col_offset}")
                    result.add(instance)
                    initializer = self.program.children[target].get("__init__")
                    if initializer:
                        self.call({("bound", json.dumps([initializer, instance]))}, args, kwargs, scope, site)
                    continue
                receiver = None
                if kind == "bound":target, receiver = json.loads(target)
                definition = self.program.defs[target]
                if not isinstance(definition.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.unresolved[(scope, site.lineno, site.col_offset)] = "callable object requires replay resolution"
                    result.add(UNKNOWN)
                    continue
                self.edges.add((scope, target))
                if target not in self.active:
                    self.active.add(target)
                    self.changed = True
                positional = list(args)
                if kind == "bound":
                    positional.insert(0, frozenset({tuple(receiver)}))
                params = [*definition.node.args.posonlyargs, *definition.node.args.args]
                captured_pos, captured_kw = self.default_values.get(target, ((), {}))
                defaults = [None] * (len(params) - len(captured_pos)) + list(captured_pos)
                for i, parameter in enumerate(params):
                    value = positional[i] if i < len(positional) else kwargs.get(parameter.arg)
                    if value is None:
                        value = defaults[i] if defaults[i] is not None else {UNKNOWN}
                    self.union(self.parameters[target], parameter.arg, value)
                for parameter, default in zip(definition.node.args.kwonlyargs, definition.node.args.kw_defaults):
                    self.union(self.parameters[target], parameter.arg, kwargs.get(parameter.arg, captured_kw.get(parameter.arg, {UNKNOWN})))
                for parameter in (definition.node.args.vararg, definition.node.args.kwarg):
                    if parameter:self.union(self.parameters[target], parameter.arg, {UNKNOWN})
                result.update(self.returns[target])
            elif kind in {"external", "builtin"}:
                if kind == "builtin" and target in {"setattr", "delattr", "exec", "eval", "compile", "__import__"}:
                    self.unresolved[(scope, site.lineno, site.col_offset)] = "runtime binding or code construction is observed during replay"
                if kind == "builtin" and target in {"globals", "locals"}:
                    result.add(("scope", self.module_scope(scope) if target == "globals" else scope))
                    continue
                if kind == "builtin" and target == "getattr" and len(args) >= 2:
                    for k, literal in args[1]:
                        if k == "literal" and isinstance(json.loads(literal), str):
                            result.update(self.attr(args[0], json.loads(literal)))
                        else:result.add(UNKNOWN)
                    if UNKNOWN in result:self.unresolved[(scope, site.lineno, site.col_offset)] = "computed attribute requires replay resolution"
                    continue
                # Passing a solver callback outside the supplied source closure is
                # explicit unresolved evidence, not a claim that it was unused.
                if any(k in {"function", "bound", "class"} for values in [*args, *kwargs.values()] for k, _ in values):
                    self.unresolved[(scope, site.lineno, site.col_offset)] = "solver callback passed to external callable"
                if kind == "external" and target == "member:clone":
                    result.add(("external", "arena"))
                else:
                    result.add(UNKNOWN)
            else:
                self.unresolved[(scope, site.lineno, site.col_offset)] = "call target depends on values not resolved from supplied sources"
                result.add(UNKNOWN)
        return frozenset(result)

    def assign(self, target, values, scope):
        if isinstance(target, ast.Name):
            self.env[scope][target.id] = frozenset(values)
        elif isinstance(target, (ast.Tuple, ast.List)):
            choices = [self.sequences[v] for k, v in values if k == "sequence" and v in self.sequences]
            if len(choices) == 1 and len(choices[0]) == len(target.elts) and len(values) == 1:
                for child, item in zip(target.elts, choices[0]):self.assign(child, item, scope)
            else:
                for child in target.elts:self.assign(child, {UNKNOWN}, scope)
                self.unresolved[(scope, target.lineno, target.col_offset)] = "destructuring binding requires replay resolution"
        elif isinstance(target, ast.Attribute):
            receivers = self.expr(target.value, scope)
            for receiver in receivers:
                key = (receiver, target.attr)
                self.attributes[key] = frozenset(values) if len(receivers) == 1 else self.attributes.get(key, EMPTY) | values
                self.unresolved[(scope, target.lineno, target.col_offset)] = "attribute update; replay identifies the concrete receiver and binding"
        elif isinstance(target, ast.Subscript):
            receivers, keys = self.expr(target.value, scope), self.expr(target.slice, scope)
            for kind, value in receivers:
                if kind != "table":
                    self.unresolved[(scope, target.lineno, target.col_offset)] = "container mutation requires replay resolution"
                    continue
                fields = self.tables.setdefault(value, {})
                if self.in_module_initialization:self.module_allocations.add(value)
                for k, literal in keys:
                    if k == "literal" and len(receivers) == len(keys) == 1:
                        fields[literal] = frozenset(values)
                        if not scope.endswith(":<module>"):
                            identity = (value, literal)
                            self.union(self.shared_table_writes[scope], identity, values)
                    else:
                        for name in fields:fields[name] |= values
                        self.unresolved[(scope, target.lineno, target.col_offset)] = "computed container mutation requires replay resolution"

    def statements(self, body, scope):
        filename = scope.split(":", 1)[0]
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                key = self.program.node_keys[id(node)]
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                if kind == "function":
                    self.default_values[key] = (tuple(self.expr(d, scope) for d in node.args.defaults),
                        {p.arg: self.expr(d, scope) for p, d in zip(node.args.kwonlyargs, node.args.kw_defaults) if d is not None})
                self.env[scope][node.name] = frozenset({(kind, key)})
                if isinstance(node, ast.ClassDef):
                    self.env[key] = {}
                    self.statements(node.body, key)
                if node.decorator_list and any(not isinstance(d, ast.Name) or d.id not in {"staticmethod", "classmethod"} for d in node.decorator_list):
                    self.env[scope][node.name] |= {UNKNOWN}
                    self.unresolved[(scope, node.lineno, node.col_offset)] = "decorated binding also requires replay resolution"
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        path = self.program.modules.get(alias.name)
                        if path and path not in self.active_modules:self.active_modules.add(path);self.changed = True
                        if path:self.edges.add((scope, path + ":<module>"))
                        self.env[scope][alias.asname or alias.name.split(".")[0]] = frozenset({("module", path) if path else ("external", alias.name)})
                else:
                    module = node.module or ""
                    if node.level:
                        prefix = PurePosixPath(filename).parent.parts
                        module = ".".join((*prefix[:len(prefix) - node.level + 1], module)).strip(".")
                    path = self.program.modules.get(module)
                    if path and path not in self.active_modules:self.active_modules.add(path);self.changed = True
                    if path:self.edges.add((scope, path + ":<module>"))
                    for alias in node.names:
                        if alias.name == "*":
                            if path:
                                self.env[scope].update({k: v for k, v in self.env[f"{path}:<module>"].items() if not k.startswith("_")})
                            else:
                                self.unresolved[(scope, node.lineno, node.col_offset)] = "star import outside the supplied source closure"
                        else:
                            self.env[scope][alias.asname or alias.name] = self.env[f"{path}:<module>"].get(alias.name, EMPTY) if path else frozenset({("external", module + "." + alias.name)})
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = self.expr(node.value, scope)
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    self.assign(target, value, scope)
            elif isinstance(node, ast.Return):
                self.union(self.returns, scope, self.expr(node.value, scope))
                break
            elif isinstance(node, ast.If):
                condition = self.expr(node.test, scope)
                if any(kind != "literal" for kind, _ in condition):
                    self.unresolved[(scope, node.lineno, node.col_offset)] = "runtime control or truth protocol is recorded during replay"
                if isinstance(node.test, ast.Constant):
                    self.statements(node.body if node.test.value else node.orelse, scope)
                else:
                    initial = dict(self.env[scope]);self.statements(node.body, scope)
                    first = dict(self.env[scope]);self.env[scope] = initial.copy();self.statements(node.orelse, scope)
                    for name in first.keys() | self.env[scope].keys():
                        self.env[scope][name] = first.get(name, EMPTY) | self.env[scope].get(name, EMPTY)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                iterable = self.expr(node.iter, scope)
                if iterable and all(k == "sequence" and not self.sequences.get(v) for k, v in iterable):
                    self.statements(node.orelse, scope)
                    continue
                elements = frozenset().union(*(frozenset().union(*self.sequences[v]) if k == "sequence" and v in self.sequences else {UNKNOWN} for k, v in iterable))
                self.assign(node.target, elements, scope)
                self.statements(node.body, scope);self.statements(node.orelse, scope)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    value = self.expr(item.context_expr, scope)
                    if item.optional_vars:self.assign(item.optional_vars, value, scope)
                self.statements(node.body, scope)
            elif isinstance(node, (ast.Try, ast.TryStar)):
                self.statements(node.body, scope)
                for handler in node.handlers:self.statements(handler.body, scope)
                self.statements(node.orelse, scope);self.statements(node.finalbody, scope)
            elif isinstance(node, ast.While):
                if isinstance(node.test, ast.Constant) and not node.test.value:
                    self.statements(node.orelse, scope)
                    continue
                self.expr(node.test, scope);self.statements(node.body, scope)
            elif isinstance(node, (ast.Global, ast.Nonlocal, ast.Delete, ast.AugAssign)):
                self.unresolved[(scope, node.lineno, node.col_offset)] = "shared or in-place binding update requires replay resolution"
            else:
                self.expr(node, scope)

    def seed_entrypoints(self):
        resolved = set()
        for root in self.roots:
            filename, qualname = root.split(":", 1)
            if qualname == "<module>":
                if filename in self.program.trees:resolved.add(root)
                continue
            parts = qualname.split(".")
            values = self.env[filename + ":<module>"].get(parts[0], EMPTY)
            for name in parts[1:]:values = self.attr(values, name)
            expanded = set()
            for kind, target in values:
                expanded.update(self.attr({(kind, target)}, "__call__") if kind == "instance" else {(kind, target)})
            values = frozenset(expanded)
            if not values:
                self.unresolved[(filename + ":<module>", 0, 0)] = "entrypoint binding is absent or unresolved"
            for kind, target in values:
                if kind not in {"function", "bound"}:
                    self.unresolved[(filename + ":<module>", 0, 0)] = "entrypoint binding requires replay resolution"
                    continue
                receiver = None
                if kind == "bound":target, receiver = json.loads(target)
                resolved.add(target)
                if target not in self.active:self.active.add(target);self.changed = True
                definition = self.program.defs[target];node = definition.node
                for parameter in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                    parent = definition.parent
                    if parameter.arg in {"self", "cls"} and parent in self.program.defs and isinstance(self.program.defs[parent].node, ast.ClassDef):
                        value = tuple(receiver) if receiver is not None else ("instance" if parameter.arg == "self" else "class", parent)
                    else:value = ("external", "arena") if parameter.arg == "env" else UNKNOWN
                    self.union(self.parameters[target], parameter.arg, {value})
                for parameter in (node.args.vararg, node.args.kwarg):
                    if parameter:self.union(self.parameters[target], parameter.arg, {UNKNOWN})
        self.resolved_roots = resolved

    def run(self):
        seen_states = set()
        accumulated_edges = set()
        while True:
            self.changed = False
            self.unresolved = {}
            self.edges = set()
            self.tables = {key: dict(value) for key, value in self.table_initials.items()}
            self.attributes = {}
            before = {scope: dict(env) for scope, env in self.env.items()}
            self.in_module_initialization = True
            for filename, tree in self.program.trees.items():
                if filename not in self.active_modules:continue
                scope = f"{filename}:<module>"
                self.env[scope] = {}
                self.statements(tree.body, scope)
            self.seed_entrypoints()
            self.in_module_initialization = False
            module_tables = {key: dict(value) for key, value in self.tables.items() if key in self.module_allocations}
            module_attributes = dict(self.attributes)
            for scope in sorted(self.active):
                self.tables = {key: dict(value) for key, value in self.table_initials.items()}
                self.tables.update({key: dict(value) for key, value in module_tables.items()})
                self.attributes = dict(module_attributes)
                self.env[scope] = dict(self.parameters[scope])
                self.statements(self.program.defs[scope].node.body, scope)
            accumulated_edges.update(self.edges)
            if not self.changed and before == dict(self.env):
                break
            state = repr((
                sorted((scope, sorted((name, sorted(values)) for name, values in env.items())) for scope, env in self.env.items()),
                sorted((scope, sorted((name, sorted(values)) for name, values in env.items())) for scope, env in self.parameters.items()),
                sorted((scope, sorted(values)) for scope, values in self.returns.items()),
                sorted((writer, sorted((key, sorted(values)) for key, values in writes.items())) for writer, writes in self.shared_table_writes.items()),
                sorted(self.active), sorted(self.active_modules),
            ))
            if state in seen_states:
                self.edges.update(accumulated_edges)
                for root in self.roots:
                    self.unresolved[(self.module_scope(root), 0, 0)] = "cyclic abstract binding requires replay resolution"
                break
            seen_states.add(state)
        graph = defaultdict(set)
        for source, target in self.edges:graph[source].add(target)
        queue = deque([[root] for root in sorted(self.resolved_roots)]);visited = set();paths = {}
        while queue:
            path = queue.popleft();key = path[-1]
            if key in visited:continue
            visited.add(key)
            current = self.program.defs.get(key)
            if current and self.old.preceding_match(current) and (self.target_files is None or current.filename in self.target_files):
                paths[key] = path
            queue.extend(path + [target] for target in sorted(graph[key]) if target not in visited)
        root_modules = {root.split(":", 1)[0] + ":<module>" for root in self.roots}
        unresolved = [{"caller": caller, "line": line, "column": column, "reason": reason}
                      for (caller, line, column), reason in sorted(self.unresolved.items()) if caller in visited or caller in root_modules]
        status = ("missing_entrypoint" if not self.resolved_roots else
                  "resolved_source_graph" if not unresolved and not self.program.errors else "partially_resolved_source_graph")
        return {"status": status,
                "entrypoints": self.roots, "resolved_entrypoints": sorted(self.resolved_roots), "paths": paths,
                "retained_entrypoint_targets": sorted(key for key, path in paths.items() if len(path) == 1),
                "direct_targets": sorted(key for key, path in paths.items() if len(path) == 2),
                "indirect_targets": sorted(key for key, path in paths.items() if len(path) > 2),
                "unchanged_definition_sha256": {key: self.program.defs[key].digest for key in paths},
                "preceding_definitions": {key: self.old.preceding_match(self.program.defs[key]) for key in paths},
                "definition_matching": "normalized-AST content identity; preceding bindings are recorded, including literal definitions rebound into new scopes",
                "unresolved_calls": unresolved, "parse_errors": self.program.errors,
                "evidence_kind": "static call paths; actual calls are recorded by the replay observer"}


def analyse_reuse(previous, current, entrypoints, *, target_files=None):
    result = SourceAnalysis(previous, current, entrypoints, target_files).run()
    return {"schema": 1, "previous_source_sha256": source_hashes(previous),
            "current_source_sha256": source_hashes(current), **result}


def code_fingerprint(code):
    """Location-independent code identity within the recorded Python version."""
    def constant(value):
        if isinstance(value, CodeType):return ["code", code_fingerprint(value)]
        if isinstance(value, bytes):return ["bytes", value.hex()]
        if isinstance(value, (tuple, frozenset)):
            values = [constant(v) for v in value]
            return [type(value).__name__, sorted(values, key=repr) if isinstance(value, frozenset) else values]
        return [type(value).__name__, repr(value)]
    record = [code.co_code.hex(), code.co_argcount, code.co_posonlyargcount,
              code.co_kwonlyargcount, code.co_flags, code.co_stacksize, code.co_nlocals,
              getattr(code, "co_exceptiontable", b"").hex(), code.co_names, code.co_varnames,
              code.co_freevars, code.co_cellvars, [constant(v) for v in code.co_consts]]
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


class ReplayCalls:
    """Observe solver calls inside an existing independent replay, without rewrites.

    runtime_paths maps snapshot filenames to the exact filenames used to compile
    them. The context records current/new-thread calls in this process. It never
    inspects framework/game bytecode or solver arguments. Attach it in the process
    executing the solver; subprocess workers require their own observer.
    """
    def __init__(self, previous, current, runtime_paths=None, entrypoints=(), entrypoint_objects=()):
        self.previous, self.current = previous, current
        self.program = Program(current)
        self.runtime_paths = runtime_paths or {name: name for name in current}
        self.roots = set(entrypoints)
        self.requested_roots = sorted(self.roots)
        if self.roots:
            resolved = analyse_reuse(previous, current, sorted(self.roots)).get("resolved_entrypoints", [])
            self.roots.update(resolved)
        self.index = {};self.counts = defaultdict(int);self.edges = defaultdict(int)
        self.errors = [];self.globals = set();self.code_cache = {}
        self.local = threading.local()
        self.scheduled_threads = {}
        self.thread_start_code = threading.Thread.start.__code__
        self.current_thread = threading.current_thread
        self.get_ident = threading.get_ident
        self.enabled = False
        self.entrypoint_events = defaultdict(int)
        self.observed_code = {}
        self.namespaces = {}
        self.context_scope = ContextVar("replay_call_scope", default=False)
        self.context_tokens = {}
        self.root_codes = {obj.__code__ if isinstance(obj, FunctionType) else obj.__func__.__code__
                           for obj in entrypoint_objects if isinstance(obj, (FunctionType, MethodType))}
        self.untracked_call_events = 0
        self.compile_modes = set()
        for filename, runtime in self.runtime_paths.items():
            if filename not in current:raise ValueError("runtime file is outside the source snapshot")
            self._index_code(filename, runtime, 0)
        self.runtime_names = set(self.runtime_paths.values())
        self.handler = self._profile

    def _index_code(self, filename, runtime, flags):
            mode = (filename, runtime, flags)
            if mode in self.compile_modes:return
            self.compile_modes.add(mode)
            compiled = compile(self.current[filename], runtime, "exec", flags=flags, dont_inherit=True)
            pending = [compiled]
            while pending:
                code = pending.pop();qualname = code.co_qualname.replace(".<locals>.", ".")
                key = self.program.locations.get((filename, qualname, code.co_firstlineno), f"{filename}:{qualname}@{code.co_firstlineno}")
                self.index[(runtime, code_fingerprint(code), qualname, code.co_firstlineno)] = key
                pending.extend(v for v in code.co_consts if isinstance(v, CodeType))
    def identify(self, frame):
        code = frame.f_code
        if code.co_filename not in self.runtime_names:
            if id(frame.f_globals) in self.globals:
                fingerprint = code_fingerprint(code)
                key = "<generated>:" + code.co_qualname + ":" + fingerprint
                self.code_cache[code] = key
                self.observed_code[key] = fingerprint
                return key
            return None
        if code in self.code_cache:return self.code_cache[code]
        qualname = code.co_qualname.replace(".<locals>.", ".")
        fingerprint = code_fingerprint(code)
        key = self.index.get((code.co_filename, fingerprint, qualname, code.co_firstlineno))
        if key is None:
            future_flags = 0
            for name in __future__.all_feature_names:
                future_flags |= getattr(__future__, name).compiler_flag
            flags = code.co_flags & future_flags
            for filename, runtime in self.runtime_paths.items():
                if runtime == code.co_filename:self._index_code(filename, runtime, flags)
            key = self.index.get((code.co_filename, fingerprint, qualname, code.co_firstlineno))
        if key is None:
            self.errors.append("runtime code differs from the supplied snapshot: " + code.co_filename + ":" + qualname)
        self.code_cache[code] = key
        if key:self.observed_code[key] = fingerprint
        if key:
            self.globals.add(id(frame.f_globals))
            for filename, runtime in self.runtime_paths.items():
                if runtime == code.co_filename:self.namespaces[filename] = frame.f_globals
        return key

    def _profile(self, frame, event, arg):
        old = self.old_sys if self.get_ident() == self.main_thread else self.old_thread
        if old is not None:old(frame, event, arg)
        if not self.enabled:
            sys.setprofile(old)
            return
        if event not in {"call", "return"}:return
        try:
            inherited = getattr(self.local, "inherited", None)
            if inherited is None:
                thread = self.current_thread()
                record = self.scheduled_threads.get(id(thread))
                inherited = bool(record and record[0]() is thread and record[1])
                self.local.inherited = inherited
            depth = getattr(self.local, "depth", 0)
            if event == "call" and frame.f_code is self.thread_start_code:
                # Scheduling metadata only: solver arguments and Thread target
                # arguments are neither read nor exported.
                worker = frame.f_locals.get("self")
                self.scheduled_threads[id(worker)] = (weakref.ref(worker), bool(depth or inherited or self.context_scope.get()))
            key = self.identify(frame)
            if key is None:
                if event == "call":self.untracked_call_events += 1
                return
            root = key.rsplit("@", 1)[0] in self.roots or frame.f_code in self.root_codes
            # A runtime binding can be a lambda, closure, decorated function or
            # alias even when source-only resolution cannot name it.
            for requested in self.requested_roots:
                filename, name = requested.split(":", 1)
                if "." in name:continue
                namespace = self.namespaces.get(filename)
                if namespace is None:continue
                value = namespace.get(name)
                if isinstance(value, FunctionType):root |= value.__code__ is frame.f_code
                elif isinstance(value, MethodType):root |= value.__func__.__code__ is frame.f_code
                elif value is not None:
                    # Raw type dictionaries avoid calling user descriptors from
                    # the observer while recognizing callable-object entries.
                    for cls in type.__getattribute__(type(value), "__mro__"):
                        method = type.__getattribute__(cls, "__dict__").get("__call__")
                        if isinstance(method, FunctionType):
                            root |= method.__code__ is frame.f_code
                            break
            if root:
                self.local.depth = depth + (1 if event == "call" else -1)
                if event == "call":
                    self.entrypoint_events[key] += 1
                    self.context_tokens[id(frame)] = self.context_scope.set(True)
                else:
                    token = self.context_tokens.pop(id(frame), None)
                    if token is not None:self.context_scope.reset(token)
            if event != "call" or (self.roots and not (getattr(self.local, "depth", 0) or inherited or self.context_scope.get())):return
            self.counts[key] += 1
            caller = frame.f_back
            while caller is not None:
                parent = self.identify(caller)
                if parent:
                    self.edges[(parent, key)] += 1;break
                caller = caller.f_back
        except Exception as exc:
            # Accounting errors are data, not failures imposed on candidate code.
            self.errors.append(type(exc).__name__)

    def __enter__(self):
        self.old_sys, self.old_thread = sys.getprofile(), threading.getprofile()
        self.main_thread = self.get_ident()
        self.enabled = True
        sys.setprofile(self.handler);threading.setprofile(self.handler)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if sys.getprofile() is not self.handler:
            self.errors.append("program replaced the profile observer")
        self.enabled = False
        sys.setprofile(self.old_sys);threading.setprofile(self.old_thread)
        self.namespaces.clear()
        self.globals.clear()
        self.context_tokens.clear()
        if self.requested_roots and not self.entrypoint_events:
            self.errors.append("declared entrypoint was not observed")
        successful_exit = exc_type is SystemExit and exc.code in (None, 0)
        self.outcome = "returned" if exc_type is None or successful_exit else "raised"

    def report(self):
        old = Program(self.previous)
        retained = {key: count for key, count in sorted(self.counts.items())
                    if key in self.program.defs and old.preceding_match(self.program.defs[key])}
        prior_bodies = defaultdict(set)
        for filename, raw in self.previous.items():
            if not filename.endswith('.py'):continue
            for flags in {mode[2] for mode in self.compile_modes}:
                try:pending = [compile(raw, filename, 'exec', flags=flags, dont_inherit=True)]
                except (SyntaxError, ValueError):continue
                while pending:
                    code = pending.pop()
                    if code.co_name != '<module>':
                        label = f"{filename}:{code.co_qualname.replace('.<locals>.', '.')}@{code.co_firstlineno}"
                        prior_bodies[code_fingerprint(code)].add(label)
                    pending.extend(value for value in code.co_consts if isinstance(value, CodeType))
        body_matches = {key: {'call_events': count, 'code_sha256': self.observed_code[key],
                              'previous_code_origins': sorted(prior_bodies[self.observed_code[key]])}
                        for key, count in sorted(self.counts.items()) if self.observed_code.get(key) in prior_bodies}
        return {"schema": 1, "evidence_kind": "observed Python calls in independent replay",
                "python": sys.version, "previous_source_sha256": source_hashes(self.previous),
                "compile_flags": sorted({flags for _, _, flags in self.compile_modes}),
                "current_source_sha256": source_hashes(self.current), "entrypoints": self.requested_roots,
                "resolved_entrypoints": sorted(self.roots),
                "entrypoint_call_events": dict(sorted(self.entrypoint_events.items())),
                "call_events": dict(sorted(self.counts.items())), "observed_retained_call_events": retained,
                "observed_prior_code_body_matches": body_matches,
                "code_body_match_scope": "compiled bodies under recorded compiler modes; defaults and closure values are not inferred from bytecode identity",
                "edges": [{"caller": a, "callee": b, "call_events": n} for (a, b), n in sorted(self.edges.items())],
                "untracked_call_events": self.untracked_call_events,
                "errors": sorted(set(self.errors)), "execution_outcome": getattr(self, "outcome", "in_progress"),
                "scope": "tracked solver sources and generated functions in their module globals, current process and new threads; call events include generator resumptions; framework/game identities and arguments are not recorded"}


def attach_replay(analysis, observation):
    """Bind observed calls to exactly the source pair being accounted for."""
    for key in ("previous_source_sha256", "current_source_sha256"):
        if analysis.get(key) != observation.get(key):
            raise ValueError("replay source binding differs: " + key)
    if observation.get("errors"):
        raise ValueError("replay observer reported binding or collection errors")
    if observation.get("execution_outcome") != "returned":
        raise ValueError("replay execution did not return successfully")
    if set(observation.get("entrypoints", [])) != set(analysis.get("entrypoints", [])):
        raise ValueError("replay entrypoint scope differs")
    return {**analysis, "observed_replay_calls": observation}
