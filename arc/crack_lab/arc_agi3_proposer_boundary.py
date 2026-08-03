#!/usr/bin/env python3
"""Fail-closed behavioral filesystem checks for ARC-AGI-3 proposers.

The ordinary taint scanner detects known private inputs.  This module enforces
the complementary capability rule: proposer-authored code and immutable tool
requests may address only their attempt workspace.  The compatibility harness
has one explicit exception, the documented raw ``gkm_arena`` import root.
That exception authorizes only the exact ``sys.path.insert(0, ROOT)`` plus
``gkm_arena`` import idiom; it does not authorize reading ROOT as a directory.

For the compatibility runner this is defense-in-depth over immutable request
records and repeatedly reopened source; it is not an OS security boundary and
must not be cited as proof that a hostile process could never execute between
polls.  Hard unattended authority belongs to the contiguous runner: its Arena
is a descriptor-confined dynamic tool and its Python probes execute from
immutable workspace snapshots in a fresh networkless container.  That release
gate remains closed unless those by-construction controls pass conformance.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


MAX_SOURCE_BYTES = 50_000_000
POLICY_SCHEMA = 1
_POLICY_SOURCE_PATH = Path(__file__).resolve()
_LOADED_POLICY_SHA256 = hashlib.sha256(_POLICY_SOURCE_PATH.read_bytes()).hexdigest()
SOURCE_SUFFIXES = frozenset({".py", ".pyw", ".sh", ".bash", ".zsh"})
SKIP_DIRECTORY_NAMES = frozenset({".git"})
RESTRICTED_RELATIVE_PARTS = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".orchestrate.lock",
})
SAFE_PROBE_ENV_RE = re.compile(r"^(?:PROBE_[A-Z0-9_]+|GKM_FRESH_REPLAY)$")
SAFE_RELATIVE_PATH_RE = re.compile(r"^[^\x00\r\n]{1,4096}$")
PARENT_REFERENCE_RE = re.compile(r"(?:^|[\s'\"=])\.\.(?:/|\\)")
HOME_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:~(?:/|\b)|\$\{?(?:HOME|PWD|OLDPWD|CODEX_HOME)\}?)",
    re.IGNORECASE,
)
SHELL_ESCAPE_RE = re.compile(
    r"(?:^|[\n;&|]|\$\()\s*(?:sudo\s+)?(?:"
    r"find|tree|du|readlink|realpath|printenv|env|cd|source|"
    r"setsid|nohup|disown|perl|ruby|node|awk|"
    r"cp|mv|install|ln|tee|truncate|dd|chmod|"
    r"curl|wget|ssh|scp|rsync|nc|ncat|netcat|"
    r"bash|zsh|sh)\b",
    re.IGNORECASE,
)
SHELL_PARENT_ENUM_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:ls|cat|head|tail|sed|rg|grep|awk|cp|mv|rm|chmod)"
    r"\s+(?:-[^\n;&|]+\s+)*(?:/|~|\.\.(?:/|\s|$))",
    re.IGNORECASE,
)
ABSOLUTE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:[^\s\"'`;|&<>]+)"
)
MPL_CACHE_ASSIGNMENT_RE = re.compile(
    r"\b(?:MPLCONFIGDIR|PYTHONPYCACHEPREFIX|XDG_CACHE_HOME)="
    r"/tmp(?:/gkm[-_][A-Za-z0-9_.-]+)?\b"
)
MPL_CACHE_MKDIR_RE = re.compile(
    r"\bmkdir\s+-p\s+/tmp/gkm[-_][A-Za-z0-9_.-]+\b"
)
DEV_NULL_REDIRECT_RE = re.compile(r"(?:\d*)?>\s*/dev/null\b")
FD_MERGE_REDIRECT_RE = re.compile(r"\b\d*>&\d\b")
ZSH_WRAPPER_RE = re.compile(r"^/bin/(?:zsh|bash|sh)\s+-lc\s+")
COMMAND_SUBSTITUTION_RE = re.compile(r"\$\((?!\()(.*?)\)", re.DOTALL)
PROCESS_SUBSTITUTION_RE = re.compile(r"(?:<|>)\(")
PYTHON_INTERPRETER_RE = re.compile(r"^(?:python(?:3(?:\.\d+)*)?)$")
PYTHON_HEREDOC_RE = re.compile(
    r"\bpython(?:3(?:\.\d+)*)?(?:\s+-[A-Za-z]+)*\s+-\s+"
    r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1\s*\n"
    r"(.*?)\n\2(?:\s|$)",
    re.DOTALL,
)
QUOTED_SHELL_DATA_RE = re.compile(
    r"'(?:[^']|'\"'\"')*'|\"(?:\\.|[^\"\\])*\"", re.DOTALL
)
SED_PROGRAM_RE = re.compile(
    r"\bsed\s+-n\s+(?:'[^']*'|\"(?:\\.|[^\"\\])*\")"
)
ARENA_INSERT_RE_TEMPLATE = (
    r"sys\.path\.insert\(\s*0\s*,\s*(['\"])%s\1\s*\)"
)
PASSIVE_EVENT_TYPES = frozenset({
    "thread.started", "turn.started", "turn.completed", "error",
})
PASSIVE_ITEM_TYPES = frozenset({"agent_message", "reasoning", "todo_list"})
ACTION_ITEM_TYPES = frozenset({"command_execution", "file_change"})
ITEM_EVENT_TYPES = frozenset({"item.started", "item.updated", "item.completed"})

FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "commands",
        "ctypes",
        "_imp",
        "_frozen_importlib",
        "_frozen_importlib_external",
        "imp",
        "glob",
        "http",
        "importlib",
        "inspect",
        "multiprocessing",
        "modulefinder",
        "marshal",
        "pickle",
        "platform",
        "posix",
        "pkgutil",
        "pydoc",
        "pty",
        "resource",
        "shutil",
        "subprocess",
        "urllib",
        "zipfile",
        "zipimport",
        "socket",
        "requests",
        "httpx",
        "aiohttp",
        "_io",
        "_socket",
    }
)
PRIVATE_HARNESS_IMPORT_ROOTS = frozenset(
    {
        "arcengine",
        "codex_campaign_policy",
        "codex_campaign_runner",
        "codex_campaign_status",
        "gkm_legs",
        "lab",
        "llm_binder",
    }
)
SHELL_CALLS = frozenset(
    {
        "os.popen",
        "os.spawnl",
        "os.spawnle",
        "os.spawnlp",
        "os.spawnlpe",
        "os.spawnv",
        "os.spawnve",
        "os.spawnvp",
        "os.spawnvpe",
        "os.system",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "subprocess.run",
    }
)
DYNAMIC_EXEC_CALLS = frozenset(
    {
        "__import__", "builtins.__import__", "compile", "builtins.compile",
        "eval", "builtins.eval", "exec", "builtins.exec",
        "runpy.run_module", "runpy.run_path",
    }
)
DIRECT_FILE_CALLS = frozenset(
    {
        "builtins.open",
        "io.open",
        "io.FileIO",
        "os.fdopen",
        "numpy.load",
        "numpy.memmap",
        "numpy.genfromtxt",
        "np.load",
        "np.memmap",
        "np.genfromtxt",
        "open",
        "os.chdir",
        "os.listdir",
        "os.open",
        "os.path.exists",
        "os.path.getsize",
        "os.path.isdir",
        "os.path.isfile",
        "os.scandir",
        "os.stat",
        "os.walk",
        "pandas.read_csv",
        "pandas.read_json",
        "pandas.read_pickle",
        "pd.read_csv",
        "pd.read_json",
        "pd.read_pickle",
    }
)
PATH_READ_METHODS = frozenset(
    {"glob", "iterdir", "open", "read_bytes", "read_text", "rglob"}
)
PATH_WRITE_METHODS = frozenset({
    "hardlink_to", "link_to", "rename", "replace", "symlink_to", "touch",
    "write_bytes", "write_text",
})
PATH_ESCAPE_METHODS = frozenset({"absolute", "expanduser", "resolve"})
PATH_ESCAPE_CALLS = frozenset(
    {
        "os.path.abspath", "os.path.expanduser", "os.path.realpath",
        "os.path.relpath", "os.readlink",
    }
)
PROCESS_ESCAPE_CALLS = frozenset(
    {
        "os.daemon", "os.fork", "os.forkpty", "os.posix_spawn",
        "os.posix_spawnp", "os.setpgrp", "os.setsid",
    }
)
FILESYSTEM_ALIAS_CALLS = frozenset({
    "os.link", "os.symlink", "pathlib.Path.hardlink_to",
    "pathlib.Path.link_to", "pathlib.Path.symlink_to",
})
SENSITIVE_MODULE_ROOTS = frozenset(
    {"builtins", "io", "os", "pathlib", "runpy", "sys"}
)
INTROSPECTIVE_ATTRIBUTES = frozenset(
    {"__bases__", "__class__", "__code__", "__dict__", "__file__",
     "__getattribute__", "__globals__", "__loader__", "__mro__", "__subclasses__", "_getframe",
     "f_globals", "f_locals", "sys.argv", "sys.meta_path", "sys.modules",
     "sys.path_hooks", "sys.path_importer_cache"}
)


@dataclass(frozen=True, order=True)
class BoundaryFinding:
    code: str
    path: str
    line: int
    detail: str

    def describe(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{self.code} in {where}: {self.detail}"


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def policy_sha256() -> str:
    """Bind the executed policy and fail if its control source drifted."""
    current = hashlib.sha256(_POLICY_SOURCE_PATH.read_bytes()).hexdigest()
    if current != _LOADED_POLICY_SHA256:
        raise RuntimeError(
            "filesystem boundary policy source changed after module import"
        )
    return _LOADED_POLICY_SHA256


def arena_module_sha256(arena_module_root: Path | str) -> str:
    """Bind the exact physical raw-arena module exposed by compatibility."""

    path = Path(arena_module_root) / "gkm_arena.py"
    raw, finding, _identity = _read_regular_nofollow(
        path,
        logical_path="gkm_arena.py",
        kind="arena_module",
        max_bytes=MAX_SOURCE_BYTES,
    )
    if finding is not None or raw is None:
        detail = finding.describe() if finding is not None else "unavailable"
        raise RuntimeError(f"raw-arena module identity is unsafe: {detail}")
    return hashlib.sha256(raw).hexdigest()


def _policy_drift_finding(logical_path: str) -> BoundaryFinding | None:
    try:
        policy_sha256()
    except (OSError, RuntimeError) as exc:
        return BoundaryFinding(
            "policy_control_drift", logical_path, 0, str(exc)
        )
    return None


def _trusted_digest(
    trusted: Mapping[str, Any], logical_path: str, digest: str
) -> bool:
    expected = trusted.get(logical_path)
    if isinstance(expected, str):
        return expected == digest
    if isinstance(expected, (set, frozenset, tuple, list)):
        return digest in expected
    return False


def _read_regular_nofollow(
    path: Path,
    *,
    logical_path: str,
    kind: str,
    max_bytes: int | None = None,
) -> tuple[bytes | None, BoundaryFinding | None, tuple[int, ...] | None]:
    """Read one stable, singly linked regular file without following links."""

    try:
        before = os.lstat(path)
    except OSError as exc:
        return None, BoundaryFinding(
            f"unreadable_{kind}", logical_path, 0, type(exc).__name__
        ), None
    if stat.S_ISLNK(before.st_mode):
        return None, BoundaryFinding(
            f"symlink_{kind}", logical_path, 0,
            f"{kind} must not be a symlink",
        ), None
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        return None, BoundaryFinding(
            f"aliased_{kind}", logical_path, 0,
            f"{kind} must be one singly linked regular file",
        ), None
    if max_bytes is not None and before.st_size > max_bytes:
        return None, BoundaryFinding(
            f"oversized_{kind}", logical_path, 0,
            f"{kind} exceeds the scan limit",
        ), None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        return None, BoundaryFinding(
            f"unreadable_{kind}", logical_path, 0, type(exc).__name__
        ), None
    try:
        opened = os.fstat(descriptor)
        if _stat_identity(opened) != _stat_identity(before):
            return None, BoundaryFinding(
                f"raced_{kind}", logical_path, 0,
                f"{kind} changed between lstat and open",
            ), None
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        after_fd = os.fstat(descriptor)
    except OSError as exc:
        return None, BoundaryFinding(
            f"unreadable_{kind}", logical_path, 0, type(exc).__name__
        ), None
    finally:
        os.close(descriptor)
    try:
        after_path = os.lstat(path)
    except OSError as exc:
        return None, BoundaryFinding(
            f"raced_{kind}", logical_path, 0, type(exc).__name__
        ), None
    if (
        _stat_identity(after_fd) != _stat_identity(before)
        or _stat_identity(after_path) != _stat_identity(before)
    ):
        return None, BoundaryFinding(
            f"raced_{kind}", logical_path, 0,
            f"{kind} changed while it was scanned",
        ), None
    raw = b"".join(blocks)
    if len(raw) != before.st_size:
        return None, BoundaryFinding(
            f"raced_{kind}", logical_path, 0,
            f"{kind} size changed while it was scanned",
        ), None
    return raw, None, _stat_identity(before)


def _dotted_name(node: ast.AST) -> str:
    names: list[str] = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
        return ".".join(reversed(names))
    return ""


def _canonical_dotted_name(
    node: ast.AST, aliases: Mapping[str, str]
) -> str:
    dotted = _dotted_name(node)
    if not dotted:
        return ""
    first, separator, remainder = dotted.partition(".")
    target = aliases.get(first, first)
    return target + (separator + remainder if separator else "")


def _literal_string(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _safe_relative(value: str) -> bool:
    if SAFE_RELATIVE_PATH_RE.fullmatch(value) is None or value.startswith(("/", "~")):
        return False
    path = PurePosixPath(value.replace("\\", "/"))
    return bool(path.parts) and all(
        part not in {"", ".", ".."}
        and part not in RESTRICTED_RELATIVE_PARTS
        for part in path.parts
    )


def _reserved_arena_shadow_name(name: str) -> bool:
    return re.fullmatch(r"gkm_arena(?:\..+)?", name) is not None


def _path_receiver_literal(
    node: ast.AST, aliases: Mapping[str, str]
) -> str | None:
    """Return the literal behind ``Path('x').read_text()`` when exact."""
    if (
        not isinstance(node, ast.Call)
        or _canonical_dotted_name(node.func, aliases) != "pathlib.Path"
    ):
        return None
    if len(node.args) != 1 or node.keywords:
        return None
    return _literal_string(node.args[0])


def scan_python_source(
    text: str,
    *,
    logical_path: str,
    arena_module_root: Path | str | None = None,
    allow_host_scaffold: bool = False,
    allow_literal_path_bindings: bool = False,
) -> tuple[BoundaryFinding, ...]:
    """Statically enforce the proposer filesystem capability in Python text."""

    policy_drift = _policy_drift_finding(logical_path)
    if policy_drift is not None:
        return (policy_drift,)

    try:
        tree = ast.parse(text, filename=logical_path)
    except SyntaxError as exc:
        return (BoundaryFinding(
            "unparseable_executable_source", logical_path,
            int(exc.lineno or 0),
            "syntax-invalid source cannot receive clean-room authority",
        ),)
    arena_root = (
        os.path.realpath(os.fspath(arena_module_root))
        if arena_module_root is not None
        else None
    )
    findings: list[BoundaryFinding] = []
    arena_sibling_roots: set[str] = set()
    if arena_root is not None:
        try:
            for entry in os.scandir(arena_root):
                if (
                    entry.name.endswith(".py")
                    and entry.name != "gkm_arena.py"
                    and entry.name.removesuffix(".py").isidentifier()
                    and entry.is_file(follow_symlinks=False)
                ):
                    arena_sibling_roots.add(entry.name.removesuffix(".py"))
                elif (
                    entry.name.isidentifier()
                    and entry.is_dir(follow_symlinks=False)
                ):
                    # The raw arena sys.path exception exposes both ordinary
                    # packages and PEP-420 namespace packages.  Enumerate the
                    # physical host-root children, not merely ``*.py`` files,
                    # and never follow a link while deriving this deny-list.
                    arena_sibling_roots.add(entry.name)
        except OSError as exc:
            findings.append(BoundaryFinding(
                "unreadable_arena_capability_root", logical_path, 0,
                type(exc).__name__,
            ))
    allowed_arena_constants: set[int] = set()
    import_aliases: dict[str, str] = {}
    literal_path_bindings: set[str] = set()
    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    arena_aliases: set[str] = set()
    arena_import_nodes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_aliases[alias.asname or alias.name.split(".", 1)[0]] = (
                    alias.name
                )
                if alias.name == "gkm_arena":
                    arena_aliases.add(alias.asname or alias.name)
                    arena_import_nodes.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name != "*":
                    import_aliases[alias.asname or alias.name] = (
                        f"{node.module}.{alias.name}"
                    )
        elif (
            allow_literal_path_bindings
            and isinstance(node, (ast.For, ast.comprehension))
            and isinstance(node.target, ast.Name)
            and isinstance(node.iter, (ast.List, ast.Tuple, ast.Set))
            and node.iter.elts
            and all(
                (value := _literal_string(element)) is not None
                and _safe_relative(value)
                for element in node.iter.elts
            )
        ):
            literal_path_bindings.add(node.target.id)

    valid_top_level_insertions: list[ast.Call] = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value, ast.Call
        ):
            continue
        node = statement.value
        if (
            _canonical_dotted_name(node.func, import_aliases)
            == "sys.path.insert"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 0
            and _literal_string(node.args[1]) == arena_root
            and arena_root is not None
        ):
            valid_top_level_insertions.append(node)
    authorized_arena_import = None
    if len(arena_import_nodes) == 1 and len(valid_top_level_insertions) == 1:
        import_node = arena_import_nodes[0]
        insertion = valid_top_level_insertions[0]
        if (
            import_node in tree.body
            and tree.body.index(parents[id(insertion)])
            < tree.body.index(import_node)
        ):
            authorized_arena_import = import_node
            allowed_arena_constants.add(id(insertion.args[1]))
    if arena_import_nodes and authorized_arena_import is None:
        findings.append(BoundaryFinding(
            "raw_arena_import_order", logical_path,
            int(getattr(arena_import_nodes[0], "lineno", 0) or 0),
            "raw Arena requires one unconditional top-level exact path "
            "insertion before one top-level module import",
        ))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _canonical_dotted_name(node.func, import_aliases)
        if (
            name == "sys.path.insert"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 0
            and _literal_string(node.args[1]) == arena_root
            and arena_root is not None
        ):
            if node not in valid_top_level_insertions:
                findings.append(BoundaryFinding(
                    "raw_arena_import_order", logical_path,
                    int(getattr(node, "lineno", 0) or 0),
                    "raw Arena path insertion must be one unconditional "
                    "top-level statement before import",
                ))

    for node in ast.walk(tree):
        line = int(getattr(node, "lineno", 0) or 0)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            normalized = value.replace("\\", "/")
            if normalized.startswith("/") and id(node) not in allowed_arena_constants:
                findings.append(BoundaryFinding(
                    "absolute_path", logical_path, line,
                    "absolute filesystem literal is outside the raw-arena capability",
                ))
            elif ".." in PurePosixPath(normalized).parts:
                findings.append(BoundaryFinding(
                    "parent_path", logical_path, line,
                    "parent traversal is forbidden",
                ))
            elif normalized.startswith("~"):
                findings.append(BoundaryFinding(
                    "home_path", logical_path, line,
                    "home expansion is forbidden",
                ))
            continue

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names] if isinstance(
                node, ast.Import
            ) else [node.module or ""]
            for module in modules:
                root = module.split(".", 1)[0]
                if root == "gkm_arena":
                    if arena_root is None:
                        findings.append(BoundaryFinding(
                            "raw_arena_import_unavailable", logical_path, line,
                            "this runner exposes Arena only through its confined tool",
                        ))
                    elif (
                        isinstance(node, ast.ImportFrom)
                        or module != "gkm_arena"
                    ):
                        findings.append(BoundaryFinding(
                            "raw_arena_import_shape", logical_path, line,
                            "raw Arena must be imported as one module capability",
                        ))
                    elif node is not authorized_arena_import:
                        findings.append(BoundaryFinding(
                            "raw_arena_import_order", logical_path, line,
                            "raw Arena import is not bound to the preceding "
                            "top-level capability insertion",
                        ))
                elif root in FORBIDDEN_IMPORT_ROOTS:
                    if not (allow_host_scaffold and root == "importlib"):
                        findings.append(BoundaryFinding(
                            "dynamic_or_process_import", logical_path, line,
                            f"import {module} can escape static workspace confinement",
                        ))
                elif root in PRIVATE_HARNESS_IMPORT_ROOTS:
                    if not (allow_host_scaffold and root == "gkm_legs"):
                        findings.append(BoundaryFinding(
                            "private_harness_import", logical_path, line,
                            f"import {module} exceeds the raw-arena capability",
                        ))
                elif root in arena_sibling_roots:
                    findings.append(BoundaryFinding(
                        "arena_sibling_import", logical_path, line,
                        f"import {module} would resolve through the raw-arena host root",
                    ))
            continue

        if isinstance(node, ast.Name) and node.id in arena_aliases:
            parent = parents.get(id(node))
            if not (
                isinstance(parent, ast.Attribute)
                and parent.value is node
            ):
                findings.append(BoundaryFinding(
                    "raw_arena_alias_escape", logical_path, line,
                    "raw Arena module may only be the receiver of an approved direct call",
                ))
            continue

        if isinstance(node, ast.Name):
            canonical = import_aliases.get(node.id, node.id)
            if node.id in {"__builtins__", "__loader__", "__spec__"}:
                findings.append(BoundaryFinding(
                    "runtime_introspection", logical_path, line,
                    f"{node.id} exposes unsealed import/runtime capabilities",
                ))
                continue
            if canonical in {
                "open", "__import__", "compile", "eval", "exec"
            }:
                parent = parents.get(id(node))
                if not (
                    isinstance(parent, ast.Call) and parent.func is node
                ):
                    findings.append(BoundaryFinding(
                        "sensitive_capability_alias", logical_path, line,
                        f"{canonical} may not be passed or rebound",
                    ))
                continue
            root = canonical.split(".", 1)[0]
            if root in SENSITIVE_MODULE_ROOTS:
                parent = parents.get(id(node))
                direct_function = canonical in (
                    SHELL_CALLS
                    | DYNAMIC_EXEC_CALLS
                    | DIRECT_FILE_CALLS
                    | PATH_ESCAPE_CALLS
                    | {"pathlib.Path"}
                )
                if not (
                    isinstance(parent, (ast.Attribute, ast.Subscript))
                    and parent.value is node
                ) and not (
                    direct_function
                    and isinstance(parent, ast.Call)
                    and parent.func is node
                ):
                    findings.append(BoundaryFinding(
                        "sensitive_capability_alias", logical_path, line,
                        f"{canonical} may not be passed, rebound, or dynamically inspected",
                    ))
                continue

        if isinstance(node, ast.Name) and node.id == "__file__":
            findings.append(BoundaryFinding(
                "runtime_path_introspection", logical_path, line,
                "__file__ can derive a path outside the attempt workspace",
            ))
            continue

        if isinstance(node, ast.Attribute):
            dotted = _canonical_dotted_name(node, import_aliases)
            if (
                isinstance(node.value, ast.Name)
                and node.value.id in arena_aliases
            ):
                allowed = {"run_program"}
                if allow_host_scaffold:
                    allowed.add("validate")
                parent = parents.get(id(node))
                if (
                    node.attr not in allowed
                    or not isinstance(parent, ast.Call)
                    or parent.func is not node
                ):
                    findings.append(BoundaryFinding(
                        "raw_arena_capability_escape", logical_path, line,
                        "raw Arena exposes only direct run_program calls"
                        + (" and supervisor validate calls" if allow_host_scaffold else ""),
                    ))
            elif dotted == "sys.path":
                parent = parents.get(id(node))
                if not (
                    isinstance(parent, ast.Attribute)
                    and parent.value is node
                    and parent.attr == "insert"
                ):
                    findings.append(BoundaryFinding(
                        "import_path_introspection", logical_path, line,
                        "sys.path may only receive the exact raw-arena insertion",
                    ))
            elif dotted in INTROSPECTIVE_ATTRIBUTES or node.attr in (
                INTROSPECTIVE_ATTRIBUTES - {"sys.modules"}
            ):
                findings.append(BoundaryFinding(
                    "runtime_introspection", logical_path, line,
                    f"{dotted or node.attr} is outside the public observation surface",
                ))
            elif dotted.split(".", 1)[0] in SENSITIVE_MODULE_ROOTS:
                parent = parents.get(id(node))
                if not (
                    isinstance(parent, ast.Attribute)
                    and parent.value is node
                ) and not (
                    isinstance(parent, (ast.Call, ast.Subscript))
                    and (
                        getattr(parent, "func", None) is node
                        or getattr(parent, "value", None) is node
                    )
                ):
                    findings.append(BoundaryFinding(
                        "sensitive_capability_alias", logical_path, line,
                        f"{dotted} may not be passed or rebound",
                    ))
            elif node.attr == "parents":
                findings.append(BoundaryFinding(
                    "runtime_path_introspection", logical_path, line,
                    "Path.parents can escape the workspace",
                ))
            continue

        if isinstance(node, ast.Call):
            name = _canonical_dotted_name(node.func, import_aliases)
            if name.startswith("sys.path."):
                if not (
                    name == "sys.path.insert"
                    and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == 0
                    and _literal_string(node.args[1]) == arena_root
                    and arena_root is not None
                    and node in valid_top_level_insertions
                ):
                    findings.append(BoundaryFinding(
                        "import_path_mutation", logical_path, line,
                        "only exact raw-arena sys.path insertion is permitted",
                    ))
                continue
            if name in SHELL_CALLS:
                findings.append(BoundaryFinding(
                    "shell_or_subprocess_escape", logical_path, line,
                    f"{name} is forbidden in proposer-authored source",
                ))
                continue
            if name in PROCESS_ESCAPE_CALLS:
                findings.append(BoundaryFinding(
                    "detached_process_escape", logical_path, line,
                    f"{name} can outlive the supervised process group",
                ))
                continue
            if name in FILESYSTEM_ALIAS_CALLS:
                findings.append(BoundaryFinding(
                    "filesystem_alias_escape", logical_path, line,
                    f"{name} can alias external bytes into the workspace",
                ))
                continue
            if name in DYNAMIC_EXEC_CALLS:
                if not (
                    allow_host_scaffold
                    and name == "exec"
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "exec_module"
                ):
                    findings.append(BoundaryFinding(
                        "dynamic_execution", logical_path, line,
                        f"{name} prevents fail-closed source confinement",
                    ))
                continue
            if name in PATH_ESCAPE_CALLS:
                findings.append(BoundaryFinding(
                    "runtime_path_introspection", logical_path, line,
                    f"{name} can reveal or traverse host paths",
                ))
                continue
            if name in {"globals", "locals", "vars", "builtins.globals",
                        "builtins.locals", "builtins.vars"}:
                findings.append(BoundaryFinding(
                    "runtime_introspection", logical_path, line,
                    f"{name} can recover unsealed capabilities",
                ))
                continue
            if name in {"getattr", "builtins.getattr"} and node.args:
                target = _canonical_dotted_name(
                    node.args[0], import_aliases
                )
                if (
                    target == "gkm_arena"
                    or target.split(".", 1)[0] in SENSITIVE_MODULE_ROOTS
                    or target in {
                        "object", "type", "BaseException", "Exception",
                        "classmethod", "staticmethod", "property",
                    }
                ):
                    findings.append(BoundaryFinding(
                        "runtime_introspection", logical_path, line,
                        "getattr cannot dynamically recover a filesystem or Arena capability",
                    ))
                    continue
            if name in {"os.getcwd", "Path.cwd", "pathlib.Path.cwd"}:
                if not (allow_host_scaffold and name == "os.getcwd"):
                    findings.append(BoundaryFinding(
                        "runtime_path_introspection", logical_path, line,
                        f"{name} exposes host path topology",
                    ))
                continue
            if name in {"os.getenv", "os.environ.get"}:
                key = _literal_string(node.args[0]) if node.args else None
                if key is None or SAFE_PROBE_ENV_RE.fullmatch(key) is None:
                    findings.append(BoundaryFinding(
                        "environment_introspection", logical_path, line,
                        "only literal PROBE_* controls are readable",
                    ))
                continue
            if name.startswith("os.environ."):
                findings.append(BoundaryFinding(
                    "environment_introspection", logical_path, line,
                    "only literal os.environ.get(PROBE_*) is permitted",
                ))
                continue
            if name in DIRECT_FILE_CALLS:
                value = _literal_string(node.args[0]) if node.args else None
                bound_safe = (
                    bool(node.args)
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in literal_path_bindings
                )
                if (value is None or not _safe_relative(value)) and not bound_safe:
                    findings.append(BoundaryFinding(
                        "dynamic_or_external_file_access", logical_path, line,
                        f"{name} requires one literal workspace-relative path",
                    ))
                if name == "os.open":
                    findings.append(BoundaryFinding(
                        "source_generation_capability", logical_path, line,
                        "os.open flags cannot be statically confined to read-only use",
                    ))
                elif name in {"open", "builtins.open", "io.open", "io.FileIO"}:
                    mode = (
                        _literal_string(node.args[1])
                        if len(node.args) > 1
                        else "r"
                    )
                    if mode is None or any(flag in mode for flag in "wax+"):
                        findings.append(BoundaryFinding(
                            "source_generation_capability", logical_path, line,
                            "runtime file writes can create transient unscanned code",
                        ))
                continue
            if isinstance(node.func, ast.Attribute):
                method = node.func.attr
                if method in PATH_WRITE_METHODS:
                    findings.append(BoundaryFinding(
                        "source_generation_capability", logical_path, line,
                        f"Path.{method} can create transient unscanned code",
                    ))
                elif method in PATH_READ_METHODS:
                    value = _path_receiver_literal(
                        node.func.value, import_aliases
                    )
                    if value is None or not _safe_relative(value):
                        findings.append(BoundaryFinding(
                            "dynamic_or_external_file_access", logical_path, line,
                            f"Path.{method} requires one literal workspace-relative base",
                        ))
                elif method in PATH_ESCAPE_METHODS:
                    findings.append(BoundaryFinding(
                        "runtime_path_introspection", logical_path, line,
                        f"Path.{method} can reveal or traverse host paths",
                    ))
            continue

        if (
            isinstance(node, ast.Subscript)
            and _canonical_dotted_name(node.value, import_aliases)
            == "os.environ"
        ):
            key = _literal_string(node.slice)
            if key is None or SAFE_PROBE_ENV_RE.fullmatch(key) is None:
                findings.append(BoundaryFinding(
                    "environment_introspection", logical_path, line,
                    "only literal PROBE_* controls are readable",
                ))

    return tuple(sorted(set(findings)))


def scan_shell_command(
    command: str,
    *,
    logical_path: str,
    line: int,
    arena_module_root: Path | str | None = None,
) -> tuple[BoundaryFinding, ...]:
    """Check one immutable proposer-authored shell command surface."""

    policy_drift = _policy_drift_finding(logical_path)
    if policy_drift is not None:
        return (policy_drift,)

    if not isinstance(command, str) or not command:
        return (BoundaryFinding(
            "malformed_command", logical_path, line,
            "command record is not nonempty text",
        ),)
    findings: list[BoundaryFinding] = []
    payload = command
    if re.search(
        r"(?:^|[/\\\s'\"])(?:\.git|__pycache__|\.pytest_cache|"
        r"\.orchestrate\.lock)(?=$|[/\\\s'\"])",
        payload,
    ):
        findings.append(BoundaryFinding(
            "hidden_control_surface", logical_path, line,
            "shell command targets a host-control or hidden execution path",
        ))
    try:
        outer = shlex.split(command, posix=True)
    except ValueError:
        return (BoundaryFinding(
            "malformed_command", logical_path, line,
            "shell command has invalid quoting",
        ),)
    if (
        len(outer) == 3
        and outer[0] in {"/bin/zsh", "/bin/bash", "/bin/sh"}
        and outer[1] == "-lc"
    ):
        payload = outer[2]
    elif command.startswith(("/bin/zsh ", "/bin/bash ", "/bin/sh ")):
        findings.append(BoundaryFinding(
            "shell_wrapper_escape", logical_path, line,
            "only the exact host-owned /bin/* -lc wrapper is permitted",
        ))

    if PARENT_REFERENCE_RE.search(payload):
        findings.append(BoundaryFinding(
            "parent_path", logical_path, line,
            "shell command contains parent traversal",
        ))
    if HOME_REFERENCE_RE.search(payload):
        findings.append(BoundaryFinding(
            "home_path", logical_path, line,
            "shell command expands host/home topology",
        ))
    if re.search(
        r"(?:^|[\s'\"/])(?:\.git|__pycache__|\.pytest_cache)(?:/|[\s'\"]|$)",
        payload,
    ):
        findings.append(BoundaryFinding(
            "hidden_execution_surface", logical_path, line,
            "hidden Git/cache paths are outside the auditable source surface",
        ))
    # Python heredoc bodies are parsed separately as Python.  Remove them from
    # shell metacharacter analysis first, so ordinary Python comparisons do
    # not masquerade as shell redirection while a real ``python - < file``
    # remains visible.
    shell_structure_source = PYTHON_HEREDOC_RE.sub(
        "python - <<__AUDITED_PYTHON_HEREDOC__", payload
    )
    structural_payload = QUOTED_SHELL_DATA_RE.sub(
        " __QUOTED_SHELL_DATA__ ", shell_structure_source
    )
    if (
        SHELL_ESCAPE_RE.search(structural_payload)
        or SHELL_PARENT_ENUM_RE.search(structural_payload)
        or re.search(r"(?:^|[;&|])\s*\.\s+", structural_payload)
    ):
        findings.append(BoundaryFinding(
            "shell_or_host_filesystem_escape", logical_path, line,
            "shell command requests a host/process/path capability",
        ))
    if re.search(r"(?<![&>])&(?![&0-9])", structural_payload):
        findings.append(BoundaryFinding(
            "detached_process_escape", logical_path, line,
            "background shell jobs can outlive the supervised process group",
        ))
    if re.search(r"(?<!<)<(?!<)", structural_payload):
        findings.append(BoundaryFinding(
            "shell_input_redirection", logical_path, line,
            "interpreter/stdin redirection is outside the statically scanned source surface",
        ))
    output_surface = FD_MERGE_REDIRECT_RE.sub("__FD_MERGE__", structural_payload)
    output_surface = DEV_NULL_REDIRECT_RE.sub("__DEV_NULL__", output_surface)
    if ">" in output_surface:
        findings.append(BoundaryFinding(
            "shell_output_redirection", logical_path, line,
            "shell-generated files can execute between behavioral audit polls",
        ))
    if re.search(r"(?:^|[\n;&|])\s*sed\b[^\n;&|]*\s-i(?:\s|$)", structural_payload):
        findings.append(BoundaryFinding(
            "shell_source_mutation", logical_path, line,
            "in-place shell source mutation is outside the audited write tool",
        ))
    if re.search(
        r"(?:^|[\s;&|])\./[^\s;&|<>]+", structural_payload
    ):
        findings.append(BoundaryFinding(
            "relative_executable_escape", logical_path, line,
            "direct relative executable dispatch is outside the approved interpreter surface",
        ))
    if re.search(
        r"\$(?:\{)?(?:MPLCONFIGDIR|PYTHONPYCACHEPREFIX|XDG_CACHE_HOME)(?:\})?",
        payload,
    ):
        findings.append(BoundaryFinding(
            "temporary_cache_capability_escape", logical_path, line,
            "temporary cache paths may be assigned but not reused as file capabilities",
        ))

    git_calls = re.findall(r"(?:^|[;&|])\s*(git\b[^;&|]*)", payload)
    for git_call in git_calls:
        if (
            re.match(r"git\s+diff\b", git_call) is None
            or re.search(r"(?:^|\s)(?:-C|--git-dir|--work-tree)(?:\s|=)", git_call)
        ):
            findings.append(BoundaryFinding(
                "git_capability_escape", logical_path, line,
                "only workspace-local git diff is permitted",
            ))

    if "`" in payload:
        findings.append(BoundaryFinding(
            "shell_substitution_escape", logical_path, line,
            "backtick command substitution is forbidden",
        ))
    for substitution in COMMAND_SUBSTITUTION_RE.findall(payload):
        unsafe_path = (
            PARENT_REFERENCE_RE.search(substitution) is not None
            or HOME_REFERENCE_RE.search(substitution) is not None
            or re.search(r"(?:^|\s)/(?:\s|$)", substitution) is not None
            or any(
                not token.endswith("/")
                for token in ABSOLUTE_TOKEN_RE.findall(substitution)
            )
        )
        if (
            re.fullmatch(
                r"\s*rg\b[^;&]*\|\s*cut\b[^;&]*\s*", substitution
            ) is None
            or unsafe_path
        ):
            findings.append(BoundaryFinding(
                "shell_substitution_escape", logical_path, line,
                "command substitution exceeds the local rg-to-cut capability",
            ))
    if PROCESS_SUBSTITUTION_RE.search(payload):
        findings.append(BoundaryFinding(
            "shell_process_substitution", logical_path, line,
            "process substitution is outside the workspace capability",
        ))
    if re.search(r"(?:^|\s)(?:\d*>>?|<)\s*[\"']?\$", payload):
        findings.append(BoundaryFinding(
            "dynamic_redirection", logical_path, line,
            "a redirection target must be literal and workspace-relative",
        ))

    inline_sources: list[str] = []
    inline_sources.extend(
        match.group(3) for match in PYTHON_HEREDOC_RE.finditer(payload)
    )
    try:
        lexer = shlex.shlex(
            payload, posix=True, punctuation_chars=";&|()<>"
        )
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        tokens = []
        findings.append(BoundaryFinding(
            "malformed_command", logical_path, line,
            "inner shell payload has invalid quoting",
        ))
    for index, token in enumerate(tokens):
        if not PYTHON_INTERPRETER_RE.fullmatch(token):
            continue
        cursor = index + 1
        while cursor < len(tokens):
            argument = tokens[cursor]
            if argument in {";", "&&", "||", "|", "&"}:
                break
            if argument == "-c":
                if cursor + 1 >= len(tokens):
                    findings.append(BoundaryFinding(
                        "malformed_inline_python", logical_path, line,
                        "python -c has no auditable source argument",
                    ))
                else:
                    inline_sources.append(tokens[cursor + 1])
                break
            if argument == "-":
                break
            if argument == "-m":
                module = tokens[cursor + 1] if cursor + 1 < len(tokens) else ""
                if module != "py_compile":
                    findings.append(BoundaryFinding(
                        "python_module_escape", logical_path, line,
                        "only the non-executing py_compile module check is permitted",
                    ))
                break
            if argument.startswith("-"):
                cursor += 1
                continue
            candidate = PurePosixPath(argument.replace("\\", "/"))
            if (
                candidate.suffix.lower() not in {".py", ".pyw"}
                or any(part in RESTRICTED_RELATIVE_PARTS for part in candidate.parts)
            ):
                findings.append(BoundaryFinding(
                    "unscannable_python_target", logical_path, line,
                    "Python file execution must name a visible .py/.pyw workspace source",
                ))
            break
    if re.search(r"\bpython(?:3(?:\.\d+)*)?\s+-[^\n]*<<", payload) and not inline_sources:
        findings.append(BoundaryFinding(
            "malformed_inline_python", logical_path, line,
            "Python heredoc could not be parsed fail closed",
        ))
    for source in inline_sources:
        findings.extend(scan_python_source(
            source,
            logical_path=f"{logical_path}:inline_python",
            arena_module_root=arena_module_root,
            allow_literal_path_bindings=True,
        ))

    reduced = payload
    reduced = MPL_CACHE_ASSIGNMENT_RE.sub("MPLCONFIGDIR=__GKM_MPL_CACHE__", reduced)
    reduced = MPL_CACHE_MKDIR_RE.sub("mkdir -p __GKM_MPL_CACHE__", reduced)
    reduced = DEV_NULL_REDIRECT_RE.sub("2>__DEV_NULL__", reduced)
    reduced = SED_PROGRAM_RE.sub("sed -n __SED_PROGRAM__", reduced)
    arena_root = (
        os.path.realpath(os.fspath(arena_module_root))
        if arena_module_root is not None
        else None
    )
    if arena_root is not None:
        reduced = re.sub(
            ARENA_INSERT_RE_TEMPLATE % re.escape(arena_root),
            "__GKM_RAW_ARENA_INSERT__",
            reduced,
        )
    for token in ABSOLUTE_TOKEN_RE.findall(reduced):
        # sed/rg regular-expression address tokens such as /^foo$/ are data,
        # not filesystem paths.  A real absolute path cannot contain these
        # unescaped regex punctuation characters.
        if token.endswith("/") or re.search(r"[\^$()\[\]*+?|]", token):
            continue
        findings.append(BoundaryFinding(
            "absolute_path", logical_path, line,
            f"shell command contains non-capability absolute token {token[:120]!r}",
        ))
    return tuple(sorted(set(findings)))


def _scan_workspace_source_payload(
    raw: bytes,
    *,
    logical_path: str,
    suffix: str,
    mode: int,
    arena_module_root: Path | str | None,
    allow_host_scaffold: bool,
) -> tuple[BoundaryFinding, ...]:
    """Scan a named source or fail closed on an executable opaque file."""

    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        return (BoundaryFinding(
            "unreadable_source", logical_path, 0, type(exc).__name__,
        ),)
    if suffix in {".py", ".pyw"}:
        return scan_python_source(
            text,
            logical_path=logical_path,
            arena_module_root=arena_module_root,
            allow_host_scaffold=allow_host_scaffold,
        )
    if suffix in {".sh", ".bash", ".zsh"}:
        return scan_shell_command(
            text,
            logical_path=logical_path,
            line=1,
            arena_module_root=arena_module_root,
        )
    # Extensionless/non-source files receive authority only when executable.
    # Recognize the interpreter from a literal shebang; every other executable
    # format is outside this static compatibility contract.
    if not stat.S_IMODE(mode) & 0o111:
        return ()
    first_line = text.splitlines()[0] if text.splitlines() else ""
    if re.fullmatch(r"#!\s*/usr/bin/env\s+python(?:3(?:\.\d+)*)?", first_line):
        return scan_python_source(
            text,
            logical_path=logical_path,
            arena_module_root=arena_module_root,
            allow_host_scaffold=allow_host_scaffold,
        )
    if re.fullmatch(
        r"#!\s*(?:/usr/bin/env\s+)?(?:/bin/)?(?:sh|bash|zsh)", first_line
    ):
        return scan_shell_command(
            text,
            logical_path=logical_path,
            line=1,
            arena_module_root=arena_module_root,
        )
    return (BoundaryFinding(
        "unscannable_executable", logical_path, 0,
        "an executable workspace file has no approved auditable source form",
    ),)


def scan_codex_transcript(
    path: Path,
    *,
    workspace_root: Path,
    arena_module_root: Path | str | None = None,
    accepted_workspace_root: str | None = None,
) -> tuple[BoundaryFinding, ...]:
    """Reopen every immutable command/file-change record fail closed."""

    selected = Path(path)
    logical = selected.name
    policy_drift = _policy_drift_finding(logical)
    if policy_drift is not None:
        return (policy_drift,)
    findings: list[BoundaryFinding] = []
    raw, read_finding, _ = _read_regular_nofollow(
        selected, logical_path=logical, kind="transcript"
    )
    if read_finding is not None or raw is None:
        return (read_finding,) if read_finding is not None else ()
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        return (BoundaryFinding(
            "non_utf8_transcript", logical, 0,
            "immutable command surface is not UTF-8",
        ),)
    workspace = Path(workspace_root).resolve()
    accepted_root: Path | None = None
    if accepted_workspace_root is not None:
        bound = (
            Path(accepted_workspace_root)
            if isinstance(accepted_workspace_root, str)
            else None
        )
        if (
            bound is None
            or not bound.is_absolute()
            or Path(os.path.abspath(os.fspath(bound))) != bound
            or re.fullmatch(r"gkm_legs_ws_[A-Za-z0-9_.-]+", bound.name) is None
        ):
            findings.append(BoundaryFinding(
                "invalid_workspace_binding", logical, 0,
                "historical workspace root binding is malformed",
            ))
        else:
            accepted_root = bound
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            findings.append(BoundaryFinding(
                "malformed_transcript", logical, line_number,
                "unrecognized line could hide a command surface",
            ))
            continue
        if not isinstance(event, dict):
            findings.append(BoundaryFinding(
                "malformed_transcript", logical, line_number,
                "JSONL record is not an object",
            ))
            continue
        event_type = event.get("type")
        item = event.get("item")
        if not isinstance(item, dict):
            if event_type not in PASSIVE_EVENT_TYPES:
                findings.append(BoundaryFinding(
                    "unknown_transcript_event", logical, line_number,
                    "item-less event is not in the sealed passive schema",
                ))
            continue
        item_type = item.get("type")
        if event_type not in ITEM_EVENT_TYPES:
            findings.append(BoundaryFinding(
                "malformed_item_event", logical, line_number,
                "item record has an unrecognized lifecycle event",
            ))
            continue
        if not isinstance(item.get("id"), str) or not item.get("id"):
            findings.append(BoundaryFinding(
                "malformed_item_event", logical, line_number,
                "item lifecycle record has no stable id",
            ))
        if item_type not in ACTION_ITEM_TYPES | PASSIVE_ITEM_TYPES:
            findings.append(BoundaryFinding(
                "unknown_item_type", logical, line_number,
                "unrecognized item surface could hide an action",
            ))
            continue
        if item_type in ACTION_ITEM_TYPES and event_type == "item.updated":
            findings.append(BoundaryFinding(
                "malformed_action_lifecycle", logical, line_number,
                "action items may only start and complete",
            ))
            continue
        if item_type == "command_execution":
            findings.extend(scan_shell_command(
                item.get("command"),
                logical_path=logical,
                line=line_number,
                arena_module_root=arena_module_root,
            ))
        elif item_type == "file_change":
            changes = item.get("changes")
            if not isinstance(changes, list):
                findings.append(BoundaryFinding(
                    "malformed_file_change", logical, line_number,
                    "file-change record has no exact changes list",
                ))
                continue
            for change in changes:
                value = change.get("path") if isinstance(change, dict) else None
                if not isinstance(value, str) or not value:
                    findings.append(BoundaryFinding(
                        "malformed_file_change", logical, line_number,
                        "changed path is missing",
                    ))
                    continue
                candidate = Path(value)
                if _reserved_arena_shadow_name(candidate.name):
                    findings.append(BoundaryFinding(
                        "reserved_arena_shadow", logical, line_number,
                        "file change may shadow the host raw-arena module",
                    ))
                elif any(
                    part in RESTRICTED_RELATIVE_PARTS
                    for part in candidate.parts
                ):
                    findings.append(BoundaryFinding(
                        "file_change_hidden_surface", logical, line_number,
                        "changed path targets a hidden/cache execution surface",
                    ))
                elif ".." in candidate.parts:
                    findings.append(BoundaryFinding(
                        "file_change_escape", logical, line_number,
                        "changed path contains parent traversal",
                    ))
                elif candidate.is_absolute():
                    try:
                        candidate.resolve().relative_to(workspace)
                    except (OSError, ValueError):
                        accepted = False
                        if accepted_root is not None:
                            try:
                                Path(os.path.abspath(os.fspath(candidate))).relative_to(
                                    accepted_root
                                )
                                accepted = True
                            except ValueError:
                                pass
                        if not accepted:
                            findings.append(BoundaryFinding(
                                "file_change_escape", logical, line_number,
                                "absolute changed path is outside the attempt workspace",
                            ))
    return tuple(sorted(set(findings)))


class LiveBoundaryMonitor:
    """Incremental boundary gate for every compatibility-runner live poll.

    Workspace files are reopened whenever their inode/size/mtime identity
    changes.  The append-only transcript is parsed only from its last complete
    line.  A replacement, truncation, malformed record, or partial terminal
    line fails closed.
    """

    def __init__(
        self,
        workspace_root: Path,
        *,
        arena_module_root: Path | str | None = None,
        trusted_host_scaffolds: Mapping[str, Any] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.arena_module_root = arena_module_root
        self.trusted_host_scaffolds = dict(trusted_host_scaffolds or {})
        self._source_cache: dict[
            str, tuple[str, tuple[BoundaryFinding, ...]]
        ] = {}
        self._transcripts: dict[
            str, tuple[int, int, int, bytes, int, str]
        ] = {}

    def scan_workspace(self) -> tuple[BoundaryFinding, ...]:
        try:
            policy_sha256()
        except (OSError, RuntimeError) as exc:
            return (BoundaryFinding(
                "policy_control_drift", str(_POLICY_SOURCE_PATH), 0,
                str(exc),
            ),)
        findings: list[BoundaryFinding] = []
        seen: set[str] = set()
        for directory, dirs, files in os.walk(
            self.workspace_root, followlinks=False
        ):
            kept_dirs: list[str] = []
            for name in sorted(dirs):
                path = Path(directory) / name
                logical = os.fspath(path.relative_to(self.workspace_root))
                try:
                    metadata = os.lstat(path)
                except OSError as exc:
                    findings.append(BoundaryFinding(
                        "unreadable_workspace_node", logical, 0,
                        type(exc).__name__,
                    ))
                    continue
                if stat.S_ISLNK(metadata.st_mode):
                    findings.append(BoundaryFinding(
                        "symlink_escape", logical, 0,
                        "workspace symlinks are forbidden",
                    ))
                elif (
                    Path(directory) == self.workspace_root
                    and _reserved_arena_shadow_name(name)
                ):
                    findings.append(BoundaryFinding(
                        "reserved_arena_shadow", logical, 0,
                        "workspace may not shadow the host raw-arena module",
                    ))
                elif name not in SKIP_DIRECTORY_NAMES:
                    kept_dirs.append(name)
            dirs[:] = kept_dirs
            for name in sorted(files):
                path = Path(directory) / name
                logical = os.fspath(path.relative_to(self.workspace_root))
                seen.add(logical)
                try:
                    metadata = os.lstat(path)
                except OSError as exc:
                    findings.append(BoundaryFinding(
                        "unreadable_workspace_node", logical, 0,
                        type(exc).__name__,
                    ))
                    continue
                if stat.S_ISLNK(metadata.st_mode):
                    findings.append(BoundaryFinding(
                        "symlink_escape", logical, 0,
                        "workspace symlinks are forbidden",
                    ))
                    continue
                if (
                    Path(directory) == self.workspace_root
                    and _reserved_arena_shadow_name(name)
                ):
                    findings.append(BoundaryFinding(
                        "reserved_arena_shadow", logical, 0,
                        "workspace may not shadow the host raw-arena module",
                    ))
                    continue
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    findings.append(BoundaryFinding(
                        "aliased_workspace_node", logical, 0,
                        "workspace evidence is nonregular or multiply linked",
                    ))
                    continue
                suffix = path.suffix.lower()
                if (
                    suffix not in SOURCE_SUFFIXES
                    and not stat.S_IMODE(metadata.st_mode) & 0o111
                ):
                    continue
                raw, read_finding, _ = _read_regular_nofollow(
                    path,
                    logical_path=logical,
                    kind="source",
                    max_bytes=MAX_SOURCE_BYTES,
                )
                if read_finding is not None or raw is None:
                    current = (read_finding,) if read_finding else ()
                    findings.extend(current)
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                cached = self._source_cache.get(logical)
                if cached is not None and cached[0] == digest:
                    findings.extend(cached[1])
                    continue
                current = _scan_workspace_source_payload(
                    raw,
                    logical_path=logical,
                    suffix=suffix,
                    mode=metadata.st_mode,
                    arena_module_root=self.arena_module_root,
                    allow_host_scaffold=_trusted_digest(
                        self.trusted_host_scaffolds, logical, digest
                    ),
                )
                self._source_cache[logical] = (digest, current)
                findings.extend(current)
        for logical in tuple(self._source_cache):
            if logical not in seen:
                self._source_cache.pop(logical, None)
        return tuple(sorted(set(findings)))

    def scan_transcript(
        self, path: Path, *, final: bool = False
    ) -> tuple[BoundaryFinding, ...]:
        selected = Path(path)
        logical = selected.name
        raw: bytes | None = None
        read_finding: BoundaryFinding | None = None
        for _ in range(3):
            raw, read_finding, stable_identity = _read_regular_nofollow(
                selected, logical_path=logical, kind="transcript"
            )
            if (
                read_finding is None
                or read_finding.code != "raced_transcript"
            ):
                break
        if read_finding is not None or raw is None:
            if (
                not final
                and read_finding is not None
                and read_finding.code == "unreadable_transcript"
                and not selected.exists()
            ):
                return ()
            return (read_finding,) if read_finding is not None else ()
        assert stable_identity is not None
        metadata_device, metadata_inode = stable_identity[:2]
        state = self._transcripts.get(os.fspath(selected))
        if state is None:
            offset, carry, line_number = 0, b"", 0
        else:
            device, inode, offset, carry, line_number, prefix_sha256 = state
            if (
                (device, inode) != (metadata_device, metadata_inode)
                or len(raw) < offset
                or hashlib.sha256(raw[:offset]).hexdigest() != prefix_sha256
            ):
                return (BoundaryFinding(
                    "replaced_transcript", logical, line_number,
                    "append-only command evidence changed identity, shrank, or rewrote its sealed prefix",
                ),)
        appended = raw[offset:]
        new_offset = len(raw)
        combined = carry + appended
        rows = combined.split(b"\n")
        new_carry = rows.pop()
        findings: list[BoundaryFinding] = []
        for raw_line in rows:
            line_number += 1
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError):
                findings.append(BoundaryFinding(
                    "malformed_transcript", logical, line_number,
                    "unrecognized line could hide a command surface",
                ))
                continue
            if not isinstance(event, dict):
                findings.append(BoundaryFinding(
                    "malformed_transcript", logical, line_number,
                    "JSONL record is not an object",
                ))
                continue
            event_type = event.get("type")
            item = event.get("item")
            if not isinstance(item, dict):
                if event_type not in PASSIVE_EVENT_TYPES:
                    findings.append(BoundaryFinding(
                        "unknown_transcript_event", logical, line_number,
                        "item-less event is not in the sealed passive schema",
                    ))
                continue
            item_type = item.get("type")
            if event_type not in ITEM_EVENT_TYPES:
                findings.append(BoundaryFinding(
                    "malformed_item_event", logical, line_number,
                    "item record has an unrecognized lifecycle event",
                ))
                continue
            if not isinstance(item.get("id"), str) or not item.get("id"):
                findings.append(BoundaryFinding(
                    "malformed_item_event", logical, line_number,
                    "item lifecycle record has no stable id",
                ))
            if item_type not in ACTION_ITEM_TYPES | PASSIVE_ITEM_TYPES:
                findings.append(BoundaryFinding(
                    "unknown_item_type", logical, line_number,
                    "unrecognized item surface could hide an action",
                ))
                continue
            if item_type in ACTION_ITEM_TYPES and event_type == "item.updated":
                findings.append(BoundaryFinding(
                    "malformed_action_lifecycle", logical, line_number,
                    "action items may only start and complete",
                ))
                continue
            if item_type == "command_execution":
                findings.extend(scan_shell_command(
                    item.get("command"),
                    logical_path=logical,
                    line=line_number,
                    arena_module_root=self.arena_module_root,
                ))
            elif item_type == "file_change":
                changes = item.get("changes")
                if not isinstance(changes, list):
                    findings.append(BoundaryFinding(
                        "malformed_file_change", logical, line_number,
                        "file-change record has no exact changes list",
                    ))
                else:
                    for change in changes:
                        value = (
                            change.get("path")
                            if isinstance(change, dict)
                            else None
                        )
                        if not isinstance(value, str) or not value:
                            findings.append(BoundaryFinding(
                                "malformed_file_change", logical,
                                line_number, "changed path is missing",
                            ))
                            continue
                        candidate = Path(value)
                        if any(
                            part in RESTRICTED_RELATIVE_PARTS
                            for part in candidate.parts
                        ):
                            findings.append(BoundaryFinding(
                                "file_change_hidden_surface", logical,
                                line_number,
                                "changed path targets a hidden/cache execution surface",
                            ))
                            continue
                        escaped = ".." in candidate.parts
                        if candidate.is_absolute() and not escaped:
                            try:
                                candidate.resolve().relative_to(
                                    self.workspace_root
                                )
                            except (OSError, ValueError):
                                escaped = True
                        if escaped:
                            findings.append(BoundaryFinding(
                                "file_change_escape", logical, line_number,
                                "changed path is outside the attempt workspace",
                            ))
        self._transcripts[os.fspath(selected)] = (
            metadata_device,
            metadata_inode,
            new_offset,
            new_carry,
            line_number,
            hashlib.sha256(raw).hexdigest(),
        )
        if final and new_carry:
            findings.append(BoundaryFinding(
                "partial_transcript", logical, line_number + 1,
                "terminal command transcript lacks a complete final record",
            ))
        return tuple(sorted(set(findings)))


def scan_workspace(
    root: Path,
    *,
    arena_module_root: Path | str | None = None,
    trusted_host_scaffolds: Mapping[str, Any] | None = None,
) -> tuple[BoundaryFinding, ...]:
    """Scan every agent-executable source and reject aliased filesystem nodes."""

    try:
        policy_sha256()
    except (OSError, RuntimeError) as exc:
        return (BoundaryFinding(
            "policy_control_drift", str(_POLICY_SOURCE_PATH), 0, str(exc),
        ),)
    selected = Path(root)
    trusted = dict(trusted_host_scaffolds or {})
    findings: list[BoundaryFinding] = []
    try:
        root_stat = os.lstat(selected)
    except OSError as exc:
        return (BoundaryFinding(
            "unreadable_workspace", str(selected), 0, type(exc).__name__,
        ),)
    if not stat.S_ISDIR(root_stat.st_mode):
        return (BoundaryFinding(
            "unsafe_workspace_root", str(selected), 0,
            "workspace root is not a physical directory",
        ),)
    for directory, dirs, files in os.walk(selected, followlinks=False):
        kept_dirs: list[str] = []
        for name in sorted(dirs):
            path = Path(directory) / name
            logical = os.fspath(path.relative_to(selected))
            try:
                metadata = os.lstat(path)
            except OSError as exc:
                findings.append(BoundaryFinding(
                    "unreadable_workspace_node", logical, 0, type(exc).__name__,
                ))
                continue
            if stat.S_ISLNK(metadata.st_mode):
                findings.append(BoundaryFinding(
                    "symlink_escape", logical, 0,
                    "workspace symlinks are forbidden",
                ))
                continue
            if (
                Path(directory) == selected
                and _reserved_arena_shadow_name(name)
            ):
                findings.append(BoundaryFinding(
                    "reserved_arena_shadow", logical, 0,
                    "workspace may not shadow the host raw-arena module",
                ))
                continue
            if name not in SKIP_DIRECTORY_NAMES:
                kept_dirs.append(name)
        dirs[:] = kept_dirs
        for name in sorted(files):
            path = Path(directory) / name
            logical = os.fspath(path.relative_to(selected))
            try:
                metadata = os.lstat(path)
            except OSError as exc:
                findings.append(BoundaryFinding(
                    "unreadable_workspace_node", logical, 0, type(exc).__name__,
                ))
                continue
            if stat.S_ISLNK(metadata.st_mode):
                findings.append(BoundaryFinding(
                    "symlink_escape", logical, 0,
                    "workspace symlinks are forbidden",
                ))
                continue
            if not stat.S_ISREG(metadata.st_mode):
                findings.append(BoundaryFinding(
                    "nonregular_workspace_node", logical, 0,
                    "workspace evidence must be regular",
                ))
                continue
            if metadata.st_nlink != 1:
                findings.append(BoundaryFinding(
                    "hardlink_escape", logical, 0,
                    "multiply linked workspace evidence is forbidden",
                ))
                continue
            if (
                Path(directory) == selected
                and _reserved_arena_shadow_name(name)
            ):
                findings.append(BoundaryFinding(
                    "reserved_arena_shadow", logical, 0,
                    "workspace may not shadow the host raw-arena module",
                ))
                continue
            suffix = path.suffix.lower()
            if (
                suffix not in SOURCE_SUFFIXES
                and not stat.S_IMODE(metadata.st_mode) & 0o111
            ):
                continue
            raw, read_finding, _ = _read_regular_nofollow(
                path,
                logical_path=logical,
                kind="source",
                max_bytes=MAX_SOURCE_BYTES,
            )
            if read_finding is not None or raw is None:
                if read_finding is not None:
                    findings.append(read_finding)
                continue
            findings.extend(_scan_workspace_source_payload(
                raw,
                logical_path=logical,
                suffix=suffix,
                mode=metadata.st_mode,
                arena_module_root=arena_module_root,
                allow_host_scaffold=_trusted_digest(
                    trusted, logical, hashlib.sha256(raw).hexdigest()
                ),
            ))
    return tuple(sorted(set(findings)))


def scan_python_file(
    path: Path,
    *,
    logical_path: str | None = None,
    arena_module_root: Path | str | None = None,
    allow_host_scaffold: bool = False,
) -> tuple[BoundaryFinding, ...]:
    """Securely reopen and scan one authoritative Python source image."""

    return scan_source_file(
        path,
        logical_path=logical_path,
        arena_module_root=arena_module_root,
        allow_host_scaffold=allow_host_scaffold,
    )


def scan_source_file(
    path: Path,
    *,
    logical_path: str | None = None,
    arena_module_root: Path | str | None = None,
    allow_host_scaffold: bool = False,
) -> tuple[BoundaryFinding, ...]:
    """Securely reopen and scan one authoritative executable source image."""

    selected = Path(path)
    logical = logical_path or selected.name
    policy_drift = _policy_drift_finding(logical)
    if policy_drift is not None:
        return (policy_drift,)
    raw, read_finding, identity = _read_regular_nofollow(
        selected,
        logical_path=logical,
        kind="source",
        max_bytes=MAX_SOURCE_BYTES,
    )
    if read_finding is not None or raw is None:
        return (read_finding,) if read_finding is not None else ()
    assert identity is not None
    return _scan_workspace_source_payload(
        raw,
        logical_path=logical,
        suffix=selected.suffix.lower(),
        mode=identity[2],
        arena_module_root=arena_module_root,
        allow_host_scaffold=allow_host_scaffold,
    )


def dynamic_tool_boundary_hits(
    operation: str,
    arguments: Mapping[str, Any],
) -> tuple[str, ...]:
    """Classify source text before a contiguous workspace write executes."""

    policy_drift = _policy_drift_finding("dynamic_tool_boundary")
    if policy_drift is not None:
        return (policy_drift.code,)

    if operation != "workspace_write" or not isinstance(arguments, Mapping):
        return ()
    path = arguments.get("path")
    text = arguments.get("text")
    if not isinstance(path, str) or not isinstance(text, str):
        return ("malformed_workspace_write",)
    if not _safe_relative(path):
        return ("workspace_write_path_escape",)
    if _reserved_arena_shadow_name(PurePosixPath(path).name):
        return ("reserved_arena_shadow",)
    suffix = Path(path).suffix.lower()
    if suffix not in SOURCE_SUFFIXES:
        if not text.startswith("#!"):
            return ()
        return ("extensionless_executable_write",)
    if suffix in {".py", ".pyw"}:
        findings = scan_python_source(
            text, logical_path=path, arena_module_root=None,
        )
    else:
        findings = scan_shell_command(
            text, logical_path=path, line=1, arena_module_root=None,
        )
    return tuple(sorted({finding.code for finding in findings}))


def first_reason(findings: Iterable[BoundaryFinding]) -> str | None:
    normalized = tuple(findings)
    return normalized[0].describe() if normalized else None


__all__ = [
    "BoundaryFinding",
    "arena_module_sha256",
    "LiveBoundaryMonitor",
    "dynamic_tool_boundary_hits",
    "first_reason",
    "scan_codex_transcript",
    "scan_python_file",
    "scan_python_source",
    "scan_shell_command",
    "scan_workspace",
]
