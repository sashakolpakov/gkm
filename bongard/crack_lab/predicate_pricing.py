"""AST-backed definition pricing for unrestricted Bongard predicates.

The unrestricted track stores reusable pixel predicates in a Python module.
This module turns such a source file into an immutable, content-addressed
pricing model.  A rule pays for the union of the definitions reachable from
its ``p_*`` predicates, so a helper (or import) shared by two predicates is
charged only once within that rule.

For the shared-library arm, pass identities from the previously promoted
model to :meth:`PredicatePricingModel.price`; unchanged nodes are then free.
For the no-share arm, use :meth:`PredicatePricingModel.price_no_share`, which
always repays the complete reachable definition set.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping


ALLOWED_IMPORT_MODULES = frozenset({"itertools", "math", "numpy"})

ALLOWED_FROM_IMPORTS: Mapping[str, frozenset[str]] = {
    "collections": frozenset({"deque"}),
    "math": frozenset({"sin"}),
    "scipy": frozenset({"ndimage"}),
    "scipy.ndimage": frozenset({
        "binary_dilation", "binary_fill_holes", "convolve",
        "gaussian_filter1d", "label",
    }),
    "scipy.spatial": frozenset({"ConvexHull", "cKDTree"}),
    "skimage.morphology": frozenset({"skeletonize"}),
}

# Backwards-compatible public summary consumed by the Phase-D policy.  Import
# admission itself is the exact module/member allowlist above, not this root set.
ALLOWED_IMPORT_ROOTS = frozenset(
    {module.split(".", 1)[0] for module in ALLOWED_IMPORT_MODULES}
    | {module.split(".", 1)[0] for module in ALLOWED_FROM_IMPORTS}
)

SAFE_BUILTIN_NAMES = frozenset({
    "Exception", "IndexError", "OverflowError", "RuntimeError", "ValueError",
    "ZeroDivisionError", "abs", "all", "any", "bool", "dict", "enumerate",
    "float", "int", "len", "list", "max", "min", "range", "round", "set",
    "sorted", "sum", "tuple", "zip",
})

_SAFE_MODULE_CALL_MAX_POSITIONAL: Mapping[str, int] = {
    "collections.deque": 1,
    "itertools.permutations": 1,
    "math.acos": 1, "math.cos": 1, "math.fsum": 1, "math.hypot": 2,
    "math.radians": 1, "math.sin": 1, "math.sqrt": 1,
    "numpy.abs": 1, "numpy.any": 1, "numpy.append": 2,
    "numpy.arange": 1, "numpy.arccos": 1, "numpy.arctan2": 2,
    "numpy.argmax": 1, "numpy.argmin": 1, "numpy.argsort": 1,
    "numpy.array": 1, "numpy.asarray": 1, "numpy.clip": 3,
    "numpy.column_stack": 1, "numpy.concatenate": 1, "numpy.convolve": 2,
    "numpy.cov": 1, "numpy.cumsum": 1, "numpy.degrees": 1,
    "numpy.diff": 1, "numpy.digitize": 2, "numpy.dot": 2,
    "numpy.empty": 1, "numpy.fft.rfft": 1, "numpy.full": 2,
    "numpy.gradient": 1, "numpy.hypot": 2, "numpy.interp": 3,
    "numpy.isfinite": 1, "numpy.isnan": 1, "numpy.linalg.eigh": 1,
    "numpy.linalg.eigvalsh": 1, "numpy.linalg.lstsq": 2,
    "numpy.linalg.norm": 1, "numpy.linspace": 3, "numpy.logical_and": 2,
    "numpy.logical_or": 2, "numpy.max": 1, "numpy.mean": 1,
    "numpy.min": 1, "numpy.minimum": 2, "numpy.nonzero": 1,
    "numpy.ones": 1, "numpy.ones_like": 1,
    "numpy.pad": 2, "numpy.roll": 2, "numpy.rot90": 2,
    "numpy.round": 1, "numpy.sign": 1, "numpy.sort": 1,
    "numpy.sqrt": 1, "numpy.stack": 1, "numpy.std": 1, "numpy.sum": 1,
    "numpy.unique": 1, "numpy.unwrap": 1, "numpy.vstack": 1,
    "numpy.where": 1, "numpy.zeros": 1, "numpy.zeros_like": 1,
    "scipy.ndimage.binary_dilation": 1, "scipy.ndimage.binary_erosion": 1,
    "scipy.ndimage.binary_fill_holes": 1, "scipy.ndimage.convolve": 2,
    "scipy.ndimage.gaussian_filter1d": 2, "scipy.ndimage.label": 1,
    "scipy.ndimage.sum": 3, "scipy.spatial.ConvexHull": 1,
    "scipy.spatial.cKDTree": 1, "skimage.morphology.skeletonize": 1,
}

# Keywords are capability too.  In particular, otherwise-pure NumPy ufuncs
# can expose uninitialised memory through ``where=`` and Qhull accepts option
# strings that request randomised rotations.  An absent entry means that the
# callable is positional-only in the predicate language.
_SAFE_MODULE_CALL_KEYWORDS: Mapping[str, frozenset[str]] = {
    "numpy.array": frozenset({"dtype"}),
    "numpy.convolve": frozenset({"mode"}),
    "numpy.diff": frozenset({"axis"}),
    "numpy.empty": frozenset({"dtype"}),
    "numpy.linalg.lstsq": frozenset({"rcond"}),
    "numpy.linalg.norm": frozenset({"axis"}),
    "numpy.min": frozenset({"axis"}),
    "numpy.ones": frozenset({"dtype"}),
    "numpy.pad": frozenset({"constant_values", "mode"}),
    "numpy.roll": frozenset({"axis"}),
    "numpy.stack": frozenset({"axis"}),
    "numpy.sum": frozenset({"axis"}),
    "numpy.zeros": frozenset({"dtype"}),
    "numpy.zeros_like": frozenset({"dtype"}),
    "scipy.ndimage.binary_dilation": frozenset({"iterations", "structure"}),
    "scipy.ndimage.binary_erosion": frozenset({"iterations"}),
    "scipy.ndimage.convolve": frozenset({"cval", "mode"}),
    "scipy.ndimage.gaussian_filter1d": frozenset({"mode", "sigma"}),
    "scipy.ndimage.label": frozenset({"structure"}),
}

_SAFE_MODULE_VALUES = frozenset({
    "math.pi", "numpy.c_", "numpy.fft", "numpy.float64", "numpy.inf",
    "numpy.int32", "numpy.linalg", "numpy.mgrid", "numpy.nan", "numpy.pi",
    "numpy.uint8", "scipy.ndimage",
})

_SAFE_INSTANCE_ATTRIBUTES = frozenset({
    "T", "equations", "label", "shape", "size", "vertices", "volume",
})
_SAFE_INSTANCE_METHOD_MAX_POSITIONAL: Mapping[str, int] = {
    "any": 1, "astype": 1, "get": 2, "max": 1, "mean": 1, "min": 1,
    "query_ball_point": 1, "ravel": 0, "reshape": 1, "std": 1, "sum": 1,
    "tolist": 0,
    # Mutators are admitted only on locally fresh objects by
    # _MutationAndAllocationSafety below.
    "append": 1, "discard": 1, "pop": 0, "popleft": 0, "sort": 0,
    "update": 1,
}
_SAFE_INSTANCE_METHOD_KEYWORDS: Mapping[str, frozenset[str]] = {
    "any": frozenset({"axis", "keepdims"}),
    "astype": frozenset(),
    "get": frozenset(),
    "max": frozenset({"axis", "keepdims"}),
    "mean": frozenset({"axis", "keepdims"}),
    "min": frozenset({"axis", "keepdims"}),
    "query_ball_point": frozenset({"r"}),
    "ravel": frozenset(),
    "reshape": frozenset(),
    "std": frozenset({"axis", "keepdims"}),
    "sum": frozenset({"axis", "keepdims"}),
    "tolist": frozenset(),
    "append": frozenset(),
    "discard": frozenset(),
    "pop": frozenset(),
    "popleft": frozenset(),
    "sort": frozenset({"key", "reverse"}),
    "update": frozenset(),
}
_MUTATING_INSTANCE_METHODS = frozenset({
    "append", "discard", "pop", "popleft", "sort", "update",
})
_FORBIDDEN_MUTATION_KEYWORDS = frozenset({"out", "output"})

# These calls allocate independent storage when used with the keyword policy
# above.  View-producing calls (notably asarray and rot90) are intentionally
# absent; their results may be read but cannot be mutated.
_FRESH_MODULE_CALLS = frozenset({
    name for name in _SAFE_MODULE_CALL_MAX_POSITIONAL
    if name.startswith(("numpy.", "scipy.", "skimage."))
}) - frozenset({
    "numpy.any", "numpy.argmax", "numpy.argmin", "numpy.asarray",
    "numpy.dot", "numpy.empty", "numpy.isfinite", "numpy.isnan",
    "numpy.max", "numpy.mean", "numpy.min", "numpy.rot90", "numpy.std",
    "numpy.sum", "scipy.ndimage.sum", "scipy.spatial.ConvexHull",
    "scipy.spatial.cKDTree",
})

MAX_STATIC_ALLOCATION_ELEMENTS = 16_384
MAX_STATIC_ITERATION_COUNT = 1_000_000
MAX_STATIC_INTEGER_BITS = 1_000_000
MAX_STATIC_PAD_WIDTH = 64
MAX_STATIC_CONTAINER_ELEMENTS = 256
MAX_STATIC_MORPHOLOGY_ITERATIONS = 64
MAX_PREDICATE_FUNCTIONS = 64
MAX_EXPANDED_NATIVE_CALLS_PER_PANEL = 128
MAX_EXPANDED_HEAVY_NATIVE_CALLS_PER_PANEL = 8
MAX_SOURCE_CHARACTERS = 1_000_000
MAX_SOURCE_UTF8_BYTES = 1_000_000

# Public, immutable contract summaries for proposer prompts and protocol docs.
ALLOWED_PURE_CALLS = frozenset(_SAFE_MODULE_CALL_MAX_POSITIONAL)
ALLOWED_INSTANCE_ATTRIBUTES = _SAFE_INSTANCE_ATTRIBUTES
ALLOWED_INSTANCE_METHODS = frozenset(_SAFE_INSTANCE_METHOD_MAX_POSITIONAL)
PREDICATE_PURITY_POLICY_ID = "bongard-predicate-purity/v2"
PREDICATE_PRICING_POLICY_ID = "bongard-predicate-pricing/v3"

_FORBIDDEN_DYNAMIC_NAMES = frozenset({
    "__import__", "breakpoint", "compile", "delattr", "eval", "exec",
    "getattr", "globals", "hasattr", "hash", "input", "locals", "open",
    "setattr", "vars",
})

_FORBIDDEN_ATTRIBUTES = frozenset({
    "DataSource", "attrgetter", "ctypes", "ctypeslib", "datasets", "dump",
    "dumps", "fromfile", "fromregex", "genfromtxt", "imread",
    "imread_collection", "imsave", "io", "load", "load_library", "loadmat",
    "loads", "loadtxt", "memmap", "methodcaller", "mmread", "mmwrite",
    "open_memmap", "os", "recfromcsv", "recfromtxt", "open", "save",
    "savemat", "savetxt", "savez", "savez_compressed", "subprocess", "sys",
    "tofile",
})


class PredicatePricingError(ValueError):
    """The source or a pricing request cannot be priced soundly."""


class UnknownPredicateError(PredicatePricingError):
    """A requested name is not a module-level ``p_*`` function."""


def _validate_predicate_source_text(source: str) -> None:
    """Bound parser input before constructing an AST or other large objects."""
    if not isinstance(source, str):
        raise PredicatePricingError("source must be a string")
    if len(source) > MAX_SOURCE_CHARACTERS:
        raise PredicatePricingError(
            "predicate source exceeds the pre-AST character limit")
    try:
        source_size = len(source.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise PredicatePricingError(
            "predicate source is not valid UTF-8 text") from exc
    if source_size > MAX_SOURCE_UTF8_BYTES:
        raise PredicatePricingError(
            "predicate source exceeds the UTF-8 byte limit")


def read_predicate_source(path: str) -> str:
    """Read one UTF-8 predicate file without ever ingesting an oversized file."""
    with open(path, "rb") as handle:
        payload = handle.read(MAX_SOURCE_UTF8_BYTES + 1)
    if len(payload) > MAX_SOURCE_UTF8_BYTES:
        raise PredicatePricingError(
            "predicate source exceeds the UTF-8 byte limit")
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PredicatePricingError(
            "predicate source is not valid UTF-8 text") from exc
    _validate_predicate_source_text(source)
    return source


def _attribute_parts(node: ast.AST) -> tuple[str, tuple[str, ...]] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return node.id, tuple(reversed(parts))


def _canonical_reference(
    node: ast.AST, import_capabilities: Mapping[str, str],
) -> str | None:
    parts = _attribute_parts(node)
    if parts is None or parts[0] not in import_capabilities:
        return None
    return ".".join((import_capabilities[parts[0]], *parts[1]))


def _import_capability_map(
    tree: ast.Module,
) -> tuple[dict[str, str], frozenset[int]]:
    """Validate exact imports and resolve their local aliases.

    A few promoted libraries import optional SciPy helpers inside the function
    that uses them.  Those imports are still safe because both the module and
    member are exact capabilities and their source is charged with that
    function.  Reusing one alias for different capabilities is rejected so
    syntactic call resolution remains unambiguous across lexical scopes.
    """
    capabilities: dict[str, str] = {}
    statement_ids: set[int] = set()
    for statement in ast.walk(tree):
        if isinstance(statement, ast.Import):
            statement_ids.add(id(statement))
            for alias in statement.names:
                if alias.name not in ALLOWED_IMPORT_MODULES:
                    raise PredicatePricingError(
                        f"predicate source imports unsupported module / "
                        f"dangerous module path "
                        f"{alias.name!r}")
                local = alias.asname or alias.name
                if local in SAFE_BUILTIN_NAMES:
                    raise PredicatePricingError(
                        f"import alias {local!r} cannot replace a certified "
                        "builtin")
                previous = capabilities.get(local)
                if previous is not None and previous != alias.name:
                    raise PredicatePricingError(
                        f"import alias {local!r} has conflicting capabilities")
                capabilities[local] = alias.name
        elif isinstance(statement, ast.ImportFrom):
            statement_ids.add(id(statement))
            if statement.level:
                raise PredicatePricingError(
                    "relative imports are forbidden in predicates")
            module = statement.module or ""
            if module == "__future__":
                if any(alias.name != "annotations" for alias in statement.names):
                    raise PredicatePricingError(
                        "only future annotations may be imported in predicates")
                continue
            allowed = ALLOWED_FROM_IMPORTS.get(module, frozenset())
            for alias in statement.names:
                if alias.name == "*":
                    raise PredicatePricingError(
                        "star imports cannot be priced because their bound "
                        "names are unknown")
                if alias.name not in allowed:
                    raise PredicatePricingError(
                        f"predicate source imports unsupported dangerous "
                        f"name/capability "
                        f"{module}.{alias.name}")
                local = alias.asname or alias.name
                if local in SAFE_BUILTIN_NAMES:
                    raise PredicatePricingError(
                        f"import alias {local!r} cannot replace a certified "
                        "builtin")
                capability = f"{module}.{alias.name}"
                previous = capabilities.get(local)
                if previous is not None and previous != capability:
                    raise PredicatePricingError(
                        f"import alias {local!r} has conflicting capabilities")
                capabilities[local] = capability
    return capabilities, frozenset(statement_ids)


def predicate_execution_builtins() -> dict[str, object]:
    """Return the minimal builtins namespace used for predicate execution."""
    namespace = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
    # Import statements require the hook, but source cannot name it: the AST
    # gate rejects every dunder load and it is absent from SAFE_BUILTIN_NAMES.
    namespace["__import__"] = builtins.__import__
    return namespace


def predicate_capability_manifest() -> dict[str, object]:
    """Return the complete, deterministic predicate-language contract.

    The returned object contains only strings, integers, tuples, and fresh
    dictionaries, so protocol code may serialize it without reaching into
    private implementation constants.  Mutating a returned dictionary cannot
    alter the policy used by this module.
    """
    return {
        "policy_id": PREDICATE_PURITY_POLICY_ID,
        "pricing_policy_id": PREDICATE_PRICING_POLICY_ID,
        "imports": {
            "modules": tuple(sorted(ALLOWED_IMPORT_MODULES)),
            "members": tuple(
                (module, tuple(sorted(members)))
                for module, members in sorted(ALLOWED_FROM_IMPORTS.items())
            ),
        },
        "builtins": tuple(sorted(SAFE_BUILTIN_NAMES)),
        "module_calls": tuple(
            (
                name,
                _SAFE_MODULE_CALL_MAX_POSITIONAL[name],
                tuple(sorted(_SAFE_MODULE_CALL_KEYWORDS.get(
                    name, frozenset()))),
            )
            for name in sorted(_SAFE_MODULE_CALL_MAX_POSITIONAL)
        ),
        "module_values": tuple(sorted(_SAFE_MODULE_VALUES)),
        "instance_attributes": tuple(sorted(_SAFE_INSTANCE_ATTRIBUTES)),
        "instance_methods": tuple(
            (
                name,
                _SAFE_INSTANCE_METHOD_MAX_POSITIONAL[name],
                tuple(sorted(_SAFE_INSTANCE_METHOD_KEYWORDS.get(
                    name, frozenset()))),
            )
            for name in sorted(_SAFE_INSTANCE_METHOD_MAX_POSITIONAL)
        ),
        "mutation": {
            "methods": tuple(sorted(_MUTATING_INSTANCE_METHODS)),
            "receiver_policy": "direct-name-locally-owned-storage-only",
            "subscript_policy": "direct-name-locally-owned-storage-only",
            "np_empty_policy": "one-dimensional-immediate-enumerate-full-fill",
            "module_parameter_and_view_mutation": "forbidden",
        },
        "resources": {
            "max_source_characters_before_ast": MAX_SOURCE_CHARACTERS,
            "max_source_utf8_bytes_before_ast": MAX_SOURCE_UTF8_BYTES,
            "file_read_policy": "binary-prefix-byte-cap-before-decode",
            "max_static_allocation_elements": MAX_STATIC_ALLOCATION_ELEMENTS,
            "max_static_iteration_count": MAX_STATIC_ITERATION_COUNT,
            "max_static_integer_bits": MAX_STATIC_INTEGER_BITS,
            "max_static_pad_width": MAX_STATIC_PAD_WIDTH,
            "max_static_container_elements": MAX_STATIC_CONTAINER_ELEMENTS,
            "max_static_morphology_iterations": (
                MAX_STATIC_MORPHOLOGY_ITERATIONS),
            "max_predicate_functions": MAX_PREDICATE_FUNCTIONS,
            "max_expanded_native_calls_per_panel": (
                MAX_EXPANDED_NATIVE_CALLS_PER_PANEL),
            "max_expanded_heavy_native_calls_per_panel": (
                MAX_EXPANDED_HEAVY_NATIVE_CALLS_PER_PANEL),
            "heavy_native_module_calls": tuple(sorted(_HEAVY_NATIVE_CALLS)),
            "heavy_native_instance_methods": tuple(
                sorted(_HEAVY_NATIVE_INSTANCE_METHODS)),
            "allowed_power_exponents": (0.5, 2),
            "while_policy": "certified-finite-worklist-only",
            "loop_native_call_policy": (
                "numpy-scipy-skimage-instance-and-helper-calls-forbidden-"
                "in-loops-comprehensions-and-lambda-callbacks"),
            "native_operator_policy": (
                "tainted-array-operators-counted-straight-line-and-forbidden-"
                "in-loops-comprehensions-and-lambda-callbacks"),
            "explicit_array_shape_policy": "source-static-required",
        },
        "unordered_containers": {
            "iteration": "forbidden",
            "escape_or_conversion": "forbidden",
            "observable_operations": ("bool", "len", "membership", "set-algebra"),
        },
    }


class _PredicateSourceSafety(ast.NodeVisitor):
    """Positive capability gate for deterministic, side-effect-free source."""

    def __init__(
        self,
        import_capabilities: Mapping[str, str],
        top_level_import_ids: frozenset[int],
        defined_functions: frozenset[str],
    ) -> None:
        self.import_capabilities = import_capabilities
        self.top_level_import_ids = top_level_import_ids
        self.defined_functions = defined_functions

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load) \
                and (node.id in _FORBIDDEN_DYNAMIC_NAMES
                     or node.id.startswith("__")):
            raise PredicatePricingError(
                f"dynamic or I/O name {node.id!r} is forbidden")

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        if id(node) not in self.top_level_import_ids:
            raise PredicatePricingError("unvalidated predicate import")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if id(node) not in self.top_level_import_ids:
            raise PredicatePricingError("unvalidated predicate import")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        raise PredicatePricingError("async predicate functions are forbidden")

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        raise PredicatePricingError("predicate functions cannot mutate globals")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:  # noqa: N802
        raise PredicatePricingError("predicate functions cannot mutate nonlocals")

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
        raise PredicatePricingError(
            "assignment expressions are forbidden in predicate source")

    def visit_Delete(self, node: ast.Delete) -> None:  # noqa: N802
        raise PredicatePricingError(
            "deletion can mutate aliased state and is forbidden")

    def visit_Try(self, node: ast.Try) -> None:  # noqa: N802
        # The verifier's deterministic line-event budget is enforced by an
        # exception raised outside proposer code.  Predicate code must not be
        # able to catch that control signal and continue unmetered.
        raise PredicatePricingError(
            "exception handling is forbidden in predicate source")

    visit_TryStar = visit_Try

    def visit_Match(self, node: ast.Match) -> None:  # noqa: N802
        # Pattern bindings are represented as strings rather than Name(Store),
        # complicating the fail-closed shadow/ownership proof for no benefit to
        # the promoted predicate language.
        raise PredicatePricingError(
            "structural pattern matching is outside the predicate subset")

    def visit_Yield(self, node: ast.Yield) -> None:  # noqa: N802
        raise PredicatePricingError(
            "generator functions are outside the bounded predicate subset")

    visit_YieldFrom = visit_Yield

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802
        if isinstance(node.op, ast.Pow) and not (
                isinstance(node.right, ast.Constant)
                and node.right.value in {0.5, 2}):
            raise PredicatePricingError(
                "predicate powers are restricted to square and square root")
        if isinstance(node.op, (ast.LShift, ast.RShift)):
            raise PredicatePricingError(
                "integer shifts are outside the bounded predicate subset")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if any(keyword.arg is None for keyword in node.keywords):
            raise PredicatePricingError(
                "dynamic keyword unpacking is forbidden in predicate calls")
        forbidden_keywords = sorted(
            keyword.arg for keyword in node.keywords
            if keyword.arg in _FORBIDDEN_MUTATION_KEYWORDS)
        if forbidden_keywords:
            raise PredicatePricingError(
                f"predicate calls cannot mutate output buffers: "
                f"{forbidden_keywords}")

        if isinstance(node.func, ast.Name) \
                and node.func.id in _FORBIDDEN_DYNAMIC_NAMES:
            raise PredicatePricingError(
                f"dynamic or I/O name {node.func.id!r} is forbidden")
        for target_part in ast.walk(node.func):
            if isinstance(target_part, ast.Name) \
                    and target_part.id.startswith("__"):
                raise PredicatePricingError(
                    f"dynamic or I/O name {target_part.id!r} is forbidden")
            if isinstance(target_part, ast.Attribute) \
                    and target_part.attr.startswith("__"):
                raise PredicatePricingError(
                    f"dynamic or file-I/O attribute "
                    f"{target_part.attr!r} is forbidden")

        canonical = _canonical_reference(node.func, self.import_capabilities)
        max_positional: int | None = None
        allowed_keywords: frozenset[str] | None = None
        if canonical is not None:
            max_positional = _SAFE_MODULE_CALL_MAX_POSITIONAL.get(canonical)
            if max_positional is None:
                leaf = canonical.rsplit(".", 1)[-1]
                if leaf in _FORBIDDEN_ATTRIBUTES:
                    raise PredicatePricingError(
                        f"dynamic or file-I/O attribute {canonical!r} is forbidden")
                raise PredicatePricingError(
                    f"module call {canonical!r} is not an allowed pure capability")
            allowed_keywords = _SAFE_MODULE_CALL_KEYWORDS.get(
                canonical, frozenset())
        elif isinstance(node.func, ast.Name):
            name = node.func.id
            if name in self.import_capabilities:
                canonical = self.import_capabilities[name]
                max_positional = _SAFE_MODULE_CALL_MAX_POSITIONAL.get(canonical)
                if max_positional is None:
                    raise PredicatePricingError(
                        f"imported call {canonical!r} is not an allowed pure "
                        "capability")
                allowed_keywords = _SAFE_MODULE_CALL_KEYWORDS.get(
                    canonical, frozenset())
            elif name not in SAFE_BUILTIN_NAMES \
                    and name not in self.defined_functions:
                raise PredicatePricingError(
                    f"call target {name!r} has no priced definition and is "
                    "not statically certified")
        elif isinstance(node.func, ast.Attribute):
            max_positional = _SAFE_INSTANCE_METHOD_MAX_POSITIONAL.get(
                node.func.attr)
            if max_positional is None:
                raise PredicatePricingError(
                    f"instance method {node.func.attr!r} is not an allowed "
                    "pure capability")
            allowed_keywords = _SAFE_INSTANCE_METHOD_KEYWORDS.get(
                node.func.attr, frozenset())
        else:
            raise PredicatePricingError(
                "dynamic call targets are forbidden in predicate source")

        starred = [arg for arg in node.args if isinstance(arg, ast.Starred)]
        if starred:
            local_helper = isinstance(node.func, ast.Name) \
                and node.func.id in self.defined_functions \
                and node.func.id not in self.import_capabilities
            safe_hypot = canonical == "numpy.hypot" \
                and len(node.args) == 1 and len(starred) == 1 \
                and self._safe_hypot_expansion(starred[0].value)
            if not (local_helper or safe_hypot):
                raise PredicatePricingError(
                    "dynamic positional unpacking is forbidden in predicate calls")
        elif max_positional is not None and len(node.args) > max_positional:
            raise PredicatePricingError(
                f"call {canonical or getattr(node.func, 'attr', '<call>')!r} "
                "has unsupported positional output/callback arguments")

        if allowed_keywords is not None:
            unsupported = sorted(
                keyword.arg for keyword in node.keywords
                if keyword.arg not in allowed_keywords)
            if unsupported:
                raise PredicatePricingError(
                    f"call {canonical or getattr(node.func, 'attr', '<call>')!r} "
                    f"uses unsupported keyword capabilities: {unsupported}")

        if isinstance(node.func, ast.Attribute) and node.func.attr == "sort":
            keys = [keyword.value for keyword in node.keywords
                    if keyword.arg == "key"]
            if keys and not isinstance(keys[0], ast.Lambda):
                raise PredicatePricingError(
                    "sort key callbacks must be an inline validated lambda")
        if isinstance(node.func, ast.Name) \
                and node.func.id in {"max", "min", "sorted"}:
            keys = [keyword.value for keyword in node.keywords
                    if keyword.arg == "key"]
            if keys and not isinstance(keys[0], ast.Lambda):
                raise PredicatePricingError(
                    f"{node.func.id} key callbacks must be an inline "
                    "validated lambda")

        if canonical == "numpy.pad":
            modes = [keyword.value for keyword in node.keywords
                     if keyword.arg == "mode"]
            if modes and not (isinstance(modes[0], ast.Constant)
                              and modes[0].value == "constant"):
                raise PredicatePricingError(
                    "numpy.pad is restricted to deterministic constant mode")
        if canonical is not None:
            for keyword in node.keywords:
                if keyword.arg == "dtype" and isinstance(
                        keyword.value, ast.Constant) \
                        and keyword.value.value in {"O", "object"}:
                    raise PredicatePricingError("object arrays are forbidden")
        self.generic_visit(node)

    @staticmethod
    def _safe_hypot_expansion(node: ast.AST) -> bool:
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Sub):
            return False
        calls = (node.left, node.right)
        if not all(isinstance(call, ast.Call)
                   and isinstance(call.func, ast.Attribute)
                   and call.func.attr in {"max", "min"}
                   and len(call.args) == 0
                   and any(keyword.arg == "axis"
                           and isinstance(keyword.value, ast.Constant)
                           and keyword.value.value == 0
                           for keyword in call.keywords)
                   for call in calls):
            return False
        left_func = calls[0].func
        right_func = calls[1].func
        return isinstance(left_func, ast.Attribute) \
            and isinstance(right_func, ast.Attribute) \
            and ast.dump(left_func.value) == ast.dump(right_func.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            raise PredicatePricingError(
                "attribute mutation is forbidden in predicate source")
        if node.attr.startswith("__"):
            raise PredicatePricingError(
                f"dynamic or file-I/O attribute {node.attr!r} is forbidden")
        canonical = _canonical_reference(node, self.import_capabilities)
        if canonical is not None:
            if canonical not in _SAFE_MODULE_CALL_MAX_POSITIONAL \
                    and canonical not in _SAFE_MODULE_VALUES:
                raise PredicatePricingError(
                    f"module attribute {canonical!r} is not an allowed pure "
                    "capability")
        elif node.attr not in _SAFE_INSTANCE_ATTRIBUTES \
                and node.attr not in _SAFE_INSTANCE_METHOD_MAX_POSITIONAL:
            raise PredicatePricingError(
                f"instance attribute {node.attr!r} is not an allowed pure "
                "capability")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        # A class body is an executable module-load program, not a lexical
        # declaration.  In particular, names in it resolve sequentially and it
        # can mutate module state before any priced predicate is called.  The
        # predicate language has no need for that ambiguity: helper functions
        # and literal module bindings cover the supported use cases.
        raise PredicatePricingError(
            "class definitions are forbidden in predicate source because "
            "class bodies execute at definition time")


class _TopLevelBindingSafety(ast.NodeVisitor):
    """A module binding may describe data, but may not execute side effects."""

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        raise PredicatePricingError(
            "top-level binding expressions cannot call functions")

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        raise PredicatePricingError("top-level lambda bindings are forbidden")

    def visit_ListComp(self, node: ast.ListComp) -> None:  # noqa: N802
        raise PredicatePricingError(
            "top-level comprehension bindings are forbidden")

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp
    visit_NamedExpr = visit_ListComp


def _check_function_import_time(
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
) -> None:
    """Reject executable function-definition syntax at module import.

    Function bodies execute only when called and are priced as part of their
    definition.  Decorator application and default/annotation expressions run
    while the module itself is loaded, however.  An unselected function could
    otherwise mutate state used by a selected predicate while remaining outside
    that predicate's dependency closure.
    """
    if statement.decorator_list:
        raise PredicatePricingError(
            "predicate function definitions cannot use decorators")
    expressions: list[ast.AST] = [
        default
        for default in (*statement.args.defaults, *statement.args.kw_defaults)
        if default is not None
    ]
    for argument in (
        *statement.args.posonlyargs,
        *statement.args.args,
        *statement.args.kwonlyargs,
    ):
        if argument.annotation is not None:
            expressions.append(argument.annotation)
    if statement.args.vararg is not None \
            and statement.args.vararg.annotation is not None:
        expressions.append(statement.args.vararg.annotation)
    if statement.args.kwarg is not None \
            and statement.args.kwarg.annotation is not None:
        expressions.append(statement.args.kwarg.annotation)
    if statement.returns is not None:
        expressions.append(statement.returns)
    expressions.extend(getattr(statement, "type_params", ()))
    for expression in expressions:
        _TopLevelBindingSafety().visit(expression)


def _check_lambda_definition_time(statement: ast.Lambda) -> None:
    """Reject executable lambda defaults evaluated when the lambda is made."""
    for default in (*statement.args.defaults, *statement.args.kw_defaults):
        if default is not None:
            _TopLevelBindingSafety().visit(default)


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _non_comment_loc(source: str) -> int:
    return sum(
        1
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def _large_literal_cost(node: ast.AST) -> int:
    """Charge collection cardinality and dense scalar payloads.

    LOC alone makes a one-line panel-hash string or giant integer nearly free.
    Small constants (up to 16 encoded bytes) remain covered by their source
    line; larger constants pay one additional unit per 16-byte block.  Call
    arguments are also charged individually: ``dict(k0=0, ..., k999=1)`` is a
    thousand-entry table even though it has no :class:`ast.Dict` node.  This is
    a transparent description-length proxy, not a ban on legitimate tables.
    """
    cost = 0
    for child in ast.walk(node):
        if isinstance(child, (ast.List, ast.Tuple, ast.Set)):
            cost += len(child.elts)
        elif isinstance(child, ast.Dict):
            cost += len(child.keys)
        elif isinstance(child, ast.Call):
            cost += len(child.args) + len(child.keywords)
        elif isinstance(child, ast.Constant):
            value = child.value
            if isinstance(value, str):
                size = len(value.encode("utf-8"))
            elif isinstance(value, bytes):
                size = len(value)
            elif isinstance(value, int) and not isinstance(value, bool):
                size = max(1, (abs(value).bit_length() + 7) // 8)
            elif isinstance(value, (float, complex)):
                size = len(repr(value).encode("ascii"))
            else:
                size = 0
            cost += max(0, (size + 15) // 16 - 1)
    return cost


def _ast_structure_cost(node: ast.AST) -> int:
    """Charge executable syntax that can be packed onto one source line.

    LOC is readable but not a sufficient description-length measure: deeply
    nested arithmetic, boolean formulae, comprehensions, and semicolon-packed
    statements can otherwise be nearly free.  Calls and literal payloads are
    charged separately by :func:`_large_literal_cost`.
    """
    cost = 0
    for child in ast.walk(node):
        if child is not node and isinstance(child, ast.stmt):
            cost += 1
        elif isinstance(child, ast.BinOp):
            cost += 1
        elif isinstance(child, ast.BoolOp):
            cost += max(1, len(child.values) - 1)
        elif isinstance(child, ast.UnaryOp):
            cost += 1
        elif isinstance(child, ast.Compare):
            cost += max(1, len(child.ops))
        elif isinstance(child, (ast.IfExp, ast.Lambda, ast.Subscript)):
            cost += 1
        elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp,
                                ast.GeneratorExp)):
            cost += 1 + len(child.generators) + sum(
                len(generator.ifs) for generator in child.generators)
    return cost


def _target_names(target: ast.AST) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(
            name for element in target.elts for name in _target_names(element)
        )
    return ()


def _defined_names(statement: ast.stmt) -> tuple[str, ...]:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return (statement.name,)
    if isinstance(statement, ast.Assign):
        return tuple(
            dict.fromkeys(
                name
                for target in statement.targets
                for name in _target_names(target)
            )
        )
    if isinstance(statement, ast.AnnAssign):
        return _target_names(statement.target)
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        names: list[str] = []
        for alias in statement.names:
            if alias.name == "*":
                raise PredicatePricingError(
                    "star imports cannot be priced because their bound names are unknown"
                )
            names.append(alias.asname or alias.name.split(".", 1)[0])
        return tuple(dict.fromkeys(names))
    return ()


def _statement_kind(statement: ast.stmt) -> str:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "function"
    if isinstance(statement, ast.ClassDef):
        return "class"
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return "import"
    return "binding"


def _statement_source(source: str, statement: ast.stmt) -> str:
    """Return the exact priced span, including function/class decorators."""
    start = statement.lineno
    decorators = getattr(statement, "decorator_list", ())
    if decorators:
        start = min(start, *(decorator.lineno for decorator in decorators))
    end = getattr(statement, "end_lineno", None)
    if end is None:  # pragma: no cover - all supported Python ASTs provide it
        raise PredicatePricingError("AST statement has no end position")
    return "\n".join(source.splitlines()[start - 1:end])


class _LocalBindingCollector(ast.NodeVisitor):
    """Collect bindings belonging to one function-like lexical scope."""

    def __init__(self) -> None:
        self.bound: set[str] = set()
        self.globals: set[str] = set()
        self.nonlocals: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802 (AST API)
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound.add(node.id)

    def visit_arg(self, node: ast.arg) -> None:  # noqa: N802
        self.bound.add(node.arg)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self.bound.update(alias.asname or alias.name.split(".", 1)[0]
                          for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        self.bound.update(alias.asname or alias.name for alias in node.names)

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        self.globals.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:  # noqa: N802
        self.nonlocals.update(node.names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
        if node.name is not None:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self.bound.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self.bound.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.bound.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        return

    def visit_ListComp(self, node: ast.ListComp) -> None:  # noqa: N802
        return

    def visit_SetComp(self, node: ast.SetComp) -> None:  # noqa: N802
        return

    def visit_DictComp(self, node: ast.DictComp) -> None:  # noqa: N802
        return

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:  # noqa: N802
        return


def _function_argument_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> frozenset[str]:
    names = {
        argument.arg
        for argument in (
            *node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
        )
    }
    if node.args.vararg is not None:
        names.add(node.args.vararg.arg)
    if node.args.kwarg is not None:
        names.add(node.args.kwarg.arg)
    return frozenset(names)


def _scope_nodes(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> tuple[ast.AST, ...]:
    """Return nodes executed in one function scope, excluding nested scopes."""
    roots: list[ast.AST] = []
    roots.extend(node.decorator_list if isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef)) else ())
    roots.extend(default for default in (
        *node.args.defaults, *node.args.kw_defaults) if default is not None)
    roots.extend(node.body if not isinstance(node, ast.Lambda) else (node.body,))
    found: list[ast.AST] = []
    pending = list(reversed(roots))
    while pending:
        current = pending.pop()
        found.append(current)
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.Lambda, ast.ClassDef)):
            continue
        pending.extend(reversed(tuple(ast.iter_child_nodes(current))))
    return tuple(found)


def _call_capability(
    node: ast.Call, import_capabilities: Mapping[str, str],
) -> str | None:
    canonical = _canonical_reference(node.func, import_capabilities)
    if canonical is None and isinstance(node.func, ast.Name):
        canonical = import_capabilities.get(node.func.id)
    return canonical


class _CallableShadowSafety:
    """Prevent lexical rebinding from changing a certified call target."""

    def __init__(
        self,
        protected_names: frozenset[str],
        reserved_capability_names: frozenset[str],
        nullable_import_names: frozenset[str],
    ) -> None:
        self.protected_names = protected_names
        self.reserved_capability_names = reserved_capability_names
        self.nullable_import_names = nullable_import_names

    def validate(self, tree: ast.Module) -> None:
        conflicting_functions = sorted({
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name in self.reserved_capability_names
        })
        if conflicting_functions:
            raise PredicatePricingError(
                "function definitions cannot replace builtin/import "
                f"capabilities: {conflicting_functions}")
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.Lambda)):
                continue
            rebound = set(_function_argument_names(function))
            scope_nodes = _scope_nodes(function)
            allowed_null_stores = {
                id(target)
                for node in scope_nodes
                if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Constant)
                and node.value.value is None
                for target in node.targets
                if isinstance(target, ast.Name)
                and target.id in self.nullable_import_names
            }
            for node in scope_nodes:
                if isinstance(node, ast.Name) and isinstance(
                        node.ctx, (ast.Store, ast.Del)) \
                        and id(node) not in allowed_null_stores:
                    rebound.add(node.id)
                elif isinstance(node, ast.ExceptHandler) and node.name:
                    rebound.add(node.name)
            shadowed = sorted(rebound & self.protected_names)
            if shadowed:
                raise PredicatePricingError(
                    "predicate scope cannot shadow certified call/import "
                    f"names: {shadowed}")


def _statement_child_blocks(statement: ast.stmt) -> tuple[list[ast.stmt], ...]:
    blocks: list[list[ast.stmt]] = []
    for field in ("body", "orelse", "finalbody"):
        value = getattr(statement, field, None)
        if isinstance(value, list):
            blocks.append(value)
    if isinstance(statement, ast.Try):
        blocks.extend(handler.body for handler in statement.handlers)
    return tuple(blocks)


def _direct_name_target(target: ast.AST) -> str | None:
    return target.id if isinstance(target, ast.Name) else None


class _MutationAndAllocationSafety:
    """Admit mutation only on storage proven local to the current call."""

    def __init__(self, import_capabilities: Mapping[str, str]) -> None:
        self.import_capabilities = import_capabilities

    def validate(self, tree: ast.Module) -> None:
        for function in ast.walk(tree):
            if isinstance(function, (ast.FunctionDef, ast.Lambda)):
                self._validate_function(function)

    def _validate_function(
        self, function: ast.FunctionDef | ast.Lambda,
    ) -> None:
        certified_empty = self._certified_empty_calls(function)
        scope_nodes = _scope_nodes(function)
        all_empty = {
            id(node)
            for node in scope_nodes
            if isinstance(node, ast.Call)
            and _call_capability(node, self.import_capabilities) == "numpy.empty"
        }
        if all_empty - certified_empty:
            raise PredicatePricingError(
                "numpy.empty is forbidden unless an immediate enumerate loop "
                "provably fills every element before any read")

        bindings: dict[str, list[bool]] = {
            name: [False] for name in _function_argument_names(function)
        }

        def bind(name: str, owned: bool) -> None:
            bindings.setdefault(name, []).append(owned)

        for node in scope_nodes:
            if isinstance(node, ast.Assign):
                owned = self._owned_expression(node.value, certified_empty)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bind(target.id, owned)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for name in _target_names(target):
                            bind(name, False)
            elif isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name):
                bind(
                    node.target.id,
                    node.value is not None
                    and self._owned_expression(node.value, certified_empty),
                )
            elif isinstance(node, (ast.For, ast.comprehension)):
                for name in _target_names(node.target):
                    bind(name, False)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if item.optional_vars is not None:
                        for name in _target_names(item.optional_vars):
                            bind(name, False)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bind(node.name, False)

        owned_names = frozenset(
            name for name, origins in bindings.items() if origins and all(origins)
        )
        for node in scope_nodes:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                canonical = _canonical_reference(
                    node.func, self.import_capabilities)
                if canonical is None and node.func.attr in _MUTATING_INSTANCE_METHODS:
                    receiver = node.func.value
                    if not isinstance(receiver, ast.Name) \
                            or receiver.id not in owned_names:
                        raise PredicatePricingError(
                            f"mutating method {node.func.attr!r} requires a "
                            "direct locally owned receiver")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    self._validate_store_target(target, owned_names)
            elif isinstance(node, ast.AnnAssign):
                self._validate_store_target(node.target, owned_names)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Subscript):
                    # ``container[i] += value`` may mutate an aliased object
                    # stored at i before assigning it back.
                    raise PredicatePricingError(
                        "augmented subscript assignment can mutate aliased state")
                if isinstance(node.target, ast.Name) \
                        and node.target.id not in owned_names:
                    raise PredicatePricingError(
                        "augmented assignment requires a locally owned value")
                self._validate_store_target(node.target, owned_names)
            elif isinstance(node, (ast.For, ast.comprehension)):
                self._validate_store_target(node.target, owned_names,
                                            binding_only=True)

    def _owned_expression(
        self, node: ast.AST, certified_empty: frozenset[int],
    ) -> bool:
        if isinstance(node, (ast.Constant, ast.List, ast.Set, ast.Dict,
                             ast.ListComp, ast.SetComp, ast.DictComp)):
            return True
        if isinstance(node, ast.Tuple):
            return all(self._owned_expression(element, certified_empty)
                       for element in node.elts)
        if isinstance(node, (ast.UnaryOp, ast.Compare)):
            return True
        if isinstance(node, ast.BinOp):
            # Python arithmetic and ndarray operators produce a new value;
            # in-place operators are represented by AugAssign and checked
            # separately.
            return True
        if isinstance(node, ast.BoolOp):
            return all(self._owned_expression(value, certified_empty)
                       for value in node.values)
        if isinstance(node, ast.IfExp):
            return self._owned_expression(node.body, certified_empty) \
                and self._owned_expression(node.orelse, certified_empty)
        if not isinstance(node, ast.Call):
            return False
        canonical = _call_capability(node, self.import_capabilities)
        if canonical == "numpy.empty":
            return id(node) in certified_empty
        if canonical in _FRESH_MODULE_CALLS \
                or canonical == "collections.deque":
            return True
        if isinstance(node.func, ast.Name) \
                and node.func.id in {"dict", "list", "set"}:
            return True
        if isinstance(node.func, ast.Attribute) \
                and node.func.attr in {
                    "any", "astype", "max", "mean", "min", "ravel",
                    "reshape", "std", "sum", "tolist",
                }:
            # astype/to-list and reductions allocate; reshape/ravel may be
            # views and are therefore deliberately *not* mutable storage.
            return node.func.attr not in {"ravel", "reshape"}
        return False

    @staticmethod
    def _validate_store_target(
        target: ast.AST,
        owned_names: frozenset[str],
        *,
        binding_only: bool = False,
    ) -> None:
        if isinstance(target, ast.Subscript) and not binding_only:
            if not isinstance(target.value, ast.Name) \
                    or target.value.id not in owned_names:
                raise PredicatePricingError(
                    "subscript mutation requires direct locally owned storage")
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                _MutationAndAllocationSafety._validate_store_target(
                    element, owned_names, binding_only=binding_only)

    def _certified_empty_calls(
        self, function: ast.FunctionDef | ast.Lambda,
    ) -> frozenset[int]:
        if isinstance(function, ast.Lambda):
            return frozenset()
        certified: set[int] = set()

        def scan(block: list[ast.stmt]) -> None:
            for index, statement in enumerate(block):
                if index + 1 < len(block):
                    call = self._empty_assignment_call(statement)
                    if call is not None and self._is_full_fill_pair(
                            statement, block[index + 1]):
                        certified.add(id(call))
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef,
                                          ast.ClassDef)):
                    continue
                for child in _statement_child_blocks(statement):
                    scan(child)

        scan(function.body)
        return frozenset(certified)

    def _empty_assignment_call(self, statement: ast.stmt) -> ast.Call | None:
        if not isinstance(statement, ast.Assign) \
                or len(statement.targets) != 1 \
                or not isinstance(statement.targets[0], ast.Name) \
                or not isinstance(statement.value, ast.Call):
            return None
        if _call_capability(statement.value, self.import_capabilities) \
                != "numpy.empty":
            return None
        return statement.value

    def _is_full_fill_pair(
        self, assignment: ast.stmt, following: ast.stmt,
    ) -> bool:
        call = self._empty_assignment_call(assignment)
        if call is None or len(call.args) != 1:
            return False
        dtype_values = [keyword.value for keyword in call.keywords
                        if keyword.arg == "dtype"]
        if dtype_values and not (
                isinstance(dtype_values[0], ast.Name)
                and dtype_values[0].id in {"bool", "float", "int"}):
            return False
        assert isinstance(assignment, ast.Assign)
        target = assignment.targets[0]
        assert isinstance(target, ast.Name)
        shape = call.args[0]
        if not (isinstance(shape, ast.Call)
                and isinstance(shape.func, ast.Name)
                and shape.func.id == "len" and len(shape.args) == 1
                and not shape.keywords and isinstance(shape.args[0], ast.Name)):
            return False
        sequence = shape.args[0]
        if not (isinstance(following, ast.For) and not following.orelse
                and isinstance(following.iter, ast.Call)
                and isinstance(following.iter.func, ast.Name)
                and following.iter.func.id == "enumerate"
                and len(following.iter.args) == 1
                and not following.iter.keywords
                and ast.dump(following.iter.args[0]) == ast.dump(sequence)
                and isinstance(following.target, (ast.Tuple, ast.List))
                and len(following.target.elts) == 2
                and isinstance(following.target.elts[0], ast.Name)
                and len(following.body) == 1):
            return False
        index_name = following.target.elts[0].id
        fill = following.body[0]
        if not (isinstance(fill, ast.Assign) and len(fill.targets) == 1
                and isinstance(fill.targets[0], ast.Subscript)
                and isinstance(fill.targets[0].value, ast.Name)
                and fill.targets[0].value.id == target.id
                and isinstance(fill.targets[0].slice, ast.Name)
                and fill.targets[0].slice.id == index_name):
            return False
        forbidden_loads = {target.id, sequence.id}
        return not any(
            isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            and node.id in forbidden_loads
            for node in ast.walk(fill.value)
        )


def _set_expression(node: ast.AST, set_names: frozenset[str]) -> bool:
    if isinstance(node, (ast.Set, ast.SetComp)):
        return True
    if isinstance(node, ast.Name):
        return node.id in set_names
    if isinstance(node, ast.Call):
        return isinstance(node.func, ast.Name) and node.func.id == "set"
    if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.BitOr, ast.BitAnd, ast.BitXor, ast.Sub)):
        return _set_expression(node.left, set_names) \
            or _set_expression(node.right, set_names)
    return False


class _UnorderedContainerSafety:
    """Forbid observations whose value can depend on hash iteration order."""

    def validate(self, tree: ast.Module) -> None:
        module_sets = self._infer_sets(tuple(tree.body), frozenset())
        for function in ast.walk(tree):
            if isinstance(function, (ast.FunctionDef, ast.Lambda)):
                nodes = _scope_nodes(function)
                set_names = self._infer_sets(nodes, module_sets)
                safe_prior_value_iters = self._prior_value_set_iterators(nodes)
                self._validate_nodes(nodes, set_names, safe_prior_value_iters)

    @staticmethod
    def _prior_value_set_iterators(nodes: Iterable[ast.AST]) -> frozenset[int]:
        safe: set[int] = set()
        for node in nodes:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "set"):
                continue
            target = node.targets[0].id
            for child in ast.walk(node.value):
                if isinstance(child, ast.comprehension) \
                        and isinstance(child.iter, ast.Name) \
                        and child.iter.id == target:
                    # Assignment evaluates its RHS before rebinding target.
                    safe.add(id(child.iter))
        return frozenset(safe)

    @staticmethod
    def _infer_sets(
        nodes: Iterable[ast.AST], initial: frozenset[str],
    ) -> frozenset[str]:
        names = set(initial)
        assignments = [
            node for node in nodes
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ]
        changed = True
        while changed:
            changed = False
            for assignment in assignments:
                value = assignment.value
                if value is None or not _set_expression(
                        value, frozenset(names)):
                    continue
                targets = assignment.targets if isinstance(
                    assignment, ast.Assign) else (assignment.target,)
                for target in targets:
                    for name in _target_names(target):
                        if name not in names:
                            names.add(name)
                            changed = True
        return frozenset(names)

    def _validate_nodes(
        self,
        nodes: Iterable[ast.AST],
        set_names: frozenset[str],
        safe_prior_value_iters: frozenset[int],
    ) -> None:
        for node in nodes:
            if isinstance(node, (ast.For, ast.comprehension)) \
                    and _set_expression(node.iter, set_names) \
                    and id(node.iter) not in safe_prior_value_iters:
                raise PredicatePricingError(
                    "iteration over an unordered set is forbidden")
            if isinstance(node, ast.Return) and node.value is not None \
                    and self._contains_set(node.value, set_names):
                raise PredicatePricingError(
                    "sets cannot escape a predicate helper")
            if isinstance(node, ast.Assign) \
                    and any(isinstance(target, (ast.Tuple, ast.List))
                            for target in node.targets) \
                    and _set_expression(node.value, set_names):
                raise PredicatePricingError(
                    "destructuring an unordered set is forbidden")
            if isinstance(node, (ast.List, ast.Tuple)) and any(
                    _set_expression(element.value if isinstance(
                        element, ast.Starred) else element, set_names)
                    for element in node.elts):
                raise PredicatePricingError(
                    "sets cannot be embedded in ordered containers")
            if isinstance(node, ast.Dict) and any(
                    value is not None and _set_expression(value, set_names)
                    for value in (*node.keys, *node.values)):
                raise PredicatePricingError(
                    "sets cannot be embedded in mappings")
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "pop" \
                    and _set_expression(node.func.value, set_names):
                raise PredicatePricingError(
                    "set.pop depends on hash iteration order")
            set_arguments = [
                argument.value if isinstance(argument, ast.Starred) else argument
                for argument in node.args
                if _set_expression(
                    argument.value if isinstance(argument, ast.Starred)
                    else argument,
                    set_names,
                )
            ]
            if not set_arguments:
                continue
            if any(isinstance(argument, ast.Starred)
                   and _set_expression(argument.value, set_names)
                   for argument in node.args):
                raise PredicatePricingError(
                    "star-expanding an unordered set is forbidden")
            safe_builtin = isinstance(node.func, ast.Name) \
                and node.func.id in {"bool", "len", "set"}
            safe_set_update = isinstance(node.func, ast.Attribute) \
                and node.func.attr == "update" \
                and _set_expression(node.func.value, set_names)
            if not (safe_builtin or safe_set_update):
                raise PredicatePricingError(
                    "unordered sets cannot be passed to an observing call")

    @staticmethod
    def _contains_set(node: ast.AST, set_names: frozenset[str]) -> bool:
        if _set_expression(node, set_names):
            return True
        if isinstance(node, (ast.List, ast.Tuple)):
            return any(_UnorderedContainerSafety._contains_set(
                element.value if isinstance(element, ast.Starred) else element,
                set_names,
            ) for element in node.elts)
        if isinstance(node, ast.Dict):
            return any(
                child is not None and _UnorderedContainerSafety._contains_set(
                    child, set_names)
                for child in (*node.keys, *node.values)
            )
        return False


def _bounded_static_int(
    node: ast.AST,
    environment: Mapping[str, int],
    *,
    magnitude_cap: int,
) -> int | None:
    """Evaluate integer-only syntax while saturating before huge arithmetic."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        value = node.value
    elif isinstance(node, ast.Constant) and isinstance(node.value, float) \
            and node.value == node.value:
        # Conservatively round finite floats away from zero.  This may reject
        # a borderline allocation that int() would truncate below the limit,
        # but it cannot underestimate a coercion-hidden resource request.
        try:
            value = int(node.value)
        except (OverflowError, ValueError):
            return None
        if value != node.value:
            value += 1 if node.value > 0 else -1
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in {"abs", "float", "int", "round"} \
            and len(node.args) == 1 and not node.keywords:
        argument = node.args[0]
        if node.func.id in {"float", "int"} \
                and isinstance(argument, ast.Constant) \
                and isinstance(argument.value, str) \
                and len(argument.value) <= 128:
            try:
                parsed = (int(argument.value) if node.func.id == "int"
                          else float(argument.value))
                value = int(parsed)
                if isinstance(parsed, float) and value != parsed:
                    value += 1 if parsed > 0 else -1
            except (OverflowError, ValueError):
                return None
        else:
            coerced = _bounded_static_int(
                argument, environment, magnitude_cap=magnitude_cap)
            if coerced is None:
                return None
            value = abs(coerced) if node.func.id == "abs" else coerced
    elif isinstance(node, ast.Name):
        return environment.get(node.id)
    elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)):
        operand = _bounded_static_int(
            node.operand, environment, magnitude_cap=magnitude_cap)
        if operand is None:
            return None
        value = operand if isinstance(node.op, ast.UAdd) else -operand
    elif isinstance(node, ast.BinOp):
        left = _bounded_static_int(
            node.left, environment, magnitude_cap=magnitude_cap)
        right = _bounded_static_int(
            node.right, environment, magnitude_cap=magnitude_cap)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            if left and abs(right) > magnitude_cap // max(1, abs(left)):
                return (magnitude_cap + 1) * (1 if left * right >= 0 else -1)
            value = left * right
        elif isinstance(node.op, ast.FloorDiv):
            if right == 0:
                return None
            value = left // right
        elif isinstance(node.op, ast.Mod):
            if right == 0:
                return None
            value = left % right
        elif isinstance(node.op, ast.Pow):
            if right < 0:
                return None
            if right == 0:
                return 1
            if abs(left) > 1 and right * max(1, abs(left).bit_length() - 1) \
                    > magnitude_cap.bit_length():
                return magnitude_cap + 1
            value = left ** right
        elif isinstance(node.op, ast.LShift):
            if right < 0 or right > magnitude_cap.bit_length():
                return magnitude_cap + 1
            value = left << right
        elif isinstance(node.op, ast.RShift):
            if right < 0:
                return None
            value = left >> right
        else:
            return None
    else:
        return None
    if abs(value) > magnitude_cap:
        return (magnitude_cap + 1) * (1 if value >= 0 else -1)
    return value


_HEAVY_NATIVE_CALLS = frozenset({
    "numpy.append", "numpy.column_stack", "numpy.concatenate",
    "numpy.convolve", "numpy.cov", "numpy.dot", "numpy.fft.rfft",
    "numpy.gradient",
    "numpy.linalg.eigh", "numpy.linalg.eigvalsh", "numpy.linalg.lstsq",
    "numpy.pad", "numpy.sort", "numpy.stack", "numpy.unique",
    "numpy.vstack",
    "scipy.ndimage.binary_dilation", "scipy.ndimage.binary_erosion",
    "scipy.ndimage.binary_fill_holes", "scipy.ndimage.convolve",
    "scipy.ndimage.gaussian_filter1d", "scipy.ndimage.label",
    "scipy.spatial.ConvexHull", "scipy.spatial.cKDTree",
    "skimage.morphology.skeletonize",
})

_HEAVY_NATIVE_INSTANCE_METHODS = frozenset({"query_ball_point", "sort"})


class _NativeWorkSafety:
    """Bound native work that Python line tracing cannot observe."""

    _LOOP_NODES = (
        ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp,
        ast.DictComp, ast.GeneratorExp,
    )
    _CHEAP_LOOP_INSTANCE_METHODS = frozenset({
        "append", "discard", "get", "pop", "popleft", "update",
    })

    def __init__(
        self, import_capabilities: Mapping[str, str],
        defined_functions: frozenset[str],
    ) -> None:
        self.import_capabilities = import_capabilities
        self.defined_functions = defined_functions

    def validate(self, tree: ast.Module) -> None:
        functions = {
            node.name: node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        predicates = tuple(sorted(
            name for name in functions if name.startswith("p_")))
        if len(predicates) > MAX_PREDICATE_FUNCTIONS:
            raise PredicatePricingError(
                f"predicate function count {len(predicates)} exceeds limit "
                f"{MAX_PREDICATE_FUNCTIONS}")

        parents = {
            child: parent for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for call in (
                node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if self._inside_loop(call, parents) and self._loop_forbidden(call):
                raise PredicatePricingError(
                    "native, instance-method, and predicate-helper calls are "
                    "forbidden inside loops and comprehensions")

        # Operators on arrays dispatch into native NumPy work without an
        # ast.Call node.  Conservatively taint all function arguments and
        # values derived from native calls/tainted names.  Such operators are
        # counted in straight-line code and forbidden in amplification scopes.
        operator_types = (ast.BinOp, ast.UnaryOp, ast.Compare, ast.AugAssign)
        array_operator_counts: dict[str, int] = {}
        for name, function in functions.items():
            tainted = self._array_tainted_names(function)
            count = 0
            for node in _scope_nodes(function):
                if isinstance(node, operator_types) \
                        and self._array_operator_maybe_native(node, tainted):
                    if self._inside_loop(node, parents):
                        raise PredicatePricingError(
                            "array-valued operators are forbidden inside "
                            "loops, comprehensions, and lambda callbacks")
                    count += 1
            array_operator_counts[name] = count
        for callback in (
                node for node in ast.walk(tree) if isinstance(node, ast.Lambda)):
            tainted = self._array_tainted_names(callback)
            if any(
                    isinstance(node, operator_types)
                    and self._array_operator_maybe_native(node, tainted)
                    for node in _scope_nodes(callback)):
                raise PredicatePricingError(
                    "array-valued operators are forbidden inside lambda "
                    "callbacks")

        direct_native: dict[str, int] = {}
        direct_heavy: dict[str, int] = {}
        helper_calls: dict[str, list[str]] = {}
        for name, function in functions.items():
            native = array_operator_counts[name]
            heavy = 0
            helpers: list[str] = []
            for node in _scope_nodes(function):
                if not isinstance(node, ast.Call):
                    continue
                canonical = _call_capability(node, self.import_capabilities)
                if canonical is not None and canonical.startswith(
                        ("numpy.", "scipy.", "skimage.")):
                    native += 1
                    heavy += int(canonical in _HEAVY_NATIVE_CALLS)
                elif isinstance(node.func, ast.Attribute) \
                        and node.func.attr in _SAFE_INSTANCE_METHOD_MAX_POSITIONAL:
                    native += 1
                    heavy += int(
                        node.func.attr in _HEAVY_NATIVE_INSTANCE_METHODS)
                elif isinstance(node.func, ast.Name) \
                        and node.func.id in functions:
                    helpers.append(node.func.id)
            direct_native[name] = native
            direct_heavy[name] = heavy
            helper_calls[name] = helpers

        memo: dict[str, tuple[int, int]] = {}

        def expanded(name: str, active: frozenset[str]) -> tuple[int, int]:
            if name in memo:
                return memo[name]
            if name in active:
                raise PredicatePricingError(
                    "recursive predicate helpers are forbidden")
            native = direct_native[name]
            heavy = direct_heavy[name]
            next_active = active | {name}
            for helper in helper_calls[name]:
                child_native, child_heavy = expanded(helper, next_active)
                native += child_native
                heavy += child_heavy
                if native > MAX_EXPANDED_NATIVE_CALLS_PER_PANEL \
                        or heavy > MAX_EXPANDED_HEAVY_NATIVE_CALLS_PER_PANEL:
                    break
            memo[name] = (native, heavy)
            return memo[name]

        native_total = heavy_total = 0
        for predicate in predicates:
            native, heavy = expanded(predicate, frozenset())
            native_total += native
            heavy_total += heavy
            if native_total > MAX_EXPANDED_NATIVE_CALLS_PER_PANEL:
                raise PredicatePricingError(
                    "expanded native calls per panel exceed the deterministic "
                    f"limit {MAX_EXPANDED_NATIVE_CALLS_PER_PANEL}")
            if heavy_total > MAX_EXPANDED_HEAVY_NATIVE_CALLS_PER_PANEL:
                raise PredicatePricingError(
                    "expanded heavy native calls per panel exceed the "
                    f"deterministic limit "
                    f"{MAX_EXPANDED_HEAVY_NATIVE_CALLS_PER_PANEL}")

    @classmethod
    def _inside_loop(
        cls, call: ast.Call, parents: Mapping[ast.AST, ast.AST],
    ) -> bool:
        current: ast.AST = call
        while current in parents:
            current = parents[current]
            if isinstance(current, cls._LOOP_NODES):
                return True
            # Inline key lambdas are callbacks invoked once per container
            # element.  Treat their bodies as an amplification scope so a
            # native call cannot hide behind sorted/min/max/list.sort.
            if isinstance(current, ast.Lambda):
                return True
            if isinstance(current, ast.FunctionDef):
                return False
        return False

    def _loop_forbidden(self, call: ast.Call) -> bool:
        canonical = _call_capability(call, self.import_capabilities)
        if canonical is not None and canonical.startswith(
                ("numpy.", "scipy.", "skimage.")):
            return True
        if isinstance(call.func, ast.Attribute) \
                and call.func.attr in _SAFE_INSTANCE_METHOD_MAX_POSITIONAL \
                and call.func.attr not in self._CHEAP_LOOP_INSTANCE_METHODS:
            return True
        return isinstance(call.func, ast.Name) \
            and call.func.id in self.defined_functions

    def _array_tainted_names(
        self, function: ast.FunctionDef | ast.Lambda,
    ) -> frozenset[str]:
        nodes = _scope_nodes(function)
        tainted = set(_function_argument_names(function))

        def expression_tainted(node: ast.AST) -> bool:
            current_taint = frozenset(tainted)
            if self._expression_maybe_array(node, current_taint):
                return True
            for candidate in ast.walk(node):
                if not isinstance(candidate, ast.Call):
                    continue
                if self._call_maybe_array(candidate, current_taint):
                    return True
            return False

        changed = True
        while changed:
            changed = False
            for node in nodes:
                value: ast.AST | None = None
                targets: Iterable[ast.AST] = ()
                if isinstance(node, ast.Assign):
                    value, targets = node.value, node.targets
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    value, targets = node.value, (node.target,)
                elif isinstance(node, ast.NamedExpr):
                    value, targets = node.value, (node.target,)
                elif isinstance(node, (ast.For, ast.comprehension)):
                    value, targets = node.iter, (node.target,)
                if value is None or not expression_tainted(value):
                    continue
                before = len(tainted)
                for target in targets:
                    tainted.update(_target_names(target))
                changed |= len(tainted) != before
        return frozenset(tainted)

    def _array_operator_maybe_native(
        self, node: ast.AST, tainted: frozenset[str],
    ) -> bool:
        if isinstance(node, ast.BinOp):
            operands: Iterable[ast.AST] = (node.left, node.right)
        elif isinstance(node, ast.UnaryOp):
            operands = (node.operand,)
        elif isinstance(node, ast.Compare):
            operands = (node.left, *node.comparators)
        elif isinstance(node, ast.AugAssign):
            operands = (node.target, node.value)
        else:
            return False
        return any(
            self._expression_maybe_array(operand, tainted)
            for operand in operands)

    def _expression_maybe_array(
        self, node: ast.AST, tainted: frozenset[str],
    ) -> bool:
        if isinstance(node, ast.Name):
            return node.id in tainted
        if isinstance(node, ast.Call):
            return self._call_maybe_array(node, tainted)
        if isinstance(node, ast.Subscript):
            # Scalar indexing of a tainted ndarray produces a scalar.  Slices
            # and tainted/fancy indices retain array-valued native semantics.
            return self._index_maybe_array(node.slice, tainted)
        if isinstance(node, ast.Attribute):
            return node.attr == "T" \
                and self._expression_maybe_array(node.value, tainted)
        if isinstance(node, (ast.Constant, ast.List, ast.Tuple, ast.Set,
                             ast.Dict)):
            return False
        return any(
            self._expression_maybe_array(child, tainted)
            for child in ast.iter_child_nodes(node))

    def _call_maybe_array(
        self, node: ast.Call, tainted: frozenset[str],
    ) -> bool:
        canonical = _call_capability(node, self.import_capabilities)
        scalar_reductions = {
            "numpy.any", "numpy.argmax", "numpy.argmin", "numpy.max",
            "numpy.mean", "numpy.min", "numpy.std", "numpy.sum",
            "numpy.linalg.norm",
        }
        if canonical in scalar_reductions:
            has_axis = bool(node.args[1:]) or any(
                keyword.arg == "axis" for keyword in node.keywords)
            return has_axis
        if canonical is not None and canonical.startswith(
                ("numpy.", "scipy.", "skimage.")):
            return True
        if isinstance(node.func, ast.Attribute):
            attribute = node.func.attr
            if attribute in {"astype", "ravel", "reshape"}:
                return True
            if attribute in {"any", "max", "mean", "min", "std", "sum"}:
                return bool(node.args) or any(
                    keyword.arg == "axis" for keyword in node.keywords)
            return False
        return isinstance(node.func, ast.Name) \
            and node.func.id in self.defined_functions

    @staticmethod
    def _index_maybe_array(
        node: ast.AST, tainted: frozenset[str],
    ) -> bool:
        if isinstance(node, ast.Slice):
            return True
        if isinstance(node, ast.Name):
            return node.id in tainted
        if isinstance(node, ast.Constant):
            return not isinstance(node.value, int)
        if isinstance(node, ast.Tuple):
            return any(
                _NativeWorkSafety._index_maybe_array(element, tainted)
                for element in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub)):
            return _NativeWorkSafety._index_maybe_array(
                node.operand, tainted)
        return True


class _ResourceSafety:
    """Reject source-static CPU/memory bombs before predicate execution."""

    _SHAPE_CALLS = frozenset({
        "numpy.empty", "numpy.full", "numpy.ones", "numpy.zeros",
    })

    def __init__(self, import_capabilities: Mapping[str, str]) -> None:
        self.import_capabilities = import_capabilities

    def validate(self, tree: ast.Module) -> None:
        globals_ = self._constant_environment(tuple(tree.body), {})
        self._validate_nodes(tuple(tree.body), globals_, module_only=True)
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.Lambda)):
                continue
            nodes = _scope_nodes(function)
            environment = self._constant_environment(nodes, globals_)
            self._validate_nodes(nodes, environment, module_only=False)
            for node in nodes:
                if isinstance(node, ast.While) \
                        and not self._certified_worklist_while(
                            node, nodes, environment):
                    raise PredicatePricingError(
                        "while loops require the certified finite visited-"
                        "worklist pattern")

    @staticmethod
    def _constant_environment(
        nodes: Iterable[ast.AST], base: Mapping[str, int],
    ) -> dict[str, int]:
        candidates: dict[str, list[int | None]] = {}
        cap = max(MAX_STATIC_ALLOCATION_ELEMENTS,
                  MAX_STATIC_ITERATION_COUNT) + 1
        for node in nodes:
            if isinstance(node, ast.Assign):
                value = _bounded_static_int(
                    node.value, base, magnitude_cap=cap)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        candidates.setdefault(target.id, []).append(value)
            elif isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name) and node.value is not None:
                candidates.setdefault(node.target.id, []).append(
                    _bounded_static_int(node.value, base, magnitude_cap=cap))
            elif isinstance(node, ast.AugAssign) and isinstance(
                    node.target, ast.Name):
                candidates.setdefault(node.target.id, []).append(None)
            elif isinstance(node, (ast.For, ast.comprehension)):
                for name in _target_names(node.target):
                    candidates.setdefault(name, []).append(None)
        result = dict(base)
        for name, values in candidates.items():
            if values and all(value is not None and value == values[0]
                              for value in values):
                assert values[0] is not None
                result[name] = values[0]
            else:
                result.pop(name, None)
        return result

    def _validate_nodes(
        self,
        nodes: Iterable[ast.AST],
        environment: Mapping[str, int],
        *,
        module_only: bool,
    ) -> None:
        for node in nodes:
            if isinstance(node, (ast.List, ast.Tuple, ast.Set)) \
                    and len(node.elts) > MAX_STATIC_CONTAINER_ELEMENTS:
                raise PredicatePricingError(
                    "source-static container elements exceed limit "
                    f"{MAX_STATIC_CONTAINER_ELEMENTS}")
            if isinstance(node, ast.Dict) \
                    and len(node.keys) > MAX_STATIC_CONTAINER_ELEMENTS:
                raise PredicatePricingError(
                    "source-static container elements exceed limit "
                    f"{MAX_STATIC_CONTAINER_ELEMENTS}")
            if isinstance(node, ast.BinOp):
                self._validate_repeat(node, environment)
                if isinstance(node.op, (ast.Pow, ast.LShift)):
                    self._validate_large_integer_operation(node, environment)
            if not isinstance(node, ast.Call):
                continue
            canonical = _call_capability(node, self.import_capabilities)
            if canonical in self._SHAPE_CALLS:
                elements = self._static_shape_elements(
                    node.args[0], environment) if node.args else None
                if elements is None:
                    raise PredicatePricingError(
                        "explicit array allocation requires a source-static "
                        "bounded shape")
                self._reject_bound(
                    elements, MAX_STATIC_ALLOCATION_ELEMENTS,
                    "array allocation elements")
            elif canonical == "numpy.linspace" and len(node.args) >= 3:
                count = _bounded_static_int(
                    node.args[2], environment,
                    magnitude_cap=MAX_STATIC_ALLOCATION_ELEMENTS + 1)
                if count is None:
                    raise PredicatePricingError(
                        "numpy.linspace requires a source-static sample count")
                self._reject_bound(
                    count, MAX_STATIC_ALLOCATION_ELEMENTS,
                    "numpy.linspace samples")
            elif canonical == "numpy.arange" and node.args:
                stop = _bounded_static_int(
                    node.args[0], environment,
                    magnitude_cap=MAX_STATIC_ALLOCATION_ELEMENTS + 1)
                if stop is None:
                    raise PredicatePricingError(
                        "numpy.arange requires a source-static element bound")
                self._reject_bound(
                    stop, MAX_STATIC_ALLOCATION_ELEMENTS,
                    "numpy.arange elements")
            elif canonical == "numpy.pad" and len(node.args) >= 2:
                padding = self._maximum_static_dimension(
                    node.args[1], environment)
                if padding is None:
                    raise PredicatePricingError(
                        "numpy.pad requires a source-static bounded pad width")
                self._reject_bound(
                    padding, MAX_STATIC_PAD_WIDTH, "numpy.pad width")
            elif canonical == "itertools.permutations":
                count = self._static_iterable_count(
                    node.args[0], environment) if node.args else None
                if count is None:
                    raise PredicatePricingError(
                        "itertools.permutations requires a statically bounded "
                        "iterable")
                permutations = 1
                for value in range(2, count + 1):
                    permutations *= value
                    if permutations > MAX_STATIC_ITERATION_COUNT:
                        break
                self._reject_bound(
                    permutations, MAX_STATIC_ITERATION_COUNT,
                    "permutation iterations")
            elif isinstance(node.func, ast.Name) and node.func.id == "range":
                count = self._static_range_count(node, environment)
                self._reject_bound(
                    count, MAX_STATIC_ITERATION_COUNT, "range iterations")
            if canonical in {
                    "scipy.ndimage.binary_dilation",
                    "scipy.ndimage.binary_erosion",
            }:
                iterations = next(
                    (keyword.value for keyword in node.keywords
                     if keyword.arg == "iterations"), None)
                if iterations is not None:
                    value = _bounded_static_int(
                        iterations, environment,
                        magnitude_cap=MAX_STATIC_MORPHOLOGY_ITERATIONS + 1)
                    if value is None or value <= 0:
                        raise PredicatePricingError(
                            "morphology iterations require a positive "
                            "source-static bound")
                    self._reject_bound(
                        value, MAX_STATIC_MORPHOLOGY_ITERATIONS,
                        "morphology iterations")

    @staticmethod
    def _reject_bound(value: int | None, bound: int, description: str) -> None:
        if value is not None and value > bound:
            raise PredicatePricingError(
                f"source-static {description} {value} exceeds limit {bound}")

    @staticmethod
    def _static_shape_elements(
        node: ast.AST, environment: Mapping[str, int],
    ) -> int | None:
        if isinstance(node, (ast.Tuple, ast.List)):
            product = 1
            for dimension in node.elts:
                value = _bounded_static_int(
                    dimension, environment,
                    magnitude_cap=MAX_STATIC_ALLOCATION_ELEMENTS + 1)
                if value is None:
                    return None
                if value < 0:
                    return 0
                product *= value
                if product > MAX_STATIC_ALLOCATION_ELEMENTS:
                    return product
            return product
        value = _bounded_static_int(
            node, environment,
            magnitude_cap=MAX_STATIC_ALLOCATION_ELEMENTS + 1)
        return max(0, value) if value is not None else None

    @staticmethod
    def _maximum_static_dimension(
        node: ast.AST, environment: Mapping[str, int],
    ) -> int | None:
        values = node.elts if isinstance(node, (ast.Tuple, ast.List)) else (node,)
        dimensions = [
            _bounded_static_int(
                value, environment, magnitude_cap=MAX_STATIC_PAD_WIDTH + 1)
            for value in values
        ]
        if any(value is None for value in dimensions):
            return None
        return max(abs(int(value)) for value in dimensions if value is not None)

    @staticmethod
    def _static_range_count(
        call: ast.Call, environment: Mapping[str, int],
    ) -> int | None:
        if not 1 <= len(call.args) <= 3 or call.keywords:
            return None
        values = [
            _bounded_static_int(
                argument, environment,
                magnitude_cap=MAX_STATIC_ITERATION_COUNT + 1)
            for argument in call.args
        ]
        if any(value is None for value in values):
            return None
        concrete = [int(value) for value in values if value is not None]
        try:
            return len(range(*concrete))
        except (OverflowError, ValueError):
            return MAX_STATIC_ITERATION_COUNT + 1

    def _static_iterable_count(
        self, node: ast.AST, environment: Mapping[str, int],
    ) -> int | None:
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return len(node.elts)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "range":
            return self._static_range_count(node, environment)
        return None

    @staticmethod
    def _validate_repeat(
        node: ast.BinOp, environment: Mapping[str, int],
    ) -> None:
        if not isinstance(node.op, ast.Mult):
            return
        sequence: ast.AST | None = None
        count_node: ast.AST | None = None
        for possible_sequence, possible_count in (
            (node.left, node.right), (node.right, node.left),
        ):
            if isinstance(possible_sequence, (
                    ast.List, ast.Tuple, ast.Set)) or (
                    isinstance(possible_sequence, ast.Constant)
                    and isinstance(possible_sequence.value, (str, bytes))):
                sequence, count_node = possible_sequence, possible_count
                break
        if sequence is None or count_node is None:
            return
        count = _bounded_static_int(
            count_node, environment,
            magnitude_cap=MAX_STATIC_ALLOCATION_ELEMENTS + 1)
        if count is None:
            raise PredicatePricingError(
                "sequence repetition requires a source-static bounded count")
        if isinstance(sequence, (ast.List, ast.Tuple, ast.Set)):
            length = len(sequence.elts)
        else:
            assert isinstance(sequence, ast.Constant)
            length = len(sequence.value)
        _ResourceSafety._reject_bound(
            max(0, count) * length, MAX_STATIC_ALLOCATION_ELEMENTS,
            "sequence repetition elements")

    @staticmethod
    def _validate_large_integer_operation(
        node: ast.BinOp, environment: Mapping[str, int],
    ) -> None:
        if isinstance(node.op, ast.Pow):
            base = _bounded_static_int(
                node.left, environment, magnitude_cap=1_000_001)
            exponent = _bounded_static_int(
                node.right, environment, magnitude_cap=MAX_STATIC_INTEGER_BITS + 1)
            if base is not None and exponent is not None and exponent >= 0 \
                    and abs(base) > 1 \
                    and exponent * max(1, abs(base).bit_length() - 1) \
                    > MAX_STATIC_INTEGER_BITS:
                raise PredicatePricingError(
                    "source-static integer power exceeds the bit-size limit")
        elif isinstance(node.op, ast.LShift):
            shift = _bounded_static_int(
                node.right, environment, magnitude_cap=MAX_STATIC_INTEGER_BITS + 1)
            if shift is not None and shift > MAX_STATIC_INTEGER_BITS:
                raise PredicatePricingError(
                    "source-static integer shift exceeds the bit-size limit")

    def _certified_worklist_while(
        self,
        loop: ast.While,
        scope_nodes: tuple[ast.AST, ...],
        environment: Mapping[str, int],
    ) -> bool:
        if not isinstance(loop.test, ast.Name) or not loop.body:
            return False
        queue = loop.test.id
        removal = loop.body[0]
        if not (isinstance(removal, ast.Assign)
                and len(removal.targets) == 1
                and isinstance(removal.targets[0], ast.Name)
                and isinstance(removal.value, ast.Call)
                and isinstance(removal.value.func, ast.Attribute)
                and isinstance(removal.value.func.value, ast.Name)
                and removal.value.func.value.id == queue
                and removal.value.func.attr in {"pop", "popleft"}
                and not removal.value.args and not removal.value.keywords):
            return False
        popped = removal.targets[0].id
        queue_bindings = [
            node for node in scope_nodes
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == queue
                    for target in node.targets)
        ]
        if len(queue_bindings) != 1:
            return False
        queue_value = queue_bindings[0].value
        if not (isinstance(queue_value, ast.Call)
                and _call_capability(queue_value, self.import_capabilities)
                == "collections.deque" and len(queue_value.args) == 1
                and not queue_value.keywords
                and isinstance(queue_value.args[0], ast.List)
                and len(queue_value.args[0].elts) == 1
                and isinstance(queue_value.args[0].elts[0], ast.Name)):
            return False
        start = queue_value.args[0].elts[0].id

        queue_calls = [
            node for node in ast.walk(loop)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == queue
        ]
        appends = [call for call in queue_calls if call.func.attr == "append"]
        if not appends or any(
                call is not removal.value and call.func.attr != "append"
                for call in queue_calls):
            return False

        append_pairs: list[tuple[ast.Call, ast.Assign, ast.If]] = []
        for append in appends:
            pair: tuple[ast.Call, ast.Assign, ast.If] | None = None
            for condition in (
                node for node in ast.walk(loop) if isinstance(node, ast.If)):
                for index, statement in enumerate(condition.body):
                    if index == 0 or not (isinstance(statement, ast.Expr)
                                          and statement.value is append):
                        continue
                    previous = condition.body[index - 1]
                    if isinstance(previous, ast.Assign):
                        pair = (append, previous, condition)
            if pair is None:
                return False
            append_pairs.append(pair)

        seen_name: str | None = None
        pair_assignments: set[int] = set()
        for append, mark, condition in append_pairs:
            if len(append.args) != 1 or append.keywords \
                    or not isinstance(append.args[0], ast.Name) \
                    or len(mark.targets) != 1 \
                    or not isinstance(mark.targets[0], ast.Subscript) \
                    or not isinstance(mark.targets[0].value, ast.Name) \
                    or not isinstance(mark.targets[0].slice, ast.Name):
                return False
            candidate = append.args[0].id
            target = mark.targets[0]
            if target.slice.id != candidate:
                return False
            if seen_name is None:
                seen_name = target.value.id
            if target.value.id != seen_name \
                    or not self._guard_is_unvisited(
                        condition.test, seen_name, candidate):
                return False
            if not self._mark_advances(mark.value, seen_name, popped):
                return False
            pair_assignments.add(id(mark))
        if seen_name is None:
            return False

        seen_bindings = [
            node for node in scope_nodes
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == seen_name
                    for target in node.targets)
        ]
        if len(seen_bindings) != 1 \
                or not self._negative_ones_allocation(seen_bindings[0].value):
            return False
        seen_stores: list[tuple[ast.AST, ast.Subscript]] = []
        for node in scope_nodes:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = (node.target,)
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Subscript) \
                        and isinstance(target.value, ast.Name) \
                        and target.value.id == seen_name:
                    seen_stores.append((node, target))
        initial = [
            (node, target) for node, target in seen_stores
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(target.slice, ast.Name)
            and target.slice.id == start
            and isinstance(node.value, ast.Constant) and node.value.value == 0
        ]
        return len(initial) == 1 and {
            id(node) for node, _ in seen_stores
        } == pair_assignments | {id(initial[0][0])} \
            and len(seen_stores) == len(pair_assignments) + 1

    def _negative_ones_allocation(self, node: ast.AST) -> bool:
        return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) \
            and isinstance(node.operand, ast.Call) \
            and _call_capability(node.operand, self.import_capabilities) \
            == "numpy.ones"

    @staticmethod
    def _guard_is_unvisited(test: ast.AST, seen: str, candidate: str) -> bool:
        # Only a syntactic conjunct is logically necessary for entry into the
        # body.  Searching ast.walk would incorrectly certify
        # ``seen[i] == -1 or True``, whose body is unconditional.
        conjuncts = list(test.values) if isinstance(
            test, ast.BoolOp) and isinstance(test.op, ast.And) else [test]
        pending = list(conjuncts)
        while pending:
            node = pending.pop()
            if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
                pending.extend(node.values)
                continue
            if not (isinstance(node, ast.Compare) and len(node.ops) == 1
                    and isinstance(node.ops[0], ast.Eq)
                    and len(node.comparators) == 1):
                continue
            pairs = ((node.left, node.comparators[0]),
                     (node.comparators[0], node.left))
            for subscript, sentinel in pairs:
                if isinstance(subscript, ast.Subscript) \
                        and isinstance(subscript.value, ast.Name) \
                        and subscript.value.id == seen \
                        and isinstance(subscript.slice, ast.Name) \
                        and subscript.slice.id == candidate \
                        and isinstance(sentinel, ast.UnaryOp) \
                        and isinstance(sentinel.op, ast.USub) \
                        and isinstance(sentinel.operand, ast.Constant) \
                        and sentinel.operand.value == 1:
                    return True
        return False

    @staticmethod
    def _mark_advances(value: ast.AST, seen: str, popped: str) -> bool:
        if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Add):
            return False
        pairs = ((value.left, value.right), (value.right, value.left))
        return any(
            isinstance(previous, ast.Subscript)
            and isinstance(previous.value, ast.Name)
            and previous.value.id == seen
            and isinstance(previous.slice, ast.Name)
            and previous.slice.id == popped
            and isinstance(increment, ast.Constant)
            and isinstance(increment.value, int)
            and increment.value > 0
            for previous, increment in pairs
        )


def _reject_recursive_helpers(tree: ast.Module) -> None:
    """Reject direct/indirect recursion and callback-hidden recursion."""
    functions = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]
    names = [node.name for node in functions]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise PredicatePricingError(
            f"function names must be globally unambiguous: {duplicates}")
    defined = frozenset(names)
    edges: dict[str, set[str]] = {name: set() for name in names}
    for function in functions:
        for node in _scope_nodes(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in defined:
                edges[function.name].add(node.func.id)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise PredicatePricingError(
                "recursive helper calls are outside the bounded predicate subset")
        if name in visited:
            return
        visiting.add(name)
        for dependency in edges[name]:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for name in names:
        visit(name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda) and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in defined
                for child in ast.walk(node.body)):
            raise PredicatePricingError(
                "callbacks cannot invoke predicate helpers recursively")


@dataclass(frozen=True, slots=True)
class _Scope:
    local_names: frozenset[str]
    global_names: frozenset[str]
    is_class: bool = False


class _ModuleLoadCollector(ast.NodeVisitor):
    """Find names whose runtime binding comes from module scope.

    This is deliberately scope-aware: a local variable called ``LIMIT`` does
    not accidentally create a dependency on a module constant of that name.
    """

    def __init__(self) -> None:
        self.names: set[str] = set()
        self._scopes: list[_Scope] = []

    @staticmethod
    def _function_scope(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> _Scope:
        collector = _LocalBindingCollector()
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            collector.visit(argument)
        if node.args.vararg is not None:
            collector.visit(node.args.vararg)
        if node.args.kwarg is not None:
            collector.visit(node.args.kwarg)
        body = node.body if not isinstance(node, ast.Lambda) else (node.body,)
        for child in body:
            collector.visit(child)
        locals_ = collector.bound - collector.globals - collector.nonlocals
        return _Scope(frozenset(locals_), frozenset(collector.globals))

    @staticmethod
    def _visit_arguments(visitor: "_ModuleLoadCollector", args: ast.arguments) -> None:
        for argument in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            if argument.annotation is not None:
                visitor.visit(argument.annotation)
        if args.vararg is not None and args.vararg.annotation is not None:
            visitor.visit(args.vararg.annotation)
        if args.kwarg is not None and args.kwarg.annotation is not None:
            visitor.visit(args.kwarg.annotation)
        for default in (*args.defaults, *args.kw_defaults):
            if default is not None:
                visitor.visit(default)

    def _is_module_load(self, name: str) -> bool:
        for index, scope in enumerate(reversed(self._scopes)):
            if name in scope.global_names:
                return True
            # A class namespace is visible while its body executes, but it is
            # not an enclosing lexical scope for a nested method or lambda.
            if scope.is_class and index:
                continue
            if name in scope.local_names:
                return False
        return True

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load) and self._is_module_load(node.id):
            self.names.add(node.id)

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_arguments(self, node.args)
        if node.returns is not None:
            self.visit(node.returns)
        self._scopes.append(self._function_scope(node))
        try:
            for child in node.body:
                self.visit(child)
        finally:
            self._scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        self._visit_arguments(self, node.args)
        self._scopes.append(self._function_scope(node))
        try:
            self.visit(node.body)
        finally:
            self._scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        local_names: set[str] = set()
        for child in node.body:
            local_names.update(_defined_names(child))
        self._scopes.append(_Scope(frozenset(local_names), frozenset(), True))
        try:
            for child in node.body:
                self.visit(child)
        finally:
            self._scopes.pop()

    def _visit_comprehension(
        self,
        value_nodes: tuple[ast.AST, ...],
        generators: list[ast.comprehension],
    ) -> None:
        # The first iterator is evaluated outside the comprehension scope.
        self.visit(generators[0].iter)
        bound: set[str] = set()
        for generator in generators:
            bound.update(_target_names(generator.target))
        self._scopes.append(_Scope(frozenset(bound), frozenset()))
        try:
            for index, generator in enumerate(generators):
                if index:
                    self.visit(generator.iter)
                for condition in generator.ifs:
                    self.visit(condition)
            for value in value_nodes:
                self.visit(value)
        finally:
            self._scopes.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:  # noqa: N802
        self._visit_comprehension((node.elt,), node.generators)

    def visit_SetComp(self, node: ast.SetComp) -> None:  # noqa: N802
        self._visit_comprehension((node.elt,), node.generators)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:  # noqa: N802
        self._visit_comprehension((node.elt,), node.generators)

    def visit_DictComp(self, node: ast.DictComp) -> None:  # noqa: N802
        self._visit_comprehension((node.key, node.value), node.generators)


@dataclass(frozen=True, slots=True)
class DefinitionNode:
    """One priced top-level source statement."""

    key: str
    kind: str
    defined_names: tuple[str, ...]
    identity: str
    cost: int
    dependencies: tuple[str, ...]
    lineno: int


@dataclass(frozen=True, slots=True)
class DefinitionPrice:
    """Immutable receipt for one rule's union-of-definitions price."""

    predicate_names: tuple[str, ...]
    used_nodes: tuple[DefinitionNode, ...]
    charged_nodes: tuple[DefinitionNode, ...]
    reused_nodes: tuple[DefinitionNode, ...]
    promoted_node_identities: frozenset[str]
    full_cost: int
    charged_cost: int

    @property
    def used_node_identities(self) -> frozenset[str]:
        return frozenset(node.identity for node in self.used_nodes)

    @property
    def charged_node_identities(self) -> frozenset[str]:
        return frozenset(node.identity for node in self.charged_nodes)


@dataclass(frozen=True, slots=True)
class PredicatePricingModel:
    """Immutable dependency graph and costs for one exact module source."""

    source_digest: str
    nodes: tuple[DefinitionNode, ...]
    predicate_names: tuple[str, ...]

    def __post_init__(self) -> None:
        keys = [node.key for node in self.nodes]
        identities = [node.identity for node in self.nodes]
        names = [name for node in self.nodes for name in node.defined_names]
        if len(keys) != len(set(keys)):
            raise PredicatePricingError("pricing model has duplicate node keys")
        if len(identities) != len(set(identities)):
            raise PredicatePricingError("pricing model has duplicate node identities")
        if len(names) != len(set(names)):
            raise PredicatePricingError("pricing model has duplicate module symbols")
        if any(not node.key or not node.identity or node.cost <= 0
               for node in self.nodes):
            raise PredicatePricingError("every definition node needs an identity and cost")
        unknown_dependencies = sorted({
            dependency
            for node in self.nodes
            for dependency in node.dependencies
            if dependency not in set(keys)
        })
        if unknown_dependencies:
            raise PredicatePricingError(
                "definition dependencies have no registered cost: "
                f"{unknown_dependencies}"
            )
        if len(self.predicate_names) != len(set(self.predicate_names)):
            raise PredicatePricingError("pricing model has duplicate predicate names")
        predicates = {
            node.defined_names[0]
            for node in self.nodes
            if node.kind == "function"
            and len(node.defined_names) == 1
            and node.defined_names[0].startswith("p_")
        }
        missing = sorted(set(self.predicate_names) - predicates)
        if missing:
            raise PredicatePricingError(
                f"predicate names have no registered function cost: {missing}"
            )

    @property
    def node_identities(self) -> frozenset[str]:
        """All identities in this source, suitable for a promoted baseline."""
        return frozenset(node.identity for node in self.nodes)

    def _predicate_request(self, predicate_names: Iterable[str]) -> tuple[str, ...]:
        if isinstance(predicate_names, (str, bytes)):
            raise PredicatePricingError(
                "predicate_names must be an iterable of names, not one string"
            )
        requested: list[str] = []
        seen: set[str] = set()
        for name in predicate_names:
            if not isinstance(name, str) or not name:
                raise PredicatePricingError("predicate names must be nonempty strings")
            if name not in seen:
                requested.append(name)
                seen.add(name)
        if not requested:
            raise PredicatePricingError("at least one predicate name is required")
        unknown = sorted(set(requested) - set(self.predicate_names))
        if unknown:
            raise UnknownPredicateError(
                f"no priced module-level p_* function(s): {unknown}"
            )
        return tuple(requested)

    def definitions_for(
        self, predicate_names: Iterable[str]
    ) -> tuple[DefinitionNode, ...]:
        """Return the source-ordered transitive union for a rule."""
        requested = self._predicate_request(predicate_names)
        node_by_key = {node.key: node for node in self.nodes}
        symbol_to_key = {
            name: node.key for node in self.nodes for name in node.defined_names
        }
        pending = [symbol_to_key[name] for name in requested]
        reached: set[str] = set()
        while pending:
            key = pending.pop()
            if key in reached:
                continue
            node = node_by_key.get(key)
            if node is None:
                raise PredicatePricingError(
                    f"definition dependency {key!r} has no registered cost"
                )
            reached.add(key)
            pending.extend(node.dependencies)
        return tuple(node for node in self.nodes if node.key in reached)

    def identities_for(self, predicate_names: Iterable[str]) -> frozenset[str]:
        return frozenset(
            node.identity for node in self.definitions_for(predicate_names)
        )

    def price(
        self,
        predicate_names: Iterable[str],
        *,
        promoted_node_identities: Iterable[str] = (),
    ) -> DefinitionPrice:
        """Price a rule after discounting exact promoted node identities."""
        if isinstance(promoted_node_identities, (str, bytes)):
            raise PredicatePricingError(
                "promoted_node_identities must be an iterable, not one string"
            )
        promoted_values: set[str] = set()
        for identity in promoted_node_identities:
            if not isinstance(identity, str) or not identity:
                raise PredicatePricingError(
                    "promoted node identities must be nonempty strings"
                )
            promoted_values.add(identity)
        promoted = frozenset(promoted_values)
        requested = self._predicate_request(predicate_names)
        used = self.definitions_for(requested)
        charged = tuple(node for node in used if node.identity not in promoted)
        reused = tuple(node for node in used if node.identity in promoted)
        return DefinitionPrice(
            predicate_names=requested,
            used_nodes=used,
            charged_nodes=charged,
            reused_nodes=reused,
            promoted_node_identities=promoted,
            full_cost=sum(node.cost for node in used),
            charged_cost=sum(node.cost for node in charged),
        )

    def price_no_share(self, predicate_names: Iterable[str]) -> DefinitionPrice:
        """Price a rule with full repayment, independent of prior rules."""
        return self.price(predicate_names, promoted_node_identities=())


def build_pricing_model(
    source: str, *, filename: str = "<predicates>"
) -> PredicatePricingModel:
    """Parse ``source`` and build its immutable predicate pricing model."""
    _validate_predicate_source_text(source)
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise PredicatePricingError(
            f"cannot price syntactically invalid source {filename!r}: {exc.msg}"
        ) from exc
    import_capabilities, import_statement_ids = _import_capability_map(tree)
    protected_module_names = frozenset(SAFE_BUILTIN_NAMES) \
        | frozenset(import_capabilities)
    for statement in tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        shadowed = sorted(set(_defined_names(statement))
                          & protected_module_names)
        if shadowed:
            raise PredicatePricingError(
                "module definitions cannot shadow certified builtin/import "
                f"names: {shadowed}")
    defined_functions = frozenset(
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    )
    _PredicateSourceSafety(
        import_capabilities,
        import_statement_ids,
        defined_functions,
    ).visit(tree)
    # Preserve import-time diagnostics before deeper body-purity checks.  A
    # default expression executes even if the affected predicate is never
    # selected.
    for statement in ast.walk(tree):
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_function_import_time(statement)
        elif isinstance(statement, ast.Lambda):
            _check_lambda_definition_time(statement)
    reserved_capabilities = frozenset(SAFE_BUILTIN_NAMES) \
        | frozenset(import_capabilities)
    _CallableShadowSafety(
        reserved_capabilities | defined_functions,
        reserved_capabilities,
        frozenset(import_capabilities),
    ).validate(tree)
    _reject_recursive_helpers(tree)
    _MutationAndAllocationSafety(import_capabilities).validate(tree)
    _UnorderedContainerSafety().validate(tree)
    _NativeWorkSafety(import_capabilities, defined_functions).validate(tree)
    _ResourceSafety(import_capabilities).validate(tree)

    statements: list[tuple[ast.stmt, tuple[str, ...], str, str, int]] = []
    symbol_owner: dict[str, str] = {}
    for statement in tree.body:
        names = _defined_names(statement)
        if not names:
            if isinstance(statement, ast.Expr) \
                    and isinstance(statement.value, ast.Constant) \
                    and isinstance(statement.value.value, str):
                continue  # module docstring: descriptive, not executable logic
            if isinstance(statement, ast.Pass):
                continue
            raise PredicatePricingError(
                f"top-level {type(statement).__name__} at line "
                f"{statement.lineno} has no statically priced definition")
        if isinstance(statement, ast.Assign):
            _TopLevelBindingSafety().visit(statement.value)
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            _TopLevelBindingSafety().visit(statement.value)
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_function_import_time(statement)
        kind = _statement_kind(statement)
        key = f"{kind}:{','.join(names)}"
        for name in names:
            previous = symbol_owner.get(name)
            if previous is not None:
                raise PredicatePricingError(
                    f"module symbol {name!r} has multiple priced definitions "
                    f"({previous!r}, {key!r})"
                )
            symbol_owner[name] = key
        text = _statement_source(source, statement)
        cost = _non_comment_loc(text) + _large_literal_cost(statement) \
            + _ast_structure_cost(statement)
        if cost <= 0:  # Every supported defining statement has at least one LOC.
            raise PredicatePricingError(f"definition {key!r} has no price")
        statements.append((statement, names, kind, text, cost))

    nodes: list[DefinitionNode] = []
    identities: set[str] = set()
    implicit_globals = SAFE_BUILTIN_NAMES
    for statement, names, kind, text, cost in statements:
        key = symbol_owner[names[0]]
        collector = _ModuleLoadCollector()
        collector.visit(statement)
        unresolved = sorted(
            collector.names - set(symbol_owner) - implicit_globals
        )
        if unresolved:
            raise PredicatePricingError(
                f"definition {key!r} uses module names with no priced "
                f"definition: {unresolved}"
            )
        dependency_keys = tuple(
            dict.fromkeys(
                symbol_owner[name]
                for name in sorted(collector.names)
                if name in symbol_owner and symbol_owner[name] != key
            )
        )
        identity = _digest(f"bongard-predicate-node/v2\0{key}\0{text}")
        if identity in identities:
            raise PredicatePricingError(f"definition identity collision for {key!r}")
        identities.add(identity)
        nodes.append(
            DefinitionNode(
                key=key,
                kind=kind,
                defined_names=names,
                identity=identity,
                cost=cost,
                dependencies=dependency_keys,
                lineno=statement.lineno,
            )
        )

    predicates = tuple(
        node.defined_names[0]
        for node in nodes
        if node.kind == "function"
        and len(node.defined_names) == 1
        and node.defined_names[0].startswith("p_")
    )
    return PredicatePricingModel(
        source_digest=_digest(source),
        nodes=tuple(nodes),
        predicate_names=predicates,
    )


# A concise synonym for callers that naturally think in parsing terms.
parse_predicate_module = build_pricing_model


__all__ = [
    "ALLOWED_FROM_IMPORTS",
    "ALLOWED_IMPORT_MODULES",
    "ALLOWED_IMPORT_ROOTS",
    "ALLOWED_INSTANCE_ATTRIBUTES",
    "ALLOWED_INSTANCE_METHODS",
    "ALLOWED_PURE_CALLS",
    "DefinitionNode",
    "DefinitionPrice",
    "MAX_EXPANDED_HEAVY_NATIVE_CALLS_PER_PANEL",
    "MAX_EXPANDED_NATIVE_CALLS_PER_PANEL",
    "MAX_PREDICATE_FUNCTIONS",
    "MAX_STATIC_ALLOCATION_ELEMENTS",
    "MAX_STATIC_CONTAINER_ELEMENTS",
    "MAX_STATIC_INTEGER_BITS",
    "MAX_STATIC_ITERATION_COUNT",
    "MAX_STATIC_MORPHOLOGY_ITERATIONS",
    "MAX_STATIC_PAD_WIDTH",
    "MAX_SOURCE_CHARACTERS",
    "MAX_SOURCE_UTF8_BYTES",
    "PREDICATE_PRICING_POLICY_ID",
    "PREDICATE_PURITY_POLICY_ID",
    "PredicatePricingError",
    "PredicatePricingModel",
    "SAFE_BUILTIN_NAMES",
    "UnknownPredicateError",
    "build_pricing_model",
    "parse_predicate_module",
    "predicate_capability_manifest",
    "predicate_execution_builtins",
    "read_predicate_source",
]
