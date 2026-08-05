"""Tests for exact AST-based predicate definition pricing."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import predicate_pricing as P


SHARED_SOURCE = """\
import math as m

WEIGHTS = [1, 2, 3, 4]

def helper(panel):
    return m.fsum(WEIGHTS) + panel.sum()

def p_low(panel):
    return helper(panel) < 20

def p_high(panel):
    return helper(panel) > 20
"""


def _keys(nodes):
    return {node.key for node in nodes}


def test_transitive_helper_constant_and_import_are_shared_once():
    model = P.build_pricing_model(SHARED_SOURCE)
    low = model.definitions_for(["p_low"])
    both = model.definitions_for(["p_low", "p_high"])

    assert _keys(low) == {
        "import:m",
        "binding:WEIGHTS",
        "function:helper",
        "function:p_low",
    }
    assert _keys(both) == _keys(low) | {"function:p_high"}
    # The shared import/constant/helper are unioned, not charged per predicate.
    assert sum(node.cost for node in both) == (
        sum(node.cost for node in low)
        + next(node.cost for node in both if node.key == "function:p_high")
    )


def test_container_literal_cardinality_is_part_of_node_cost():
    small = P.build_pricing_model(
        "TABLE = [1]\ndef p_a(panel):\n    return TABLE[0]\n"
    )
    large = P.build_pricing_model(
        "TABLE = [1, 2, 3, 4, 5]\ndef p_a(panel):\n    return TABLE[0]\n"
    )
    small_table = next(node for node in small.nodes if node.key == "binding:TABLE")
    large_table = next(node for node in large.nodes if node.key == "binding:TABLE")
    assert large_table.cost - small_table.cost == 4
    assert large.price_no_share(["p_a"]).full_cost \
        > small.price_no_share(["p_a"]).full_cost


def test_dense_string_bytes_and_integer_payloads_are_not_one_line_loopholes():
    compact = P.build_pricing_model(
        "TOKEN = 'ordinary'\ndef p_a(panel):\n    return TOKEN == 'ordinary'\n"
    )
    long_string = P.build_pricing_model(
        f"TOKEN = {'x' * 512!r}\ndef p_a(panel):\n    return bool(TOKEN)\n"
    )
    long_bytes = P.build_pricing_model(
        f"TOKEN = {bytes(range(256))!r}\ndef p_a(panel):\n    return bool(TOKEN)\n"
    )
    giant_integer = P.build_pricing_model(
        f"TOKEN = {2 ** 4096}\ndef p_a(panel):\n    return bool(TOKEN)\n"
    )
    compact_cost = compact.price_no_share(["p_a"]).full_cost
    assert long_string.price_no_share(["p_a"]).full_cost > compact_cost + 20
    assert long_bytes.price_no_share(["p_a"]).full_cost > compact_cost + 10
    assert giant_integer.price_no_share(["p_a"]).full_cost > compact_cost + 20


def test_call_constructed_lookup_payload_is_charged_per_entry():
    small = P.build_pricing_model(
        "def p_table(panel):\n"
        "    return dict(k0=0)['k0']\n"
    )
    entries = ", ".join(f"k{index}={index % 2}" for index in range(100))
    large = P.build_pricing_model(
        "def p_table(panel):\n"
        f"    return dict({entries})['k0']\n"
    )
    small_cost = small.price_no_share(["p_table"]).full_cost
    large_cost = large.price_no_share(["p_table"]).full_cost
    assert large_cost - small_cost == 99


def test_changed_helper_identity_invalidates_baseline_promotion():
    before_source = """\
def helper(panel):
    return panel.sum()

def p_a(panel):
    return helper(panel) > 1
"""
    after_source = before_source.replace("panel.sum()", "panel.mean()")
    before = P.build_pricing_model(before_source)
    after = P.build_pricing_model(after_source)

    old_helper = next(node for node in before.nodes if node.key == "function:helper")
    new_helper = next(node for node in after.nodes if node.key == "function:helper")
    old_predicate = next(node for node in before.nodes if node.key == "function:p_a")
    new_predicate = next(node for node in after.nodes if node.key == "function:p_a")
    assert old_helper.identity != new_helper.identity
    assert old_predicate.identity == new_predicate.identity

    receipt = after.price(
        ["p_a"], promoted_node_identities=before.identities_for(["p_a"])
    )
    assert _keys(receipt.charged_nodes) == {"function:helper"}
    assert _keys(receipt.reused_nodes) == {"function:p_a"}
    assert receipt.charged_cost == new_helper.cost


def test_shared_first_use_differs_exactly_from_no_share_repayment():
    baseline_source = SHARED_SOURCE.replace(
        "\ndef p_high(panel):\n    return helper(panel) > 20\n", "\n"
    )
    baseline = P.build_pricing_model(baseline_source)
    current = P.build_pricing_model(SHARED_SOURCE)
    promoted = baseline.identities_for(["p_low"])

    shared = current.price(["p_high"], promoted_node_identities=promoted)
    no_share = current.price_no_share(["p_high"])
    assert _keys(shared.charged_nodes) == {"function:p_high"}
    assert no_share.charged_cost == no_share.full_cost
    assert shared.full_cost == no_share.full_cost
    assert no_share.charged_cost - shared.charged_cost == sum(
        node.cost for node in shared.reused_nodes
    )


def test_repeated_predicate_names_and_helpers_charge_once_per_rule():
    model = P.build_pricing_model(SHARED_SOURCE)
    once = model.price_no_share(["p_low", "p_high"])
    repeated = model.price_no_share(
        ["p_low", "p_low", "p_high", "p_low", "p_high"]
    )
    assert repeated.predicate_names == ("p_low", "p_high")
    assert repeated.used_nodes == once.used_nodes
    assert repeated.charged_cost == once.charged_cost
    assert len([node for node in repeated.used_nodes
                if node.key == "function:helper"]) == 1


def test_local_shadow_does_not_create_false_module_dependency():
    source = """\
LIMIT = [1, 2, 3]

def p_a(panel):
    LIMIT = panel.sum()
    return LIMIT > 0
"""
    model = P.build_pricing_model(source)
    assert _keys(model.definitions_for(["p_a"])) == {"function:p_a"}


def test_missing_predicate_and_bad_requests_are_rejected():
    model = P.build_pricing_model(SHARED_SOURCE)
    with pytest.raises(P.UnknownPredicateError):
        model.price_no_share(["p_missing"])
    with pytest.raises(P.UnknownPredicateError):
        model.price_no_share(["helper"])
    with pytest.raises(P.PredicatePricingError):
        model.price_no_share([])
    with pytest.raises(P.PredicatePricingError):
        model.price_no_share("p_low")
    with pytest.raises(P.PredicatePricingError, match="no priced definition"):
        P.build_pricing_model(
            "def p_unpriced(panel):\n    return missing_helper(panel)\n"
        )


def test_malformed_model_cannot_hide_a_missing_predicate_cost():
    with pytest.raises(P.PredicatePricingError, match="no registered function cost"):
        P.PredicatePricingModel(
            source_digest="0" * 64,
            nodes=(),
            predicate_names=("p_missing",),
        )


def test_syntax_errors_and_star_imports_are_rejected():
    with pytest.raises(P.PredicatePricingError):
        P.build_pricing_model("def p_a(:\n    pass\n", filename="bad.py")
    with pytest.raises(P.PredicatePricingError, match="star imports"):
        P.build_pricing_model("from somewhere import *\n")


def test_unpriced_top_level_execution_and_dynamic_namespace_access_are_rejected():
    with pytest.raises(P.PredicatePricingError, match="no statically priced"):
        P.build_pricing_model(
            "STATE = []\n"
            "STATE.append(1)\n"
            "def p_a(panel):\n    return len(STATE)\n"
        )
    with pytest.raises(P.PredicatePricingError, match="dynamic or I/O"):
        P.build_pricing_model(
            "def p_a(panel):\n    return 0.0\n"
            "globals()['p_a'] = lambda panel: float(panel.sum())\n"
        )
    with pytest.raises(P.PredicatePricingError, match="dynamic or I/O name"):
        P.build_pricing_model(
            "EXEC = exec\n"
            "def p_a(panel):\n    return 0.0\n"
        )
    with pytest.raises(P.PredicatePricingError, match="dynamic or I/O name"):
        P.build_pricing_model(
            "SIDE_EFFECT = __builtins__['exec']('pass')\n"
            "def p_a(panel):\n    return 0.0\n"
        )
    with pytest.raises(P.PredicatePricingError, match="cannot call functions"):
        P.build_pricing_model(
            "SIDE_EFFECT = sum([1, 2, 3])\n"
            "def p_a(panel):\n    return 0.0\n"
        )


def test_import_time_defaults_decorators_and_classes_are_rejected():
    with pytest.raises(P.PredicatePricingError, match="cannot call functions"):
        P.build_pricing_model(
            "TABLE = {}\n"
            "def configure():\n"
            "    TABLE['answer'] = 7\n"
            "def p_bootstrap(panel, configured=configure()):\n"
            "    return 0\n"
            "def p_answer(panel):\n"
            "    return TABLE['answer']\n"
        )
    with pytest.raises(P.PredicatePricingError, match="cannot use decorators"):
        P.build_pricing_model(
            "def identity(fn):\n"
            "    return fn\n"
            "@identity\n"
            "def p_a(panel):\n"
            "    return 0.0\n"
        )
    with pytest.raises(P.PredicatePricingError, match="class definitions"):
        P.build_pricing_model(
            "TABLE = [1, 2, 3, 4, 5]\n"
            "class Lookup:\n"
            "    selected = TABLE\n"
            "    TABLE = ()\n"
            "def p_a(panel):\n"
            "    return Lookup.selected[0]\n"
        )


@pytest.mark.parametrize("source", [
    (
        "from operator import attrgetter as fetch\n"
        "import numpy as np\n"
        "def p_a(panel):\n"
        "    return fetch('load')(np)('answers.npy')[0]\n"
    ),
    (
        "import operator\n"
        "def p_a(panel):\n"
        "    return operator.methodcaller('read')('answers.npy')\n"
    ),
])
def test_dynamic_operator_attribute_helpers_cannot_bypass_io_guard(source):
    with pytest.raises(P.PredicatePricingError, match="dangerous|file-I/O"):
        P.build_pricing_model(source)


def test_process_randomized_builtin_hash_is_forbidden():
    with pytest.raises(P.PredicatePricingError, match="dynamic or I/O.*'hash'"):
        P.build_pricing_model(
            "def p_unstable(panel):\n"
            "    return hash(panel.tobytes())\n"
        )


def test_forbidden_imports_and_file_io_cannot_bypass_source_boundary():
    with pytest.raises(P.PredicatePricingError, match="unsupported module"):
        P.build_pricing_model(
            "import os\ndef p_a(panel):\n    return os.getcwd()\n")
    with pytest.raises(P.PredicatePricingError, match="file-I/O attribute"):
        P.build_pricing_model(
            "import numpy as np\n"
            "def p_a(panel):\n    return np.load('answers.npy')[0]\n")
    with pytest.raises(P.PredicatePricingError, match="dynamic or file-I/O"):
        P.build_pricing_model(
            "import numpy as np\n"
            "def p_a(panel):\n    return np.__builtins__['open']('x')\n")
    with pytest.raises(P.PredicatePricingError, match="dangerous name"):
        P.build_pricing_model(
            "from numpy import load as read_answer\n"
            "def p_a(panel):\n    return read_answer('answers.npy')[0]\n")
    with pytest.raises(P.PredicatePricingError, match="dangerous module path"):
        P.build_pricing_model(
            "import numpy.ctypeslib as bridge\n"
            "def p_a(panel):\n    return bridge.load_library('x', '.')\n")
    with pytest.raises(P.PredicatePricingError, match="dynamic or file-I/O"):
        P.build_pricing_model(
            "import numpy as np\n"
            "def p_a(panel):\n    return np.lib._datasource.open('answers.npy')\n")
    with pytest.raises(P.PredicatePricingError, match="dynamic or file-I/O"):
        P.build_pricing_model(
            "import math\n"
            "def p_a(panel):\n    return math.__loader__.load_module('os')\n")


def test_model_and_receipts_are_immutable():
    model = P.build_pricing_model(SHARED_SOURCE)
    receipt = model.price_no_share(["p_low"])
    with pytest.raises((AttributeError, TypeError)):
        model.source_digest = "changed"
    with pytest.raises((AttributeError, TypeError)):
        receipt.charged_cost = 0


def test_capability_manifest_is_complete_deterministic_and_detached():
    first = P.predicate_capability_manifest()
    second = P.predicate_capability_manifest()
    assert first == second
    assert P.PREDICATE_PURITY_POLICY_ID == "bongard-predicate-purity/v2"
    assert P.PREDICATE_PRICING_POLICY_ID == "bongard-predicate-pricing/v3"
    assert first["policy_id"] == P.PREDICATE_PURITY_POLICY_ID
    assert first["pricing_policy_id"] == P.PREDICATE_PRICING_POLICY_ID
    assert json.loads(json.dumps(first))["policy_id"] \
        == P.PREDICATE_PURITY_POLICY_ID
    assert "numpy.random.default_rng" not in {
        row[0] for row in first["module_calls"]
    }
    assert first["resources"]["native_operator_policy"].startswith(
        "tainted-array-operators-counted")
    assert "numpy.linalg.eigvalsh" in first["resources"][
        "heavy_native_module_calls"]
    first["imports"]["modules"] = ()
    assert P.predicate_capability_manifest() == second


@pytest.mark.parametrize("source", [
    "from skimage import data\ndef p_a(panel):\n    return 0.0\n",
    "from skimage.data import download_all\n"
    "def p_a(panel):\n    return download_all()\n",
    "import scipy\ndef p_a(panel):\n    return 0.0\n",
    "import numpy as np\n"
    "def p_a(panel):\n    return np.random.default_rng().random()\n",
    "import numpy as np\n"
    "def p_a(panel):\n    np.seterr(all='ignore')\n    return 0.0\n",
    "import numpy as np\n"
    "def p_a(panel):\n    np.set_printoptions(threshold=1)\n    return 0.0\n",
])
def test_positive_import_and_module_api_gate_rejects_unlisted_capabilities(source):
    with pytest.raises(P.PredicatePricingError):
        P.build_pricing_model(source)


@pytest.mark.parametrize("builtin_name", [
    "help", "exit", "quit", "dir", "type", "object",
])
def test_ambient_or_introspective_builtins_are_not_predicate_capabilities(
        builtin_name):
    with pytest.raises(P.PredicatePricingError, match="not statically certified"):
        P.build_pricing_model(
            f"def p_a(panel):\n    return {builtin_name}(panel)\n"
        )


@pytest.mark.parametrize("body", [
    "panel[0, 0] = 0",
    "alias = np.asarray(panel)\nalias[0, 0] = 0",
    "alias = np.rot90(panel)\nalias[0, 0] = 0",
])
def test_panel_and_view_mutation_are_rejected(body):
    source = "import numpy as np\n\ndef p_a(panel):\n" + "\n".join(
        f"    {line}" for line in body.splitlines()
    ) + "\n    return float(panel.sum())\n"
    with pytest.raises(P.PredicatePricingError, match="locally owned"):
        P.build_pricing_model(source)


def test_module_state_and_forwarded_parameter_mutation_are_rejected():
    with pytest.raises(P.PredicatePricingError, match="locally owned"):
        P.build_pricing_model(
            "CACHE = {}\n"
            "def p_a(panel):\n"
            "    CACHE[0] = float(panel.sum())\n"
            "    return CACHE[0]\n"
        )
    with pytest.raises(P.PredicatePricingError, match="locally owned"):
        P.build_pricing_model(
            "def mutate(value):\n"
            "    value[0, 0] = 0\n"
            "def p_a(panel):\n"
            "    mutate(panel)\n"
            "    return float(panel.sum())\n"
        )


def test_locally_allocated_scratch_mutation_remains_supported():
    model = P.build_pricing_model(
        "import numpy as np\n"
        "def p_a(panel):\n"
        "    values = []\n"
        "    values.append(float(panel.sum()))\n"
        "    scratch = np.zeros(2)\n"
        "    scratch[0] = values[0]\n"
        "    scratch[1] = 1.0\n"
        "    return float(scratch.sum())\n"
    )
    assert model.predicate_names == ("p_a",)


def test_numpy_empty_requires_the_exact_full_initialization_certificate():
    with pytest.raises(P.PredicatePricingError, match="numpy.empty"):
        P.build_pricing_model(
            "import numpy as np\n"
            "def p_a(panel):\n"
            "    return float(np.empty(1).sum())\n"
        )
    with pytest.raises(P.PredicatePricingError, match="numpy.empty"):
        P.build_pricing_model(
            "import numpy as np\n"
            "def p_a(panel):\n"
            "    pts = np.nonzero(panel)[0]\n"
            "    counts = np.empty(len(pts), dtype=int)\n"
            "    counts[0] = 1\n"
            "    return float(counts.sum())\n"
        )
    with pytest.raises(P.PredicatePricingError, match="source-static"):
        P.build_pricing_model(
            "import numpy as np\n"
            "def p_a(panel):\n"
            "    pts = np.nonzero(panel)[0]\n"
            "    counts = np.empty(len(pts), dtype=int)\n"
            "    for i, point in enumerate(pts):\n"
            "        counts[i] = int(point)\n"
            "    return float(counts.sum())\n"
        )
@pytest.mark.parametrize("body", [
    "for value in {'alpha', 'beta'}:\n        return float(len(value))",
    "values = {'alpha', 'beta'}\nreturn float(len(list(values)))",
    "values = {'alpha', 'beta'}\nfirst, second = values\n"
    "return float(len(first))",
    "values = {'alpha', 'beta'}\nreturn float(len(values.pop()))",
])
def test_hash_order_observations_are_rejected(body):
    source = "def p_a(panel):\n" + "\n".join(
        f"    {line}" for line in body.splitlines()
    ) + "\n    return 0.0\n"
    with pytest.raises(P.PredicatePricingError, match="unordered|set.pop"):
        P.build_pricing_model(source)


def test_order_independent_local_set_use_remains_supported():
    model = P.build_pricing_model(
        "def p_a(panel):\n"
        "    labels = set(range(3))\n"
        "    labels.discard(2)\n"
        "    return float(len(labels) + (1 in labels))\n"
    )
    assert model.predicate_names == ("p_a",)


@pytest.mark.parametrize("source", [
    "import numpy as np\ndef p_a(panel):\n"
    "    return float(np.zeros((10000000000,)).sum())\n",
    "import numpy as np\ndef p_a(panel):\n"
    "    size = 10000000000\n"
    "    return float(np.ones(size).sum())\n",
    "def p_a(panel):\n"
    "    return float(sum(1 for _ in range(10000000000)))\n",
    "def p_a(panel):\n"
    "    values = [0] * 10000000000\n"
    "    return float(len(values))\n",
    "import itertools\ndef p_a(panel):\n"
    "    return float(sum(1 for _ in itertools.permutations(range(20))))\n",
    "def p_a(panel):\n    while True:\n        pass\n    return 0.0\n",
    "import numpy as np\ndef p_a(panel):\n"
    "    return float(np.pad(panel, 100000).sum())\n",
    "from scipy.ndimage import binary_dilation\n"
    "def p_a(panel):\n"
    "    return float(binary_dilation(panel, iterations=100000).sum())\n",
    "def p_a(panel):\n    return float(int(panel.sum()) ** 100)\n",
])
def test_source_static_resource_bombs_and_unbounded_loops_are_rejected(source):
    with pytest.raises(P.PredicatePricingError):
        P.build_pricing_model(source)


@pytest.mark.parametrize("source", [
    "import numpy as np\n"
    "def p_a(panel):\n"
    "    matrix = np.ones((1000, 1000))\n"
    "    for _ in range(100):\n"
    "        np.linalg.eigvalsh(matrix)\n"
    "    return 0.0\n",
    "import numpy as np\n"
    "def p_a(panel):\n"
    "    matrix = np.ones((128, 128))\n"
    "    for _ in range(100):\n"
    "        np.linalg.eigvalsh(matrix)\n"
    "    return 0.0\n",
    "import numpy as np\n"
    "def helper(matrix):\n"
    "    return float(np.linalg.eigvalsh(matrix).sum())\n"
    "def p_a(panel):\n"
    "    matrix = np.ones((128, 128))\n"
    "    total = 0.0\n"
    "    for _ in range(100):\n"
    "        total += helper(matrix)\n"
    "    return total\n",
    "import numpy as np\n"
    "def p_a(panel):\n"
    "    n = int(len(panel)) * 100000\n"
    "    return float(np.ones(n).sum())\n",
    "import numpy as np\n"
    "def p_a(panel):\n"
    "    matrix = np.ones((128, 128))\n"
    "    values = sorted(range(100), "
    "key=lambda _: np.linalg.eigvalsh(matrix).sum())\n"
    "    return float(len(values))\n",
    "def p_a(panel):\n"
    "    accumulator = panel\n"
    "    for _ in range(1000000):\n"
    "        accumulator = accumulator + panel\n"
    "    return float(accumulator.sum())\n",
])
def test_native_work_and_dynamic_allocation_bombs_are_rejected(source):
    with pytest.raises(P.PredicatePricingError):
        P.build_pricing_model(source)


def test_expanded_heavy_native_call_budget_counts_all_predicates():
    source = "import numpy as np\n" + "\n".join(
        f"def p_{index}(panel):\n"
        "    matrix = np.ones((8, 8))\n"
        "    return float(np.linalg.eigvalsh(matrix).sum())"
        for index in range(P.MAX_EXPANDED_HEAVY_NATIVE_CALLS_PER_PANEL + 1)
    ) + "\n"
    with pytest.raises(P.PredicatePricingError, match="heavy native calls"):
        P.build_pricing_model(source)


def test_straight_line_array_operators_count_against_native_budget():
    source = (
        "def p_a(panel):\n"
        "    value = panel\n"
        + "".join(
            "    value = value + panel\n"
            for _ in range(P.MAX_EXPANDED_NATIVE_CALLS_PER_PANEL + 1))
        + "    return float(value.sum())\n"
    )
    with pytest.raises(P.PredicatePricingError, match="native calls"):
        P.build_pricing_model(source)


@pytest.mark.parametrize("expression", [
    "np.sqrt(panel, panel)",
    "np.sqrt(panel, out=panel)",
    "np.sqrt(panel, where=(panel > 0))",
])
def test_numpy_output_and_uninitialized_where_channels_are_rejected(expression):
    with pytest.raises(P.PredicatePricingError):
        P.build_pricing_model(
            "import numpy as np\n"
            "def p_a(panel):\n"
            f"    return float({expression}.sum())\n"
        )


def test_call_capability_aliases_cannot_be_shadowed():
    with pytest.raises(P.PredicatePricingError, match="shadow"):
        P.build_pricing_model(
            "import numpy as np\n"
            "def p_a(panel):\n"
            "    np = panel\n"
            "    np.sort()\n"
            "    return 0.0\n"
        )
    with pytest.raises(P.PredicatePricingError, match="shadow"):
        P.build_pricing_model(
            "def p_a(panel):\n"
            "    len = panel.sort\n"
            "    len()\n"
            "    return 0.0\n"
        )


def test_one_line_ast_structure_increases_definition_price():
    simple = P.build_pricing_model(
        "def p_a(panel):\n    return float(panel.sum())\n"
    )
    arithmetic = P.build_pricing_model(
        "def p_a(panel):\n"
        "    return float((((panel.sum() + 1) * 2 - 3) / 4) ** 2)\n"
    )
    simple_cost = simple.price_no_share(["p_a"]).full_cost
    assert arithmetic.price_no_share(["p_a"]).full_cost > simple_cost + 4

    flat = P.build_pricing_model(
        "def p_a(panel):\n    return float(sum(range(5)))\n"
    )
    comprehension = P.build_pricing_model(
        "def p_a(panel):\n"
        "    return float(sum(x * x for x in range(5) if x > 0))\n"
    )
    assert comprehension.price_no_share(["p_a"]).full_cost \
        > flat.price_no_share(["p_a"]).full_cost


def test_source_size_is_bounded_before_ast_parsing(monkeypatch):
    oversized = "x" * (P.MAX_SOURCE_CHARACTERS + 1)

    def forbidden_parse(*args, **kwargs):
        raise AssertionError("oversized source reached ast.parse")

    monkeypatch.setattr(P.ast, "parse", forbidden_parse)
    with pytest.raises(P.PredicatePricingError, match="character limit"):
        P.build_pricing_model(oversized)


def test_source_size_limit_counts_utf8_bytes():
    source = "é" * (P.MAX_SOURCE_UTF8_BYTES // 2 + 1)
    with pytest.raises(P.PredicatePricingError, match="UTF-8 byte limit"):
        P.build_pricing_model(source)


@pytest.mark.parametrize("callback", ["max", "min", "sorted"])
def test_named_key_callbacks_cannot_hide_recursive_helper_calls(callback):
    source = (
        "def helper(value):\n"
        f"    return {callback}([value], key=helper)\n"
        "def p_a(panel):\n"
        "    return float(helper(float(panel.sum())))\n"
    )
    with pytest.raises(P.PredicatePricingError, match="callback"):
        P.build_pricing_model(source)


@pytest.mark.parametrize("nested_definition", [
    "    @p_a\n"
    "    def nested(value):\n"
    "        return value\n",
    "    def nested(value=p_a(panel)):\n"
    "        return value\n",
    "    callback = lambda value=p_a(panel): value\n",
])
def test_nested_definition_time_decorators_and_defaults_are_rejected(
        nested_definition):
    source = (
        "def p_a(panel):\n"
        + nested_definition
        + "    return 0.0\n"
    )
    with pytest.raises(P.PredicatePricingError,
                       match="decorators|cannot call functions"):
        P.build_pricing_model(source)


@pytest.mark.parametrize("shape", [
    "int(1e12)",
    "int(1000000000000.5)",
    "int('10000000000')",
    "int(float('1e12'))",
    "(int(1000000.0), int(2.0))",
])
def test_static_allocation_limits_see_through_builtin_integer_coercions(shape):
    with pytest.raises(P.PredicatePricingError, match="allocation elements"):
        P.build_pricing_model(
            "import numpy as np\n"
            "def p_a(panel):\n"
            f"    return float(np.zeros({shape}).sum())\n"
        )


def _worklist_source(condition):
    return (
        "import numpy as np\n"
        "from collections import deque\n"
        "def p_a(panel):\n"
        "    n = 3\n"
        "    seen = -np.ones(n, dtype=int)\n"
        "    start = 0\n"
        "    seen[start] = 0\n"
        "    queue = deque([start])\n"
        "    while queue:\n"
        "        current = queue.popleft()\n"
        "        candidate = current\n"
        f"        if {condition}:\n"
        "            seen[candidate] = seen[current] + 1\n"
        "            queue.append(candidate)\n"
        "    return float(seen.sum())\n"
    )


def test_worklist_unvisited_guard_must_be_logically_necessary():
    with pytest.raises(P.PredicatePricingError, match="worklist"):
        P.build_pricing_model(
            _worklist_source("seen[candidate] == -1 or True")
        )
    model = P.build_pricing_model(
        _worklist_source(
            "candidate is not None and seen[candidate] == -1")
    )
    assert model.predicate_names == ("p_a",)


def test_bounded_predicate_file_reader_stops_before_ingesting_oversize(tmp_path):
    oversized = tmp_path / "oversized.py"
    oversized.write_bytes(b"x" * (P.MAX_SOURCE_UTF8_BYTES + 1))
    with pytest.raises(P.PredicatePricingError, match="UTF-8 byte limit"):
        P.read_predicate_source(str(oversized))

    invalid = tmp_path / "invalid.py"
    invalid.write_bytes(b"\xff")
    with pytest.raises(P.PredicatePricingError, match="valid UTF-8"):
        P.read_predicate_source(str(invalid))


def test_predicate_exception_handlers_cannot_catch_verifier_budget_signal():
    with pytest.raises(P.PredicatePricingError, match="exception handling"):
        P.build_pricing_model(
            "def p_a(panel):\n"
            "    try:\n"
            "        return float(panel.sum())\n"
            "    except Exception:\n"
            "        return 0.0\n"
        )
