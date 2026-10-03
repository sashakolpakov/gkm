"""Finite labelled-graph pushouts, with a conservative executable attachment gate.

We designate injective graph morphisms as cofibrations in this experiment.
This is not a claim that transducers carry a Quillen model structure. Graph
pushouts are exact; determinism and preservation of old executions are separate
checks. The first executable grammar permits tail reuse, not calls that return.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from pattern_fsa import PatternGenome, PatternRule, PrimitiveSet


@dataclass(frozen=True)
class Graph:
    states: int
    rules: tuple[PatternRule, ...]
    primitives: PrimitiveSet

    def __post_init__(self):
        if type(self.states) is not int or self.states < 0:
            raise ValueError("invalid state count")
        for rule in self.rules:
            if not (0 <= rule.state < self.states and 0 <= rule.next_state < self.states):
                raise ValueError("transition endpoint outside graph")
            if rule.observation not in self.primitives.observations:
                raise ValueError("observation outside signature")
            if any(action not in self.primitives.actions for action in rule.actions):
                raise ValueError("action outside signature")

    def payload(self):
        return asdict(self)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()

    @property
    def complexity(self):
        return sum(1 + len(rule.actions) for rule in self.rules)

    def genome(self):
        # Token identities are never inspected; alphabet_size is metadata only.
        if len({rule.key for rule in self.rules}) != len(self.rules):
            raise ValueError("pushout is not a deterministic transducer")
        return PatternGenome(self.states, 1, list(self.rules))


@dataclass(frozen=True)
class Morphism:
    states: tuple[int, ...]
    rules: tuple[int, ...]

    def validate(self, source: Graph, target: Graph, *, injective=False):
        if source.primitives != target.primitives:
            raise ValueError("primitive signature mismatch")
        for values, count, bound in (
            (self.states, source.states, target.states),
            (self.rules, len(source.rules), len(target.rules)),
        ):
            if len(values) != count or any(type(v) is not int or not 0 <= v < bound for v in values):
                raise ValueError("invalid morphism map")
            if injective and len(set(values)) != len(values):
                raise ValueError("cofibration must be injective")
        for index, rule in enumerate(source.rules):
            image = target.rules[self.rules[index]]
            if image != PatternRule(self.states[rule.state], rule.observation,
                                    rule.actions, self.states[rule.next_state]):
                raise ValueError("morphism changes a labelled transition")


def _quotient(left_count, right_count, left_interface, right_interface):
    """Number the disjoint union modulo exactly the interface identifications."""
    parents = list(range(left_count + right_count))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left, right in zip(left_interface, right_interface):
        parents[root(left_count + right)] = root(left)
    numbers = {}
    images = []
    for index in range(len(parents)):
        representative = root(index)
        images.append(numbers.setdefault(representative, len(numbers)))
    return tuple(images[:left_count]), tuple(images[left_count:]), len(numbers)


@dataclass(frozen=True)
class Pushout:
    boundary: Graph
    left: Graph
    right: Graph
    into_left: Morphism
    into_right: Morphism
    graph: Graph
    from_left: Morphism
    from_right: Morphism

    def mediate(self, target: Graph, left_map: Morphism, right_map: Morphism):
        """Construct the unique map from the quotient for an agreeing cocone."""
        left_map.validate(self.left, target)
        right_map.validate(self.right, target)
        images = []
        for count, l_in, r_in, l_out, r_out in (
            (self.graph.states, self.from_left.states, self.from_right.states,
             left_map.states, right_map.states),
            (len(self.graph.rules), self.from_left.rules, self.from_right.rules,
             left_map.rules, right_map.rules),
        ):
            result = [None] * count
            for projection, destination in ((l_in, l_out), (r_in, r_out)):
                for src, dst in zip(projection, destination):
                    if result[src] is not None and result[src] != dst:
                        raise ValueError("cocone does not agree on boundary")
                    result[src] = dst
            if any(value is None for value in result):
                raise ValueError("quotient has an unaccounted element")
            images.append(tuple(result))
        result = Morphism(*images)
        result.validate(self.graph, target)
        return result

    def certificate(self):
        return {
            "category": "finite directed multigraphs with observation/action labels",
            "cofibrations": "injective state and transition maps",
            "boundary": self.boundary.payload(), "left": self.left.payload(),
            "right": self.right.payload(), "pushout": self.graph.payload(),
            "hashes": {key: getattr(self, key).digest
                       for key in ("boundary", "left", "right", "graph")},
            "maps": {key: asdict(getattr(self, key)) for key in
                     ("into_left", "into_right", "from_left", "from_right")},
        }


def pushout(boundary: Graph, left: Graph, right: Graph,
            into_left: Morphism, into_right: Morphism) -> Pushout:
    into_left.validate(boundary, left, injective=True)
    into_right.validate(boundary, right, injective=True)
    ls, rs, states = _quotient(left.states, right.states, into_left.states, into_right.states)
    le, re, edge_count = _quotient(len(left.rules), len(right.rules),
                                 into_left.rules, into_right.rules)
    rules = [None] * edge_count
    for source, state_map, edge_map in ((left, ls, le), (right, rs, re)):
        for index, rule in enumerate(source.rules):
            image = PatternRule(state_map[rule.state], rule.observation,
                                rule.actions, state_map[rule.next_state])
            slot = edge_map[index]
            if rules[slot] is not None and rules[slot] != image:
                raise ValueError("incompatible interface labels")
            rules[slot] = image
    graph = Graph(states, tuple(rules), left.primitives)
    result = Pushout(boundary, left, right, into_left, into_right,
                     graph, Morphism(ls, le), Morphism(rs, re))
    result.from_left.validate(left, graph, injective=True)
    result.from_right.validate(right, graph, injective=True)
    # The identity cocone also checks coverage and interface commutation.
    if result.mediate(graph, result.from_left, result.from_right) != Morphism(
        tuple(range(states)), tuple(range(edge_count))
    ):
        raise ValueError("invalid quotient projections")
    return result


def attach(library: Graph, fresh_states: int, rules: tuple[PatternRule, ...],
           ports: tuple[int, ...]) -> tuple[Pushout, int]:
    """Glue local states onto selected library states without adding old outgoing rules.

    Local states 0..fresh_states-1 are new; following states are interface ports.
    The new entry is local state 0. Missing old rules remain missing: even implicit
    halting is preserved at every old entry, for every input, not just examples.
    """
    if type(fresh_states) is not int or not 1 <= fresh_states <= 8:
        raise ValueError("attachment requires 1..8 fresh states")
    if any(type(port) is not int or not 0 <= port < library.states for port in ports):
        raise ValueError("invalid interface port")
    if len(set(ports)) != len(ports):
        raise ValueError("duplicate interface port")
    if len(rules) > 64 or any(len(rule.actions) > 8 for rule in rules):
        raise ValueError("glue exceeds grammar bounds")
    if any(rule.state >= fresh_states for rule in rules):
        raise ValueError("glue may not add outgoing rules at retained states")
    boundary = Graph(len(ports), (), library.primitives)
    glue = Graph(fresh_states + len(ports), rules, library.primitives)
    result = pushout(boundary, library, glue, Morphism(ports, ()),
                     Morphism(tuple(range(fresh_states, glue.states)), ()))
    result.graph.genome()  # reject conflicting transitions rather than last-write-wins
    return result, result.from_right.states[0]


def parse_attachment(payload: dict, library: Graph):
    if not isinstance(payload, dict) or set(payload) != {"library_hash", "fresh_states", "ports", "rules"}:
        raise ValueError("unexpected attachment fields")
    if payload["library_hash"] != library.digest:
        raise ValueError("proposal is not bound to this library")
    if not isinstance(payload["ports"], list) or not isinstance(payload["rules"], list):
        raise ValueError("ports and rules must be arrays")
    if len(payload["ports"]) > library.states or len(payload["rules"]) > 64:
        raise ValueError("proposal exceeds size limit")
    rules = []
    for item in payload["rules"]:
        if not isinstance(item, dict) or set(item) != {"state", "observation", "actions", "next_state"}:
            raise ValueError("invalid rule fields")
        if not isinstance(item["actions"], list) or not 1 <= len(item["actions"]) <= 8:
            raise ValueError("invalid action list")
        values = [item["state"], item["observation"], item["next_state"], *item["actions"]]
        if any(type(value) is not int for value in values):
            raise ValueError("rule values must be integers, not coerced strings or booleans")
        rules.append(PatternRule(**item))
    return attach(library, payload["fresh_states"], tuple(rules), tuple(payload["ports"]))


def read_graph(payload):
    if set(payload) != {"states", "rules", "primitives"}:
        raise ValueError("invalid graph fields")
    signature = dict(payload["primitives"])
    signature["actions"] = tuple(signature["actions"])
    return Graph(payload["states"], tuple(PatternRule(**rule) for rule in payload["rules"]),
                 PrimitiveSet(**signature))


def verify_certificate(certificate, expected_left=None):
    """Rebuild the quotient and every map from serialized evidence, not just hashes."""
    boundary, left, right = [read_graph(certificate[key]) for key in ("boundary", "left", "right")]
    if expected_left is not None and left.digest != expected_left.digest:
        raise ValueError("certificate substituted the retained library")
    maps = [Morphism(tuple(certificate["maps"][key]["states"]),
                     tuple(certificate["maps"][key]["rules"]))
            for key in ("into_left", "into_right")]
    rebuilt = pushout(boundary, left, right, *maps)
    if json.dumps(rebuilt.certificate(), sort_keys=True) != json.dumps(certificate, sort_keys=True):
        raise ValueError("certificate is not the claimed labelled-graph pushout")
    return rebuilt
