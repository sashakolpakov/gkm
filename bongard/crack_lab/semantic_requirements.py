"""Auditable semantic requirement registry.

This module decides whether an executable cone is allowed to claim a semantic
term.  It is intentionally separate from scoring: admissibility is a hard
gate, Kolmogorov/MDL selection happens after this gate.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass


RICH_PROXY_LEGS = frozenset({
    "bbox_fill",
    "bbox_aspect",
    "total_ink",
    "largest_area",
    "closure_ratio",
    "symmetry_residual",
    "object_count",
})

SCALAR_OK_TERMS = frozenset({
    "elongated",
    "aspect",
    "sparse",
    "filled",
    "fill",
    "large",
    "small",
    "area",
    "open",
    "closed",
    "symmetric",
    "asymmetric",
    "count",
})


@dataclass(frozen=True)
class SemanticRequirement:
    term: str
    aliases: tuple[str, ...]
    accepted_types: tuple[str, ...]
    primitive_required_types: tuple[str, ...] = ()
    required_roles: tuple[str, ...] = ()
    constraints: dict[str, int | float | str] | None = None
    unresolved_relation: str | None = None
    missing_legs: tuple[str, ...] = ()

    def all_terms(self) -> tuple[str, ...]:
        return (self.term,) + self.aliases


@dataclass(frozen=True)
class MissingLeg:
    semantic_term: str
    required_witness_types: tuple[str, ...]
    available_terminal_types: tuple[str, ...]
    unresolved_relation: str | None = None
    attempted_paths: tuple[str, ...] = ()
    missing_legs: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        rel = f"\nunresolved relation: {self.unresolved_relation}" if self.unresolved_relation else ""
        missing = "\nmissing:\n- " + "\n- ".join(self.missing_legs) if self.missing_legs else ""
        paths = "\navailable paths terminate at:\n- " + "\n- ".join(self.available_terminal_types)
        return (
            "MISSING_LEG\n"
            f"semantic term: {self.semantic_term}\n"
            f"required: {' + '.join(self.required_witness_types)}"
            f"{paths}{rel}{missing}"
        )


SEMANTIC_REQUIREMENTS: tuple[SemanticRequirement, ...] = (
    SemanticRequirement(
        "triangle",
        ("triangular", "three-sided", "three sided", "three-sided polygon"),
        ("TriangleWitness",),
        ("ContourWitness", "PolygonWitness", "TriangleWitness"),
        constraints={"side_count": 3},
        missing_legs=("extract_contours", "fit_polygon", "classify_triangle"),
    ),
    SemanticRequirement(
        "quadrilateral",
        ("square", "rectangle", "rectangular", "four-sided", "four sided", "four-sided polygon"),
        ("QuadrilateralWitness",),
        ("ContourWitness", "PolygonWitness", "QuadrilateralWitness"),
        constraints={"side_count": 4},
        missing_legs=("extract_contours", "fit_polygon", "classify_quadrilateral"),
    ),
    SemanticRequirement(
        "circle",
        ("circular", "round", "rounded circle"),
        ("CircleWitness",),
        ("ContourWitness", "CircleWitness"),
        missing_legs=("extract_contours", "fit_circle"),
    ),
    SemanticRequirement(
        "two circles",
        ("two circular lobes", "pair of circles", "circle pair", "two intersecting circles"),
        ("CirclePairWitness",),
        ("CircleWitness", "CirclePairWitness"),
        missing_legs=("fit_multiple_circles",),
    ),
    SemanticRequirement(
        "intersect",
        ("intersecting", "intersection", "crossing", "overlap"),
        ("IntersectionWitness", "CircleIntersectionWitness"),
        ("ContactWitness", "IntersectionWitness"),
        unresolved_relation="intersection",
        missing_legs=("detect_intersection", "circle_pair_intersection"),
    ),
    SemanticRequirement(
        "attachment",
        ("attached", "touching", "joined", "connected at", "shared endpoint", "shared point"),
        ("ContactWitness",),
        ("PartGraphWitness", "ContactWitness"),
        unresolved_relation="attachment",
        missing_legs=("build_part_graph", "detect_attachment"),
    ),
    SemanticRequirement(
        "four blades",
        ("pinwheel", "four triangular blades", "radial four", "four-fold", "four fold"),
        ("RadialArrangementWitness",),
        ("PartGraphWitness", "RadialArrangementWitness"),
        constraints={"part_count": 4},
        missing_legs=("build_part_graph", "detect_radial_arrangement"),
    ),
    SemanticRequirement(
        "bird-like",
        ("bird", "wings", "paired appendages"),
        ("PartGraphWitness",),
        ("PartGraphWitness", "ContactWitness", "CurveWitness", "SymmetryWitness"),
        required_roles=("body", "left_appendage", "right_appendage"),
        missing_legs=("build_part_graph", "detect_attachment", "decompose_curve_into_arcs_and_lines", "reflection_symmetry"),
    ),
    SemanticRequirement(
        "fish-like",
        ("fish", "fish shape", "tail and body"),
        ("PartGraphWitness",),
        ("PartGraphWitness", "ContactWitness", "SymmetryWitness"),
        required_roles=("body", "tail"),
        missing_legs=("build_part_graph", "detect_attachment", "reflection_symmetry"),
    ),
    SemanticRequirement(
        "lamp-like",
        ("lamp", "advanced lamp"),
        ("PartGraphWitness",),
        ("PartGraphWitness", "ContactWitness", "PolygonWitness"),
        required_roles=("shade", "stem"),
        missing_legs=("build_part_graph", "detect_attachment", "fit_polygon"),
    ),
)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _contains_alias(haystack: str, alias: str) -> bool:
    alias_n = _norm(alias)
    if not alias_n:
        return False
    return re.search(rf"(^| )({re.escape(alias_n)})( |$)", haystack) is not None


def explicit_terms(hypothesis) -> tuple[str, ...]:
    values = []
    for attr in ("semantic_requirements", "witness_requirements", "relations"):
        values.extend(str(x) for x in getattr(hypothesis, attr, ()) if str(x).strip())
    return tuple(values)


def requirements_for_hypothesis(hypothesis) -> tuple[SemanticRequirement, ...]:
    text = _norm(" ".join((hypothesis.description, " ".join(explicit_terms(hypothesis)))))
    found: list[SemanticRequirement] = []
    for req in SEMANTIC_REQUIREMENTS:
        if any(_contains_alias(text, alias) for alias in req.all_terms()):
            found.append(req)
    return tuple(found)


def scalar_terms_for_hypothesis(hypothesis) -> frozenset[str]:
    text = _norm(hypothesis.description + " " + " ".join(explicit_terms(hypothesis)))
    return frozenset(term for term in SCALAR_OK_TERMS if _contains_alias(text, term))


def is_rich_requirement(req: SemanticRequirement) -> bool:
    return bool(req.primitive_required_types)
