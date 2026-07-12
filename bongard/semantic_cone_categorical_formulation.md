# Bongard Semantic Cone: Categorical Formulation

This note formalizes the semantic-cone design for Bongard predicate search.
It is intentionally shorter than the implementation specbook: its purpose is
to fix the mathematical object that the code must enforce.

## 1. The Failure Mode

The current paired response format,

```text
{"semantic": H, "code": q}
```

is only a product of an English/structured hypothesis and an executable
predicate. It does not prove that `q` is a realization of `H`.

The observed failure is exactly this missing morphism: the proposer can name
the right semantic invariant, then implement an unrelated panel statistic, and
the verifier may select that accidental statistic because it separates the
support set.

The semantic track must therefore require factorization, not mere pairing.

## 2. Categories

Let `PanelC` be the operational category of panel representations.

Objects include:

- raw raster panels;
- binarized panels;
- parsed scenes;
- visual objects;
- object descriptors;
- relations;
- Boolean decisions.

Morphisms include parsing, normalization, object selection, descriptor
extraction, relation evaluation, and declared nuisance transformations such as
translation, rotation, reflection, scale, stroke-width perturbation, and object
permutation.

Let `SemanticC` be the typed relational category of semantic objects.

Objects include:

- `Scene`;
- `Object`;
- `ObjectSet`;
- `ObjectPair`;
- `ShapeDescriptor`;
- `Measurement`;
- `Relation`;
- `Bool`.

The leg registry is a typed subcategory, or more precisely a small typed
multicategory when legs take several arguments. A registered leg is an arrow
with a contract:

```text
leg : A_1 x ... x A_n -> B
```

plus declared invariances, equivariances, failure modes, and complexity cost.

## 3. Semantic Realization

A parser/realizer functor maps panels into semantic structure:

```text
S : PanelC -> SemanticC
```

A semantic hypothesis is not a free-form sentence. It is a finite typed diagram:

```text
D_H : J -> SemanticC
```

with named variables, selector bindings, relation nodes, measurement nodes, and
a Boolean decision node.

For a panel `x`, the executable semantic predicate must be obtained by filling
this diagram from `S(x)` and evaluating the decision node.

## 4. Task Cone

A semantic solution for a task is a cone from the realized panel object into the
hypothesis diagram:

```text
          S(x)
        /  |  \
       v   v   v
      D_H(j1), D_H(j2), ...
```

Operationally, the cone consists of:

- parser leg;
- selector bindings;
- descriptor and measurement legs;
- relation legs;
- threshold or comparison parameters;
- Boolean composition.

The final predicate is admissible only when it factors through this cone:

```text
p_H = beta . r . b . S
```

where:

- `S` parses or realizes the panel;
- `b` applies selector and variable bindings;
- `r` evaluates the declared measurements and relations;
- `beta` evaluates the declared Boolean rule.

In pure semantic mode, arbitrary code

```text
q : Panel -> Bool
```

is rejected unless the compiler can exhibit the factorization above using
registered typed legs and declared bindings.

## 5. Naturality

For every declared label-preserving morphism `m` in `PanelC`, the following
square should commute up to declared tolerance:

```text
Panel  --p_H-->  Bool
  |              |
  m              id
  v              v
Panel  --p_H-->  Bool
```

That is:

```text
p_H(m(x)) = p_H(x)
```

For relation-level concepts, naturality must be checked before the final Boolean
node as well:

```text
relation_H(S(m(x))) ~= relation_H(S(x))
```

The verifier should report the first noncommuting node, not just final accuracy.

## 5a. Cofibrations are Gluing Morphisms

Composite semantic structure is built by attaching a patch `P` to a source
witness `A` along an interface `I`, i.e. by a pushout

```text
I  ---->  P
|         |
v         v
A  --->  A ⊔_I P
```

The cofibration is the canonical morphism `A -> A ⊔_I P`. Two consequences
fix the verified contract:

1. It is NOT an inclusion on the nose. The gluing identifies structure along
   `I`, so identifiers may be renamed, coordinates re-expressed, and derived
   fields recomputed. The contract therefore verifies *glue-equivalence*: a
   single consistent bijection of witness identifiers plus numeric tolerance
   on geometry. Field-for-field equality (an earlier, wrong formulation)
   rejects legitimate gluings the moment a part is renumbered.
2. Projections split the gluing only up to that identification:
   `proj(A ⊔_I P) ≅ A` over `I`, never `proj(...) == A`.

Cofibration specs carry no concept content and are never library constants:
the proposer generates each spec inside its cone IR (binding diagram nodes
as source and target of the gluing), the compiler checks the declared nodes,
types and attachment leg — a missing attachment leg is a MISSING_LEG demand,
which keeps representation poverty visible — and the verifier checks
glue-equivalence on the actual witness values of every panel. Hard-coded
specs are admissible only as unit-test fixtures.

## 5b. Generality of the Gate

Nothing problem-specific is admissible anywhere in the harness: no concept
lexicon, no per-concept requirement table, no hard-coded composite gluing.
Term admissibility is audited against the leg registry itself: every
declared term must be anchored by structure inside the score's dependency
cone (witness types, legs, contract vocabulary) or by a declared gluing; a
term naming registry-expressible structure the score does not execute is
rejected as weakening; a term the registry cannot express at all surfaces
as MISSING_LEG rather than riding on an unrelated scalar. Cone invariance
(Section 5) is likewise executed generally: declared nuisance morphisms
with exact pixel actions are applied to the panels and the decision must
commute; morphisms without an exact action are reported as unchecked.

## 6. Contrast

Contrast interventions are not label-preserving morphisms. They are controlled
semantic edits expected to move the object across the concept boundary.

For a declared contrast intervention `c`:

```text
p_H(c(x)) != p_H(x)
```

when the edit is valid and the original panel is not already ambiguous.

This separates invariance from discriminative power. A statistic that survives
support-set CV but ignores declared contrast is a shortcut, not a semantic cone.

## 7. Thresholds and Cross-Validation

Numeric thresholds are parameters of arrows in the cone, not literals baked into
new task code.

For a measurement leg

```text
m_H : Object -> Measurement
```

and order comparison

```text
theta : Measurement -> Bool
```

the threshold parameter of `theta` must be fit inside each held-out fold. The
reported rule is admissible only when the fold-wise parameters define a stable
common interval or an explicitly charged tolerance.

This is the categorical version of the circle-residual issue: the semantic
object may be "points lie on a circle", while the numeric cutoff is a learned
parameter of the measurement arrow.

## 8. Failure Taxonomy

The categorical formulation localizes failures.

- Wrong semantic hypothesis: no low-risk cone exists for `D_H`.
- Missing leg: the required arrow is absent from the registry.
- Bad binding: the diagram is valid, but selector arrows choose the wrong
  objects.
- Bad leg implementation: the arrow exists but violates its contract or
  naturality tests.
- Shortcut predicate: executable `q` separates panels but does not factor
  through `D_H`.
- Underspecified semantics: many non-isomorphic cones fit support data with
  similar risk and complexity.
- Search exhaustion: admissible diagrams were not found within the declared
  frontier, but no structural impossibility was shown.

This is why a correct verbal answer with wrong code must be classified as a
compiler, leg, or binding failure rather than a semantic failure.

## 9. Model Selection

Kolmogorov/MDL selection remains the outer selector. For a cone `C` over
hypothesis `H` and promoted library `L`, choose by conditional free energy:

```text
F(C | L) =
  R_support
  + R_loo
  + R_naturality
  + R_contrast
  + R_counterfactual
  + lambda * K(C | L)
```

where `K(C | L)` charges:

- new legs;
- leg calls;
- bindings;
- diagram nodes and edges;
- parameters and precision;
- residuals;
- exceptions.

Previously promoted legs are not recharged, but calls, bindings, and new
parameters are charged per use.

## 10. Implementation Rule

The semantic track should no longer ask the proposer for a final arbitrary
`p_*(panel)` classifier.

The proposer should return candidate typed diagrams. The harness should:

1. type-check the semantic IR;
2. compile it through registered legs;
3. enumerate bindings and fold-fit parameters;
4. verify support, LOO, naturality, contrast, and regressions;
5. select among admissible cones by conditional description length plus risk.

Unrestricted `p_*(panel)` search remains useful, but it is a separate control
track. A control-track success is not a semantic-pure success.

## 11. One-Sentence Rule

A semantic Bongard predicate is a priced, typed, replay-verified cone whose
decision map factors through declared semantic objects and relations, commutes
with declared nuisance morphisms, responds to declared contrast interventions,
and wins against competitors by conditional description length plus risk.
