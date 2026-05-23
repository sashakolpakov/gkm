# Abstraction Emergence Related Work

This note collects references for the predicate-macro / abstraction-emergence line of the manuscript. The framing should be conservative: predicate invention and reusable abstraction learning are established ideas. The possible contribution here is whether typed finite-state predicate automata emerge and remain useful under a free-energy / complexity accounting protocol on Bongard-style action-program tasks.

## Positioning Claim

Do not claim novelty for predicate invention itself. Claim the narrower experimental program:

```text
Can a population of sparse deterministic automata discover reusable predicate automata
under multi-task pressure, when the predicate library has explicit complexity cost and
later task solvers get cheaper by calling those predicates?
```

The empirical question is whether abstraction actually appears and transfers in this substrate, rather than only sounding plausible.

## Predicate Invention And ILP

- Stephen Muggleton and Wray Buntine, **Machine Invention of First-Order Predicates by Inverting Resolution**, 1988.  
  Early predicate-invention anchor. The useful citation point is that representation vocabulary was already recognized as a bottleneck: a learner may need to invent its own intermediate predicates rather than rely on the teacher's supplied primitives.
  DOI: https://doi.org/10.1016/B978-0-934613-64-4.50040-2

- Stephen Muggleton, **Predicate Invention and Utilization**, 1994.  
  Focused discussion of why invented predicates matter for ILP and theory formation. Useful when framing our automata predicates as theoretical/internal concepts rather than direct observations.
  DOI: https://doi.org/10.1080/09528139408953784

- Stephen Muggleton and Luc De Raedt, **Inductive Logic Programming: Theory and Methods**, 1994.  
  Classic ILP foundation. Useful for defining the broader field: learn symbolic hypotheses from examples plus background knowledge. Predicate invention appears early as a central problem.
  URL: https://www.dcc.fc.up.pt/~ines/aulas/1920/TAIA/1-s2.0-0743106694900353-main.pdf

- Andrew Cropper, Sebastijan Dumancic, **Inductive Logic Programming at 30**, 2021.  
  Modern survey. It explicitly treats predicate invention as one of the main directions in contemporary ILP. Useful citation when admitting that predicate invention is an old, hard problem rather than our invention.
  URL: https://arxiv.org/abs/2102.10556

- Andrew Cropper, Rolf Morel, **Predicate Invention by Learning From Failures**, 2021.  
  Introduces POPPI and frames predicate invention as discovering novel high-level concepts. Useful for our claim that intermediate predicates can improve learning when useful and should not be too costly when unnecessary.
  URL: https://arxiv.org/abs/2104.14426

- Stephen H. Muggleton, Dianhuan Lin, Niels Pahlavi, Alireza Tamaddoni-Nezhad, **Meta-interpretive learning: application to grammatical inference**, 2014.  
  Metagol/MIL learns recursive logic programs and supports predicate invention. Relevant because our substrate is closer to grammatical/automata induction than generic neural classification.
  DOI: https://doi.org/10.1007/s10994-013-5358-3
  PDF: https://www.doc.ic.ac.uk/~shm/Papers/metagol.pdf

- Later MIL systems may be useful if the manuscript goes deeper into search efficiency. For now, the important point is that predicate invention is known, powerful, and computationally difficult.

## Program Induction And Library Learning

- Kevin Ellis et al., **Learning Libraries of Subroutines for Neurally-Guided Bayesian Program Induction**, 2018.  
  Early EC-style system: solve tasks, compress solved programs into reusable library abstractions, then use those abstractions to solve harder tasks. Directly relevant to our multi-task compression story.
  URL: https://people.csail.mit.edu/asolar/papers/EllisMSST18.pdf

- Kevin Ellis et al., **DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning**, 2020.  
  Core reference for library learning as expertise development. It alternates problem solving with abstraction/library growth. Our automata-macro idea is close in spirit but changes the substrate, objective accounting, and task family.
  URL: https://arxiv.org/abs/2006.08381

- Matthew Bowers et al., **Top-Down Synthesis for Library Learning / Stitch**, 2022.  
  Compression/refactoring over a corpus of programs to find reusable lambda abstractions. Useful for stating that abstraction discovery can be formulated as compression over many solved programs.
  URL: https://arxiv.org/abs/2211.16605

- Gabriel Grand et al., **LILO: Learning Interpretable Libraries by Compressing and Documenting Code**, 2023.  
  Modern neurosymbolic library learning. LILO combines synthesis, compression, and documentation. Useful contrast: LILO learns program libraries for synthesis; our proposed experiment learns typed predicate automata for Bongard action/program scenes under explicit free-energy costs.
  URL: https://arxiv.org/abs/2310.19791

## Automata Learning And Grammatical Inference

- E. Mark Gold, **Language Identification in the Limit**, 1967.  
  Foundational formal model of grammar/language induction. Useful backdrop for automata/regular-language learnability and why examples alone can be ambiguous.
  DOI: https://doi.org/10.1016/S0019-9958(67)91165-5

- Dana Angluin, **Learning Regular Sets from Queries and Counterexamples**, 1987.  
  L* algorithm for learning regular languages with membership and equivalence queries. Important because our counterexample/archive idea rhymes with active automata learning, although our setting is evolutionary/free-energy rather than query-optimal learning with a teacher.
  DOI: https://doi.org/10.1016/0890-5401(87)90052-6

- Weiss, Goldberg, Yahav, **Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples**, 2017.  
  Related if we discuss automata extraction/interpretability rather than evolution. Lower priority for the manuscript but useful context.
  URL: https://arxiv.org/abs/1711.09576

## Macro-Actions And Hierarchical RL

- Richard Sutton, Doina Precup, Satinder Singh, **Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning**, 1999.  
  Options framework: temporally extended actions with policies and termination. Useful analogy for automata using automata, but our predicates are object-to-bool sensory abstractions, not primarily action policies.
  URL: https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf

- Pierre-Luc Bacon, Jean Harb, Doina Precup, **The Option-Critic Architecture**, 2017.  
  Learns both option policies and termination conditions end-to-end. Useful comparison for autonomous abstraction discovery, and as a warning: learned options often collapse to trivial behavior without appropriate pressure.
  URL: https://arxiv.org/abs/1609.05140

- George Konidaris and Andrew Barto, **Skill Discovery in Continuous Reinforcement Learning Domains using Skill Chaining**, 2009.  
  Example of discovering reusable skills/options from experience. Useful only as broad analogy for developmental skill libraries.
  URL: https://papers.nips.cc/paper/3683-skill-discovery-in-continuous-reinforcement-learning-domains-using-skill-chaining

## Compression, MDL, And Free-Energy Framing

- Jorma Rissanen, **Modeling by Shortest Data Description**, 1978.  
  Classic MDL origin. This is the cleanest citation for the idea that reusable abstractions should be justified by total description-length savings.
  DOI: https://doi.org/10.1016/0005-1098(78)90005-5

- Peter Grünwald, **The Minimum Description Length Principle**, 2007; Grünwald and Roos, **Minimum Description Length Revisited**, 2019.  
  General MDL background. Useful when formalizing library cost plus task cost.
  URL: https://arxiv.org/abs/1908.08484

- Ray Solomonoff, **A Formal Theory of Inductive Inference**, 1964.  
  Broader algorithmic-probability backdrop for complexity-weighted induction. Probably not necessary in the main related work unless the manuscript leans into universal induction.
  URL: https://mlanthology.org/misc/1964/solomonoff1964misc-formal/

- Jürgen Schmidhuber, **Driven by Compression Progress**, 2008.  
  Useful only if discussing open-endedness/curiosity via compression progress. Not central to the predicate-macro experiment, but relevant to the thesis's open-ended evolution framing.
  URL: https://arxiv.org/abs/0812.4360

## Bongard And Visual Concept Learning

- Mikhail Bongard, **Pattern Recognition**, 1970.  
  Original source of Bongard problems. Cite historically if the manuscript includes a benchmark background section.

- Nie et al., **Bongard-LOGO: A New Benchmark for Human-Level Concept Learning and Reasoning**, 2020.  
  Primary external benchmark. Important because it provides synthetic visual concepts and program-guided LOGO generation, letting us separate symbolic action-program reasoning from pixel perception.
  URL: https://arxiv.org/abs/2010.00763

- Depeweg et al., **Solving Bongard Problems with a Visual Language and Pragmatic Reasoning**, 2018.  
  Bayesian/pragmatic reasoning over a hand-designed visual language for Bongard problems. Useful precedent for explicit symbolic predicates in Bongard solving.
  URL: https://arxiv.org/abs/1804.04452

- Sonwane et al., **Using Program Synthesis and Inductive Logic Programming to solve Bongard Problems**, 2021.  
  Very close in spirit: DreamCoder-like program construction plus ILP for Bongard concepts. Important comparison and possible reviewer target. Our distinction must be clear: finite-state predicate automata, explicit library/free-energy accounting, and abstraction emergence across task families.
  URL: https://arxiv.org/abs/2110.09947

- Jiang et al., **Bongard-HOI: Benchmarking Few-Shot Visual Reasoning for Human-Object Interactions**, 2022.  
  Harder natural-image Bongard benchmark with human-object interactions and hard negatives. Later target only; not the first predicate-macro substrate.
  URL: https://arxiv.org/abs/2205.13803

- Wu et al., **Bongard-OpenWorld: Few-Shot Reasoning for Free-form Visual Concepts in the Real World**, 2023.  
  Open-world real-image Bongard variant. Useful later, but too confounded with perception/open vocabulary for the initial free-energy predicate story.
  URL: https://arxiv.org/abs/2310.10207

- **Symbolic Grounding Reveals Representational Bottlenecks in Abstract Visual Reasoning**, 2026.  
  Recent Bongard-LOGO symbolic-input diagnostic. Useful support for our current result: action programs / structured symbolic inputs reveal representational bottlenecks rather than merely pixel-recognition failures.
  URL: https://arxiv.org/abs/2604.21346

## What This Implies For Our Manuscript

### Avoid claiming

- Predicate invention is new.
- Library learning is new.
- Macro-actions/options are new.
- Bongard symbolic solving is new.
- Metadata predicates solving Abstract Bongard-LOGO demonstrates concept discovery.

### Plausible contribution

- Typed finite-state predicate automata as reusable calls inside sparse task automata.
- Explicit free-energy accounting over both library complexity and task-solver complexity.
- A multi-task test for abstraction emergence: macros should appear only when shared structure makes them cheaper than duplicated inline logic.
- Controlled Bongard-LOGO action-program substrate where perception is delayed and predicate formation can be measured directly.
- Honest negative-result protocol: report when macro predicates improve action-only performance but fail to match privileged metadata.

### Clean experiment to run next

```text
Task family A: one predicate target, e.g. convex vs non-convex.
Task family B: predicate composed with another condition, e.g. convex AND has_curve.
Task family C: action-policy variant, e.g. if convex then output/route/action X else Y.

Compare:
1. no library: each task evolves raw automata independently;
2. shared library allowed: predicate automata can be created and called;
3. forced library ablation: same call syntax but no library compression benefit;
4. oracle metadata upper bound: attr:convex etc. supplied directly.
```

Expected evidence pattern:

```text
single task: inline solution may be cheaper;
multi-task: shared predicate library should reduce total free energy;
transfer: learned predicate should help new tasks using the same latent property;
failure: if macros do not transfer, the abstraction story is not yet working.
```
