#!/usr/bin/env python3
"""Colimit-cone experiment on the multi-channel foraging substrate.

Tests whether free-energy selection with sum-over-legs / shared-leg-discounting
cone accounting builds a behavior library only when reuse pays. The design,
predictions, pitfalls, and falsification criteria are in
COLIMIT_CONE_APPROACH.md; the substrate is cone_foraging.py.

Stages (Section 7 of the design document):

    A  evolve inline champions per task
    B  lift candidate legs from inline champions (encapsulation mutation)
    C  evolve gluings per condition / lambda / candidate leg
    D  joint free-energy selection, validation elbow, hidden evaluation
    E  transfer with frozen library plus budget-matched discovery rates
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
for _domain in ("cone", "arc"):
    _p = REPO_ROOT / _domain
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cone_foraging as cf  # noqa: E402

SCENARIOS: Dict[str, Tuple[Tuple[str, ...], Optional[str]]] = {
    "single": (("forage",), "homing"),
    "multi": (("forage", "homing"), "forage_then_home"),
    "two_phase": (("forage_then_home",), "homing"),
    # multi_seq adds compositional pressure (reconciliation log R10, option 2):
    # the sequencing task needs the first call to hand control back, so the
    # RETURN boundary must earn its keep inside the support set itself.
    "multi_seq": (("forage", "homing", "forage_then_home"), None),
}

CSV_HEADER = (
    "scenario,condition,lambda,train_loss,val_loss,hidden_loss,"
    "train_solved,val_solved,hidden_solved,complexity,free_energy,leg,static_calls,leg_return,call_champion_seen"
)


@dataclass
class TaskData:
    task: cf.TaskSpec
    train: List[cf.ConeLevel]
    validation: List[cf.ConeLevel]
    hidden: List[cf.ConeLevel]


@dataclass
class Config:
    """One candidate cone: a library choice plus per-task genomes at one lambda."""

    lambda_value: float
    leg: Optional[cf.Leg]
    genomes: Dict[str, cf.ConeGenome]
    condition: str
    saw_call_champion: bool = False
    extra_legs: Tuple[cf.Leg, ...] = ()
    free_leg_indices: frozenset = frozenset()

    def library(self) -> List[cf.Leg]:
        library = list(self.extra_legs)
        if self.leg is not None:
            library.append(self.leg)
        return library


@dataclass
class ConfigScore:
    config: Config
    train_loss: float
    val_loss: float
    complexity: float
    free_energy: float
    train_solved: int
    val_solved: int
    static_calls: int


@dataclass
class SelectedRow:
    scenario: str
    condition: str
    lambda_value: float
    train_loss: float
    val_loss: float
    hidden_loss: float
    train_solved: int
    val_solved: int
    hidden_solved: int
    task_count: int
    complexity: float
    free_energy: float
    leg: str
    static_calls: int
    leg_return: bool
    call_champion_seen: bool
    rules: Dict[str, List[str]] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS) + ("all",), default="all")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--lambda-values", default="0.001,0.003,0.01")
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--generations", type=int, default=45)
    parser.add_argument("--state-count", type=int, default=3)
    parser.add_argument("--max-rules", type=int, default=14)
    parser.add_argument("--max-rule-length", type=int, default=2)
    parser.add_argument("--mutation-rate", type=float, default=0.10)
    parser.add_argument("--call-cost", type=float, default=cf.DEFAULT_CALL_COST)
    parser.add_argument("--train-count", type=int, default=6)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--hidden-count", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=44)
    parser.add_argument("--val-tolerance", type=float, default=0.02)
    parser.add_argument("--discovery-replicates", type=int, default=5)
    parser.add_argument("--discovery-population", type=int, default=36)
    parser.add_argument("--discovery-generations", type=int, default=14)
    parser.add_argument(
        "--joint",
        action="store_true",
        help="also run cold joint cone evolution (shared_joint / no_share_joint rows)",
    )
    parser.add_argument("--joint-population", type=int, default=150)
    parser.add_argument("--joint-generations", type=int, default=80)
    parser.add_argument(
        "--growth",
        action="store_true",
        help="run the staged library-growth experiment (sequential colimit)",
    )
    parser.add_argument("--quick", action="store_true", help="small budgets for a fast smoke run")
    parser.add_argument("--output-dir", default="output/colimit_cone")
    parser.add_argument("--show-rules", action="store_true")
    return parser.parse_args()


def make_task_data(args: argparse.Namespace, task_name: str, seed_offset: int) -> TaskData:
    task = cf.TASKS[task_name]
    base = args.seed + 1000 * seed_offset
    return TaskData(
        task=task,
        train=cf.make_cone_levels(base + 1, args.train_count, task),
        validation=cf.make_cone_levels(base + 2, args.validation_count, task),
        hidden=cf.make_cone_levels(base + 3, args.hidden_count, task),
    )


def evolve_task(
    args: argparse.Namespace,
    data: TaskData,
    allowed_actions: Sequence[int],
    library: Sequence[cf.Leg],
    condition: str,
    lambda_value: float,
    seed: int,
    population: Optional[int] = None,
    generations: Optional[int] = None,
) -> cf.EvolutionResult:
    return cf.evolve_cone_task(
        data.task,
        data.train,
        allowed_actions,
        library,
        condition,
        lambda_value,
        seed=seed,
        population_size=population or args.population,
        generations=generations or args.generations,
        state_count=args.state_count,
        max_rules=args.max_rules,
        max_rule_length=args.max_rule_length,
        mutation_rate=args.mutation_rate,
        call_cost=args.call_cost,
        max_steps=args.max_steps,
    )


def score_config(
    args: argparse.Namespace,
    config: Config,
    task_data: Dict[str, TaskData],
    charge_library: bool = True,
) -> ConfigScore:
    library = config.library()
    train_losses, val_losses = [], []
    train_solved = val_solved = 0
    for name, genome in config.genomes.items():
        data = task_data[name]
        train_eval = cf.evaluate_cone_task(genome, library, data.train, data.task, max_steps=args.max_steps)
        val_eval = cf.evaluate_cone_task(genome, library, data.validation, data.task, max_steps=args.max_steps)
        train_losses.append(train_eval.loss)
        val_losses.append(val_eval.loss)
        train_solved += int(train_eval.solved)
        val_solved += int(val_eval.solved)
    complexity = cf.cone_complexity(
        list(config.genomes.values()),
        library,
        config.condition,
        call_cost=args.call_cost,
        charge_library=charge_library,
        free_legs=config.free_leg_indices,
    )
    train_loss = sum(train_losses)
    return ConfigScore(
        config=config,
        train_loss=train_loss,
        val_loss=sum(val_losses),
        complexity=complexity,
        free_energy=train_loss + config.lambda_value * complexity,
        train_solved=train_solved,
        val_solved=val_solved,
        static_calls=sum(len(genome.call_references()) for genome in config.genomes.values()),
    )


def select_by_validation_elbow(scores: Sequence[ConfigScore], tolerance: float) -> ConfigScore:
    best_val = min(score.val_loss for score in scores)
    allowed = [score for score in scores if score.val_loss <= best_val + tolerance]
    return min(allowed, key=lambda score: (score.complexity, score.val_loss, score.config.lambda_value))


def hidden_evaluation(
    args: argparse.Namespace,
    score: ConfigScore,
    task_data: Dict[str, TaskData],
) -> Tuple[float, int]:
    library = score.config.library()
    hidden_loss = 0.0
    hidden_solved = 0
    for name, genome in score.config.genomes.items():
        data = task_data[name]
        evaluation = cf.evaluate_cone_task(genome, library, data.hidden, data.task, max_steps=args.max_steps)
        hidden_loss += evaluation.loss
        hidden_solved += int(evaluation.solved)
    return hidden_loss, hidden_solved


def make_row(
    args: argparse.Namespace,
    scenario: str,
    condition: str,
    score: ConfigScore,
    task_data: Dict[str, TaskData],
    call_champion_seen: bool,
) -> SelectedRow:
    hidden_loss, hidden_solved = hidden_evaluation(args, score, task_data)
    library = score.config.library()
    used = cf.legs_used(list(score.config.genomes.values()), library)
    used_legs = [library[idx] for idx in sorted(used)]
    leg_name = "+".join(leg.name for leg in used_legs) if used_legs else "none"
    leg_return = any(cf.genome_has_return(leg.genome) for leg in used_legs)
    rules = {name: genome.describe() for name, genome in score.config.genomes.items()}
    for leg in used_legs:
        rules[f"leg:{leg.name}"] = leg.genome.describe()
    return SelectedRow(
        scenario=scenario,
        condition=condition,
        lambda_value=score.config.lambda_value,
        train_loss=score.train_loss,
        val_loss=score.val_loss,
        hidden_loss=hidden_loss,
        train_solved=score.train_solved,
        val_solved=score.val_solved,
        hidden_solved=hidden_solved,
        task_count=len(score.config.genomes),
        complexity=score.complexity,
        free_energy=score.free_energy,
        leg=leg_name,
        static_calls=score.static_calls,
        leg_return=leg_return,
        call_champion_seen=call_champion_seen,
        rules=rules,
    )


def print_row(row: SelectedRow) -> None:
    print(
        f"{row.scenario},{row.condition},{row.lambda_value:.4f},"
        f"{row.train_loss:.4f},{row.val_loss:.4f},{row.hidden_loss:.4f},"
        f"{row.train_solved}/{row.task_count},{row.val_solved}/{row.task_count},{row.hidden_solved}/{row.task_count},"
        f"{row.complexity:.2f},{row.free_energy:.4f},{row.leg},{row.static_calls},{row.leg_return},{row.call_champion_seen}"
    )


def dedupe_legs(legs: Sequence[cf.Leg]) -> List[cf.Leg]:
    seen = set()
    unique = []
    for leg in legs:
        signature = tuple(sorted((rule.state, rule.observation, rule.actions, rule.next_state) for rule in leg.genome.rules))
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(leg)
    return unique


def run_scenario(
    args: argparse.Namespace,
    scenario: str,
    lambda_values: Sequence[float],
) -> Tuple[List[SelectedRow], List[Dict[str, object]]]:
    support_names, transfer_name = SCENARIOS[scenario]
    task_data = {
        name: make_task_data(args, name, seed_offset=idx + 1)
        for idx, name in enumerate(support_names)
    }

    # Stage A: inline champions per task per lambda.
    inline_configs: List[Config] = []
    for lambda_idx, lambda_value in enumerate(lambda_values):
        genomes = {}
        for task_idx, name in enumerate(support_names):
            result = evolve_task(
                args,
                task_data[name],
                cf.inline_actions(),
                [],
                "inline",
                lambda_value,
                seed=args.seed + 13 * (task_idx + 1) + 101 * (lambda_idx + 1),
            )
            genomes[name] = result.genome
        inline_configs.append(Config(lambda_value, None, genomes, "inline"))
    inline_scores = [score_config(args, config, task_data) for config in inline_configs]
    inline_selected = select_by_validation_elbow(inline_scores, args.val_tolerance)

    rows = [make_row(args, scenario, "inline", inline_selected, task_data, call_champion_seen=False)]

    # Stage B: candidate legs lifted from the selected inline champions.
    candidates = dedupe_legs(
        [
            cf.lift_leg(genome, f"lift_{name}")
            for name, genome in inline_selected.config.genomes.items()
        ]
    )

    # Stages C/D: gluing evolution and joint selection per searched condition.
    selected_leg: Optional[cf.Leg] = None
    for condition in ("shared", "no_share"):
        per_lambda_best: List[ConfigScore] = []
        call_champion_seen = False
        for lambda_idx, lambda_value in enumerate(lambda_values):
            configs: List[Config] = [
                Config(lambda_value, None, dict(inline_configs[lambda_idx].genomes), condition)
            ]
            for candidate_idx, candidate in enumerate(candidates):
                genomes = {}
                saw_call = False
                for task_idx, name in enumerate(support_names):
                    result = evolve_task(
                        args,
                        task_data[name],
                        cf.controller_actions(1),
                        [candidate],
                        condition,
                        lambda_value,
                        seed=args.seed
                        + 7 * (task_idx + 1)
                        + 211 * (lambda_idx + 1)
                        + 1009 * (candidate_idx + 1)
                        + (97 if condition == "no_share" else 0),
                    )
                    genomes[name] = result.genome
                    saw_call = saw_call or result.saw_call_champion
                call_champion_seen = call_champion_seen or saw_call
                configs.append(Config(lambda_value, candidate, genomes, condition, saw_call_champion=saw_call))
            scores = [score_config(args, config, task_data) for config in configs]
            per_lambda_best.append(min(scores, key=lambda s: (s.free_energy, s.train_loss, s.complexity)))
        selected = select_by_validation_elbow(per_lambda_best, args.val_tolerance)
        rows.append(make_row(args, scenario, condition, selected, task_data, call_champion_seen))
        if condition == "shared" and selected.config.leg is not None and selected.static_calls > 0:
            selected_leg = selected.config.leg

    # Cold joint cone evolution: leg bodies and gluings from random init,
    # true shared accounting charged during local selection (no lifting).
    if args.joint:
        for condition_label, condition in (("shared_joint", "shared"), ("no_share_joint", "no_share")):
            scores: List[ConfigScore] = []
            call_seen = False
            for lambda_idx, lambda_value in enumerate(lambda_values):
                result = cf.evolve_joint_cone(
                    {name: (task_data[name].task, task_data[name].train) for name in support_names},
                    condition,
                    lambda_value,
                    seed=args.seed + 31_000 + 101 * lambda_idx + (97 if condition == "no_share" else 0),
                    population_size=args.joint_population,
                    generations=args.joint_generations,
                    state_count=args.state_count,
                    max_rules=args.max_rules,
                    max_rule_length=args.max_rule_length,
                    mutation_rate=args.mutation_rate,
                    call_cost=args.call_cost,
                    max_steps=args.max_steps,
                )
                call_seen = call_seen or result.saw_call_champion
                config = Config(
                    lambda_value,
                    cf.Leg("joint_evolved", result.legs[0]),
                    dict(result.controllers),
                    condition,
                    saw_call_champion=result.saw_call_champion,
                )
                scores.append(score_config(args, config, task_data))
            selected = select_by_validation_elbow(scores, args.val_tolerance)
            rows.append(make_row(args, scenario, condition_label, selected, task_data, call_seen))

    # Witness: hand-written representability floor, charged with shared accounting.
    witness_leg = cf.witness_seek_leg()
    witness_config = Config(
        lambda_values[len(lambda_values) // 2],
        witness_leg,
        {name: cf.witness_gluing(task_data[name].task) for name in support_names},
        "witness",
    )
    witness_score = score_config(args, witness_config, task_data)
    rows.append(make_row(args, scenario, "witness", witness_score, task_data, call_champion_seen=True))

    # Stage E: transfer with the frozen library selected by the shared condition.
    if transfer_name is None:
        return rows, []
    transfer_rows, discovery_records = run_transfer(
        args, scenario, transfer_name, selected_leg, lambda_values
    )
    rows.extend(transfer_rows)
    return rows, discovery_records


def run_transfer(
    args: argparse.Namespace,
    scenario: str,
    transfer_name: str,
    selected_leg: Optional[cf.Leg],
    lambda_values: Sequence[float],
) -> Tuple[List[SelectedRow], List[Dict[str, object]]]:
    transfer_data = {transfer_name: make_task_data(args, transfer_name, seed_offset=9)}
    scenario_label = f"{scenario}_transfer"
    rows: List[SelectedRow] = []
    mid_lambda = lambda_values[len(lambda_values) // 2]

    # Inline transfer baseline.
    inline_scores = []
    for lambda_idx, lambda_value in enumerate(lambda_values):
        result = evolve_task(
            args,
            transfer_data[transfer_name],
            cf.inline_actions(),
            [],
            "inline",
            lambda_value,
            seed=args.seed + 5000 + 101 * lambda_idx,
        )
        config = Config(lambda_value, None, {transfer_name: result.genome}, "inline")
        inline_scores.append(score_config(args, config, transfer_data))
    inline_selected = select_by_validation_elbow(inline_scores, args.val_tolerance)
    rows.append(make_row(args, scenario_label, "inline", inline_selected, transfer_data, call_champion_seen=False))

    # Shared / no_share transfer with the frozen library.
    if selected_leg is not None:
        for condition in ("shared", "no_share"):
            scores = []
            call_seen = False
            for lambda_idx, lambda_value in enumerate(lambda_values):
                result = evolve_task(
                    args,
                    transfer_data[transfer_name],
                    cf.controller_actions(1),
                    [selected_leg],
                    condition,
                    lambda_value,
                    seed=args.seed + 6000 + 101 * lambda_idx + (97 if condition == "no_share" else 0),
                )
                call_seen = call_seen or result.saw_call_champion
                config = Config(lambda_value, selected_leg, {transfer_name: result.genome}, condition)
                # Marginal cost: the library was already paid for by the support
                # tasks, matching the repository's transfer protocol.
                scores.append(score_config(args, config, transfer_data, charge_library=False))
            selected = select_by_validation_elbow(scores, args.val_tolerance)
            rows.append(make_row(args, scenario_label, condition, selected, transfer_data, call_seen))

    # Witness transfer.
    witness_leg = cf.witness_seek_leg()
    witness_config = Config(
        mid_lambda,
        witness_leg,
        {transfer_name: cf.witness_gluing(cf.TASKS[transfer_name])},
        "witness",
    )
    witness_score = score_config(args, witness_config, transfer_data, charge_library=False)
    rows.append(make_row(args, scenario_label, "witness", witness_score, transfer_data, call_champion_seen=True))

    # Budget-matched discovery rates (the search half of the cone claim).
    discovery_records: List[Dict[str, object]] = []
    if selected_leg is not None:
        for condition, allowed, library in (
            ("gluing", cf.controller_actions(1), [selected_leg]),
            ("inline", cf.inline_actions(), []),
        ):
            successes = 0
            for replicate in range(args.discovery_replicates):
                result = evolve_task(
                    args,
                    transfer_data[transfer_name],
                    allowed,
                    library,
                    "shared" if condition == "gluing" else "inline",
                    mid_lambda,
                    seed=args.seed + 9000 + 31 * replicate + (17 if condition == "gluing" else 0),
                    population=args.discovery_population,
                    generations=args.discovery_generations,
                )
                config = Config(mid_lambda, library[0] if library else None, {transfer_name: result.genome},
                                "shared" if condition == "gluing" else "inline")
                score = score_config(args, config, transfer_data, charge_library=False)
                if score.train_solved == 1 and score.val_solved == 1:
                    successes += 1
            discovery_records.append(
                {
                    "scenario": scenario,
                    "transfer_task": transfer_name,
                    "search": condition,
                    "replicates": args.discovery_replicates,
                    "successes": successes,
                    "population": args.discovery_population,
                    "generations": args.discovery_generations,
                    "lambda": mid_lambda,
                }
            )
    return rows, discovery_records


def run_growth(args: argparse.Namespace, lambda_values: Sequence[float]) -> List[SelectedRow]:
    """Staged library growth: phase 1 evolves a seek leg cold on the sequencing
    support set; phase 2 freezes it (definition already paid) and cold-evolves
    one new leg on the hazard task family. The sequential-colimit predictions:
    phase-2 gluings call the legacy leg under new bindings (cross-generation
    reuse), the new leg is selected only under shared discounting, and the
    no-legacy ablation pays more or fails more."""
    rows: List[SelectedRow] = []
    mid_lambda = lambda_values[len(lambda_values) // 2]

    def joint_kwargs() -> Dict[str, object]:
        return dict(
            population_size=args.joint_population,
            generations=args.joint_generations,
            state_count=args.state_count,
            max_rules=args.max_rules,
            max_rule_length=args.max_rule_length,
            mutation_rate=args.mutation_rate,
            call_cost=args.call_cost,
            max_steps=args.max_steps,
        )

    # Phase 1: cold cone on the sequencing support set (shared accounting),
    # lambda-swept and elbow-selected like every other stage. A single
    # unvetted phase-1 run produced a mediocre legacy leg that poisoned all of
    # phase 2 (see the leg-robustness study: leg quality gates the cone).
    phase1_names = ("forage", "homing", "forage_then_home")
    phase1_data = {
        name: make_task_data(args, name, seed_offset=idx + 1)
        for idx, name in enumerate(phase1_names)
    }
    def naturality_probe(leg: cf.Leg) -> Tuple[bool, float]:
        """Library admission test: the same leg, called under FOOD and HOME
        bindings, on phase-1 validation levels (hidden levels stay untouched).
        Only behaviorally natural legs deserve to become diagram objects: the
        leg-robustness study showed probe failure predicts downstream
        discovery failure, and an already-paid unnatural leg is a trap for
        marginal accounting."""
        total_loss = 0.0
        solved = True
        for task_name, channel in (("forage", cf.FOOD_CHANNEL), ("homing", cf.HOME_CHANNEL)):
            controller = cf.ConeGenome(
                state_count=2,
                rules=[cf.ConeRule(0, cf.ANY_OBS, (cf.call_action(0, channel),), 1)],
            )
            evaluation = cf.evaluate_cone_task(
                controller, [leg], phase1_data[task_name].validation,
                phase1_data[task_name].task, max_steps=args.max_steps,
            )
            total_loss += evaluation.loss
            solved = solved and evaluation.solved
        return solved, total_loss

    phase1_candidates: List[Tuple[bool, float, ConfigScore]] = []
    phase1_call_seen = False
    for lambda_idx, lambda_value in enumerate(lambda_values):
        for restart in range(2):
            phase1 = cf.evolve_joint_cone(
                {name: (phase1_data[name].task, phase1_data[name].train) for name in phase1_names},
                "shared",
                lambda_value,
                seed=args.seed + 41_000 + 101 * lambda_idx + 7177 * restart,
                **joint_kwargs(),
            )
            phase1_call_seen = phase1_call_seen or phase1.saw_call_champion
            leg = cf.Leg("legacy_seek", phase1.legs[0])
            config = Config(
                lambda_value, leg, dict(phase1.controllers),
                "shared", saw_call_champion=phase1.saw_call_champion,
            )
            natural, probe_loss = naturality_probe(leg)
            phase1_candidates.append((natural, probe_loss, score_config(args, config, phase1_data)))
    admission_passed = any(natural for natural, _loss, _score in phase1_candidates)
    admitted = (
        [item for item in phase1_candidates if item[0]] if admission_passed else phase1_candidates
    )
    phase1_selected = min(admitted, key=lambda item: (item[1], item[2].complexity))[2]
    if not admission_passed:
        print("growth_phase1: WARNING no phase-1 leg passed the naturality admission probe")
    legacy = phase1_selected.config.leg
    rows.append(
        make_row(
            args, "growth_phase1", "shared_joint", phase1_selected, phase1_data, phase1_call_seen,
        )
    )

    # Phase 2: hazard task family with the legacy leg frozen and callable.
    phase2_names = ("flee", "forage_flee", "flee_then_home")
    phase2_data = {
        name: make_task_data(args, name, seed_offset=idx + 20)
        for idx, name in enumerate(phase2_names)
    }
    phase2_levels = {
        name: (phase2_data[name].task, phase2_data[name].train) for name in phase2_names
    }

    joint_conditions = (
        # label, accounting, frozen legs, free indices
        ("shared_growth", "shared", (legacy,), frozenset({0})),
        ("shared_no_legacy", "shared", (), frozenset()),
        ("no_share_growth", "no_share", (legacy,), frozenset()),
    )
    for label, condition, frozen, free_indices in joint_conditions:
        scores: List[ConfigScore] = []
        call_seen = False
        for lambda_idx, lambda_value in enumerate(lambda_values):
            result = cf.evolve_joint_cone(
                phase2_levels,
                condition,
                lambda_value,
                seed=args.seed + 43_000 + 101 * lambda_idx + 7 * len(frozen)
                + (97 if condition == "no_share" else 0),
                frozen_legs=frozen,
                evolved_legs=1,
                **joint_kwargs(),
            )
            call_seen = call_seen or result.saw_call_champion
            config = Config(
                lambda_value,
                cf.Leg("new_leg", result.legs[0]),
                dict(result.controllers),
                condition,
                saw_call_champion=result.saw_call_champion,
                extra_legs=tuple(frozen),
                free_leg_indices=free_indices,
            )
            scores.append(score_config(args, config, phase2_data))
        selected = select_by_validation_elbow(scores, args.val_tolerance)
        rows.append(make_row(args, "growth_phase2", label, selected, phase2_data, call_seen))

    # Inline baseline for phase 2.
    inline_scores: List[ConfigScore] = []
    for lambda_idx, lambda_value in enumerate(lambda_values):
        genomes = {}
        for task_idx, name in enumerate(phase2_names):
            result = evolve_task(
                args, phase2_data[name], cf.inline_actions(), [], "inline", lambda_value,
                seed=args.seed + 47_000 + 13 * (task_idx + 1) + 101 * lambda_idx,
            )
            genomes[name] = result.genome
        inline_scores.append(
            score_config(args, Config(lambda_value, None, genomes, "inline"), phase2_data)
        )
    inline_selected = select_by_validation_elbow(inline_scores, args.val_tolerance)
    rows.append(
        make_row(args, "growth_phase2", "inline", inline_selected, phase2_data, call_champion_seen=False)
    )

    # Witness floor for phase 2 (full cost, both legs charged).
    witness_config = Config(
        mid_lambda,
        cf.witness_flee_leg(),
        {name: cf.witness_gluing(phase2_data[name].task, seek_index=0, flee_index=1) for name in phase2_names},
        "witness",
        extra_legs=(cf.witness_seek_leg(),),
    )
    rows.append(
        make_row(
            args, "growth_phase2", "witness",
            score_config(args, witness_config, phase2_data),
            phase2_data, call_champion_seen=True,
        )
    )
    return rows


def main() -> None:
    args = parse_args()
    if args.quick:
        args.population = min(args.population, 40)
        args.generations = min(args.generations, 18)
        args.discovery_replicates = min(args.discovery_replicates, 3)
    lambda_values = tuple(float(item) for item in args.lambda_values.split(",") if item.strip())
    scenarios = tuple(SCENARIOS) if args.scenario == "all" else (args.scenario,)

    print(CSV_HEADER)
    all_rows: List[SelectedRow] = []
    all_discovery: List[Dict[str, object]] = []
    for scenario in scenarios:
        rows, discovery = run_scenario(args, scenario, lambda_values)
        all_rows.extend(rows)
        all_discovery.extend(discovery)
        for row in rows:
            print_row(row)

    if args.growth:
        growth_rows = run_growth(args, lambda_values)
        all_rows.extend(growth_rows)
        for row in growth_rows:
            print_row(row)

    if all_discovery:
        print("\ndiscovery,scenario,transfer_task,search,successes,replicates,population,generations,lambda")
        for record in all_discovery:
            print(
                f"discovery,{record['scenario']},{record['transfer_task']},{record['search']},"
                f"{record['successes']},{record['replicates']},{record['population']},"
                f"{record['generations']},{record['lambda']:.4f}"
            )

    if args.show_rules:
        for row in all_rows:
            print(f"\n{row.scenario} / {row.condition} / leg={row.leg}")
            for owner, rules in row.rules.items():
                print(f"  [{owner}]")
                for rule in rules:
                    print(f"    {rule}")

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "rows": [
            {key: value for key, value in vars(row).items()}
            for row in all_rows
        ],
        "discovery": all_discovery,
        "settings": {
            "lambda_values": list(lambda_values),
            "population": args.population,
            "generations": args.generations,
            "call_cost": args.call_cost,
            "seed": args.seed,
        },
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
