#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

import autorun
from research.phase4_decision_policy import (
    answer_score_from_phase4_result,
    choose_next_candidate,
    dump_candidate_scores,
    judge_scout,
)
from research.phase4_evidence import append_evidence, load_evidence
from research.phase4_search_space import (
    INCUMBENT_CANDIDATE_NAME,
    rigorous_edge_joint_champion_challenger_staged_hard_base,
)

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_REPLICATION_SEEDS = [103, 104, 105]
SCOUT_SEEDS = [106, 107, 108, 109]
INCUMBENT_NAME = INCUMBENT_CANDIDATE_NAME


def incumbent_phase4() -> dict[str, Any]:
    return rigorous_edge_joint_champion_challenger_staged_hard_base()


def build_experiment(name: str, seed: int, phase4_updates: dict[str, Any]) -> autorun.Experiment:
    return autorun.Experiment(
        f"{name} seed{seed}",
        {
            "profile": "robustness",
            "seed": seed,
            "phase4": copy.deepcopy(phase4_updates),
        },
    )


def existing_ok_descriptions() -> set[str]:
    return {row.get("description", "") for row in autorun.parse_results_table() if row.get("status") == "ok"}


def run_single(base_config: dict[str, Any], exp: autorun.Experiment, run_index: int) -> tuple[str, dict[str, Any] | None, int]:
    run_id, results, returncode = autorun.run_experiment(base_config, exp, run_index)
    status = "ok" if results is not None else f"failed({returncode})"
    autorun.append_result(run_id, status, results, exp.name)
    return run_id, results, returncode


def load_phase4_result_for_description(description: str) -> dict[str, Any] | None:
    for row in autorun.parse_results_table():
        if row.get("description") != description or row.get("status") != "ok":
            continue
        results = autorun.load_run_results(row["run_id"])
        if results is None:
            return None
        return results.get("phase4")
    return None


def ensure_incumbent(base_config: dict[str, Any], run_index: int, log_handle) -> tuple[int, dict[str, Any]]:
    description = f"{INCUMBENT_NAME} seed{SCOUT_SEEDS[0]}"
    incumbent = load_phase4_result_for_description(description)
    if incumbent is not None:
        return run_index, incumbent
    exp = build_experiment(INCUMBENT_NAME, SCOUT_SEEDS[0], incumbent_phase4())
    log_handle.write(f"run {run_index}: {exp.name}\n")
    log_handle.flush()
    _, results, _ = run_single(base_config, exp, run_index)
    if results is None:
        raise RuntimeError("Incumbent scout failed; cannot continue autoresearch loop")
    append_evidence(
        {
            "timestamp": time.time(),
            "candidate_name": INCUMBENT_NAME,
            "stage": "scout",
            "hypothesis_id": "H1",
            "decision": "baseline",
            "seed": SCOUT_SEEDS[0],
            "answer_score": answer_score_from_phase4_result(results["phase4"]),
            "description": exp.name,
        }
    )
    return run_index + 1, results["phase4"]


def next_unused_seed(ok_descriptions: set[str], base_name: str, preferred: list[int]) -> int | None:
    for seed in preferred:
        if f"{base_name} seed{seed}" not in ok_descriptions:
            return seed
    return None


def run_replication_block(base_config: dict[str, Any], candidate_name: str, phase4_updates: dict[str, Any], run_index: int, log_handle) -> tuple[int, list[dict[str, Any]]]:
    ok_descriptions = existing_ok_descriptions()
    block_results: list[dict[str, Any]] = []
    for seed in BLOCKED_REPLICATION_SEEDS:
        exp = build_experiment(candidate_name, seed, phase4_updates)
        if exp.name in ok_descriptions:
            result = load_phase4_result_for_description(exp.name)
            if result is not None:
                block_results.append(result)
            continue
        log_handle.write(f"run {run_index}: {exp.name}\n")
        log_handle.flush()
        _, results, _ = run_single(base_config, exp, run_index)
        if results is not None:
            block_results.append(results["phase4"])
        run_index += 1
    return run_index, block_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    autorun.ensure_results_header()
    history_rows = autorun.parse_results_table()
    evidence_rows = load_evidence()
    base_config = yaml.safe_load(autorun.BASE_CONFIG.read_text())
    run_index = autorun.next_run_index(history_rows)

    if args.dry_run:
        print(dump_candidate_scores(history_rows, evidence_rows))
        return 0

    with autorun.AUTORUN_LOG.open("a") as log_handle:
        log_handle.write(f"start {time.strftime('%Y-%m-%d %H:%M:%S')} strategy=phase4_autoresearch max_cycles={args.max_cycles}\n")
        run_index, incumbent_phase4_result = ensure_incumbent(base_config, run_index, log_handle)
        incumbent_score = answer_score_from_phase4_result(incumbent_phase4_result)

        completed_cycles = 0
        while completed_cycles < args.max_cycles:
            history_rows = autorun.parse_results_table()
            evidence_rows = load_evidence()
            candidate_score = choose_next_candidate(history_rows, evidence_rows)
            if candidate_score is None:
                log_handle.write("no candidate available\n")
                break
            candidate = candidate_score.candidate
            ok_descriptions = existing_ok_descriptions()
            scout_seed = next_unused_seed(ok_descriptions, candidate.name, SCOUT_SEEDS)
            if scout_seed is None:
                log_handle.write(f"no scout seed available for {candidate.name}\n")
                break
            scout_exp = build_experiment(candidate.name, scout_seed, candidate.phase4_updates)
            log_handle.write(f"run {run_index}: {scout_exp.name}\n")
            log_handle.flush()
            _, results, _ = run_single(base_config, scout_exp, run_index)
            run_index += 1
            if results is None:
                append_evidence(
                    {
                        "timestamp": time.time(),
                        "candidate_name": candidate.name,
                        "stage": "scout",
                        "hypothesis_id": candidate.hypothesis_id,
                        "decision": "kill",
                        "seed": scout_seed,
                        "reason": "run_failed",
                        "description": scout_exp.name,
                    }
                )
                completed_cycles += 1
                continue

            scout_phase4 = results["phase4"]
            decision, scout_score = judge_scout(incumbent_score, scout_phase4)
            evidence_entry: dict[str, Any] = {
                "timestamp": time.time(),
                "candidate_name": candidate.name,
                "stage": "scout",
                "hypothesis_id": candidate.hypothesis_id,
                "intervention_type": candidate.intervention_type,
                "decision": decision,
                "seed": scout_seed,
                "answer_score": scout_score,
                "incumbent_answer_score": incumbent_score,
                "description": scout_exp.name,
                "rationale": candidate.rationale,
                "expected_effect": candidate.expected_effect,
            }
            append_evidence(evidence_entry)

            if decision == "promote":
                run_index, block_results = run_replication_block(base_config, candidate.name, candidate.phase4_updates, run_index, log_handle)
                if block_results:
                    block_score = sum(answer_score_from_phase4_result(result) for result in block_results) / len(block_results)
                    replicate_decision = "promote" if block_score >= incumbent_score + 0.10 else "kill"
                    append_evidence(
                        {
                            "timestamp": time.time(),
                            "candidate_name": candidate.name,
                            "stage": "replication",
                            "hypothesis_id": candidate.hypothesis_id,
                            "decision": replicate_decision,
                            "answer_score": block_score,
                            "incumbent_answer_score": incumbent_score,
                            "seeds": BLOCKED_REPLICATION_SEEDS,
                            "description": candidate.name,
                        }
                    )
                    if replicate_decision == "promote":
                        incumbent_score = block_score
            completed_cycles += 1
        log_handle.write(f"end {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
