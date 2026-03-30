#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT.parent / ".venv" / "bin" / "python"
BASE_CONFIG = ROOT / "config" / "production_strict_retrieval_datafix_rankreg_choice_embed256.yaml"
ARTIFACTS = ROOT / "artifacts" / "strict_retrieval_sweep"
RESULTS_TSV = ARTIFACTS / "results.tsv"


@dataclass
class SweepExperiment:
    name: str
    updates: dict[str, Any]
    embed_override: int | None = None


def nested_update(data: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            nested_update(data[key], value)
        else:
            data[key] = value
    return data


def slugify(name: str) -> str:
    cleaned = []
    for char in name.lower():
        cleaned.append(char if char.isalnum() else "-")
    slug = "".join(cleaned)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def ensure_results_header() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if RESULTS_TSV.exists():
        return
    RESULTS_TSV.write_text(
        "run_id\tstatus\tavg_known\tavg_known_exact\tavg_known_paraphrase\tavg_margin\tavg_ood\tgap\tsynthesis\tavg_known_conf\tavg_ood_conf\n"
    )


def append_result(run_id: str, status: str, summary: dict[str, Any] | None) -> None:
    if summary is None:
        row = f"{run_id}\t{status}\t0\t0\t0\t0\t0\t0\t0\t0\t0\n"
    else:
        row = (
            f"{run_id}\t{status}\t"
            f"{summary.get('avg_known_similarity', 0):.4f}\t"
            f"{summary.get('avg_known_exact_similarity', 0):.4f}\t"
            f"{summary.get('avg_known_paraphrase_similarity', 0):.4f}\t"
            f"{summary.get('avg_known_margin', 0):.4f}\t"
            f"{summary.get('avg_ignorant_similarity', 0):.4f}\t"
            f"{summary.get('ignorance_gap', 0):.4f}\t"
            f"{summary.get('synthesis_similarity', 0):.4f}\t"
            f"{summary.get('avg_known_confidence', 0):.4f}\t"
            f"{summary.get('avg_ood_confidence', 0):.4f}\n"
        )
    with RESULTS_TSV.open("a") as handle:
        handle.write(row)


def load_base_config() -> dict[str, Any]:
    return yaml.safe_load(BASE_CONFIG.read_text())


def experiment_queue() -> list[SweepExperiment]:
    return [
        SweepExperiment(
            "embed256 baseline",
            {},
            embed_override=256,
        ),
        SweepExperiment(
            "embed256 rankreg 0.01",
            {"phase4": {"rank_reg_weight": 0.01}},
            embed_override=256,
        ),
        SweepExperiment(
            "embed256 rankreg 0.02",
            {"phase4": {"rank_reg_weight": 0.02}},
            embed_override=256,
        ),
        SweepExperiment(
            "embed256 rankreg 0.00",
            {"phase4": {"rank_reg_weight": 0.0}},
            embed_override=256,
        ),
        SweepExperiment(
            "embed256 lower ood/clf",
            {"phase4": {"ood_weight": 0.05, "clf_weight": 0.15}},
            embed_override=256,
        ),
        SweepExperiment(
            "embed256 higher ood/clf",
            {"phase4": {"ood_weight": 0.2, "clf_weight": 0.35}},
            embed_override=256,
        ),
        SweepExperiment(
            "embed256 margin weight 1.0",
            {"phase4": {"retrieval_margin_weight": 1.0}},
            embed_override=256,
        ),
        SweepExperiment(
            "embed256 margin weight 0.4",
            {"phase4": {"retrieval_margin_weight": 0.4}},
            embed_override=256,
        ),
        SweepExperiment(
            "embed256 query/pred spread 0.2",
            {"phase4": {"query_spread_weight": 0.2, "pred_spread_weight": 0.2}},
            embed_override=256,
        ),
        SweepExperiment(
            "embed256 query multiview",
            {"phase4": {"query_multiview_weight": 0.1, "query_multiview_prediction_weight": 0.1}},
            embed_override=256,
        ),
        SweepExperiment(
            "embed256 momentum queue",
            {"phase4": {"use_momentum_queue": True, "momentum_queue_weight": 0.1, "momentum_queue_prediction_weight": 0.1}},
            embed_override=256,
        ),
        SweepExperiment(
            "embed256 family prototypes",
            {
                "phase4": {
                    "use_family_prototypes": True,
                    "prototype_weight": 0.1,
                    "prototype_target": "family",
                    "prototype_code_weight": 0.1,
                    "prototype_prediction_weight": 0.1,
                    "prototype_repulsion_weight": 0.05,
                }
            },
            embed_override=256,
        ),
    ]


def run_command(command: list[str]) -> int:
    import subprocess

    return subprocess.call(command, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-experiments", type=int, default=999)
    parser.add_argument("--match", type=str, default="")
    args = parser.parse_args()

    ensure_results_header()
    queue = experiment_queue()
    if args.match:
        match = args.match.lower()
        queue = [exp for exp in queue if match in exp.name.lower()]
    queue = queue[: args.max_experiments]
    if not queue:
        return 0

    for exp in queue:
        run_id = slugify(exp.name)
        run_dir = ARTIFACTS / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        config = load_base_config()
        nested_update(config, exp.updates)
        config_path = run_dir / "config.yaml"
        config["profile"] = f"strict-retrieval-sweep-{run_id}"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))

        model_path = run_dir / "model.pt"
        print(f"[sweep] {exp.name}")
        train_cmd = [
            str(PYTHON),
            "train_production.py",
            "--config",
            str(config_path),
            "--size",
            "15000000",
            "--output",
            str(model_path),
            "--device",
            "cuda",
        ]
        rc = run_command(train_cmd)
        if rc != 0:
            append_result(run_id, f"train_failed({rc})", None)
            continue

        summary_path = run_dir / "summary.json"
        test_cmd = [
            str(PYTHON),
            "test_2.7b.py",
            "15000000",
            str(model_path),
            "--json",
            "--embed-dim",
            str(config["phase4"].get("embed_dim_override", 0) or 0),
            "--encoder-layers",
            str(config["phase4"].get("encoder_layers_override", 0) or 0),
            "--encoder-heads",
            str(config["phase4"].get("encoder_heads_override", 0) or 0),
            "--predictor-layers",
            str(config["phase4"].get("predictor_layers_override", 0) or 0),
            "--predictor-heads",
            str(config["phase4"].get("predictor_heads_override", 0) or 0),
            "--decoder-layers",
            str(config["phase4"].get("decoder_layers_override", 0) or 0),
            "--decoder-heads",
            str(config["phase4"].get("decoder_heads_override", 0) or 0),
            "--decoder-hidden-dim",
            str(config["phase4"].get("decoder_hidden_dim_override", 0) or 0),
        ]
        import subprocess
        try:
            result = subprocess.check_output(test_cmd, cwd=ROOT)
        except subprocess.CalledProcessError as exc:
            append_result(run_id, f"test_failed({exc.returncode})", None)
            continue
        output = result.decode("utf-8").strip()
        if not output:
            append_result(run_id, "test_failed(empty)", None)
            continue
        # Extract last JSON object printed
        json_start = output.rfind("{")
        if json_start == -1:
            append_result(run_id, "test_failed(no_json)", None)
            continue
        try:
            summary = json.loads(output[json_start:])
        except json.JSONDecodeError:
            append_result(run_id, "test_failed(bad_json)", None)
            continue
        summary_path.write_text(json.dumps(summary, indent=2))
        append_result(run_id, "ok", summary)
        time.sleep(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
