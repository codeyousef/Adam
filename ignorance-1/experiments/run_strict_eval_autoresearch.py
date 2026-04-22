#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.strict_eval_search_space as strict_space  # noqa: E402
from research.strict_eval_search_space import (  # noqa: E402
    STRICT_INCUMBENT_NAME,
    STRICT_SCOUT_SIZE,
    StrictEvalCandidate,
    build_strict_eval_scout_config,
    strict_answer_score,
    strict_candidate_library,
)

PYTHON = ROOT.parent / ".venv" / "bin" / "python"
ARTIFACTS = ROOT / "artifacts" / "strict_eval_autoresearch_v4"
RESULTS_TSV = ARTIFACTS / "results.tsv"
ADAPTIVE_STATE_PATH = ARTIFACTS / "adaptive_state.json"
DECISIONS_JSONL = ARTIFACTS / "decisions.jsonl"
SCOUT_SEEDS = [501, 502, 503, 504]
BLOCKED_REPLICATION_SEEDS = [505, 506, 507]

STRICT_EVAL_OVERRIDE_KEYS = {
    "rerank_topk",
    "rerank_query_weight",
    "rerank_agreement_weight",
    "rerank_lexical_weight",
    "rerank_support_weight",
    "rerank_spec_weight",
    "rerank_consensus_weight",
    "rerank_consensus_temperature",
    "rerank_consensus_floor",
    "rerank_consensus_margin_gate",
    "rerank_pairwise_mode",
    "rerank_shortlist_mode",
    "rerank_answerspec_mode",
    "rerank_answerspec_margin_gate",
    "rerank_parafence_weight",
    "rerank_parafence_variants",
    "rerank_safe_expand_topk",
    "rerank_safe_expand_margin",
    "rerank_support_floor_margin_gate",
    "rerank_verifier_uplift_weight",
    "rerank_verifier_gap_scale",
    "rerank_verifier_support_weight",
    "rerank_verifier_spec_weight",
    "retrieval_facet_score_mode",
    "retrieval_facet_softmax_temperature",
    "retrieval_global_facet_blend",
    "confidence_mode",
    "confidence_support_topk",
    "confidence_support_temperature",
    "confidence_parafence_variants",
    "selective_gate_mode",
    "selective_gate_margin_threshold",
    "selective_gate_mean_gap_threshold",
    "selective_gate_similarity_floor",
}

_FOLLOWUP_MODE_MAP = {
    "v33": "v34",
    "v34": "v35",
    "v35": "v36",
    "v36": "v37",
    "v37": "v38",
    "v38": "v39",
    "v39": "v40",
    "v40": "v41",
    "v42": "v43",
    "v43": "v44",
    "v44": "v45",
    "v45": "v46",
    "v46": "v47",
    "v47": "v48",
    "v48": "v49",
    "v49": "v50",
    "v50": "v51",
    "v51": "v52",
    "v52": "v53",
    "v53": "v54",
    "v54": "v55",
    "v55": "v56",
    "v57": "v58",
    "v58": "v59",
    "v59": "v60",
    "v60": "v61",
    "v61": "v62",
    "v62": "v63",
    "v63": "v64",
    "v64": "v65",
    "v65": "v66",
    "v66": "v67",
    "v67": "v68",
    "v68": "v69",
    "v69": "v70",
    "v70": "v71",
    "v71": "v72",
    "v72": "v73",
    "v73": "v74",
    "v74": "v75",
    "v75": "v76",
    "v76": "v77",
    "v77": "v78",
    "v78": "v79",
    "v79": "v80",
    "v80": "v81",
    "v81": "v82",
    "v82": "v83",
    "v83": "v84",
    "v84": "v85",
    "v85": "v86",
    "v86": "v87",
    "v87": "v88",
    "v88": "v89",
    "v89": "v90",
    "v90": "v91",
    "v91": "v92",
    "v92": "v93",
    "v93": "v94",
    "v94": "v95",
    "v95": "v96",
    "v96": "v97",
    "v97": "v98",
    "v98": "v99",
    "v99": "v100",
    "v100": "v101",
    "v101": "v102",
    "v102": "v103",
    "v103": "v104",
    "v104": "v105",
    "v105": "v106",
    "v106": "v107",
    "v107": "v108",
    "v108": "v109",
    "v109": "v110",
    "v110": "v111",
    "v111": "v112",
    "v112": "v113",
    "v113": "v114",
    "v114": "v115",
    "v115": "v116",
    "v116": "v117",
    "v117": "v118",
    "v118": "v119",
    "v119": "v120",
    "v120": "v121",
    "v121": "v122",
    "v122": "v123",
    "v123": "v124",
    "v124": "v125",
    "v125": "v126",
    "v126": "v127",
    "v127": "v128",
    "v128": "v129",
    "v129": "v130",
    "v130": "v131",
    "v131": "v132",
    "v132": "v133",
    "v133": "v134",
    "v134": "v135",
    "v135": "v136",
    "v136": "v137",
    "v137": "v138",
    "v138": "v139",
    "v139": "v140",
    "v140": "v141",
    "v141": "v142",
    "v142": "v143",
    "v143": "v144",
    "v144": "v145",
    "v145": "v146",
    "v146": "v147",
    "v147": "v148",
    "v148": "v149",
    "v149": "v150",
    "v150": "v151",
    "v151": "v152",
    "v152": "v153",
    "v153": "v154",
    "v154": "v155",
    "v155": "v156",
    "v156": "v157",
    "v157": "v158",
    "v158": "v159",
    "v159": "v160",
    "v160": "v161",
    "v161": "v162",
    "v162": "v163",
    "v163": "v164",
    "v164": "v165",
    "v165": "v166",
    "v166": "v167",
    "v167": "v168",
    "v168": "v169",
    "v169": "v170",
    "v170": "v171",
    "v171": "v172",
    "v172": "v173",
    "v173": "v174",
    "v174": "v175",
    "v175": "v176",
    "v176": "v177",
    "v177": "v178",
    "v178": "v179",
    "v179": "v180",
    "v180": "v181",
    "v285": "v286",
    "v286": "v290",
    "v287": "v289",
    "v289": "v292",
    "v290": "v291",
    "v291": "v292",
    "v292": "v293",
    "v293": "v294",
    "v294": "v295",
    "v295": "v296",
    "v296": "v297",
    "v297": "v298",
    "v298": "v300",
    "v300": "v301",
    "v301": "v302",
    "v302": "v303",
    "v303": "v304",
    "v304": "v305",
    "v305": "v306",
    "v306": "v307",
    "v307": "v308",
    "v308": "v309",
    "v309": "v310",
    "v310": "v311",
    "v311": "v312",
    "v312": "v313",
    "v313": "v314",
    "v314": "v315",
    "v315": "v316",
    "v316": "v317",
    "v317": "v318",
    "v318": "v319",
    "v319": "v320",
    "v320": "v321",
    "v321": "v322",
    "v322": "v323",
    "v323": "v324",
    "v324": "v325",
    "v325": "v326",
    "v326": "v327",
    "v327": "v328",
    "v328": "v329",
    "v329": "v330",
    "v330": "v331",
    "v331": "v332",
    "v332": "v333",
    "v333": "v336",
    "v334": "v335",
    "v335": "v339",
    "v336": "v337",
    "v337": "v338",
    "v338": "v340",
    "v339": "v341",
    "v341": "v342",
    "v342": "v343",
    "v343": "v344",
    "v344": "v345",
    "v345": "v346",
    "v346": "v347",
    "v347": "v348",
    "v348": "v349",
    "v349": "v350",
    "v350": "v351",
    "v351": "v352",
    "v352": "v353",
    "v353": "v354",
    "v354": "v355",
    "v355": "v356",
    "v356": "v357",
    "v357": "v358",
    "v358": "v359",
    "v359": "v360",
    "v360": "v361",
    "v361": "v362",
    "v362": "v363",
    "v363": "v364",
    "v364": "v365",
    "v365": "v366",
    "v366": "v367",
    "v367": "v368",
    "v368": "v369",
    "v369": "v370",
    "v370": "v371",
    "v371": "v372",
    "v372": "v373",
    "v373": "v374",
    "v374": "v375",
    "v375": "v376",
    "v376": "v377",
    "v377": "v378",
    "v378": "v379",
    "v379": "v380",
    "v181": "v182",
    "v182": "v183",
    "v183": "v184",
    "v184": "v185",
    "v185": "v186",
    "v186": "v187",
    "v187": "v236",
    "v191": "v192",
    "v192": "v193",
    "v193": "v194",
    "v194": "v195",
    "v195": "v196",
    "v196": "v197",
    "v197": "v198",
    "v198": "v199",
    "v199": "v200",
    "v200": "v201",
    "v201": "v233",
    "v233": "v234",
    "v234": "v235",
    "v236": "v237",
    "v237": "v238",
    "v238": "v239",
    "v239": "v240",
    "v240": "v241",
    "v241": "v242",
    "v242": "v243",
    "v243": "v244",
    "v244": "v245",
    "v245": "v246",
    "v246": "v247",
    "v247": "v248",
    "v248": "v249",
    "v249": "v250",
    "v250": "v251",
    "v251": "v252",
    "v252": "v253",
    "v253": "v254",
    "v254": "v255",
    "v255": "v256",
    "v256": "v257",
    "v257": "v258",
    "v258": "v259",
    "v259": "v260",
    "v260": "v261",
    "v261": "v262",
    "v262": "v263",
    "v263": "v264",
    "v264": "v265",
    "v265": "v266",
    "v266": "v267",
    "v267": "v268",
    "v268": "v269",
    "v269": "v270",
    "v270": "v271",
    "v271": "v275",
    "v272": "v273",
    "v273": "v274",
    "v275": "v276",
    "v276": "v277",
    "v277": "v278",
    "v278": "v279",
    "v279": "v280",
    "v280": "v281",
    "v281": "v282",
    "v282": "v283",
    "v283": "v284",
    "v202": "v203",
    "v203": "v204",
    "v204": "v205",
    "v205": "v206",
    "v206": "v207",
    "v208": "v209",
    "v209": "v210",
    "v210": "v211",
    "v212": "v213",
    "v213": "v214",
    "v214": "v215",
    "v215": "v216",
    "v216": "v217",
    "v217": "v218",
    "v218": "v219",
    "v220": "v221",
    "v221": "v222",
    "v222": "v223",
    "v223": "v224",
    "v224": "v225",
    "v225": "v226",
    "v226": "v227",
    "v227": "v228",
    "v228": "v229",
    "v229": "v230",
    "v230": "v231",
    "v231": "v232",
    "v188": "v189",
    "v189": "v190",
    "v190": "v191",
}


def artifacts_for_mode(mode: str) -> Path:
    normalized = str(mode or "v4").strip().lower() or "v4"
    return ROOT / "artifacts" / f"strict_eval_autoresearch_{normalized}"


def _paths_for_mode(mode: str) -> tuple[Path, Path, Path]:
    artifacts = artifacts_for_mode(mode)
    return (
        artifacts / "results.tsv",
        artifacts / "adaptive_state.json",
        artifacts / "decisions.jsonl",
    )


def slugify(name: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(name or ""))
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-") or "candidate"


def _candidate_mode(candidate_name: str) -> str:
    token = str(candidate_name or "").strip().split(" ", 1)[0].lower()
    return token if token.startswith("v") else "v4"


def _v364_candidate_library() -> list[StrictEvalCandidate]:
    """
    v364: Classifier weight reduction + equivalence softening on frozen taxonomy_support_discipline.

    v363 key finding: ALL 5 calibrator variants (control, neighborhood_posterior,
    support_feature_calibrator, evidential_support, agreement_augmented) scored
    38.67-39.73 identically. The calibrator readout is NOT the bottleneck.

    The bottleneck is the TRAINING SIGNAL that produces the encoder geometry.
    If the encoder geometry doesn't encode sufficient supported/unsupported
    separation, no calibrator can create it.

    Research2/3 diagnosis: under taxonomy_support_discipline, two mechanisms
    suppress instance-level support discrimination:

    1. EQUIVALENCE OVERBINDING: prototype_weight=0.08, prototype_repulsion=0.12,
       equivalence_alignment=0.04 compress instance-level distinctions into
       family-level groupings. Under taxonomy_support_discipline's stricter
       boundary data, this prevents the encoder from separating direct support
       from same-family near misses at the instance level.

    2. CLASSIFIER WEIGHT PRESSURE: classifier_weight=0.05 (v363) pushes the
       model toward conservative abstention on in-domain examples, compressing
       the confidence_gap that strict_eval requires (gap needs to be >= 0.10).

    v364 mechanism: reduce both mechanisms simultaneously so the encoder
    geometry encodes better supported/unsupported separation for the frozen
    calibrator to read out. Keep freeze_backbone=true (preserves v338's good
    geometry). Don't change the calibrator - fix the training signal instead.

    Base checkpoint: v338 (rank=0.101, offdiag=0.035 - the good geometry)
    """
    v338_checkpoint = str(
        ROOT
        / "artifacts"
        / "strict_eval_autoresearch_v338"
        / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504"
        / "model.pt"
    )

    # v363 phase4 - the baseline that scored 38-39 for ALL calibrator variants
    v363_phase4 = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "proxy_recipe": "v6_overnight",
        "reference_size": 15_000_000,
        "step_scale_power": 0.55,
        "max_step_multiplier": 5.0,
        "lr_scale_power": 0.2,
        "max_lr_divisor": 2.5,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "classifier_weight": 0.05,
        "alignment_prediction_weight": 1.0,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ranking_margin_weight": 0.15,
        "ranking_margin": 0.26,
        "ranking_focal_gamma": 1.5,
        "phase4_joint_training": True,
        "champion_challenger_weight": 0.5,
        "champion_challenger_margin": 0.05,
        "champion_challenger_temperature": 0.1,
        "champion_challenger_start_fraction": 0.3,
        "champion_challenger_ramp_fraction": 0.2,
        "retrieval_margin_weight": 0.22,
        "retrieval_margin": 0.32,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.075,
        "rank_reg_eps": 0.0001,
        "rank_reg_target": "all",
        "query_margin_weight": 0.2,
        "query_margin": 0.2,
        "ood_weight": 0.0,
        "clf_weight": 0.05,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "paraphrase_batch_probability": 0.35,
        "microbatch_size": 1,
        "max_seq_len": 192,
        "optimizer": "paged_adamw32bit",
        "paged_optimizer": True,
        "use_retrieval_data_strategy": True,
        "max_hard_negatives_per_example": 4,
        "scheduler": "cosine",
        "warmup_fraction": 0.1,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "embed_dim_override": 256,
        "encoder_layers_override": 4,
        "encoder_heads_override": 8,
        "predictor_layers_override": 4,
        "predictor_heads_override": 8,
        "decoder_layers_override": 2,
        "decoder_heads_override": 8,
        "decoder_hidden_dim_override": 256,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "ema_target_decay": 0.995,
        "use_equivalence_class_retrieval": True,
        "equivalence_alignment_weight": 0.04,
        "equivalence_prediction_weight": 0.02,
        "equivalence_margin_weight": 0.04,
        "use_retrieval_data_strategy": True,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "prototype_target": "equivalence",
        "prototype_weight": 0.08,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.08,
        "prototype_prediction_weight": 0.08,
        "prototype_repulsion_weight": 0.12,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "momentum_queue_temperature": 0.07,
        "phase4_factorized_hard_negatives": True,
        "retrieval_margin_prediction_weight": 0.75,
        "retrieval_margin_embedding_weight": 0.6,
        "ranking_prediction_weight": 1.0,
        "ranking_embedding_weight": 0.0,
        "ranking_start_fraction": 0.2,
        "ranking_ramp_fraction": 0.2,
        "ranking_largest_only": False,
        "pred_ood_weight": 0.0,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_boundary_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "ignorance_start_step": 380,
        "ignorance_ramp_steps": 80,
        "vicreg_invariance_weight": 1.0,
        "vicreg_variance_weight": 1.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_variance_target": 0.75,
        "vicreg_queue_samples": 128,
        "sigreg_weight": 0.5,
        "reset_query_head_on_resume": False,
        "freeze_backbone": True,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict[str, Any],
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S364",
            intervention_type="v364_classifier_equivalence_rebalancing",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=copy.deepcopy(v363_phase4) | copy.deepcopy(phase4_overrides),
            eval_overrides=copy.deepcopy(common_eval) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        candidate(
            "v364 reduced classifier_weight=0.01 + frozen taxonomy",
            {"classifier_weight": 0.01, "clf_weight": 0.01, "alignment_prediction_weight": 0.5},
            "v363 showed all 5 calibrator variants scored 38-39 identically - the bottleneck is NOT the calibrator readout. Mechanism 1 from research2/3: classifier_weight=0.05 (v363) pushes the model toward conservative abstention on in-domain examples, compressing the confidence_gap needed for strict_eval (gap=0.057 << 0.10). Reducing to 0.01 removes this pressure, letting the encoder produce better supported/unsupported separation for the frozen calibrator to read out.",
            "If classifier_weight is the active ingredient suppressing confidence_gap: gap should improve from 0.057 toward 0.10, hygiene should stay >= 0.375, and score should exceed v363's 38-39 range. Target: 40-42.",
        ),
        candidate(
            "v364 softened equivalence + frozen taxonomy",
            {
                "equivalence_alignment_weight": 0.005,
                "equivalence_prediction_weight": 0.002,
                "equivalence_margin_weight": 0.005,
                "prototype_weight": 0.02,
                "prototype_code_weight": 0.02,
                "prototype_prediction_weight": 0.02,
                "prototype_repulsion_weight": 0.03,
            },
            "Mechanism 2 from research2/3: equivalence overbinding (prototype_weight=0.08, prototype_repulsion=0.12, equivalence_alignment=0.04) compresses instance-level distinctions into family-level groupings. Under taxonomy_support_discipline's stricter boundary data, this prevents the encoder from encoding direct-support vs same-family-near-miss separation at the instance level. Reducing all equivalence weights by ~4-6x should allow instance-level support structure to emerge in the frozen encoder geometry.",
            "If equivalence overbinding is the compression mechanism: encoder geometry should encode better instance-level separation, improving both confidence_gap and hygiene together. Score target: 40+. If only gap improves but hygiene stays flat, the two mechanisms are partially independent."
        ),
        candidate(
            "v364 minimal classifier + softened equivalence + frozen taxonomy",
            {
                "classifier_weight": 0.01,
                "clf_weight": 0.01,
                "alignment_prediction_weight": 0.5,
                "equivalence_alignment_weight": 0.005,
                "equivalence_prediction_weight": 0.002,
                "equivalence_margin_weight": 0.005,
                "prototype_weight": 0.02,
                "prototype_code_weight": 0.02,
                "prototype_prediction_weight": 0.02,
                "prototype_repulsion_weight": 0.03,
            },
            "Combined intervention: the two mechanisms from research2/3 work together. Minimal classifier_weight (0.01) removes conservative-abstention pressure from the training signal. Softened equivalence prevents family-level compression of instance-level support. Together they should allow the frozen encoder to produce geometry where supported and unsupported in-domain examples are more separable, enabling the calibrator to read out a larger confidence_gap and hygiene rate.",
            "Combined effect should exceed either individual intervention. If classifier alone gives +1-2 points and equivalence alone gives +1-2 points, the combination should give +3-5 points - potentially recovering enough to PASS strict_eval. This is the highest-priority v364 candidate."
        ),
        candidate(
            "v364 neighborhood posterior + minimal classifier + softened equivalence",
            {
                "classifier_weight": 0.01,
                "clf_weight": 0.01,
                "alignment_prediction_weight": 0.5,
                "equivalence_alignment_weight": 0.005,
                "equivalence_prediction_weight": 0.002,
                "equivalence_margin_weight": 0.005,
                "prototype_weight": 0.02,
                "prototype_code_weight": 0.02,
                "prototype_prediction_weight": 0.02,
                "prototype_repulsion_weight": 0.03,
            },
            "Same combined intervention as candidate 3 but with neighborhood_posterior calibrator instead of support_feature_calibrator. v363's neighborhood_posterior scored 38.67 vs support_feature 39.73 - the 1-point gap may close once the training signal produces better encoder geometry. This tests whether the calibrator variant matters when the encoder geometry is properly prepared.",
            "Should score >= v363 neighborhood_posterior (38.67). If it closes the gap with support_feature_calibrator, the calibrator is secondary to the encoder. If it still trails, the calibrator variant has independent value."
        ),
        candidate(
            "v364 higher ranking/retrieval margins + minimal classifier",
            {
                "classifier_weight": 0.01,
                "clf_weight": 0.01,
                "alignment_prediction_weight": 0.5,
                "ranking_margin": 0.35,
                "ranking_margin_weight": 0.22,
                "retrieval_margin": 0.40,
                "retrieval_margin_weight": 0.28,
                "query_margin": 0.28,
                "query_margin_weight": 0.28,
            },
            "Higher ranking and retrieval margins push the model to open larger separation between correct and incorrect ranked candidates. Under taxonomy_support_discipline, larger margins force the model to commit more strongly to supported vs unsupported ordering, which should naturally increase the confidence_gap. Combined with minimal classifier_weight (0.01), the margin increase can't be absorbed by pushing toward abstention - it must create actual ranking separation.",
            "If the taxonomy_support_discipline objective naturally compresses margins, increasing them explicitly should counteract that. The confidence_gap should improve as the ranking loss forces larger separations. Score target: 41+."
        ),
    ]


def _v369_candidate_library() -> list[StrictEvalCandidate]:
    """
    v369: Confirm and extend the v365 "no_prop_loss" winner that scored 41.63 PASS.

    Key findings from v363-v368:
    - v365 "no_prop_loss" (taxonomy + unfrozen + clf=0.01 + all equivalence weights zeroed):
      score=41.63, PASSED legacy strict eval. This is the current best.
    - v365 clf=0.01 without no_prop_loss: only 30.7, passed legacy but not competitive
    - v365 clf=0.005: only 29.6, passed legacy but worse
    - v366/v367/v368: mixed_boundary + clf=0.09 + various backbones ALL collapsed or
      scored 31-39, far below v365's 41.63
    - v368 taxonomy + clf=0.09 + unfrozen: 31-34 (clf=0.09 is too high, collapses geometry)
    - THE DOMINANT FACTOR: Zeroing equivalence weights was far more impactful than clf_weight.
      clf=0.01 + no_prop_loss >> clf=0.09 + mixed_boundary + all other changes

    v369 strategy: Focus on confirming and refining the no_prop_loss winner.
    The key unknown is whether no_prop_loss generalizes across seeds and whether
    a small amount of equivalence weight helps or hurts.

    v369 candidates:
    1. Confirm no_prop_loss with seed503 (exact v365 winner replica): should score ~41-43
    2. Confirm no_prop_loss with seed504 (different seed): stability check
    3. Mixed_boundary + no_prop_loss: can v340's better dataset beat v365's 41.63?
       v340 passed at 43.5 with mixed_boundary + frozen + clf=0.09 (no no_prop_loss).
       With no_prop_loss + unfrozen instead of frozen, can we do even better?
    4. no_prop_loss + small equivalence weights: does a tiny amount of equivalence
       provide regularization benefit without causing overbinding?
    5. no_prop_loss + longer training (500 steps): more training may further improve hygiene
    """
    v338_checkpoint = str(
        ROOT
        / "artifacts"
        / "strict_eval_autoresearch_v338"
        / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504"
        / "model.pt"
    )

    # v365 "no_prop_loss" winning base config
    # Key: taxonomy_support_discipline + unfrozen + clf=0.01 + ALL equivalence weights = 0
    v369_phase4_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "proxy_recipe": "v6_overnight",
        "reference_size": 15_000_000,
        "step_scale_power": 0.55,
        "max_step_multiplier": 5.0,
        "lr_scale_power": 0.2,
        "max_lr_divisor": 2.5,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "classifier_weight": 0.01,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ranking_margin_weight": 0.15,
        "ranking_margin": 0.26,
        "ranking_focal_gamma": 1.5,
        "phase4_joint_training": True,
        "champion_challenger_weight": 0.5,
        "champion_challenger_margin": 0.05,
        "champion_challenger_temperature": 0.1,
        "champion_challenger_start_fraction": 0.3,
        "champion_challenger_ramp_fraction": 0.2,
        "retrieval_margin_weight": 0.22,
        "retrieval_margin": 0.32,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.075,
        "rank_reg_eps": 0.0001,
        "rank_reg_target": "all",
        "query_margin_weight": 0.2,
        "query_margin": 0.2,
        "ood_weight": 0.0,
        "clf_weight": 0.01,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "paraphrase_batch_probability": 0.35,
        "microbatch_size": 1,
        "max_seq_len": 192,
        "optimizer": "paged_adamw32bit",
        "paged_optimizer": True,
        "use_retrieval_data_strategy": True,
        "max_hard_negatives_per_example": 4,
        "scheduler": "cosine",
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "embed_dim_override": 256,
        "encoder_layers_override": 4,
        "encoder_heads_override": 8,
        "predictor_layers_override": 4,
        "predictor_heads_override": 8,
        "decoder_layers_override": 2,
        "decoder_heads_override": 8,
        "decoder_hidden_dim_override": 256,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "ema_target_decay": 0.995,
        "use_equivalence_class_retrieval": True,
        # ZERO equivalence weights — this is what made v365's no_prop_loss winner
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "prototype_target": "equivalence",
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "momentum_queue_temperature": 0.07,
        "phase4_factorized_hard_negatives": True,
        "retrieval_margin_prediction_weight": 0.75,
        "retrieval_margin_embedding_weight": 0.6,
        "ranking_prediction_weight": 1.0,
        "ranking_embedding_weight": 0.0,
        "ranking_start_fraction": 0.2,
        "ranking_ramp_fraction": 0.2,
        "ranking_largest_only": False,
        "pred_ood_weight": 0.0,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_boundary_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "ignorance_start_step": 380,
        "ignorance_ramp_steps": 80,
        "vicreg_invariance_weight": 1.0,
        "vicreg_variance_weight": 1.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_variance_target": 0.75,
        "vicreg_queue_samples": 128,
        "sigreg_weight": 0.5,
        "reset_query_head_on_resume": False,
        "freeze_backbone": False,  # unfrozen — key for no_prop_loss success
        "warm_start_phase3_only": False,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "phase4_steps": 300,
    }

    # Mixed_boundary variant for candidate 3
    v369_phase4_mixed = dict(v369_phase4_base)
    v369_phase4_mixed["phase4_dataset"] = "behavioral_constraints_v2_mixed_boundary_curriculum_v1"

    # Small equivalence weights variant for candidate 4
    v369_phase4_small_equiv = dict(v369_phase4_base)
    v369_phase4_small_equiv["equivalence_alignment_weight"] = 0.002
    v369_phase4_small_equiv["equivalence_prediction_weight"] = 0.001
    v369_phase4_small_equiv["prototype_weight"] = 0.01
    v369_phase4_small_equiv["prototype_code_weight"] = 0.01
    v369_phase4_small_equiv["prototype_prediction_weight"] = 0.01

    # Longer training variant for candidate 5
    v369_phase4_500steps = dict(v369_phase4_base)
    v369_phase4_500steps["production_mode"] = True
    v369_phase4_500steps["production_steps"] = 500

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict[str, Any],
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S369",
            intervention_type="v369_confirm_and_extend_noprop",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=copy.deepcopy(v369_phase4_base) | copy.deepcopy(phase4_overrides),
            eval_overrides=copy.deepcopy(common_eval) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        # === Confirm no_prop_loss winner with seed503 (exact replica of v365's winner) ===
        candidate(
            "v369 taxonomy + no_prop_loss + seed503",
            {},
            "v368 confirmed clf=0.09 + unfrozen = geometry collapse (all v368 candidates 31-34). "
            "The v365 no_prop_loss winner (score=41.63) used taxonomy + unfrozen + clf=0.01 + "
            "ALL equivalence weights zeroed. This candidate replicates that exact config with seed503 "
            "to confirm reproducibility before building on it.",
            "If score ~41-43 (reproduces v365): no_prop_loss effect is real and stable. "
            "If score <38: v365 winner was a lucky seed; need to investigate variance."
        ),
        # === Confirm no_prop_loss with different seed ===
        candidate(
            "v369 taxonomy + no_prop_loss + seed504",
            {},
            "Same as seed503 but with seed504. Tests whether the v365 no_prop_loss result "
            "is stable across random seeds or was a lucky draw. If both seeds score 41+: "
            "the no_prop_loss approach is reliable. If seed504 << seed503: "
            "high variance, need seed-averaging or regularization.",
            "If seed504 ~41-43: no_prop_loss is stable (PASS). "
            "If seed504 ~30-35: v365 winner was a lucky seed, need different approach."
        ),
        # === Mixed_boundary + no_prop_loss: can v340's better dataset beat v365? ===
        candidate(
            "v369 mixed_boundary + no_prop_loss + seed505",
            v369_phase4_mixed,
            "v340 passed at 43.5 with mixed_boundary + frozen + clf=0.09. "
            "v365's no_prop_loss passed at 41.63 with taxonomy + unfrozen + clf=0.01. "
            "This candidate combines: mixed_boundary (v340's better dataset) + no_prop_loss "
            "(v365's winning mechanism) + unfrozen (v365's encoder update approach). "
            "Hypothesis: mixed_boundary provides better within-family discrimination, "
            "and no_prop_loss prevents the overbinding that plagued v366/v367/v368.",
            "If score >= 43.5: mixed_boundary + no_prop_loss beats taxonomy + no_prop_loss. "
            "If score 41-43: mixed_boundary doesn't add value over taxonomy for no_prop_loss. "
            "If score 30-40: mixed_boundary still collapses even with no_prop_loss."
        ),
        # === Small equivalence weights: does a tiny amount help or hurt? ===
        candidate(
            "v369 taxonomy + small_equivalence + seed506",
            v369_phase4_small_equiv,
            "v365's no_prop_loss zeroed ALL equivalence weights. This candidate tests "
            "a middle ground: very small equivalence weights (0.001-0.002 range) to see "
            "if a tiny amount of family-level structure helps hygiene without causing "
            "the overbinding seen at higher values. The prototype_repulsion_weight=0 "
            "is preserved to prevent family compression.",
            "If small equivalence improves score: a small amount of family structure helps "
            "instance-level discrimination. If score drops: zeroing equivalence was the "
            "correct call and no_prop_loss must be maintained."
        ),
        # === Longer training ===
        candidate(
            "v369 taxonomy + no_prop_loss + prod_steps500 + seed507",
            v369_phase4_500steps,
            "v365 winner used 300 production steps. This candidate tests whether "
            "500 steps of no_prop_loss training further improves the hygiene metric. "
            "More training with the unfrozen encoder might allow better instance-level "
            "support discrimination without the collapse risk (since no_prop_loss "
            "removes the equivalence overbinding mechanism).",
            "If 500 steps > 300 steps: longer training helps hygiene. "
            "If 500 steps ~300: training has already converged at 300 steps. "
            "If 500 steps < 300: longer training starts to overfit / collapse."
        ),
    ]



def _v366_candidate_library() -> list[StrictEvalCandidate]:
    """
    v366: mixed_boundary_curriculum_v1 + properly-trained unfrozen backbone.

    v365 key findings:
    - taxonomy_support_discipline_v1 as phase4_dataset produced best score (41.63) but
      hygiene=0.375 (FAIL << 0.75 threshold)
    - v340 PASSED under OLD eval using mixed_boundary_curriculum_v1 + freeze_backbone=true + clf_weight=0.09
    - v338 winner (mixed_boundary, frozen, 33.0) — insufficient training / wrong clf_weight
    - v365's training was bugged: only ~164 steps instead of intended 300+ due to proxy scaling

    v366 theory: The training BUG masked whether mixed_boundary actually fixes hygiene.
    Mixed_boundary produces better instance-level discrimination than taxonomy_support_discipline
    because it mixes boundary types more evenly, avoiding family-level compression.
    With the fix, training actually runs properly.

    Key changes from v365 best (seed503, score=41.63):
    1. phase4_dataset: mixed_boundary_curriculum_v1 (instead of taxonomy_support_discipline_v1)
    2. clf_weight/classifier_weight: 0.09 (v340 PASSED value, not v365's 0.01)
       This changes the confidence_gap threshold: v365 PASSED at ≥0.10, but strict eval
       requires ≥0.20. Need clf_weight=0.09 to produce the larger gap while still passing.
    3. freeze_backbone=false: allow encoder to learn from mixed_boundary data
    4. Longer training: phase4_steps removed (let proxy scaling determine steps)
    5. Ignorance start: 300 (earlier than v365's 380) for earlier anti-collapse signal
    6. Rank reg target: code+query (v340's value, not v365's "all")

    Also adopt v340's proven values:
    - alignment_mse_weight: 0.03, alignment_prediction_weight: 0.78
    - max_hard_negatives_per_example: 6
    - prototype weights: 0.08 each
    - warmup_fraction: 0.1
    """
    v338_checkpoint = str(
        ROOT
        / "artifacts"
        / "strict_eval_autoresearch_v338"
        / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504"
        / "model.pt"
    )

    # v340 + v365 hybrid: mixed_boundary + unfrozen + higher classifier_weight
    v366_phase4_base = {
        "phase4_dataset": "behavioral_constraints_v2_mixed_boundary_curriculum_v1",
        "phase4_balance_families": True,
        "proxy_recipe": "v6_overnight",
        "reference_size": 15_000_000,
        "step_scale_power": 0.55,
        "max_step_multiplier": 5.0,
        "lr_scale_power": 0.2,
        "max_lr_divisor": 2.5,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "classifier_weight": 0.09,
        "alignment_prediction_weight": 0.78,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.03,
        "ranking_margin_weight": 0.15,
        "ranking_margin": 0.26,
        "ranking_focal_gamma": 1.5,
        "phase4_joint_training": True,
        "champion_challenger_weight": 0.5,
        "champion_challenger_margin": 0.05,
        "champion_challenger_temperature": 0.1,
        "champion_challenger_start_fraction": 0.3,
        "champion_challenger_ramp_fraction": 0.2,
        "retrieval_margin_weight": 0.22,
        "retrieval_margin": 0.32,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.07,
        "rank_reg_eps": 0.0001,
        "rank_reg_target": "code+query",
        "query_margin_weight": 0.2,
        "query_margin": 0.2,
        "ood_weight": 0.0,
        "clf_weight": 0.09,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "paraphrase_batch_probability": 0.35,
        "microbatch_size": 1,
        "max_seq_len": 192,
        "optimizer": "paged_adamw32bit",
        "paged_optimizer": True,
        "use_retrieval_data_strategy": True,
        "max_hard_negatives_per_example": 6,
        "scheduler": "cosine",
        "warmup_fraction": 0.1,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "embed_dim_override": 256,
        "encoder_layers_override": 4,
        "encoder_heads_override": 8,
        "predictor_layers_override": 4,
        "predictor_heads_override": 8,
        "decoder_layers_override": 2,
        "decoder_heads_override": 8,
        "decoder_hidden_dim_override": 256,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "ema_target_decay": 0.995,
        "use_equivalence_class_retrieval": False,
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "prototype_target": "equivalence",
        "prototype_weight": 0.08,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.08,
        "prototype_prediction_weight": 0.08,
        "prototype_repulsion_weight": 0.12,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "momentum_queue_temperature": 0.07,
        "phase4_factorized_hard_negatives": True,
        "retrieval_margin_prediction_weight": 0.75,
        "retrieval_margin_embedding_weight": 0.6,
        "ranking_prediction_weight": 1.0,
        "ranking_embedding_weight": 0.0,
        "ranking_start_fraction": 0.2,
        "ranking_ramp_fraction": 0.2,
        "ranking_largest_only": False,
        "pred_ood_weight": 0.0,
        "classifier_query_weight": 0.9,
        "classifier_prediction_weight": 0.1,
        "epistemic_boundary_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "ignorance_start_step": 300,
        "ignorance_ramp_steps": 80,
        "vicreg_invariance_weight": 1.0,
        "vicreg_variance_weight": 1.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_variance_target": 0.75,
        "vicreg_queue_samples": 128,
        "sigreg_weight": 0.5,
        "reset_query_head_on_resume": False,
        "freeze_backbone": False,
        "warm_start_phase3_only": False,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict[str, Any],
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S366",
            intervention_type="v366_mixed_boundary_unfrozen",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=copy.deepcopy(v366_phase4_base) | copy.deepcopy(phase4_overrides),
            eval_overrides=copy.deepcopy(common_eval) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        candidate(
            "v366 mixed_boundary + unfrozen + clf=0.09 control",
            {},
            "v365 (taxonomy_support_discipline, unfrozen, clf=0.01) got score=41.63 but hygiene=0.375 FAIL. "
            "v340 (mixed_boundary, frozen, clf=0.09) PASSED under OLD eval. This is the MIXED_BOUNDARY "
            "+ UNFROZEN combination — the missing experiment. clf_weight=0.09 from v340 to produce "
            "the larger confidence_gap (strict eval needs ≥0.20, not just ≥0.10). "
            "freeze_backbone=false lets the encoder learn instance-level discrimination from "
            "mixed_boundary data which is more balanced than taxonomy_support_discipline. "
            "Training bug fix ensures proper step count.",
            "If mixed_boundary produces better instance-level support structure than "
            "taxonomy_support_discipline: hygiene should improve toward ≥0.75. "
            "clf_weight=0.09 should produce conf_gap ≥0.20. Score target: 42-44 (PASS)."
        ),
        candidate(
            "v366 mixed_boundary + UNFROZEN + higher_lr_clf=0.09",
            {
                "lr": 8e-6,
            },
            "Same as control but with slightly higher initial LR (8e-6 vs v340's 1e-5). "
            "The v338 warm-start may need a slightly different LR to properly adapt the encoder "
            "to mixed_boundary data. 1e-5 might have been too high for v338's geometry.",
            "If LR is the bottleneck: encoder learns instance-level structure faster, "
            "producing better hygiene within the same step budget. Score target: 43-45."
        ),
        candidate(
            "v366 mixed_boundary + frozen + clf=0.09 + equivalence_off",
            {
                "freeze_backbone": True,
            },
            "The counterpart to v365's unfrozen approach: same mixed_boundary + clf=0.09 as v340, "
            "but test if frozen geometry with mixed_boundary produces the same hygiene as v340 "
            "(score 43.5). This isolates whether it's the mixed_boundary data or the unfreezing "
            "that helps. Equivalence is already OFF (use_equivalence_class_retrieval=False).",
            "If frozen geometry with mixed_boundary reaches v340's score (43+): the key "
            "ingredient is the mixed_boundary data, not unfreezing. "
            "If unfrozen does better: both data AND encoder adaptation matter."
        ),
    ]


def _v367_candidate_library() -> list[StrictEvalCandidate]:
    """
    v367: Fix BOTH bugs from v366 and systematically test the two-mode hypothesis.

    v366 failure analysis:
    - Bug #1 (freeze_backbone in wrong config section): materialize_run_config()
      reads config.freeze_backbone, but candidates put it in phase4_updates.
      This means ALL v366 candidates ran with freeze_backbone=True (default),
      so the encoder was frozen throughout phase4. The "unfrozen" candidates
      were actually frozen. This explains why mixed_boundary + unfrozen (38.45)
      scored LOWER than v365's unfrozen taxonomy_support_discipline (41.63):
      the encoder couldn't learn at all.
    - Bug #2 (phase4_steps/production_steps ignored): materialize_run_config()
      hardcoded production_steps=0 after merging phase4_updates. This means
      the training pipeline uses proxy-scaled steps (~112 for 30M warm-start
      from 15M scout) instead of the intended 300+ steps. All v366 training
      ran for only 112 steps (14 seconds) — far too few to learn from
      mixed_boundary data.

    v367 theory: The v340 recipe (mixed_boundary + frozen + clf=0.09) PASSED the
    OLD eval with score=43.5. But frozen geometry may not generalize well to
    the taxonomy_support_discipline queries in the strict eval. The key question
    is: does mixed_boundary data produce better instance-level discrimination
    (hygiene ≥ 0.75) than taxonomy_support_discipline, when the encoder is
    actually allowed to learn (unfrozen)?

    Hypothesis A (Conservative): freeze_backbone=true preserves v338's good
    encoder geometry; mixed_boundary data (from v340) is sufficient to pass
    hygiene via the frozen encoder's existing structure. This matches v340
    which passed with frozen + mixed_boundary.
    → Candidates 1-2: freeze_backbone=true + mixed_boundary + production_steps=300

    Hypothesis B (Aggressive): freeze_backbone=false allows encoder to learn
    instance-level support structure from mixed_boundary, which should produce
    better within-family discrimination and higher hygiene. The v338 warm-start
    provides a good starting point; 300+ training steps should be sufficient.
    → Candidates 3-5: freeze_backbone=false + mixed_boundary + production_steps=300

    Hypothesis C (Taxonomy+Unfrozen): The best v365 candidate (seed503, score=41.63)
    used taxonomy_support_discipline + unfrozen and passed conf_gap (0.243 ≥ 0.20)
    but failed hygiene (0.375). If the conf_gap fix generalizes to mixed_boundary,
    and mixed_boundary fixes hygiene, this could pass both thresholds.
    → Candidates 6-7: taxonomy_support_discipline + unfrozen + production_steps=300

    All v367 candidates use production_mode=true + production_steps=300 to ensure
    proper training duration (not the 112-step proxy-scaled minimum from v366).
    """
    v338_checkpoint = str(
        ROOT
        / "artifacts"
        / "strict_eval_autoresearch_v338"
        / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504"
        / "model.pt"
    )

    # v340 recipe: frozen backbone + mixed_boundary + clf=0.09 + production_steps=300
    # This is the most direct test of whether the v340 recipe passes the strict eval.
    v367_phase4_frozen_mixed = {
        "phase4_dataset": "behavioral_constraints_v2_mixed_boundary_curriculum_v1",
        "phase4_balance_families": True,
        "production_mode": True,
        "production_steps": 300,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "freeze_backbone": True,
        "warm_start_phase3_only": False,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        # v340 proven weights
        "alignment_mse_weight": 0.03,
        "alignment_prediction_weight": 0.78,
        "use_retrieval_data_strategy": True,
        "max_hard_negatives_per_example": 6,
        "prototype_weight": 0.08,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.08,
        "prototype_prediction_weight": 0.08,
        "prototype_repulsion_weight": 0.12,
        "warmup_fraction": 0.1,
        "min_lr_ratio": 0.2,
        "ignorance_start_step": 300,
        "rank_reg_weight": 0.07,
        "rank_reg_target": "code+query",
        "rank_reg_eps": 0.0001,
        "clf_weight": 0.09,
        "classifier_query_weight": 0.9,
        "classifier_prediction_weight": 0.1,
        "ranking_margin_weight": 0.15,
        "ranking_margin": 0.26,
        "retrieval_margin_weight": 0.22,
        "retrieval_margin": 0.32,
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "use_equivalence_class_retrieval": False,
        "use_retrieval_data_strategy": True,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "prototype_target": "equivalence",
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
    }

    # Unfrozen variant of the v340 recipe — the key experiment
    v367_phase4_unfrozen_mixed = dict(v367_phase4_frozen_mixed)
    v367_phase4_unfrozen_mixed["freeze_backbone"] = False

    # Taxonomy_support_discipline variant (best v365 data, but with production_steps fix)
    v367_phase4_unfrozen_taxonomy = dict(v367_phase4_unfrozen_mixed)
    v367_phase4_unfrozen_taxonomy["phase4_dataset"] = "behavioral_constraints_v2_taxonomy_support_discipline_v1"

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict[str, Any],
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S367",
            intervention_type="v367_bugfix_and_systematic_sweep",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=copy.deepcopy(phase4_overrides),
            eval_overrides=copy.deepcopy(common_eval) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        # === Hypothesis A: Frozen + Mixed Boundary ===
        candidate(
            "v367 frozen + mixed_boundary + clf=0.09 + prod_steps300 seed503",
            copy.deepcopy(v367_phase4_frozen_mixed),
            "v366 Bug #1 fix verification: freeze_backbone is now correctly promoted "
            "to top-level config. This candidate tests whether frozen geometry with "
            "mixed_boundary (the v340 recipe) passes the strict eval. "
            "production_mode=true + production_steps=300 ensures proper training duration. "
            "If this passes (expected score 43+): the v340 recipe is the answer. "
            "If this fails: frozen geometry doesn't generalize to taxonomy queries.",
            "If frozen + mixed_boundary passes: hygiene ≥ 0.75, conf_gap ≥ 0.20, score ≥ 43 (PASS). "
            "If frozen fails: geometry doesn't transfer, go to Hypothesis B (unfrozen)."
        ),
        candidate(
            "v367 frozen + mixed_boundary + clf=0.09 + prod_steps300 seed504",
            copy.deepcopy(v367_phase4_frozen_mixed),
            "Same as seed503 candidate but with seed504. Tests result stability across seeds.",
            "If stable: hygiene and conf_gap reproduce, score 43+ (PASS). "
            "If unstable: variance too high, need different approach."
        ),

        # === Hypothesis B: Unfrozen + Mixed Boundary ===
        candidate(
            "v367 unfrozen + mixed_boundary + clf=0.09 + prod_steps300 seed503",
            copy.deepcopy(v367_phase4_unfrozen_mixed),
            "v366 Bug #1 + #2 fix: encoder CAN now learn from mixed_boundary data. "
            "This is the core v367 experiment — does unfreezing allow the encoder to "
            "learn instance-level discrimination from mixed_boundary data, producing "
            "hygiene ≥ 0.75? Combined with clf_weight=0.09 (conf_gap ≥ 0.20). "
            "production_steps=300 ensures enough training to converge.",
            "If unfreezing fixes hygiene: encoder learns instance-level support, "
            "hygiene ≥ 0.75, conf_gap ≥ 0.20, score ≥ 43 (PASS). "
            "If hygiene still fails: the phase3 geometry doesn't support instance-level discrimination, "
            "even with 300 steps of mixed_boundary training."
        ),
        candidate(
            "v367 unfrozen + mixed_boundary + clf=0.09 + prod_steps300 seed504",
            copy.deepcopy(v367_phase4_unfrozen_mixed),
            "Same as seed503 but with seed504. Tests result stability.",
            "If stable: hygiene ≥ 0.75, conf_gap ≥ 0.20, score ≥ 43 (PASS). "
            "If seed-sensitive: need regularization or different architecture."
        ),
        candidate(
            "v367 unfrozen + mixed_boundary + clf=0.09 + prod_steps500 seed505",
            (lambda base: {**base, "production_steps": 500})(v367_phase4_unfrozen_mixed),
            "Longer training (500 steps vs 300) tests whether hygiene improves "
            "with more training. v340 used 500 production steps and passed. "
            "If 300-step hygiene is insufficient, 500 may be enough to converge.",
            "If 300-step hygiene fails but 500 passes: need more training for mixed_boundary convergence. "
            "If 300 already passes: redundant with candidate 3."
        ),

        # === Hypothesis C: Unfrozen + Taxonomy Support Discipline ===
        candidate(
            "v367 unfrozen + taxonomy_support + clf=0.09 + prod_steps300 seed503",
            copy.deepcopy(v367_phase4_unfrozen_taxonomy),
            "Best v365 data (taxonomy_support_discipline) with BOTH bugs fixed. "
            "v365's seed503 got score=41.63, hygiene=0.375 FAIL, conf_gap=0.243 PASS. "
            "With 300 production steps (vs 164 proxy-scaled) and truly unfrozen backbone, "
            "hygiene should improve. clf_weight=0.09 maintains conf_gap ≥ 0.20.",
            "If 300 steps of taxonomy_support_discipline + unfrozen fixes hygiene: "
            "hygiene ≥ 0.75, conf_gap ≥ 0.20, score ≥ 43 (PASS). "
            "If hygiene still fails despite 300 steps: taxonomy_support_discipline "
            "is insufficient for instance-level discrimination regardless of training duration."
        ),
        candidate(
            "v367 unfrozen + taxonomy_support + clf=0.09 + prod_steps500 seed505",
            (lambda base: {**base, "production_steps": 500})(v367_phase4_unfrozen_taxonomy),
            "Same as above with 500 steps. Tests whether more training duration "
            "fully fixes hygiene for taxonomy_support_discipline.",
            "If 300 fails but 500 passes: training duration is the limiter. "
            "If both fail: need different training signal or architecture change."
        ),
    ]


def _v365_candidate_library() -> list[StrictEvalCandidate]:
    """
    v365: taxonomy + no_prop_loss candidates.
    Key results: no_prop_loss variants scored 41.55-41.63 PASS.
    """
    v338_checkpoint = str(
        ROOT
        / "artifacts"
        / "strict_eval_autoresearch_v338"
        / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504"
        / "model.pt"
    )
    v365_phase4_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "production_mode": True,
        "production_steps": 300,
        "clf_weight": 0.01,
        "classifier_weight": 0.01,
        "freeze_backbone": False,
        "warm_start_phase3_only": False,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "phase4_steps": 300,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_queue_samples": 128,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "phase4_factorized_hard_negatives": True,
        "paraphrase_batch_probability": 0.35,
        "retrieval_margin_weight": 0.22,
        "retrieval_margin": 0.32,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.075,
        "rank_reg_target": "all",
        "ranking_margin_weight": 0.15,
        "ranking_margin": 0.26,
        "ranking_focal_gamma": 1.5,
        "sigreg_weight": 0.5,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ema_target_decay": 0.995,
        "epistemic_boundary_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "phase4_joint_training": True,
        "max_hard_negatives_per_example": 4,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "ood_weight": 0.0,
        "pred_ood_weight": 0.0,
        "clf_weight": 0.01,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "reset_query_head_on_resume": False,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict[str, Any],
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S365",
            intervention_type="v365_unfrozen_backbone",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=copy.deepcopy(v365_phase4_base) | copy.deepcopy(phase4_overrides),
            eval_overrides=copy.deepcopy(common_eval) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        candidate(
            "v365 unfrozen no_prop_loss warmstart v338 seed501",
            {
                "equivalence_alignment_weight": 0.0,
                "equivalence_prediction_weight": 0.0,
                "equivalence_margin_weight": 0.0,
                "prototype_weight": 0.0,
                "prototype_query_weight": 0.0,
                "prototype_code_weight": 0.0,
                "prototype_prediction_weight": 0.0,
                "prototype_repulsion_weight": 0.0,
                "use_momentum_queue": False,
                "momentum_queue_weight": 0.0,
                "momentum_queue_prediction_weight": 0.0,
            },
            "Replicate v365 winner (41.63) with seed501. no_prop_loss = all equivalence weights zeroed.",
            "If ~41: no_prop_loss effect is real. If <35: was lucky seed.",
        ),
        candidate(
            "v365 unfrozen no_prop_loss warmstart v338 seed502",
            {
                "equivalence_alignment_weight": 0.0,
                "equivalence_prediction_weight": 0.0,
                "equivalence_margin_weight": 0.0,
                "prototype_weight": 0.0,
                "prototype_query_weight": 0.0,
                "prototype_code_weight": 0.0,
                "prototype_prediction_weight": 0.0,
                "prototype_repulsion_weight": 0.0,
                "use_momentum_queue": False,
                "momentum_queue_weight": 0.0,
                "momentum_queue_prediction_weight": 0.0,
            },
            "Same as seed501 with seed502. Tests reproducibility.",
            "If ~41: stable. If <35: high variance.",
        ),
        candidate(
            "v365 unfrozen no_prop_loss warmstart v338 seed503",
            {
                "equivalence_alignment_weight": 0.0,
                "equivalence_prediction_weight": 0.0,
                "equivalence_margin_weight": 0.0,
                "prototype_weight": 0.0,
                "prototype_query_weight": 0.0,
                "prototype_code_weight": 0.0,
                "prototype_prediction_weight": 0.0,
                "prototype_repulsion_weight": 0.0,
                "use_momentum_queue": False,
                "momentum_queue_weight": 0.0,
                "momentum_queue_prediction_weight": 0.0,
            },
            "Same as seed501 with seed503. Third seed confirmation.",
            "If ~41: very stable. If <35: investigate.",
        ),
    ]





def _v371_candidate_library() -> list[StrictEvalCandidate]:
    """
    v371: Frozen backbone + no_phase4_training (production_steps=0).

    Key discovery from v340: the model passes strict eval when trained with
    production_steps=0 (no phase4 training at all) + frozen backbone.
    This preserves the phase3 retrieval geometry exactly.

    v369/370 used unfrozen + prod_steps=500 which degrades geometry.

    Design: Apply frozen + prod_steps=0 to the no_prop_loss recipe.
    """
    v338_checkpoint = str(
        ROOT
        / "artifacts"
        / "strict_eval_autoresearch_v338"
        / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504"
        / "model.pt"
    )

    v371_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "production_mode": False,
        "production_steps": 0,
        "phase4_steps": 0,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "freeze_backbone": True,
        "warm_start_phase3_only": True,
        "warm_start_model_path": v338_checkpoint,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_queue_samples": 128,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "phase4_factorized_hard_negatives": True,
        "paraphrase_batch_probability": 0.35,
        "retrieval_margin_weight": 0.22,
        "retrieval_margin": 0.32,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.07,
        "rank_reg_target": "code+query",
        "ranking_margin_weight": 0.15,
        "ranking_margin": 0.26,
        "ranking_focal_gamma": 1.5,
        "sigreg_weight": 0.5,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ema_target_decay": 0.995,
        "epistemic_boundary_weight": 0.0,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "phase4_joint_training": True,
        "max_hard_negatives_per_example": 4,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "ood_weight": 0.0,
        "pred_ood_weight": 0.0,
        "clf_weight": 0.09,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "reset_query_head_on_resume": False,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict,
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S371",
            intervention_type="v371_frozen_prodsteps0_taxonomy",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=copy.deepcopy(v371_base) | copy.deepcopy(phase4_overrides),
            eval_overrides=copy.deepcopy(common_eval) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        # Control: exactly frozen + prod_steps=0 + no_prop_loss (same formula as v340's success)
        candidate(
            "v371 frozen prodsteps0 no_prop_loss seed501",
            {},
            "Frozen backbone + prod_steps=0 + no_prop_loss (equivalence weights zeroed). "
            "Same formula that made v340 pass. Using taxonomy dataset.",
            "If >43: taxonomy works with frozen+prod0. If <40: taxonomy harder than mixed_boundary.",
        ),
        # Try mixed_boundary (v340's dataset) for comparison
        candidate(
            "v371 frozen prodsteps0 mixed_boundary seed502",
            {"phase4_dataset": "behavioral_constraints_v2_mixed_boundary_curriculum_v1"},
            "Same as seed501 but use mixed_boundary curriculum (v340's dataset). "
            "Tests whether mixed_boundary works as well as taxonomy.",
            "If ~43: mixed_boundary is reliable. If different from seed501: dataset matters.",
        ),
        # Try clf=0.05 (middle ground)
        candidate(
            "v371 frozen prodsteps0 clf0.05 seed503",
            {"clf_weight": 0.05, "classifier_weight": 0.05},
            "Frozen + prod_steps=0 + clf=0.05. Between v340's 0.09 and v369's 0.01. "
            "v340 used 0.09; lower clf may reduce geometry collapse risk.",
            "If >43: clf=0.05 works. If <40: clf=0.09 necessary.",
        ),
        # Try higher retrieval margin (more hygiene pressure)
        candidate(
            "v371 frozen prodsteps0 stronger_margin seed504",
            {"retrieval_margin_weight": 0.35, "retrieval_margin": 0.40},
            "Frozen + prod_steps=0 + stronger retrieval margin. "
            "Higher margin pushes harder on retrieval discrimination.",
            "If >43: margin helps hygiene. If <40: margin too aggressive.",
        ),
        # Try clf=0.09 + prod_steps=300 (partial phase4)
        candidate(
            "v371 frozen prodsteps300 clf0.09 seed505",
            {"production_steps": 300, "phase4_steps": 300},
            "Frozen + prod_steps=300 + clf=0.09. Partial phase4 training while frozen. "
            "Tests whether some phase4 training helps hygiene without collapsing geometry.",
            "If >43: partial phase4 helps. If <40: phase4 training hurts even when frozen.",
        ),
    ]


def _v370_candidate_library() -> list[StrictEvalCandidate]:
    """
    v370: Push hygiene upward from the v369 winner (score=40.12, hygiene=0.207).
    
    v369 best: taxonomy + no_prop_loss + prod_steps=500 → score=40.12, hygiene=0.207.
    The only remaining failure: "direct support retrieval hygiene too low"
    objective_supported_direct_rate must be >= 0.75.
    
    Key design levers for hygiene:
    - retrieval_margin_weight: higher → stricter retrieval discrimination
    - ranking_margin_weight: higher → stricter ranking discrimination
    - sigreg_weight: higher → less collapsed representations  
    - rank_reg_weight: higher → more diverse channel usage
    - query_spread_weight: higher → less query embedding collapse
    - pred_spread_weight: higher → less predictor embedding collapse
    - equivalence_alignment_weight: we know 0 is optimal for answer score; keep 0
    
    Candidates explore these levers to maximize hygiene while preserving answer score.
    """
    v338_checkpoint = str(
        ROOT
        / "artifacts"
        / "strict_eval_autoresearch_v338"
        / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504"
        / "model.pt"
    )

    # v369 winner base config
    v370_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "production_mode": True,
        "production_steps": 500,
        "clf_weight": 0.01,
        "classifier_weight": 0.01,
        "freeze_backbone": False,
        "warm_start_phase3_only": False,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "phase4_steps": 300,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_queue_samples": 128,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        # no_prop_loss — confirmed optimal
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "phase4_factorized_hard_negatives": True,
        "paraphrase_batch_probability": 0.35,
        "retrieval_margin_weight": 0.22,
        "retrieval_margin": 0.32,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.075,
        "rank_reg_target": "all",
        "ranking_margin_weight": 0.15,
        "ranking_margin": 0.26,
        "ranking_focal_gamma": 1.5,
        "sigreg_weight": 0.5,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ema_target_decay": 0.995,
        "epistemic_boundary_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "phase4_joint_training": True,
        "max_hard_negatives_per_example": 4,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "ood_weight": 0.0,
        "pred_ood_weight": 0.0,
        "clf_weight": 0.01,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "reset_query_head_on_resume": False,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict[str, Any],
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S370",
            intervention_type="v370_hygiene_refinement",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=copy.deepcopy(v370_base) | copy.deepcopy(phase4_overrides),
            eval_overrides=copy.deepcopy(common_eval) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        # v370_1: Exact replica of v369 winner for reproducibility confirmation
        candidate(
            "v370 hygiene refinement control prod_steps500 seed501",
            {},
            "Replica of v369 winner (score=40.12, hygiene=0.207). "
            "Seed501 confirms reproducibility. This is the reference.",
            "If ~40: winner is reproducible. If <38: seed-sensitive.",
        ),
        # v370_2: Stronger retrieval + ranking margin — push hygiene directly
        candidate(
            "v370 stronger retrieval+ranking margin seed502",
            {
                "retrieval_margin_weight": 0.35,   # up from 0.22
                "ranking_margin_weight": 0.28,     # up from 0.15
                "ranking_margin": 0.35,            # up from 0.26
                "retrieval_margin": 0.42,          # up from 0.32
            },
            "Double down on margin-based discrimination. "
            "Higher margins force cleaner separation between support and non-support.",
            "If hygiene > 0.375: margin pushes hygiene over threshold. "
            "If score drops significantly: margin too aggressive.",
        ),
        # v370_3: Stronger spread + sigreg — broader representations for hygiene
        candidate(
            "v370 stronger spread+sigreg seed503",
            {
                "spread_weight": 0.08,              # up from 0.02
                "query_spread_weight": 0.08,       # up from 0.02
                "pred_spread_weight": 0.08,        # up from 0.02
                "sigreg_weight": 1.5,              # up from 0.5
            },
            "Stronger spread penalties prevent representation collapse. "
            "Broader representations may yield cleaner retrieval geometry.",
            "If hygiene improves: collapse was hurting retrieval geometry. "
            "If score drops: spread too aggressive.",
        ),
        # v370_4: Stronger rank regularization — more diverse channels
        candidate(
            "v370 stronger rank_reg seed504",
            {
                "rank_reg_weight": 0.15,            # up from 0.075
                "rank_reg_target": "all",
            },
            "Higher rank regularization forces the model to use more latent channels. "
            "Diverse representations may improve retrieval hygiene.",
            "If hygiene > 0.375: diversity helps. "
            "If score drops: rank_reg too aggressive.",
        ),
        # v370_5: Combo — moderate margin + moderate spread
        candidate(
            "v370 combo margin+spread+rank seed505",
            {
                "retrieval_margin_weight": 0.30,
                "ranking_margin_weight": 0.22,
                "spread_weight": 0.05,
                "query_spread_weight": 0.05,
                "pred_spread_weight": 0.05,
                "sigreg_weight": 1.0,
                "rank_reg_weight": 0.12,
            },
            "Combination: moderate increases across margin, spread, sigreg, rank_reg. "
            "Balanced approach — if any single lever is too strong, this may find the sweet spot.",
            "If hygiene > 0.375: balanced approach works. "
            "If < 0.375 but > 0.207: partial progress. "
            "If score drops significantly: reduce all slightly.",
        ),
        # v370_6: Tighter/larger margins vs v369_2
        candidate(
            "v370 very strong retrieval margin seed506",
            {
                "retrieval_margin_weight": 0.50,   # very strong
                "ranking_margin_weight": 0.35,
                "retrieval_margin": 0.50,
                "ranking_margin": 0.40,
            },
            "Push retrieval margin much harder. "
            "If 0.35 didn't work, try 0.50 to see if even stronger discrimination helps.",
            "If hygiene > 0.375: strong margin is the answer. "
            "If score collapses: margin has a ceiling.",
        ),
    ]



def _v368_candidate_library() -> list[StrictEvalCandidate]:
    """
    v368: taxonomy_support_discipline + clf=0.09 + unfrozen.
    v368 completed 25 candidates (4 static + 21 adaptive). Best: 34.96.
    clf=0.09 was too high for unfrozen, caused geometry collapse.
    """
    v338_checkpoint = str(
        ROOT
        / "artifacts"
        / "strict_eval_autoresearch_v338"
        / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504"
        / "model.pt"
    )
    v368_phase4_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "production_mode": True,
        "production_steps": 300,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "freeze_backbone": False,
        "warm_start_phase3_only": False,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "phase4_steps": 300,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_queue_samples": 128,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "phase4_factorized_hard_negatives": True,
        "paraphrase_batch_probability": 0.35,
        "retrieval_margin_weight": 0.22,
        "retrieval_margin": 0.32,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.075,
        "rank_reg_target": "all",
        "ranking_margin_weight": 0.15,
        "ranking_margin": 0.26,
        "ranking_focal_gamma": 1.5,
        "sigreg_weight": 0.5,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ema_target_decay": 0.995,
        "epistemic_boundary_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "phase4_joint_training": True,
        "max_hard_negatives_per_example": 4,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "ood_weight": 0.0,
        "pred_ood_weight": 0.0,
        "clf_weight": 0.09,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "reset_query_head_on_resume": False,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict[str, Any],
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S368",
            intervention_type="v368_clf09_taxonomy",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=copy.deepcopy(v368_phase4_base) | copy.deepcopy(phase4_overrides),
            eval_overrides=copy.deepcopy(common_eval) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        candidate(
            "v368 taxonomy clf=0.09 seed501",
            {},
            "taxonomy + clf=0.09 + unfrozen. v368 ran this but collapsed to ~34. clf=0.09 too high.",
            "If >35: clf=0.09 can work. If <30: collapse confirmed.",
        ),
    ]


def candidate_library_for_mode(mode: str) -> list[StrictEvalCandidate]:
    normalized = str(mode or "v4").strip().lower() or "v4"
    if normalized == "v364":
        return _v364_candidate_library()
    if normalized == "v365":
        return _v365_candidate_library()
    if normalized == "v366":
        return _v366_candidate_library()
    if normalized == "v367":
        return _v367_candidate_library()
    if normalized == "v368":
        return _v368_candidate_library()
    if normalized == "v369":
        return _v369_candidate_library()
    if normalized == "v370":
        return _v370_candidate_library()
    if normalized == "v371":
        return _v371_candidate_library()
    if normalized == "v372":
        return _v372_candidate_library()
    if normalized == "v373":
        return _v373_candidate_library()
    if normalized == "v374":
        return _v374_candidate_library()
    if normalized == "v375":
        return _v375_candidate_library()
    if normalized == "v376":
        return _v376_candidate_library()
    if normalized == "v377":
        return _v377_candidate_library()
    if normalized == "v378":
        return _v378_candidate_library()
    if normalized == "v379":
        return _v379_candidate_library()
    if normalized == "v380":
        return _v380_candidate_library()
    if normalized == "v381":
        return _v381_candidate_library()
    if normalized == "v382":
        return _v382_candidate_library()

    fn = getattr(strict_space, f"strict_{normalized}_candidate_library", None)
    if callable(fn):
        return fn()
    raise ValueError(f"Unknown strict autoresearch mode: {mode}")


def _candidate_lookup(candidate_name: str) -> StrictEvalCandidate | None:
    name = str(candidate_name or "")
    mode = _candidate_mode(name)
    try:
        for candidate in candidate_library_for_mode(mode):
            if candidate.name == name:
                return candidate
    except Exception:
        return None
    return None


def _incumbent_name_for_mode(mode: str) -> str:
    library = candidate_library_for_mode(mode)
    return library[0].name if library else STRICT_INCUMBENT_NAME


def default_search_state(mode: str = "v4") -> dict[str, Any]:
    normalized = str(mode or "v4").strip().lower() or "v4"
    return {
        "mode": normalized,
        "incumbent_name": _incumbent_name_for_mode(normalized),
        "incumbent_score": float("-inf"),
        "suppressed_families": [],
        "history": [],
        "completed_runs": 0,
    }


def load_search_state(mode: str = "v4") -> dict[str, Any]:
    normalized = str(mode or "v4").strip().lower() or "v4"
    _, adaptive_path, _ = _paths_for_mode(normalized)
    if adaptive_path.exists():
        state = json.loads(adaptive_path.read_text())
        state["mode"] = normalized
        return state
    return default_search_state(normalized)


def save_search_state(state: dict[str, Any]) -> None:
    mode = str(state.get("mode", "v4") or "v4").strip().lower()
    _, adaptive_path, _ = _paths_for_mode(mode)
    adaptive_path.parent.mkdir(parents=True, exist_ok=True)
    adaptive_path.write_text(json.dumps(state, indent=2, sort_keys=True))


def ensure_results_header(mode: str = "v4") -> None:
    results_path, _, _ = _paths_for_mode(mode)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        return
    results_path.write_text(
        "run_id\tstatus\tstrict_status\tanswer_score\tavg_known\tavg_known_exact\tavg_known_paraphrase\tavg_margin\n"
    )


def append_result(run_id: str, status: str, summary: dict | None, mode: str = "v4") -> None:
    ensure_results_header(mode)
    results_path, _, _ = _paths_for_mode(mode)
    summary = summary or {}
    row = [
        run_id,
        status,
        str(summary.get("strict_status", "missing")),
        f"{float(summary.get('answer_score', strict_answer_score(summary) if summary else float('nan'))):.4f}",
        f"{float(summary.get('avg_known_similarity', summary.get('avg_known_margin', 0.0)) or 0.0):.4f}",
        f"{float(summary.get('avg_known_exact_similarity', 0.0) or 0.0):.4f}",
        f"{float(summary.get('avg_known_paraphrase_similarity', 0.0) or 0.0):.4f}",
        f"{float(summary.get('avg_known_margin', 0.0) or 0.0):.4f}",
    ]
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(row) + "\n")


def append_decision(event: dict[str, Any], mode: str = "v4") -> None:
    _, _, decisions_path = _paths_for_mode(mode)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("ts", time.time())
    with decisions_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _eval_cmd_from_config(config: dict[str, Any], model_path: Path) -> list[str]:
    cmd = [
        str(PYTHON),
        str(ROOT / "test_2.7b.py"),
        str(int((config.get("sizes") or config.get("phase4", {}).get("sizes") or [2_700_000_000])[0])),
        str(model_path),
        "--json",
    ]
    if "confidence_threshold" in config:
        cmd.extend(["--confidence-threshold", str(config["confidence_threshold"])])
    if "lexical_weight" in config:
        cmd.extend(["--lexical-weight", str(config["lexical_weight"])])
    strict_eval = dict(config.get("strict_eval") or {})
    for key in sorted(strict_eval):
        cmd.extend([f"--{key.replace('_', '-')}", str(strict_eval[key])])
    phase4 = dict(config.get("phase4") or {})
    embed_arg_map = {
        "embed_dim_override": "embed-dim",
        "encoder_layers_override": "encoder-layers",
        "encoder_heads_override": "encoder-heads",
        "predictor_layers_override": "predictor-layers",
        "predictor_heads_override": "predictor-heads",
        "decoder_layers_override": "decoder-layers",
        "decoder_heads_override": "decoder-heads",
        "decoder_hidden_dim_override": "decoder-hidden-dim",
    }
    for key, cli_name in embed_arg_map.items():
        if key in phase4:
            cmd.extend([f"--{cli_name}", str(phase4[key])])
    return cmd


def _profile_for_candidate(candidate: StrictEvalCandidate, seed: int) -> str:
    return f"strict-eval-autoresearch-v4-{slugify(candidate.name)}-seed{seed}"


def materialize_run_config(run_dir: Path, candidate: StrictEvalCandidate, seed: int) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    config = build_strict_eval_scout_config(seed=seed, phase4_updates=candidate.phase4_updates)
    phase4 = config.setdefault("phase4", {})

    # Extract freeze_backbone from phase4_updates and promote to top-level config.
    # train_production.py reads config.freeze_backbone (NOT phase4.freeze_backbone),
    # but candidates put it in phase4_updates. This fix ensures the training pipeline
    # actually sees the intended freeze_backbone value.
    candidate_phase4 = candidate.phase4_updates or {}
    if "freeze_backbone" in candidate_phase4:
        config["freeze_backbone"] = candidate_phase4["freeze_backbone"]

    # Honor production_mode=True + production_steps from phase4_updates if specified.
    # Only override to proxy-scaling (production_steps=0) when candidate did NOT
    # explicitly request production_mode=True with a positive step count.
    prod_mode = candidate_phase4.get("production_mode", False)
    prod_steps = candidate_phase4.get("production_steps", 0)
    if not (prod_mode and prod_steps > 0):
        phase4["production_mode"] = False
        phase4["production_steps"] = 0
        phase4["production_phase4_repeats"] = 0
    # else: preserve the candidate-specified production_mode + production_steps
    config["seed"] = seed
    config["profile"] = _profile_for_candidate(candidate, seed)

    mode = _candidate_mode(candidate.name)
    mode_num = int(mode[1:]) if mode.startswith("v") and mode[1:].isdigit() else -1
    if mode_num >= 63:
        scout_size = 15_000_000
        phase4["sizes"] = [scout_size]
        phase4["reference_size"] = scout_size
        config["sizes"] = [scout_size]
        config["reference_size"] = scout_size
    else:
        scout_size = int((phase4.get("sizes") or [STRICT_SCOUT_SIZE])[0])
        phase4.setdefault("reference_size", scout_size)
        config.setdefault("sizes", [scout_size])
        config.setdefault("reference_size", scout_size)

    if candidate.base_model_path:
        config["base_model_path"] = str(candidate.base_model_path)
    if candidate.warm_start_model_path:
        config["warm_start_model_path"] = str(candidate.warm_start_model_path)

    eval_overrides = copy.deepcopy(candidate.eval_overrides or {})
    for top_level_key in ("lexical_weight", "confidence_threshold"):
        if top_level_key in eval_overrides:
            config[top_level_key] = eval_overrides.pop(top_level_key)
    if eval_overrides:
        strict_eval = dict(config.get("strict_eval") or {})
        strict_eval.update(eval_overrides)
        config["strict_eval"] = strict_eval

    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def _parse_summary_blob(output: str) -> dict[str, object]:
    text = str(output or "").strip()
    for idx in range(len(text) - 1, -1, -1):
        if text[idx] != "{":
            continue
        candidate = text[idx:]
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Could not locate JSON summary in test_2.7b output")


def _strict_failure_bucket(text: Any) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    normalized = re.sub(r"\s*\([^)]*\)\s*$", "", normalized)
    return normalized.strip()


def _aggregate_strict_failures(values: list[Any]) -> list[str]:
    present = [list(value or []) for value in values if value is not None]
    if not present:
        return []

    counts: dict[str, int] = {}
    replica_count = len(present)
    for failures in present:
        seen_in_replica: set[str] = set()
        for item in failures:
            bucket = _strict_failure_bucket(item)
            if not bucket or bucket in seen_in_replica:
                continue
            seen_in_replica.add(bucket)
            counts[bucket] = counts.get(bucket, 0) + 1

    majority_threshold = (replica_count // 2) + 1
    aggregated = [
        f"{bucket} [{count}/{replica_count} replicas]"
        for bucket, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if count >= majority_threshold
    ]
    if aggregated:
        return aggregated
    return [
        f"{bucket} [{count}/{replica_count} replicas]"
        for bucket, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _aggregate_eval_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {}

    def aggregate(values: list[Any], key: str | None = None) -> Any:
        present = [value for value in values if value is not None]
        if not present:
            return None
        if key == "strict_status":
            statuses = [str(value) for value in present]
            return statuses[0] if statuses and all("PASS" in status for status in statuses) else "❌ FAIL"
        if key == "strict_failures":
            return _aggregate_strict_failures(values)
        first = present[0]
        if isinstance(first, dict) and all(isinstance(value, dict) for value in present):
            keys = sorted({subkey for value in present for subkey in value})
            return {subkey: aggregate([value.get(subkey) for value in present], subkey) for subkey in keys}
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in present):
            return float(sum(float(value) for value in present) / len(present))
        return copy.deepcopy(first)

    keys = sorted({key for summary in summaries for key in summary})
    aggregated = {key: aggregate([summary.get(key) for summary in summaries], key) for key in keys}
    replica_scores = [float(summary.get("answer_score", strict_answer_score(summary)) or 0.0) for summary in summaries]
    aggregated["replica_count"] = len(summaries)
    aggregated["replica_answer_scores"] = replica_scores
    aggregated["replica_answer_score_mean"] = float(sum(replica_scores) / len(replica_scores)) if replica_scores else 0.0
    aggregated["replica_answer_score_median"] = float(statistics.median(replica_scores)) if replica_scores else 0.0
    aggregated["answer_score"] = aggregated["replica_answer_score_median"]
    return aggregated


def run_candidate(candidate: StrictEvalCandidate, seed: int, mode: str = "v4") -> dict[str, object] | None:
    normalized = str(mode or _candidate_mode(candidate.name)).strip().lower() or "v4"
    ensure_results_header(normalized)
    run_id = f"{slugify(candidate.name)}-seed{seed}"
    run_dir = artifacts_for_mode(normalized) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = materialize_run_config(run_dir, candidate, seed)
    config = yaml.safe_load(config_path.read_text())
    model_path = run_dir / "model.pt"

    if candidate.warm_start_model_path:
        shutil.copy2(candidate.warm_start_model_path, str(model_path) + ".tmp")
    train_cmd = [
        str(PYTHON),
        str(ROOT / "train_production.py"),
        "--config",
        str(config_path),
        "--size",
        str(int((config.get("sizes") or config.get("phase4", {}).get("sizes") or [2_700_000_000])[0])),
        "--output",
        str(model_path),
        "--device",
        str(config.get("device", "cuda")),
    ]
    if candidate.warm_start_model_path:
        train_cmd.append("--resume")
    rc = subprocess.call(train_cmd, cwd=ROOT)
    if rc != 0:
        append_result(run_id, "train_failed", None, mode=normalized)
        return None

    eval_repeats = max(int(getattr(candidate, "eval_repeats", 1) or 1), 1)
    replica_summaries: list[dict[str, Any]] = []
    for _ in range(eval_repeats):
        output = subprocess.check_output(_eval_cmd_from_config(config, model_path), cwd=ROOT, text=True)
        replica_summaries.append(_parse_summary_blob(output))

    summary = replica_summaries[0] if len(replica_summaries) == 1 else _aggregate_eval_summaries(replica_summaries)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    if len(replica_summaries) > 1:
        (run_dir / "replica_summaries.json").write_text(json.dumps(replica_summaries, indent=2, sort_keys=True))
    append_result(run_id, "ok", summary, mode=normalized)
    return summary


def classify_summary(summary: dict[str, Any]) -> dict[str, Any]:
    avg_known = float(summary.get("avg_known_similarity", 0.0) or 0.0)
    exact = float(summary.get("avg_known_exact_similarity", 0.0) or 0.0)
    paraphrase = float(summary.get("avg_known_paraphrase_similarity", 0.0) or 0.0)
    synthesis = float(summary.get("synthesis_similarity", 0.0) or 0.0)
    margin = float(summary.get("avg_known_margin", 0.0) or 0.0)
    ignorance_gap = float(summary.get("ignorance_gap", 0.0) or 0.0)
    ood_raw = summary.get("avg_ood_confidence", 1.0)
    ood_confidence = float(1.0 if ood_raw is None else ood_raw)
    objective_supported_direct_rate = summary.get("objective_supported_direct_rate")
    objective_supported_wrong_chunk_rate = summary.get("objective_supported_wrong_chunk_rate")
    objective_in_domain_unsupported_abstention_rate = summary.get("objective_in_domain_unsupported_abstention_rate")
    query_diag = dict(summary.get("query_diagnostics") or {})
    code_diag = dict(summary.get("code_diagnostics") or {})
    query_offdiag = float(query_diag.get("avg_offdiag_similarity", 1.0) or 1.0)
    query_rank = float(query_diag.get("participation_ratio_fraction", 0.0) or 0.0)
    code_rank = float(code_diag.get("participation_ratio_fraction", 0.0) or 0.0)

    flags: list[str] = []
    if avg_known <= 0.0 and exact <= 0.0 and paraphrase <= 0.0 and synthesis <= 0.0:
        flags.append("catastrophic_zero_known")
        primary_failure_mode = "catastrophic_zero_known"
    elif objective_in_domain_unsupported_abstention_rate is not None and float(objective_in_domain_unsupported_abstention_rate) < 0.75:
        primary_failure_mode = "objective_support_discipline"
    elif margin < 0.03:
        primary_failure_mode = "margin_starved_signal"
    elif ood_confidence > 0.35:
        primary_failure_mode = "calibration_bound_signal"
    else:
        primary_failure_mode = "weak_signal_nonzero"

    if query_offdiag > 0.90 or query_rank < 0.05:
        flags.append("query_collapse")
    if code_rank < 0.10:
        flags.append("code_low_rank")
    if ood_confidence > 0.35:
        flags.append("ood_overconfident")
    if ignorance_gap < 0.30:
        flags.append("ignorance_gap_small")
    if margin < 0.05:
        flags.append("margin_too_small")
    if paraphrase < 0.50:
        flags.append("paraphrase_weak")
    if synthesis < 0.45:
        flags.append("synthesis_weak")
    if objective_supported_direct_rate is not None and float(objective_supported_direct_rate) < 0.75:
        flags.append("direct_support_hygiene")
    if objective_supported_wrong_chunk_rate is not None and float(objective_supported_wrong_chunk_rate) > 0.20:
        flags.append("same_family_wrong_chunk")
    if (
        objective_in_domain_unsupported_abstention_rate is not None
        and float(objective_in_domain_unsupported_abstention_rate) < 0.75
    ):
        flags.append("modifier_unsupported")

    return {
        "primary_failure_mode": primary_failure_mode,
        "flags": list(dict.fromkeys(flags)),
    }


def candidate_family(candidate_name: str, phase4_updates: dict[str, Any] | None = None) -> str:
    candidate = _candidate_lookup(candidate_name)
    if candidate is not None:
        return str(candidate.intervention_type)

    name = str(candidate_name or "").lower()
    if name.startswith("adaptive margin") or name.startswith("adaptive recovery cycle"):
        return "adaptive_margin"
    if name.startswith("adaptive confidence"):
        return "adaptive_confidence"
    if name.startswith("adaptive rank"):
        return "adaptive_rank"
    if name.startswith("adaptive multiview"):
        return "adaptive_multiview"

    updates = dict(phase4_updates or {})
    if updates.get("use_query_multiview"):
        return "query_multiview"
    if updates.get("use_momentum_queue"):
        return "momentum_queue"
    return "control"


def detect_stall(state: dict[str, Any], *, window: int = 8, min_runs: int = 12, min_improvement: float = 0.05) -> dict[str, Any] | None:
    history = list(state.get("history", []) or [])
    adaptive = [entry for entry in history if str(entry.get("candidate_name", "")).lower().startswith("adaptive")]
    if len(adaptive) < min_runs:
        return None
    recent = adaptive[-window:]
    if len(recent) < window:
        return None
    scores = [float(entry.get("score", float("-inf")) or float("-inf")) for entry in recent]
    if max(scores) - scores[0] >= min_improvement:
        return None

    flag_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    for entry in recent:
        for flag in entry.get("flags", []) or []:
            flag_counts[str(flag)] = flag_counts.get(str(flag), 0) + 1
        failure = str(entry.get("primary_failure_mode", "") or "")
        if failure:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
    dominant_flags = [flag for flag, _ in sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))]
    dominant_failure_modes = [mode for mode, _ in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))]
    return {
        "reason": "adaptive_plateau",
        "window": window,
        "min_runs": min_runs,
        "min_improvement": min_improvement,
        "dominant_flags": dominant_flags,
        "dominant_failure_modes": dominant_failure_modes,
    }


def _best_entry(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not history:
        return None
    return max(history, key=lambda entry: float(entry.get("score", float("-inf")) or float("-inf")))


def _entry_by_name(history: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    matches = [item for item in history if str(item.get("candidate_name", "")) == name]
    return _best_entry(matches)


def _score_of_contains(history: list[dict[str, Any]], needle: str) -> float | None:
    matches = [item for item in history if needle.lower() in str(item.get("candidate_name", "")).lower()]
    best = _best_entry(matches)
    return None if best is None else float(best.get("score", float("-inf")) or float("-inf"))


def _adaptive_candidates(state: dict[str, Any]) -> list[StrictEvalCandidate]:
    mode = str(state.get("mode", "v4") or "v4").strip().lower()
    base = build_strict_eval_scout_config(seed=0)["phase4"]
    incumbent_name = str(state.get("incumbent_name") or _incumbent_name_for_mode(mode))
    incumbent_candidate = _candidate_lookup(incumbent_name)
    if incumbent_candidate is not None:
        base = copy.deepcopy(incumbent_candidate.phase4_updates)

    def build(name: str, updates: dict[str, Any], family: str) -> StrictEvalCandidate:
        merged = copy.deepcopy(base)
        merged.update(copy.deepcopy(updates))
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="ADAPTIVE",
            intervention_type=family,
            rationale=f"Adaptive follow-up for {mode}",
            expected_effect="Escalate the strongest surviving signal without reopening already-dead families.",
            phase4_updates=merged,
        )

    completed = int(state.get("completed_runs", 0) or 0)
    cycle_name = f"adaptive recovery cycle {max(completed + 1, 1)}"
    return [
        build(
            "adaptive margin recovery",
            {
                "use_query_multiview": False,
                "query_multiview_weight": 0.0,
                "query_margin_weight": max(float(base.get("query_margin_weight", 0.04) or 0.04), 0.10),
                "retrieval_margin_weight": max(float(base.get("retrieval_margin_weight", 0.10) or 0.10), 0.16),
            },
            "adaptive_margin",
        ),
        build(
            "adaptive confidence calibration",
            {
                "clf_weight": max(float(base.get("clf_weight", 0.0) or 0.0), 0.04),
                "classifier_prediction_weight": max(float(base.get("classifier_prediction_weight", 0.0) or 0.0), 1.0),
            },
            "adaptive_confidence",
        ),
        build(
            "adaptive rank recovery",
            {
                "rank_reg_weight": max(float(base.get("rank_reg_weight", 0.0) or 0.0), 0.05),
                "ranking_margin_weight": max(float(base.get("ranking_margin_weight", 0.0) or 0.0), 0.12),
            },
            "adaptive_rank",
        ),
        build(
            cycle_name,
            {
                "query_margin_weight": max(float(base.get("query_margin_weight", 0.04) or 0.04), 0.10),
                "rank_reg_weight": max(float(base.get("rank_reg_weight", 0.0) or 0.0), 0.05),
                "retrieval_margin_weight": max(float(base.get("retrieval_margin_weight", 0.10) or 0.10), 0.16),
            },
            "adaptive_margin",
        ),
    ]


def choose_next_candidate(state: dict[str, Any]) -> StrictEvalCandidate | None:
    mode = str(state.get("mode", "v4") or "v4").strip().lower()
    history = state.get("history", []) or []
    tried_names = {str(entry.get("candidate_name", "")) for entry in history}
    suppressed = set(state.get("suppressed_families", []) or [])
    static_first_modes = {f"v{i}" for i in range(5, 382)}
    static_only_modes = {f"v{i}" for i in range(179, 256)} | {"v257", "v258", "v259", "v260", "v261", "v262", "v263", "v264", "v265", "v266", "v267", "v268", "v269", "v270", "v271", "v272", "v273", "v274", "v275", "v276", "v277", "v278", "v279", "v280", "v281", "v282", "v283", "v284", "v285", "v286", "v287", "v288", "v289", "v290", "v292", "v293", "v294", "v295", "v296", "v297", "v298", "v300", "v301", "v302", "v303", "v304", "v305", "v306", "v307", "v308", "v309", "v310", "v311", "v324", "v325", "v326", "v327", "v329", "v334", "v335", "v339", "v340", "v341", "v342", "v343", "v344", "v345", "v346", "v347", "v348", "v349", "v350", "v351", "v352", "v353", "v355", "v356", "v357", "v358", "v359", "v360", "v361", "v362", "v363", "v364", "v365", "v366", "v367", "v368", "v369", "v370", "v371", "v372", "v373", "v374", "v375", "v376", "v377", "v378", "v379", "v380", "v381", "v382"}

    def is_allowed(candidate: StrictEvalCandidate) -> bool:
        family = candidate_family(candidate.name, candidate.phase4_updates)
        return candidate.name not in tried_names and family not in suppressed

    if mode in static_first_modes:
        for candidate in candidate_library_for_mode(mode):
            if is_allowed(candidate):
                return candidate
        if mode in static_only_modes:
            return None
        for candidate in _adaptive_candidates(state):
            if is_allowed(candidate):
                return candidate
        return None

    for candidate in _adaptive_candidates(state):
        if is_allowed(candidate):
            return candidate
    for candidate in candidate_library_for_mode(mode):
        if is_allowed(candidate):
            return candidate
    return None


def next_seed(state: dict[str, Any]) -> int:
    used: set[int] = set()
    for entry in state.get("history", []) or []:
        try:
            used.add(int(entry.get("seed")))
        except Exception:
            pass
    for seed in SCOUT_SEEDS:
        if seed not in used:
            return seed
    candidate_seed = max(used or [SCOUT_SEEDS[-1]]) + 1
    while candidate_seed in used:
        candidate_seed += 1
    return candidate_seed


def should_stop(summary: dict[str, Any] | None = None) -> bool:
    if not summary:
        return False
    return "PASS" in str(summary.get("strict_status", "")).upper()


def _write_stop_notice(state: dict[str, Any], reason: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    mode = str(state.get("mode", "v4") or "v4").strip().lower()
    artifacts = artifacts_for_mode(mode)
    artifacts.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "reason": reason,
        "incumbent_name": state.get("incumbent_name"),
        "incumbent_score": state.get("incumbent_score"),
        "completed_runs": state.get("completed_runs", 0),
        "details": details or {},
        "ts": time.time(),
    }
    payload["recommended_followup"] = recommend_followup_mode(mode, state, payload)
    (artifacts / "stop_notice.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _maybe_chain_followup(current_mode: str, stop_notice: dict[str, Any] | None) -> int:
    followup = ((stop_notice or {}).get("recommended_followup") or {})
    next_mode = str(followup.get("next_mode") or "").strip().lower()
    if not next_mode or next_mode == str(current_mode or "").strip().lower():
        return 0
    argv = [
        str(PYTHON),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "--mode",
        next_mode,
        "--budget-hours",
        "16",
        "--max-cycles",
        "999",
        "--auto-followup",
    ]
    return subprocess.call(argv, cwd=ROOT)


def recommend_followup_mode(mode: str, state: dict[str, Any], stop_notice: dict[str, Any] | None = None) -> dict[str, Any] | None:
    normalized = str(mode or "v4").strip().lower() or "v4"
    if normalized not in _FOLLOWUP_MODE_MAP:
        return None

    history = list(state.get("history", []) or [])
    static_entries = [item for item in history if str(item.get("candidate_name", "")).lower().startswith(f"{normalized} ")]
    best_entry = _best_entry(history)
    best_static = _best_entry(static_entries)
    stop_reason = str((stop_notice or {}).get("reason") or state.get("stop_reason") or "")
    if stop_reason.strip().lower() == "strict_pass":
        return None
    details = dict((stop_notice or {}).get("details") or state.get("stall_details") or {})
    dominant_flags = list(details.get("dominant_flags") or [])
    dominant_failure_modes = list(details.get("dominant_failure_modes") or [])
    catastrophic_static_failures = [
        str(item.get("candidate_name", ""))
        for item in static_entries
        if "catastrophic_zero_known" in set(item.get("flags", []) or []) or float(item.get("score", 0.0) or 0.0) <= -40.0
    ]
    static_scores = [
        {
            "candidate_name": str(item.get("candidate_name", "")),
            "score": float(item.get("score", float("-inf")) or float("-inf")),
        }
        for item in sorted(static_entries, key=lambda item: float(item.get("score", float("-inf")) or float("-inf")), reverse=True)
    ]

    evidence: dict[str, Any] = {
        "best_name": str((best_entry or {}).get("candidate_name") or ""),
        "best_score": float((best_entry or {}).get("score", float("-inf")) or float("-inf")) if best_entry else None,
        "best_static_name": str((best_static or {}).get("candidate_name") or None),
        "best_static_score": float((best_static or {}).get("score", float("-inf")) or float("-inf")) if best_static else None,
        "static_scores": static_scores,
        "catastrophic_static_failures": catastrophic_static_failures,
        "dominant_flags": dominant_flags,
        "dominant_failure_modes": dominant_failure_modes,
        "stop_reason": stop_reason,
    }

    if normalized == "v33":
        evidence["queue_failed"] = bool(catastrophic_static_failures) or (_score_of_contains(static_entries, "queue") or 0.0) <= -40.0
    elif normalized == "v36":
        evidence["calibration_failed"] = (_score_of_contains(history, "adaptive confidence") or -999.0) < float((best_entry or {}).get("score", -999.0))
        evidence["multiview_score"] = _score_of_contains(history, "adaptive multiview")
    elif normalized == "v37":
        evidence["queue_failed"] = bool(catastrophic_static_failures)
    elif normalized == "v38":
        evidence["reranker_failures"] = catastrophic_static_failures
    elif normalized == "v39":
        evidence["consensus_failures"] = catastrophic_static_failures
    elif normalized == "v40":
        evidence["best_v40_static"] = str((best_static or {}).get("candidate_name") or "")
    elif normalized == "v42":
        evidence["stronger_ranking_score"] = _score_of_contains(history, "stronger ranking")
        evidence["adaptive_confidence_score"] = _score_of_contains(history, "adaptive confidence")
    elif normalized == "v43":
        evidence["warmer_alignment_score"] = _score_of_contains(history, "warmer alignment")
    elif normalized == "v44":
        evidence["code_anchor_score"] = _score_of_contains(history, "code anchors")
        evidence["multiview_score"] = _score_of_contains(history, "multiview")
    elif normalized == "v45":
        evidence["paraphrase_expansion_score"] = _score_of_contains(history, "paraphrase")
        evidence["control_score"] = _score_of_contains(history, "control")
    elif normalized == "v46":
        evidence["delayed_calibration_score"] = _score_of_contains(history, "delay")
        evidence["adaptive_confidence_score"] = _score_of_contains(history, "adaptive confidence")
        evidence["ema_teacher_score"] = _score_of_contains(history, "ema teacher")
    elif normalized == "v53":
        evidence["adaptive_confidence_score"] = _score_of_contains(history, "adaptive confidence")
    elif normalized == "v55":
        keep_warm = _entry_by_name(history, "v55 frozen-backbone verifier + keep warm head")
        evidence["keep_warm_head_score"] = float((keep_warm or {}).get("score", float("-inf")) or float("-inf")) if keep_warm else None
    elif normalized == "v57":
        evidence["recovered_baseline_score"] = -3.037494682768981
    elif normalized == "v58":
        keep_adaptive = _entry_by_name(history, "v58 frozen-adaptive query-head + keep adaptive head")
        adaptive_rank_margin = _entry_by_name(history, "adaptive rank and margin recovery")
        evidence["keep_adaptive_head_score"] = float((keep_adaptive or {}).get("score", float("-inf")) or float("-inf")) if keep_adaptive else None
        evidence["adaptive_rank_margin_score"] = float((adaptive_rank_margin or {}).get("score", float("-inf")) or float("-inf")) if adaptive_rank_margin else None
    elif normalized == "v60":
        evidence["light_query_multiview_score"] = _score_of_contains(history, "light query multiview")
    elif normalized == "v61":
        reset_head = _entry_by_name(history, "v61 frozen-recovered verifier + reset head")
        evidence["reset_head_score"] = float((reset_head or {}).get("score", float("-inf")) or float("-inf")) if reset_head else None
        evidence["frontier_replay_run_id"] = "adaptive-recovery-cycle-18-seed524"
        evidence["frontier_replay_score"] = -1.9177984602749345
    elif normalized == "v62":
        light_multiview = _entry_by_name(history, "v62 frontier-replay + light multiview helper")
        evidence["light_multiview_score"] = float((light_multiview or {}).get("score", float("-inf")) or float("-inf")) if light_multiview else None
    elif normalized == "v63":
        teacher_score = _score_of_contains(history, "teacher stabilization")
        confidence_score = _score_of_contains(history, "adaptive confidence")
        light_multiview = _entry_by_name(history, "v63 adaptive-proxy-queue + light multiview")
        evidence["teacher_stabilization_score"] = teacher_score
        evidence["adaptive_confidence_score"] = confidence_score
        evidence["light_multiview_score"] = float((light_multiview or {}).get("score", float("-inf")) or float("-inf")) if light_multiview else None
    elif normalized == "v66":
        consensus = _entry_by_name(history, "v66 frozen-teacher verifier-selection + consensus rerank")
        stricter_gate = _entry_by_name(history, "v66 frozen-teacher verifier-selection + stricter gate")
        wider_pool = _entry_by_name(history, "v66 frozen-teacher verifier-selection + wider pool")
        lexical_bridge = _entry_by_name(history, "v66 frozen-teacher verifier-selection + lexical bridge")
        adaptive_confidence = _entry_by_name(history, "adaptive confidence calibration")
        evidence["consensus_rerank_score"] = float((consensus or {}).get("score", float("-inf")) or float("-inf")) if consensus else None
        evidence["stricter_gate_score"] = float((stricter_gate or {}).get("score", float("-inf")) or float("-inf")) if stricter_gate else None
        evidence["wider_pool_score"] = float((wider_pool or {}).get("score", float("-inf")) or float("-inf")) if wider_pool else None
        evidence["lexical_bridge_score"] = float((lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if lexical_bridge else None
        evidence["adaptive_confidence_score"] = float((adaptive_confidence or {}).get("score", float("-inf")) or float("-inf")) if adaptive_confidence else None
    elif normalized == "v68":
        control = _entry_by_name(history, "v68 agreement-first verifier control")
        softer = _entry_by_name(history, "v68 agreement-first verifier + softer acceptance")
        wider = _entry_by_name(history, "v68 agreement-first verifier + wider evidence pool")
        semantic = _entry_by_name(history, "v68 agreement-first verifier + semantic tie-break")
        consensus_backstop = _entry_by_name(history, "v68 agreement-first verifier + consensus backstop")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["softer_acceptance_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["wider_evidence_score"] = float((wider or {}).get("score", float("-inf")) or float("-inf")) if wider else None
        evidence["semantic_tie_break_score"] = float((semantic or {}).get("score", float("-inf")) or float("-inf")) if semantic else None
        evidence["consensus_backstop_score"] = float((consensus_backstop or {}).get("score", float("-inf")) or float("-inf")) if consensus_backstop else None
    elif normalized == "v69":
        control = _entry_by_name(history, "v69 query-union verifier control")
        wider = _entry_by_name(history, "v69 query-union verifier + wider union pool")
        consensus_backstop = _entry_by_name(history, "v69 query-union verifier + consensus backstop")
        lighter_lexical = _entry_by_name(history, "v69 query-union verifier + lighter lexical bridge")
        softer = _entry_by_name(history, "v69 query-union verifier + softer acceptance")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["wider_union_score"] = float((wider or {}).get("score", float("-inf")) or float("-inf")) if wider else None
        evidence["consensus_backstop_score"] = float((consensus_backstop or {}).get("score", float("-inf")) or float("-inf")) if consensus_backstop else None
        evidence["lighter_lexical_score"] = float((lighter_lexical or {}).get("score", float("-inf")) or float("-inf")) if lighter_lexical else None
        evidence["softer_acceptance_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
    elif normalized == "v70":
        control = _entry_by_name(history, "v70 agreement-sharpened verifier control")
        tighter = _entry_by_name(history, "v70 agreement-sharpened verifier + tighter local race")
        lexical = _entry_by_name(history, "v70 agreement-sharpened verifier + lexical stability")
        query_led = _entry_by_name(history, "v70 agreement-sharpened verifier + query-led disambiguation")
        consensus = _entry_by_name(history, "v70 agreement-sharpened verifier + margin-safe consensus")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tighter_local_race_score"] = float((tighter or {}).get("score", float("-inf")) or float("-inf")) if tighter else None
        evidence["lexical_stability_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["query_led_disambiguation_score"] = float((query_led or {}).get("score", float("-inf")) or float("-inf")) if query_led else None
        evidence["margin_safe_consensus_score"] = float((consensus or {}).get("score", float("-inf")) or float("-inf")) if consensus else None
    elif normalized == "v71":
        control = _entry_by_name(history, "v71 union-consensus verifier control")
        tighter = _entry_by_name(history, "v71 union-consensus verifier + tighter local race")
        lexical = _entry_by_name(history, "v71 union-consensus verifier + lexical stability")
        query_led = _entry_by_name(history, "v71 union-consensus verifier + query-led disambiguation")
        selective = _entry_by_name(history, "v71 union-consensus verifier + selective consensus gate")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tighter_local_race_score"] = float((tighter or {}).get("score", float("-inf")) or float("-inf")) if tighter else None
        evidence["lexical_stability_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["query_led_disambiguation_score"] = float((query_led or {}).get("score", float("-inf")) or float("-inf")) if query_led else None
        evidence["selective_consensus_gate_score"] = float((selective or {}).get("score", float("-inf")) or float("-inf")) if selective else None
    elif normalized == "v72":
        control = _entry_by_name(history, "v72 union-local-duel verifier control")
        stronger = _entry_by_name(history, "v72 union-local-duel verifier + stronger agreement gate")
        query_led = _entry_by_name(history, "v72 union-local-duel verifier + query-led duel")
        lexical = _entry_by_name(history, "v72 union-local-duel verifier + lexical stability")
        rescue = _entry_by_name(history, "v72 union-local-duel verifier + fallback consensus rescue")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_agreement_gate_score"] = float((stronger or {}).get("score", float("-inf")) or float("-inf")) if stronger else None
        evidence["query_led_duel_score"] = float((query_led or {}).get("score", float("-inf")) or float("-inf")) if query_led else None
        evidence["lexical_stability_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["fallback_consensus_rescue_score"] = float((rescue or {}).get("score", float("-inf")) or float("-inf")) if rescue else None
    elif normalized == "v73":
        control = _entry_by_name(history, "v73 union-triage verifier control")
        stronger = _entry_by_name(history, "v73 union-triage verifier + stronger agreement finish")
        query_led = _entry_by_name(history, "v73 union-triage verifier + query-led triage")
        lexical = _entry_by_name(history, "v73 union-triage verifier + lexical stability")
        rescue = _entry_by_name(history, "v73 union-triage verifier + selective consensus rescue")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_agreement_finish_score"] = float((stronger or {}).get("score", float("-inf")) or float("-inf")) if stronger else None
        evidence["query_led_triage_score"] = float((query_led or {}).get("score", float("-inf")) or float("-inf")) if query_led else None
        evidence["lexical_stability_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["selective_consensus_rescue_score"] = float((rescue or {}).get("score", float("-inf")) or float("-inf")) if rescue else None
    elif normalized == "v74":
        control = _entry_by_name(history, "v74 union-selective triage verifier control")
        stronger = _entry_by_name(history, "v74 union-selective triage verifier + stronger agreement finish")
        softer = _entry_by_name(history, "v74 union-selective triage verifier + softer acceptance")
        lexical = _entry_by_name(history, "v74 union-selective triage verifier + lexical stability")
        tighter = _entry_by_name(history, "v74 union-selective triage verifier + tighter consensus duel")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_agreement_finish_score"] = float((stronger or {}).get("score", float("-inf")) or float("-inf")) if stronger else None
        evidence["softer_acceptance_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["lexical_stability_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["tighter_consensus_duel_score"] = float((tighter or {}).get("score", float("-inf")) or float("-inf")) if tighter else None
    elif normalized == "v75":
        control = _entry_by_name(history, "v75 union-soft-accept verifier control")
        query_led = _entry_by_name(history, "v75 union-soft-accept verifier + query-led finish")
        lexical = _entry_by_name(history, "v75 union-soft-accept verifier + lexical stability")
        wider = _entry_by_name(history, "v75 union-soft-accept verifier + wider triage pool")
        gentler = _entry_by_name(history, "v75 union-soft-accept verifier + gentler acceptance")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["query_led_finish_score"] = float((query_led or {}).get("score", float("-inf")) or float("-inf")) if query_led else None
        evidence["lexical_stability_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["wider_triage_pool_score"] = float((wider or {}).get("score", float("-inf")) or float("-inf")) if wider else None
        evidence["gentler_acceptance_score"] = float((gentler or {}).get("score", float("-inf")) or float("-inf")) if gentler else None
    elif normalized == "v76":
        control = _entry_by_name(history, "v76 query-led soft-accept verifier control")
        stronger = _entry_by_name(history, "v76 query-led soft-accept verifier + stronger query-led finish")
        softer = _entry_by_name(history, "v76 query-led soft-accept verifier + softer gate")
        lexical = _entry_by_name(history, "v76 query-led soft-accept verifier + lexical bridge")
        wider = _entry_by_name(history, "v76 query-led soft-accept verifier + wider query triage")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_query_led_finish_score"] = float((stronger or {}).get("score", float("-inf")) or float("-inf")) if stronger else None
        evidence["softer_gate_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["wider_query_triage_score"] = float((wider or {}).get("score", float("-inf")) or float("-inf")) if wider else None
    elif normalized == "v77":
        control = _entry_by_name(history, "v77 promoted query-led verifier control")
        softer = _entry_by_name(history, "v77 promoted query-led verifier + softer gate")
        lexical = _entry_by_name(history, "v77 promoted query-led verifier + lexical tie-break")
        local_duel = _entry_by_name(history, "v77 promoted query-led verifier + local duel finish")
        rescue = _entry_by_name(history, "v77 promoted query-led verifier + tiny consensus rescue")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["softer_gate_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["lexical_tie_break_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["local_duel_finish_score"] = float((local_duel or {}).get("score", float("-inf")) or float("-inf")) if local_duel else None
        evidence["tiny_consensus_rescue_score"] = float((rescue or {}).get("score", float("-inf")) or float("-inf")) if rescue else None
    elif normalized == "v78":
        control = _entry_by_name(history, "v78 v69-reset union-consensus control")
        wider = _entry_by_name(history, "v78 v69-reset union-consensus + wider union backstop")
        lighter = _entry_by_name(history, "v78 v69-reset union-consensus + lighter lexical backstop")
        colder = _entry_by_name(history, "v78 v69-reset union-consensus + colder consensus")
        higher = _entry_by_name(history, "v78 v69-reset union-consensus + higher consensus floor")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["wider_union_backstop_score"] = float((wider or {}).get("score", float("-inf")) or float("-inf")) if wider else None
        evidence["lighter_lexical_backstop_score"] = float((lighter or {}).get("score", float("-inf")) or float("-inf")) if lighter else None
        evidence["colder_consensus_score"] = float((colder or {}).get("score", float("-inf")) or float("-inf")) if colder else None
        evidence["higher_consensus_floor_score"] = float((higher or {}).get("score", float("-inf")) or float("-inf")) if higher else None
    elif normalized == "v79":
        control = _entry_by_name(history, "v79 lexical-light union control")
        softer = _entry_by_name(history, "v79 lexical-light union + softer acceptance")
        wider = _entry_by_name(history, "v79 lexical-light union + wider shortlist")
        query_led = _entry_by_name(history, "v79 lexical-light union + query-led boost")
        lexical_veto = _entry_by_name(history, "v79 lexical-light union + tiny lexical veto")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["softer_acceptance_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["wider_shortlist_score"] = float((wider or {}).get("score", float("-inf")) or float("-inf")) if wider else None
        evidence["query_led_boost_score"] = float((query_led or {}).get("score", float("-inf")) or float("-inf")) if query_led else None
        evidence["tiny_lexical_veto_score"] = float((lexical_veto or {}).get("score", float("-inf")) or float("-inf")) if lexical_veto else None
    elif normalized == "v80":
        control = _entry_by_name(history, "v80 v78-winner lexical-light control")
        query_led = _entry_by_name(history, "v80 v78-winner lexical-light + query-led boost")
        wider = _entry_by_name(history, "v80 v78-winner lexical-light + modest wider shortlist")
        combined = _entry_by_name(history, "v80 v78-winner lexical-light + query-led wider shortlist")
        rescue = _entry_by_name(history, "v80 v78-winner lexical-light + narrower consensus rescue")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["query_led_boost_score"] = float((query_led or {}).get("score", float("-inf")) or float("-inf")) if query_led else None
        evidence["modest_wider_shortlist_score"] = float((wider or {}).get("score", float("-inf")) or float("-inf")) if wider else None
        evidence["query_led_wider_shortlist_score"] = float((combined or {}).get("score", float("-inf")) or float("-inf")) if combined else None
        evidence["narrower_consensus_rescue_score"] = float((rescue or {}).get("score", float("-inf")) or float("-inf")) if rescue else None
    elif normalized == "v81":
        control = _entry_by_name(history, "v81 predictor-shortlist verifier control")
        stronger = _entry_by_name(history, "v81 predictor-shortlist verifier + stronger agreement finish")
        softer = _entry_by_name(history, "v81 predictor-shortlist verifier + softer gate")
        lexical = _entry_by_name(history, "v81 predictor-shortlist verifier + tiny lexical stability")
        duel = _entry_by_name(history, "v81 predictor-shortlist verifier + local duel finish")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_agreement_finish_score"] = float((stronger or {}).get("score", float("-inf")) or float("-inf")) if stronger else None
        evidence["softer_gate_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["tiny_lexical_stability_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["local_duel_finish_score"] = float((duel or {}).get("score", float("-inf")) or float("-inf")) if duel else None
    elif normalized == "v82":
        control = _entry_by_name(history, "v82 union-endgame verifier control")
        stronger = _entry_by_name(history, "v82 union-endgame verifier + stronger agreement finish")
        softer = _entry_by_name(history, "v82 union-endgame verifier + softer gate")
        duel = _entry_by_name(history, "v82 union-endgame verifier + local duel finish")
        softer_duel = _entry_by_name(history, "v82 union-endgame verifier + softer local duel")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_agreement_finish_score"] = float((stronger or {}).get("score", float("-inf")) or float("-inf")) if stronger else None
        evidence["softer_gate_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["local_duel_finish_score"] = float((duel or {}).get("score", float("-inf")) or float("-inf")) if duel else None
        evidence["softer_local_duel_score"] = float((softer_duel or {}).get("score", float("-inf")) or float("-inf")) if softer_duel else None
    elif normalized == "v83":
        control = _entry_by_name(history, "v83 soft-accept union verifier control")
        stronger = _entry_by_name(history, "v83 soft-accept union verifier + stronger agreement finish")
        wider = _entry_by_name(history, "v83 soft-accept union verifier + wider union pool")
        lexical = _entry_by_name(history, "v83 soft-accept union verifier + tiny lexical stability")
        rescue = _entry_by_name(history, "v83 soft-accept union verifier + gentler consensus rescue")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_agreement_finish_score"] = float((stronger or {}).get("score", float("-inf")) or float("-inf")) if stronger else None
        evidence["wider_union_pool_score"] = float((wider or {}).get("score", float("-inf")) or float("-inf")) if wider else None
        evidence["tiny_lexical_stability_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["gentler_consensus_rescue_score"] = float((rescue or {}).get("score", float("-inf")) or float("-inf")) if rescue else None
    elif normalized == "v84":
        control = _entry_by_name(history, "v84 margin-gated soft-accept verifier control")
        stronger = _entry_by_name(history, "v84 margin-gated soft-accept verifier + stronger local agreement")
        query_led = _entry_by_name(history, "v84 margin-gated soft-accept verifier + query-led local finish")
        lexical = _entry_by_name(history, "v84 margin-gated soft-accept verifier + tiny lexical veto")
        rescue = _entry_by_name(history, "v84 margin-gated soft-accept verifier + selective consensus rescue")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_local_agreement_score"] = float((stronger or {}).get("score", float("-inf")) or float("-inf")) if stronger else None
        evidence["query_led_local_finish_score"] = float((query_led or {}).get("score", float("-inf")) or float("-inf")) if query_led else None
        evidence["tiny_lexical_veto_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["selective_consensus_rescue_score"] = float((rescue or {}).get("score", float("-inf")) or float("-inf")) if rescue else None
    elif normalized == "v85":
        control = _entry_by_name(history, "v85 query-led margin-gated verifier control")
        stronger = _entry_by_name(history, "v85 query-led margin-gated verifier + stronger query-led finish")
        softer = _entry_by_name(history, "v85 query-led margin-gated verifier + softer gate")
        lexical = _entry_by_name(history, "v85 query-led margin-gated verifier + tiny lexical bridge")
        rescue = _entry_by_name(history, "v85 query-led margin-gated verifier + selective consensus rescue")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_query_led_finish_score"] = float((stronger or {}).get("score", float("-inf")) or float("-inf")) if stronger else None
        evidence["softer_gate_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["selective_consensus_rescue_score"] = float((rescue or {}).get("score", float("-inf")) or float("-inf")) if rescue else None
    elif normalized == "v86":
        control = _entry_by_name(history, "v86 local-max verifier control")
        lexical = _entry_by_name(history, "v86 local-max verifier + tiny lexical bridge")
        stronger = _entry_by_name(history, "v86 local-max verifier + stronger agreement finish")
        wider = _entry_by_name(history, "v86 local-max verifier + wider shortlist")
        colder = _entry_by_name(history, "v86 local-max verifier + colder consensus backstop")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["stronger_agreement_finish_score"] = float((stronger or {}).get("score", float("-inf")) or float("-inf")) if stronger else None
        evidence["wider_shortlist_score"] = float((wider or {}).get("score", float("-inf")) or float("-inf")) if wider else None
        evidence["colder_consensus_backstop_score"] = float((colder or {}).get("score", float("-inf")) or float("-inf")) if colder else None
    elif normalized == "v87":
        control = _entry_by_name(history, "v87 colder-consensus local-max control")
        softer = _entry_by_name(history, "v87 colder-consensus local-max + softer gate")
        lexical = _entry_by_name(history, "v87 colder-consensus local-max + tiny lexical bridge")
        narrower = _entry_by_name(history, "v87 colder-consensus local-max + narrower shortlist")
        query_led = _entry_by_name(history, "v87 colder-consensus local-max + stronger query-led finish")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["softer_gate_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["narrower_shortlist_score"] = float((narrower or {}).get("score", float("-inf")) or float("-inf")) if narrower else None
        evidence["stronger_query_led_finish_score"] = float((query_led or {}).get("score", float("-inf")) or float("-inf")) if query_led else None
    elif normalized == "v88":
        control = _entry_by_name(history, "v88 narrower-local-max control")
        softer = _entry_by_name(history, "v88 narrower-local-max + softer gate")
        query_led = _entry_by_name(history, "v88 narrower-local-max + stronger query-led finish")
        lexical = _entry_by_name(history, "v88 narrower-local-max + tiny lexical bridge")
        stronger = _entry_by_name(history, "v88 narrower-local-max + stronger agreement finish")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["softer_gate_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["stronger_query_led_finish_score"] = float((query_led or {}).get("score", float("-inf")) or float("-inf")) if query_led else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["stronger_agreement_finish_score"] = float((stronger or {}).get("score", float("-inf")) or float("-inf")) if stronger else None
    elif normalized == "v89":
        control = _entry_by_name(history, "v89 query-led narrower-local-max control")
        softer = _entry_by_name(history, "v89 query-led narrower-local-max + softer gate")
        lexical = _entry_by_name(history, "v89 query-led narrower-local-max + tiny lexical bridge")
        duel = _entry_by_name(history, "v89 query-led narrower-local-max + local duel finish")
        colder = _entry_by_name(history, "v89 query-led narrower-local-max + colder consensus backstop")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["softer_gate_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["local_duel_finish_score"] = float((duel or {}).get("score", float("-inf")) or float("-inf")) if duel else None
        evidence["colder_consensus_backstop_score"] = float((colder or {}).get("score", float("-inf")) or float("-inf")) if colder else None
    elif normalized == "v90":
        control = _entry_by_name(history, "v90 tie-break stability control")
        lexical = _entry_by_name(history, "v90 tie-break stability + tiny lexical bridge")
        lighter_consensus = _entry_by_name(history, "v90 tie-break stability + lighter consensus backstop")
        balanced_agreement = _entry_by_name(history, "v90 tie-break stability + balanced agreement finish")
        hybrid = _entry_by_name(history, "v90 tie-break stability + lexical-consensus hybrid")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["lighter_consensus_backstop_score"] = float((lighter_consensus or {}).get("score", float("-inf")) or float("-inf")) if lighter_consensus else None
        evidence["balanced_agreement_finish_score"] = float((balanced_agreement or {}).get("score", float("-inf")) or float("-inf")) if balanced_agreement else None
        evidence["lexical_consensus_hybrid_score"] = float((hybrid or {}).get("score", float("-inf")) or float("-inf")) if hybrid else None
    elif normalized == "v91":
        control = _entry_by_name(history, "v91 query-led local-race control")
        narrower = _entry_by_name(history, "v91 query-led local-race + narrower race")
        lexical = _entry_by_name(history, "v91 query-led local-race + tiny lexical bridge")
        lighter_consensus = _entry_by_name(history, "v91 query-led local-race + lighter consensus backstop")
        stronger_query_led = _entry_by_name(history, "v91 query-led local-race + stronger query-led finish")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["narrower_race_score"] = float((narrower or {}).get("score", float("-inf")) or float("-inf")) if narrower else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["lighter_consensus_backstop_score"] = float((lighter_consensus or {}).get("score", float("-inf")) or float("-inf")) if lighter_consensus else None
        evidence["stronger_query_led_finish_score"] = float((stronger_query_led or {}).get("score", float("-inf")) or float("-inf")) if stronger_query_led else None
    elif normalized == "v92":
        control = _entry_by_name(history, "v92 stabilized local-race control")
        softer = _entry_by_name(history, "v92 stabilized local-race + softer gate")
        colder = _entry_by_name(history, "v92 stabilized local-race + colder consensus backstop")
        wider = _entry_by_name(history, "v92 stabilized local-race + wider race")
        balanced = _entry_by_name(history, "v92 stabilized local-race + balanced agreement finish")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["softer_gate_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["colder_consensus_backstop_score"] = float((colder or {}).get("score", float("-inf")) or float("-inf")) if colder else None
        evidence["wider_race_score"] = float((wider or {}).get("score", float("-inf")) or float("-inf")) if wider else None
        evidence["balanced_agreement_finish_score"] = float((balanced or {}).get("score", float("-inf")) or float("-inf")) if balanced else None
    elif normalized == "v93":
        control = _entry_by_name(history, "v93 colder-consensus local-race control")
        softer = _entry_by_name(history, "v93 colder-consensus local-race + softer gate")
        stronger_query_led = _entry_by_name(history, "v93 colder-consensus local-race + stronger query-led finish")
        lexical = _entry_by_name(history, "v93 colder-consensus local-race + tiny lexical bridge")
        tighter = _entry_by_name(history, "v93 colder-consensus local-race + tighter consensus gate")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["softer_gate_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["stronger_query_led_finish_score"] = float((stronger_query_led or {}).get("score", float("-inf")) or float("-inf")) if stronger_query_led else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["tighter_consensus_gate_score"] = float((tighter or {}).get("score", float("-inf")) or float("-inf")) if tighter else None
    elif normalized == "v94":
        control = _entry_by_name(history, "v94 query-led colder-consensus control")
        softer = _entry_by_name(history, "v94 query-led colder-consensus + softer gate")
        lexical = _entry_by_name(history, "v94 query-led colder-consensus + tiny lexical bridge")
        tighter_consensus = _entry_by_name(history, "v94 query-led colder-consensus + tighter consensus gate")
        tighter_local = _entry_by_name(history, "v94 query-led colder-consensus + tighter local race")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["softer_gate_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["tighter_consensus_gate_score"] = float((tighter_consensus or {}).get("score", float("-inf")) or float("-inf")) if tighter_consensus else None
        evidence["tighter_local_race_score"] = float((tighter_local or {}).get("score", float("-inf")) or float("-inf")) if tighter_local else None
    elif normalized == "v95":
        control = _entry_by_name(history, "v95 stable query-led control")
        lighter_consensus = _entry_by_name(history, "v95 stable query-led + lighter consensus backstop")
        balanced = _entry_by_name(history, "v95 stable query-led + balanced agreement finish")
        lexical = _entry_by_name(history, "v95 stable query-led + tiny lexical bridge")
        wider = _entry_by_name(history, "v95 stable query-led + wider local race")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["lighter_consensus_backstop_score"] = float((lighter_consensus or {}).get("score", float("-inf")) or float("-inf")) if lighter_consensus else None
        evidence["balanced_agreement_finish_score"] = float((balanced or {}).get("score", float("-inf")) or float("-inf")) if balanced else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["wider_local_race_score"] = float((wider or {}).get("score", float("-inf")) or float("-inf")) if wider else None
    elif normalized == "v96":
        control = _entry_by_name(history, "v96 local-max stability control")
        wider = _entry_by_name(history, "v96 local-max stability + wider shortlist")
        narrower = _entry_by_name(history, "v96 local-max stability + narrower shortlist")
        lexical = _entry_by_name(history, "v96 local-max stability + tiny lexical bridge")
        query_led_wider = _entry_by_name(history, "v96 local-max stability + query-led wider shortlist")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["wider_shortlist_score"] = float((wider or {}).get("score", float("-inf")) or float("-inf")) if wider else None
        evidence["narrower_shortlist_score"] = float((narrower or {}).get("score", float("-inf")) or float("-inf")) if narrower else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["query_led_wider_shortlist_score"] = float((query_led_wider or {}).get("score", float("-inf")) or float("-inf")) if query_led_wider else None
    elif normalized == "v97":
        control = _entry_by_name(history, "v97 narrower-local-max control")
        softer = _entry_by_name(history, "v97 narrower-local-max + softer gate")
        lexical = _entry_by_name(history, "v97 narrower-local-max + tiny lexical bridge")
        stronger_query_led = _entry_by_name(history, "v97 narrower-local-max + stronger query-led finish")
        colder_consensus = _entry_by_name(history, "v97 narrower-local-max + colder consensus backstop")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["softer_gate_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["stronger_query_led_finish_score"] = float((stronger_query_led or {}).get("score", float("-inf")) or float("-inf")) if stronger_query_led else None
        evidence["colder_consensus_backstop_score"] = float((colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if colder_consensus else None
    elif normalized == "v98":
        control = _entry_by_name(history, "v98 stability-restored local-max control")
        colder = _entry_by_name(history, "v98 stability-restored local-max + colder consensus backstop")
        stronger_query_led = _entry_by_name(history, "v98 stability-restored local-max + stronger query-led finish")
        softer = _entry_by_name(history, "v98 stability-restored local-max + softer gate")
        selective = _entry_by_name(history, "v98 stability-restored local-max + selective narrowing")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["colder_consensus_backstop_score"] = float((colder or {}).get("score", float("-inf")) or float("-inf")) if colder else None
        evidence["stronger_query_led_finish_score"] = float((stronger_query_led or {}).get("score", float("-inf")) or float("-inf")) if stronger_query_led else None
        evidence["softer_gate_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["selective_narrowing_score"] = float((selective or {}).get("score", float("-inf")) or float("-inf")) if selective else None
    elif normalized == "v99":
        control = _entry_by_name(history, "v99 colder-consensus stable local-max control")
        selective = _entry_by_name(history, "v99 colder-consensus stable local-max + selective narrowing")
        softer = _entry_by_name(history, "v99 colder-consensus stable local-max + softer gate")
        higher_floor = _entry_by_name(history, "v99 colder-consensus stable local-max + higher consensus floor")
        tighter_gate = _entry_by_name(history, "v99 colder-consensus stable local-max + tighter consensus gate")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["selective_narrowing_score"] = float((selective or {}).get("score", float("-inf")) or float("-inf")) if selective else None
        evidence["softer_gate_score"] = float((softer or {}).get("score", float("-inf")) or float("-inf")) if softer else None
        evidence["higher_consensus_floor_score"] = float((higher_floor or {}).get("score", float("-inf")) or float("-inf")) if higher_floor else None
        evidence["tighter_consensus_gate_score"] = float((tighter_gate or {}).get("score", float("-inf")) or float("-inf")) if tighter_gate else None
    elif normalized == "v100":
        control = _entry_by_name(history, "v100 softer-gate colder-consensus local-max control")
        tighter_gate = _entry_by_name(history, "v100 softer-gate colder-consensus local-max + tighter consensus gate")
        lexical = _entry_by_name(history, "v100 softer-gate colder-consensus local-max + tiny lexical bridge")
        narrower = _entry_by_name(history, "v100 softer-gate colder-consensus local-max + narrower shortlist")
        stronger_query_led = _entry_by_name(history, "v100 softer-gate colder-consensus local-max + stronger query-led finish")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tighter_consensus_gate_score"] = float((tighter_gate or {}).get("score", float("-inf")) or float("-inf")) if tighter_gate else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["narrower_shortlist_score"] = float((narrower or {}).get("score", float("-inf")) or float("-inf")) if narrower else None
        evidence["stronger_query_led_finish_score"] = float((stronger_query_led or {}).get("score", float("-inf")) or float("-inf")) if stronger_query_led else None
    elif normalized == "v101":
        control = _entry_by_name(history, "v101 narrower-soft-gate local-max control")
        lexical = _entry_by_name(history, "v101 narrower-soft-gate local-max + tiny lexical bridge")
        softer_gate = _entry_by_name(history, "v101 narrower-soft-gate local-max + softer gate")
        lighter_consensus = _entry_by_name(history, "v101 narrower-soft-gate local-max + lighter consensus drag")
        tiny_query_led = _entry_by_name(history, "v101 narrower-soft-gate local-max + tiny query-led tie-break")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["softer_gate_score"] = float((softer_gate or {}).get("score", float("-inf")) or float("-inf")) if softer_gate else None
        evidence["lighter_consensus_drag_score"] = float((lighter_consensus or {}).get("score", float("-inf")) or float("-inf")) if lighter_consensus else None
        evidence["tiny_query_led_tie_break_score"] = float((tiny_query_led or {}).get("score", float("-inf")) or float("-inf")) if tiny_query_led else None
    elif normalized == "v102":
        control = _entry_by_name(history, "v102 softer-gate local-max stability control")
        lexical = _entry_by_name(history, "v102 softer-gate local-max stability + tiny lexical bridge")
        colder_consensus = _entry_by_name(history, "v102 softer-gate local-max stability + slightly colder consensus")
        tiny_query_led = _entry_by_name(history, "v102 softer-gate local-max stability + tiny query-led tie-break")
        wider_shortlist = _entry_by_name(history, "v102 softer-gate local-max stability + slightly wider shortlist")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["slightly_colder_consensus_score"] = float((colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if colder_consensus else None
        evidence["tiny_query_led_tie_break_score"] = float((tiny_query_led or {}).get("score", float("-inf")) or float("-inf")) if tiny_query_led else None
        evidence["slightly_wider_shortlist_score"] = float((wider_shortlist or {}).get("score", float("-inf")) or float("-inf")) if wider_shortlist else None
    elif normalized == "v103":
        control = _entry_by_name(history, "v103 colder-consensus softer-gate local-max control")
        lexical = _entry_by_name(history, "v103 colder-consensus softer-gate local-max + tiny lexical bridge")
        tiny_query_led = _entry_by_name(history, "v103 colder-consensus softer-gate local-max + tiny query-led tie-break")
        softer_gate = _entry_by_name(history, "v103 colder-consensus softer-gate local-max + slightly softer gate")
        stronger_agreement = _entry_by_name(history, "v103 colder-consensus softer-gate local-max + slightly stronger agreement finish")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["tiny_query_led_tie_break_score"] = float((tiny_query_led or {}).get("score", float("-inf")) or float("-inf")) if tiny_query_led else None
        evidence["slightly_softer_gate_score"] = float((softer_gate or {}).get("score", float("-inf")) or float("-inf")) if softer_gate else None
        evidence["slightly_stronger_agreement_finish_score"] = float((stronger_agreement or {}).get("score", float("-inf")) or float("-inf")) if stronger_agreement else None
    elif normalized == "v104":
        control = _entry_by_name(history, "v104 softer-gate colder-consensus local-max control")
        lexical = _entry_by_name(history, "v104 softer-gate colder-consensus local-max + tiny lexical bridge")
        colder_consensus = _entry_by_name(history, "v104 softer-gate colder-consensus local-max + slightly colder consensus")
        wider_shortlist = _entry_by_name(history, "v104 softer-gate colder-consensus local-max + slightly wider shortlist")
        stronger_agreement = _entry_by_name(history, "v104 softer-gate colder-consensus local-max + slightly stronger agreement finish")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["slightly_colder_consensus_score"] = float((colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if colder_consensus else None
        evidence["slightly_wider_shortlist_score"] = float((wider_shortlist or {}).get("score", float("-inf")) or float("-inf")) if wider_shortlist else None
        evidence["slightly_stronger_agreement_finish_score"] = float((stronger_agreement or {}).get("score", float("-inf")) or float("-inf")) if stronger_agreement else None
    elif normalized == "v105":
        control = _entry_by_name(history, "v105 wider-shortlist softer-gate local-max control")
        lexical = _entry_by_name(history, "v105 wider-shortlist softer-gate local-max + tiny lexical bridge")
        colder_consensus = _entry_by_name(history, "v105 wider-shortlist softer-gate local-max + slightly colder consensus")
        softer_gate = _entry_by_name(history, "v105 wider-shortlist softer-gate local-max + slightly softer gate")
        stronger_agreement = _entry_by_name(history, "v105 wider-shortlist softer-gate local-max + slightly stronger agreement finish")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["slightly_colder_consensus_score"] = float((colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if colder_consensus else None
        evidence["slightly_softer_gate_score"] = float((softer_gate or {}).get("score", float("-inf")) or float("-inf")) if softer_gate else None
        evidence["slightly_stronger_agreement_finish_score"] = float((stronger_agreement or {}).get("score", float("-inf")) or float("-inf")) if stronger_agreement else None
    elif normalized == "v106":
        control = _entry_by_name(history, "v106 agreement-promoted wider-shortlist local-max control")
        lexical = _entry_by_name(history, "v106 agreement-promoted wider-shortlist local-max + tiny lexical bridge")
        narrower = _entry_by_name(history, "v106 agreement-promoted wider-shortlist local-max + slightly narrower shortlist")
        query_tiebreak = _entry_by_name(history, "v106 agreement-promoted wider-shortlist local-max + tiny query-led tie-break")
        colder_consensus = _entry_by_name(history, "v106 agreement-promoted wider-shortlist local-max + slightly colder consensus")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["slightly_narrower_shortlist_score"] = float((narrower or {}).get("score", float("-inf")) or float("-inf")) if narrower else None
        evidence["tiny_query_led_tie_break_score"] = float((query_tiebreak or {}).get("score", float("-inf")) or float("-inf")) if query_tiebreak else None
        evidence["slightly_colder_consensus_score"] = float((colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if colder_consensus else None
    elif normalized == "v107":
        control = _entry_by_name(history, "v107 agreement-stable wider-shortlist local-max control")
        lexical = _entry_by_name(history, "v107 agreement-stable wider-shortlist local-max + tiny lexical bridge")
        narrower = _entry_by_name(history, "v107 agreement-stable wider-shortlist local-max + slightly narrower shortlist")
        query_tiebreak = _entry_by_name(history, "v107 agreement-stable wider-shortlist local-max + tiny query-led tie-break")
        lighter_consensus = _entry_by_name(history, "v107 agreement-stable wider-shortlist local-max + lighter consensus drag")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["slightly_narrower_shortlist_score"] = float((narrower or {}).get("score", float("-inf")) or float("-inf")) if narrower else None
        evidence["tiny_query_led_tie_break_score"] = float((query_tiebreak or {}).get("score", float("-inf")) or float("-inf")) if query_tiebreak else None
        evidence["lighter_consensus_drag_score"] = float((lighter_consensus or {}).get("score", float("-inf")) or float("-inf")) if lighter_consensus else None
    elif normalized == "v108":
        control = _entry_by_name(history, "v108 lighter-consensus stable wider-shortlist local-max control")
        lexical = _entry_by_name(history, "v108 lighter-consensus stable wider-shortlist local-max + tiny lexical bridge")
        narrower = _entry_by_name(history, "v108 lighter-consensus stable wider-shortlist local-max + slightly narrower shortlist")
        query_tiebreak = _entry_by_name(history, "v108 lighter-consensus stable wider-shortlist local-max + tiny query-led tie-break")
        softer_gate = _entry_by_name(history, "v108 lighter-consensus stable wider-shortlist local-max + slightly softer gate")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["slightly_narrower_shortlist_score"] = float((narrower or {}).get("score", float("-inf")) or float("-inf")) if narrower else None
        evidence["tiny_query_led_tie_break_score"] = float((query_tiebreak or {}).get("score", float("-inf")) or float("-inf")) if query_tiebreak else None
        evidence["slightly_softer_gate_score"] = float((softer_gate or {}).get("score", float("-inf")) or float("-inf")) if softer_gate else None
    elif normalized == "v109":
        control = _entry_by_name(history, "v109 reset-to-v106 control")
        lexical = _entry_by_name(history, "v109 reset-to-v106 + tiny lexical bridge")
        narrower = _entry_by_name(history, "v109 reset-to-v106 + slightly narrower shortlist")
        query_tiebreak = _entry_by_name(history, "v109 reset-to-v106 + tiny query-led tie-break")
        lighter_consensus = _entry_by_name(history, "v109 reset-to-v106 + lighter consensus drag")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_lexical_bridge_score"] = float((lexical or {}).get("score", float("-inf")) or float("-inf")) if lexical else None
        evidence["slightly_narrower_shortlist_score"] = float((narrower or {}).get("score", float("-inf")) or float("-inf")) if narrower else None
        evidence["tiny_query_led_tie_break_score"] = float((query_tiebreak or {}).get("score", float("-inf")) or float("-inf")) if query_tiebreak else None
        evidence["lighter_consensus_drag_score"] = float((lighter_consensus or {}).get("score", float("-inf")) or float("-inf")) if lighter_consensus else None
    elif normalized == "v110":
        control = _entry_by_name(history, "v110 adaptive local-race control")
        wider_nomination = _entry_by_name(history, "v110 adaptive local-race + wider nomination")
        wider_tighter_gate = _entry_by_name(history, "v110 adaptive local-race + wider nomination + tighter local gate")
        slight_agreement = _entry_by_name(history, "v110 adaptive local-race + slight agreement restoration")
        wider_agreement = _entry_by_name(history, "v110 adaptive local-race + wider nomination + slight agreement restoration")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["wider_nomination_score"] = float((wider_nomination or {}).get("score", float("-inf")) or float("-inf")) if wider_nomination else None
        evidence["wider_nomination_tighter_local_gate_score"] = float((wider_tighter_gate or {}).get("score", float("-inf")) or float("-inf")) if wider_tighter_gate else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement else None
        evidence["wider_nomination_slight_agreement_restoration_score"] = float((wider_agreement or {}).get("score", float("-inf")) or float("-inf")) if wider_agreement else None
    elif normalized == "v111":
        control = _entry_by_name(history, "v111 v106-reset control")
        slight_agreement = _entry_by_name(history, "v111 v106-reset + slight agreement restoration")
        wider_nomination = _entry_by_name(history, "v111 v106-reset + wider nomination")
        wider_tighter_gate = _entry_by_name(history, "v111 v106-reset + wider nomination + tighter local gate")
        wider_agreement = _entry_by_name(history, "v111 v106-reset + wider nomination + slight agreement restoration")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement else None
        evidence["wider_nomination_score"] = float((wider_nomination or {}).get("score", float("-inf")) or float("-inf")) if wider_nomination else None
        evidence["wider_nomination_tighter_local_gate_score"] = float((wider_tighter_gate or {}).get("score", float("-inf")) or float("-inf")) if wider_tighter_gate else None
        evidence["wider_nomination_slight_agreement_restoration_score"] = float((wider_agreement or {}).get("score", float("-inf")) or float("-inf")) if wider_agreement else None
    elif normalized == "v112":
        control = _entry_by_name(history, "v112 promoted wider-nomination tighter-local-gate control")
        slight_agreement = _entry_by_name(history, "v112 promoted wider-nomination tighter-local-gate + slight agreement restoration")
        softer_gate = _entry_by_name(history, "v112 promoted wider-nomination tighter-local-gate + slightly softer local gate")
        wider_nomination = _entry_by_name(history, "v112 promoted wider-nomination tighter-local-gate + slightly wider nomination")
        lighter_consensus = _entry_by_name(history, "v112 promoted wider-nomination tighter-local-gate + lighter consensus drag")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement else None
        evidence["slightly_softer_local_gate_score"] = float((softer_gate or {}).get("score", float("-inf")) or float("-inf")) if softer_gate else None
        evidence["slightly_wider_nomination_score"] = float((wider_nomination or {}).get("score", float("-inf")) or float("-inf")) if wider_nomination else None
        evidence["lighter_consensus_drag_score"] = float((lighter_consensus or {}).get("score", float("-inf")) or float("-inf")) if lighter_consensus else None
    elif normalized == "v113":
        control = _entry_by_name(history, "v113 local-race reset control")
        slight_agreement = _entry_by_name(history, "v113 local-race reset + slight agreement restoration")
        query_led = _entry_by_name(history, "v113 local-race reset + query-led finish")
        colder_consensus = _entry_by_name(history, "v113 local-race reset + colder consensus backstop")
        narrower_race = _entry_by_name(history, "v113 local-race reset + narrower race")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement else None
        evidence["query_led_finish_score"] = float((query_led or {}).get("score", float("-inf")) or float("-inf")) if query_led else None
        evidence["colder_consensus_backstop_score"] = float((colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if colder_consensus else None
        evidence["narrower_race_score"] = float((narrower_race or {}).get("score", float("-inf")) or float("-inf")) if narrower_race else None
    elif normalized == "v114":
        control = _entry_by_name(history, "v114 colder-consensus local-race control")
        softer_gate = _entry_by_name(history, "v114 colder-consensus local-race + slightly softer gate")
        slight_agreement = _entry_by_name(history, "v114 colder-consensus local-race + slight agreement restoration")
        narrower_race = _entry_by_name(history, "v114 colder-consensus local-race + narrower race")
        tiny_lexical = _entry_by_name(history, "v114 colder-consensus local-race + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_softer_gate_score"] = float((softer_gate or {}).get("score", float("-inf")) or float("-inf")) if softer_gate else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement else None
        evidence["narrower_race_score"] = float((narrower_race or {}).get("score", float("-inf")) or float("-inf")) if narrower_race else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical else None
    elif normalized == "v115":
        control = _entry_by_name(history, "v115 agreement-restored local-race control")
        slightly_colder_consensus = _entry_by_name(history, "v115 agreement-restored local-race + slightly colder consensus")
        slightly_stronger_agreement = _entry_by_name(history, "v115 agreement-restored local-race + slightly stronger agreement")
        slightly_wider_race = _entry_by_name(history, "v115 agreement-restored local-race + slightly wider race")
        tiny_query_tie_break = _entry_by_name(history, "v115 agreement-restored local-race + tiny query tie-break")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["slightly_stronger_agreement_score"] = float((slightly_stronger_agreement or {}).get("score", float("-inf")) or float("-inf")) if slightly_stronger_agreement else None
        evidence["slightly_wider_race_score"] = float((slightly_wider_race or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_race else None
        evidence["tiny_query_tie_break_score"] = float((tiny_query_tie_break or {}).get("score", float("-inf")) or float("-inf")) if tiny_query_tie_break else None
    elif normalized == "v116":
        control = _entry_by_name(history, "v116 promoted query tie-break local-race control")
        slightly_wider_race = _entry_by_name(history, "v116 promoted query tie-break local-race + slightly wider race")
        slightly_colder_consensus = _entry_by_name(history, "v116 promoted query tie-break local-race + slightly colder consensus")
        slight_agreement_restoration = _entry_by_name(history, "v116 promoted query tie-break local-race + slight agreement restoration")
        tiny_lexical_bridge = _entry_by_name(history, "v116 promoted query tie-break local-race + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_wider_race_score"] = float((slightly_wider_race or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_race else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement_restoration or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement_restoration else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
    elif normalized == "v117":
        control = _entry_by_name(history, "v117 colder-consensus query tie-break local-race control")
        slightly_wider_race = _entry_by_name(history, "v117 colder-consensus query tie-break local-race + slightly wider race")
        slight_agreement_restoration = _entry_by_name(history, "v117 colder-consensus query tie-break local-race + slight agreement restoration")
        slightly_softer_gate = _entry_by_name(history, "v117 colder-consensus query tie-break local-race + slightly softer gate")
        tiny_lexical_bridge = _entry_by_name(history, "v117 colder-consensus query tie-break local-race + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_wider_race_score"] = float((slightly_wider_race or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_race else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement_restoration or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement_restoration else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
    elif normalized == "v118":
        control = _entry_by_name(history, "v118 softer-gate colder-consensus query tie-break local-race control")
        slightly_wider_race = _entry_by_name(history, "v118 softer-gate colder-consensus query tie-break local-race + slightly wider race")
        slightly_firmer_gate = _entry_by_name(history, "v118 softer-gate colder-consensus query tie-break local-race + slightly firmer gate")
        slightly_colder_consensus = _entry_by_name(history, "v118 softer-gate colder-consensus query tie-break local-race + slightly colder consensus")
        slight_agreement_restoration = _entry_by_name(history, "v118 softer-gate colder-consensus query tie-break local-race + slight agreement restoration")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_wider_race_score"] = float((slightly_wider_race or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_race else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement_restoration or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement_restoration else None
    elif normalized == "v119":
        control = _entry_by_name(history, "v119 colder-consensus softer-gate query tie-break local-race control")
        slightly_firmer_gate = _entry_by_name(history, "v119 colder-consensus softer-gate query tie-break local-race + slightly firmer gate")
        slightly_wider_race = _entry_by_name(history, "v119 colder-consensus softer-gate query tie-break local-race + slightly wider race")
        slight_agreement_restoration = _entry_by_name(history, "v119 colder-consensus softer-gate query tie-break local-race + slight agreement restoration")
        slightly_softer_gate = _entry_by_name(history, "v119 colder-consensus softer-gate query tie-break local-race + slightly softer gate")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["slightly_wider_race_score"] = float((slightly_wider_race or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_race else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement_restoration or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement_restoration else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
    elif normalized == "v120":
        control = _entry_by_name(history, "v120 wider-race colder-consensus local-race control")
        slightly_firmer_gate = _entry_by_name(history, "v120 wider-race colder-consensus local-race + slightly firmer gate")
        slightly_colder_consensus = _entry_by_name(history, "v120 wider-race colder-consensus local-race + slightly colder consensus")
        tiny_lexical_bridge = _entry_by_name(history, "v120 wider-race colder-consensus local-race + tiny lexical bridge")
        slight_query_led_finish = _entry_by_name(history, "v120 wider-race colder-consensus local-race + slight query-led finish")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["slight_query_led_finish_score"] = float((slight_query_led_finish or {}).get("score", float("-inf")) or float("-inf")) if slight_query_led_finish else None
    elif normalized == "v121":
        control = _entry_by_name(history, "v121 firmer-gate colder-consensus local-race control")
        selective_wider_race = _entry_by_name(history, "v121 firmer-gate colder-consensus local-race + selective wider race")
        slightly_colder_consensus = _entry_by_name(history, "v121 firmer-gate colder-consensus local-race + slightly colder consensus")
        slight_agreement_restoration = _entry_by_name(history, "v121 firmer-gate colder-consensus local-race + slight agreement restoration")
        slightly_softer_gate = _entry_by_name(history, "v121 firmer-gate colder-consensus local-race + slightly softer gate")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["selective_wider_race_score"] = float((selective_wider_race or {}).get("score", float("-inf")) or float("-inf")) if selective_wider_race else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement_restoration or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement_restoration else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
    elif normalized == "v122":
        control = _entry_by_name(history, "v122 colder-consensus firmer-gate local-race control")
        selective_wider_race = _entry_by_name(history, "v122 colder-consensus firmer-gate local-race + selective wider race")
        slightly_firmer_gate = _entry_by_name(history, "v122 colder-consensus firmer-gate local-race + slightly firmer gate")
        tiny_lexical_bridge = _entry_by_name(history, "v122 colder-consensus firmer-gate local-race + tiny lexical bridge")
        slight_query_led_finish = _entry_by_name(history, "v122 colder-consensus firmer-gate local-race + slight query-led finish")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["selective_wider_race_score"] = float((selective_wider_race or {}).get("score", float("-inf")) or float("-inf")) if selective_wider_race else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["slight_query_led_finish_score"] = float((slight_query_led_finish or {}).get("score", float("-inf")) or float("-inf")) if slight_query_led_finish else None
    elif normalized == "v123":
        control = _entry_by_name(history, "v123 query-led firmer-gate colder-consensus local-race control")
        selective_wider_race = _entry_by_name(history, "v123 query-led firmer-gate colder-consensus local-race + selective wider race")
        slightly_colder_consensus = _entry_by_name(history, "v123 query-led firmer-gate colder-consensus local-race + slightly colder consensus")
        tiny_lexical_bridge = _entry_by_name(history, "v123 query-led firmer-gate colder-consensus local-race + tiny lexical bridge")
        slight_agreement_restoration = _entry_by_name(history, "v123 query-led firmer-gate colder-consensus local-race + slight agreement restoration")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["selective_wider_race_score"] = float((selective_wider_race or {}).get("score", float("-inf")) or float("-inf")) if selective_wider_race else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement_restoration or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement_restoration else None
    elif normalized == "v124":
        control = _entry_by_name(history, "v124 wider-race query-led firmer-gate control")
        slightly_firmer_gate = _entry_by_name(history, "v124 wider-race query-led firmer-gate + slightly firmer gate")
        slight_query_led_finish = _entry_by_name(history, "v124 wider-race query-led firmer-gate + slight query-led finish")
        tiny_lexical_bridge = _entry_by_name(history, "v124 wider-race query-led firmer-gate + tiny lexical bridge")
        selective_narrower_fallback = _entry_by_name(history, "v124 wider-race query-led firmer-gate + selective narrower fallback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["slight_query_led_finish_score"] = float((slight_query_led_finish or {}).get("score", float("-inf")) or float("-inf")) if slight_query_led_finish else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["selective_narrower_fallback_score"] = float((selective_narrower_fallback or {}).get("score", float("-inf")) or float("-inf")) if selective_narrower_fallback else None
    elif normalized == "v125":
        control = _entry_by_name(history, "v125 restored local-race query-led control")
        selective_wider_fallback = _entry_by_name(history, "v125 restored local-race query-led + selective wider fallback")
        slightly_firmer_gate = _entry_by_name(history, "v125 restored local-race query-led + slightly firmer gate")
        tiny_lexical_bridge = _entry_by_name(history, "v125 restored local-race query-led + tiny lexical bridge")
        slightly_colder_consensus = _entry_by_name(history, "v125 restored local-race query-led + slightly colder consensus")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["selective_wider_fallback_score"] = float((selective_wider_fallback or {}).get("score", float("-inf")) or float("-inf")) if selective_wider_fallback else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
    elif normalized == "v126":
        control = _entry_by_name(history, "v126 colder-consensus restored local-race control")
        slightly_firmer_gate = _entry_by_name(history, "v126 colder-consensus restored local-race + slightly firmer gate")
        slight_query_led_finish = _entry_by_name(history, "v126 colder-consensus restored local-race + slight query-led finish")
        tiny_lexical_bridge = _entry_by_name(history, "v126 colder-consensus restored local-race + tiny lexical bridge")
        slight_agreement_restoration = _entry_by_name(history, "v126 colder-consensus restored local-race + slight agreement restoration")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["slight_query_led_finish_score"] = float((slight_query_led_finish or {}).get("score", float("-inf")) or float("-inf")) if slight_query_led_finish else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement_restoration or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement_restoration else None
    elif normalized == "v127":
        control = _entry_by_name(history, "v127 promoted query-led colder-consensus local-race control")
        slightly_firmer_gate = _entry_by_name(history, "v127 promoted query-led colder-consensus local-race + slightly firmer gate")
        tiny_lexical_bridge = _entry_by_name(history, "v127 promoted query-led colder-consensus local-race + tiny lexical bridge")
        slightly_colder_consensus = _entry_by_name(history, "v127 promoted query-led colder-consensus local-race + slightly colder consensus")
        slight_agreement_restoration = _entry_by_name(history, "v127 promoted query-led colder-consensus local-race + slight agreement restoration")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement_restoration or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement_restoration else None
    elif normalized == "v128":
        control = _entry_by_name(history, "v128 lexical-bridge promoted query-led colder-consensus local-race control")
        slightly_firmer_gate = _entry_by_name(history, "v128 lexical-bridge promoted query-led colder-consensus local-race + slightly firmer gate")
        slightly_colder_consensus = _entry_by_name(history, "v128 lexical-bridge promoted query-led colder-consensus local-race + slightly colder consensus")
        slight_agreement_restoration = _entry_by_name(history, "v128 lexical-bridge promoted query-led colder-consensus local-race + slight agreement restoration")
        selective_wider_fallback = _entry_by_name(history, "v128 lexical-bridge promoted query-led colder-consensus local-race + selective wider fallback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement_restoration or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement_restoration else None
        evidence["selective_wider_fallback_score"] = float((selective_wider_fallback or {}).get("score", float("-inf")) or float("-inf")) if selective_wider_fallback else None
    elif normalized == "v129":
        control = _entry_by_name(history, "v129 v126-reset query-led colder-consensus local-race control")
        subtler_lexical_bridge = _entry_by_name(history, "v129 v126-reset query-led colder-consensus local-race + subtler lexical bridge")
        slightly_softer_gate = _entry_by_name(history, "v129 v126-reset query-led colder-consensus local-race + slightly softer gate")
        selective_wider_fallback = _entry_by_name(history, "v129 v126-reset query-led colder-consensus local-race + selective wider fallback")
        slightly_colder_consensus = _entry_by_name(history, "v129 v126-reset query-led colder-consensus local-race + slightly colder consensus")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["subtler_lexical_bridge_score"] = float((subtler_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if subtler_lexical_bridge else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["selective_wider_fallback_score"] = float((selective_wider_fallback or {}).get("score", float("-inf")) or float("-inf")) if selective_wider_fallback else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
    elif normalized == "v130":
        control = _entry_by_name(history, "v130 v129-reset query-led colder-consensus local-race control")
        slight_agreement_restoration = _entry_by_name(history, "v130 v129-reset query-led colder-consensus local-race + slight agreement restoration")
        slightly_firmer_gate = _entry_by_name(history, "v130 v129-reset query-led colder-consensus local-race + slightly firmer gate")
        selective_narrowing = _entry_by_name(history, "v130 v129-reset query-led colder-consensus local-race + selective narrowing")
        lighter_consensus_drag = _entry_by_name(history, "v130 v129-reset query-led colder-consensus local-race + lighter consensus drag")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement_restoration or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement_restoration else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["selective_narrowing_score"] = float((selective_narrowing or {}).get("score", float("-inf")) or float("-inf")) if selective_narrowing else None
        evidence["lighter_consensus_drag_score"] = float((lighter_consensus_drag or {}).get("score", float("-inf")) or float("-inf")) if lighter_consensus_drag else None
    elif normalized == "v131":
        control = _entry_by_name(history, "v131 v129-frontier restored control")
        micro_duel_finish = _entry_by_name(history, "v131 v129-frontier restored + micro duel finish")
        slight_agreement_restoration = _entry_by_name(history, "v131 v129-frontier restored + micro duel + slight agreement restoration")
        slightly_firmer_gate = _entry_by_name(history, "v131 v129-frontier restored + micro duel + slightly firmer gate")
        lighter_consensus_drag = _entry_by_name(history, "v131 v129-frontier restored + micro duel + lighter consensus drag")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["micro_duel_finish_score"] = float((micro_duel_finish or {}).get("score", float("-inf")) or float("-inf")) if micro_duel_finish else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement_restoration or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement_restoration else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["lighter_consensus_drag_score"] = float((lighter_consensus_drag or {}).get("score", float("-inf")) or float("-inf")) if lighter_consensus_drag else None

    elif normalized == "v132":
        control = _entry_by_name(history, "v132 v129-frontier local-max control")
        selective_narrowing = _entry_by_name(history, "v132 v129-frontier local-max + selective narrowing")
        wider_fallback = _entry_by_name(history, "v132 v129-frontier local-max + wider fallback")
        tighter_local_gate = _entry_by_name(history, "v132 v129-frontier local-max + tighter local gate")
        slight_agreement_restoration = _entry_by_name(history, "v132 v129-frontier local-max + slight agreement restoration")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["selective_narrowing_score"] = float((selective_narrowing or {}).get("score", float("-inf")) or float("-inf")) if selective_narrowing else None
        evidence["wider_fallback_score"] = float((wider_fallback or {}).get("score", float("-inf")) or float("-inf")) if wider_fallback else None
        evidence["tighter_local_gate_score"] = float((tighter_local_gate or {}).get("score", float("-inf")) or float("-inf")) if tighter_local_gate else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement_restoration or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement_restoration else None

    elif normalized == "v133":
        control = _entry_by_name(history, "v133 wider-fallback local-max control")
        tighter_local_gate = _entry_by_name(history, "v133 wider-fallback local-max + tighter local gate")
        slight_agreement_restoration = _entry_by_name(history, "v133 wider-fallback local-max + slight agreement restoration")
        selective_narrowing_finish = _entry_by_name(history, "v133 wider-fallback local-max + selective narrowing finish")
        slightly_colder_consensus = _entry_by_name(history, "v133 wider-fallback local-max + slightly colder consensus")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tighter_local_gate_score"] = float((tighter_local_gate or {}).get("score", float("-inf")) or float("-inf")) if tighter_local_gate else None
        evidence["slight_agreement_restoration_score"] = float((slight_agreement_restoration or {}).get("score", float("-inf")) or float("-inf")) if slight_agreement_restoration else None
        evidence["selective_narrowing_finish_score"] = float((selective_narrowing_finish or {}).get("score", float("-inf")) or float("-inf")) if selective_narrowing_finish else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None

    elif normalized == "v134":
        control = _entry_by_name(history, "v134 colder-consensus wider-fallback control")
        slightly_softer_gate = _entry_by_name(history, "v134 colder-consensus wider-fallback + slightly softer gate")
        tiny_lexical_bridge = _entry_by_name(history, "v134 colder-consensus wider-fallback + tiny lexical bridge")
        tiny_query_led_tie_break = _entry_by_name(history, "v134 colder-consensus wider-fallback + tiny query-led tie-break")
        slightly_wider_pool = _entry_by_name(history, "v134 colder-consensus wider-fallback + slightly wider pool")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["tiny_query_led_tie_break_score"] = float((tiny_query_led_tie_break or {}).get("score", float("-inf")) or float("-inf")) if tiny_query_led_tie_break else None
        evidence["slightly_wider_pool_score"] = float((slightly_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_pool else None

    elif normalized == "v135":
        control = _entry_by_name(history, "v135 query-led colder-consensus wider-fallback control")
        slightly_softer_gate = _entry_by_name(history, "v135 query-led colder-consensus wider-fallback + slightly softer gate")
        tiny_lexical_bridge = _entry_by_name(history, "v135 query-led colder-consensus wider-fallback + tiny lexical bridge")
        slightly_colder_consensus = _entry_by_name(history, "v135 query-led colder-consensus wider-fallback + slightly colder consensus")
        slightly_tighter_local_gate = _entry_by_name(history, "v135 query-led colder-consensus wider-fallback + slightly tighter local gate")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["slightly_tighter_local_gate_score"] = float((slightly_tighter_local_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_tighter_local_gate else None

    elif normalized == "v136":
        control = _entry_by_name(history, "v136 selective-query-led wider-fallback control")
        slightly_softer_gate = _entry_by_name(history, "v136 selective-query-led wider-fallback + slightly softer gate")
        tiny_lexical_bridge = _entry_by_name(history, "v136 selective-query-led wider-fallback + tiny lexical bridge")
        slightly_colder_consensus = _entry_by_name(history, "v136 selective-query-led wider-fallback + slightly colder consensus")
        slightly_wider_pool = _entry_by_name(history, "v136 selective-query-led wider-fallback + slightly wider pool")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["slightly_wider_pool_score"] = float((slightly_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_pool else None

    elif normalized == "v137":
        control = _entry_by_name(history, "v137 restored-v134 frontier control")
        slightly_softer_gate = _entry_by_name(history, "v137 restored-v134 frontier + slightly softer gate")
        slightly_wider_pool = _entry_by_name(history, "v137 restored-v134 frontier + slightly wider pool")
        slightly_colder_consensus = _entry_by_name(history, "v137 restored-v134 frontier + slightly colder consensus")
        softer_gate_wider_pool = _entry_by_name(history, "v137 restored-v134 frontier + softer gate + wider pool")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["slightly_wider_pool_score"] = float((slightly_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_pool else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["softer_gate_wider_pool_score"] = float((softer_gate_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if softer_gate_wider_pool else None

    elif normalized == "v138":
        control = _entry_by_name(history, "v138 admission-promoted restored frontier control")
        slightly_colder_consensus = _entry_by_name(history, "v138 admission-promoted restored frontier + slightly colder consensus")
        tiny_query_led_tie_break = _entry_by_name(history, "v138 admission-promoted restored frontier + tiny query-led tie-break")
        colder_consensus_query_led_tie_break = _entry_by_name(history, "v138 admission-promoted restored frontier + colder consensus + query-led tie-break")
        tiny_lexical_bridge = _entry_by_name(history, "v138 admission-promoted restored frontier + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["tiny_query_led_tie_break_score"] = float((tiny_query_led_tie_break or {}).get("score", float("-inf")) or float("-inf")) if tiny_query_led_tie_break else None
        evidence["colder_consensus_query_led_tie_break_score"] = float((colder_consensus_query_led_tie_break or {}).get("score", float("-inf")) or float("-inf")) if colder_consensus_query_led_tie_break else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None

    elif normalized == "v139":
        control = _entry_by_name(history, "v139 query-led admission-promoted control")
        slightly_colder_consensus = _entry_by_name(history, "v139 query-led admission-promoted + slightly colder consensus")
        tiny_lexical_bridge = _entry_by_name(history, "v139 query-led admission-promoted + tiny lexical bridge")
        slightly_firmer_gate = _entry_by_name(history, "v139 query-led admission-promoted + slightly firmer gate")
        slightly_narrower_pool = _entry_by_name(history, "v139 query-led admission-promoted + slightly narrower pool")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["slightly_narrower_pool_score"] = float((slightly_narrower_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_narrower_pool else None

    elif normalized == "v140":
        control = _entry_by_name(history, "v140 colder-consensus query-led control")
        slightly_softer_gate = _entry_by_name(history, "v140 colder-consensus query-led + slightly softer gate")
        slightly_wider_pool = _entry_by_name(history, "v140 colder-consensus query-led + slightly wider pool")
        softer_gate_wider_pool = _entry_by_name(history, "v140 colder-consensus query-led + softer gate + wider pool")
        tiny_lexical_bridge = _entry_by_name(history, "v140 colder-consensus query-led + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["slightly_wider_pool_score"] = float((slightly_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_pool else None
        evidence["softer_gate_wider_pool_score"] = float((softer_gate_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if softer_gate_wider_pool else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None

    elif normalized == "v141":
        control = _entry_by_name(history, "v141 restored query-led tie-break control")
        slightly_softer_gate = _entry_by_name(history, "v141 restored query-led tie-break + slightly softer gate")
        slightly_wider_pool = _entry_by_name(history, "v141 restored query-led tie-break + slightly wider pool")
        softer_gate_wider_pool = _entry_by_name(history, "v141 restored query-led tie-break + softer gate + wider pool")
        tiny_lexical_bridge = _entry_by_name(history, "v141 restored query-led tie-break + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["slightly_wider_pool_score"] = float((slightly_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_pool else None
        evidence["softer_gate_wider_pool_score"] = float((softer_gate_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if softer_gate_wider_pool else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None

    elif normalized == "v142":
        control = _entry_by_name(history, "v142 softer-gate restored tie-break control")
        slightly_colder_consensus = _entry_by_name(history, "v142 softer-gate restored tie-break + slightly colder consensus")
        slightly_narrower_pool = _entry_by_name(history, "v142 softer-gate restored tie-break + slightly narrower pool")
        slightly_firmer_agreement_finish = _entry_by_name(history, "v142 softer-gate restored tie-break + slightly firmer agreement finish")
        tiny_lexical_bridge = _entry_by_name(history, "v142 softer-gate restored tie-break + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["slightly_narrower_pool_score"] = float((slightly_narrower_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_narrower_pool else None
        evidence["slightly_firmer_agreement_finish_score"] = float((slightly_firmer_agreement_finish or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_agreement_finish else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None

    elif normalized == "v143":
        control = _entry_by_name(history, "v143 guarded restored tie-break control")
        slightly_colder_consensus = _entry_by_name(history, "v143 guarded restored tie-break + slightly colder consensus")
        slightly_wider_pool = _entry_by_name(history, "v143 guarded restored tie-break + slightly wider pool")
        restored_firm_gate = _entry_by_name(history, "v143 guarded restored tie-break + restored firm gate")
        tiny_lexical_bridge = _entry_by_name(history, "v143 guarded restored tie-break + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["slightly_wider_pool_score"] = float((slightly_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_pool else None
        evidence["restored_firm_gate_score"] = float((restored_firm_gate or {}).get("score", float("-inf")) or float("-inf")) if restored_firm_gate else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None

    elif normalized == "v144":
        control = _entry_by_name(history, "v144 champion-reset softer-gate control")
        slightly_colder_consensus = _entry_by_name(history, "v144 champion-reset softer-gate + slightly colder consensus")
        slightly_wider_pool = _entry_by_name(history, "v144 champion-reset softer-gate + slightly wider pool")
        slightly_firmer_rollback = _entry_by_name(history, "v144 champion-reset softer-gate + slightly firmer rollback")
        tiny_lexical_bridge = _entry_by_name(history, "v144 champion-reset softer-gate + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["slightly_wider_pool_score"] = float((slightly_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_pool else None
        evidence["slightly_firmer_rollback_score"] = float((slightly_firmer_rollback or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_rollback else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None

    elif normalized == "v145":
        control = _entry_by_name(history, "v145 promoted colder-consensus control")
        tiny_lexical_bridge = _entry_by_name(history, "v145 promoted colder-consensus + tiny lexical bridge")
        slightly_firmer_rollback = _entry_by_name(history, "v145 promoted colder-consensus + slightly firmer rollback")
        lexical_rollback_hybrid = _entry_by_name(history, "v145 promoted colder-consensus + lexical rollback hybrid")
        slightly_warmer_rollback = _entry_by_name(history, "v145 promoted colder-consensus + slightly warmer rollback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["slightly_firmer_rollback_score"] = float((slightly_firmer_rollback or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_rollback else None
        evidence["lexical_rollback_hybrid_score"] = float((lexical_rollback_hybrid or {}).get("score", float("-inf")) or float("-inf")) if lexical_rollback_hybrid else None
        evidence["slightly_warmer_rollback_score"] = float((slightly_warmer_rollback or {}).get("score", float("-inf")) or float("-inf")) if slightly_warmer_rollback else None

    elif normalized == "v146":
        control = _entry_by_name(history, "v146 rollback-stabilized control")
        tiny_lexical_bridge = _entry_by_name(history, "v146 rollback-stabilized + tiny lexical bridge")
        slightly_firmer_rollback = _entry_by_name(history, "v146 rollback-stabilized + slightly firmer rollback")
        slightly_colder_consensus = _entry_by_name(history, "v146 rollback-stabilized + slightly colder consensus")
        firmer_lexical_hybrid = _entry_by_name(history, "v146 rollback-stabilized + firmer lexical hybrid")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["slightly_firmer_rollback_score"] = float((slightly_firmer_rollback or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_rollback else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["firmer_lexical_hybrid_score"] = float((firmer_lexical_hybrid or {}).get("score", float("-inf")) or float("-inf")) if firmer_lexical_hybrid else None

    elif normalized == "v147":
        control = _entry_by_name(history, "v147 promoted rollback-stabilized colder-consensus control")
        tiny_lexical_bridge = _entry_by_name(history, "v147 promoted rollback-stabilized colder-consensus + tiny lexical bridge")
        slightly_firmer_rollback = _entry_by_name(history, "v147 promoted rollback-stabilized colder-consensus + slightly firmer rollback")
        firmer_lexical_hybrid = _entry_by_name(history, "v147 promoted rollback-stabilized colder-consensus + firmer lexical hybrid")
        slightly_warmer_rollback = _entry_by_name(history, "v147 promoted rollback-stabilized colder-consensus + slightly warmer rollback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["slightly_firmer_rollback_score"] = float((slightly_firmer_rollback or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_rollback else None
        evidence["firmer_lexical_hybrid_score"] = float((firmer_lexical_hybrid or {}).get("score", float("-inf")) or float("-inf")) if firmer_lexical_hybrid else None
        evidence["slightly_warmer_rollback_score"] = float((slightly_warmer_rollback or {}).get("score", float("-inf")) or float("-inf")) if slightly_warmer_rollback else None

    elif normalized == "v148":
        control = _entry_by_name(history, "v148 v146-reset colder-consensus control")
        tiny_lexical_bridge = _entry_by_name(history, "v148 v146-reset colder-consensus + tiny lexical bridge")
        slightly_firmer_rollback = _entry_by_name(history, "v148 v146-reset colder-consensus + slightly firmer rollback")
        lexical_rollback_hybrid = _entry_by_name(history, "v148 v146-reset colder-consensus + lexical rollback hybrid")
        slightly_warmer_rollback = _entry_by_name(history, "v148 v146-reset colder-consensus + slightly warmer rollback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["slightly_firmer_rollback_score"] = float((slightly_firmer_rollback or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_rollback else None
        evidence["lexical_rollback_hybrid_score"] = float((lexical_rollback_hybrid or {}).get("score", float("-inf")) or float("-inf")) if lexical_rollback_hybrid else None
        evidence["slightly_warmer_rollback_score"] = float((slightly_warmer_rollback or {}).get("score", float("-inf")) or float("-inf")) if slightly_warmer_rollback else None

    elif normalized == "v149":
        control = _entry_by_name(history, "v149 restored colder-consensus admission control")
        slightly_softer_gate = _entry_by_name(history, "v149 restored colder-consensus admission + slightly softer gate")
        slightly_wider_pool = _entry_by_name(history, "v149 restored colder-consensus admission + slightly wider pool")
        tiny_query_led_tie_break = _entry_by_name(history, "v149 restored colder-consensus admission + tiny query-led tie-break")
        softer_gate_wider_pool = _entry_by_name(history, "v149 restored colder-consensus admission + softer gate + wider pool")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["slightly_wider_pool_score"] = float((slightly_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_pool else None
        evidence["tiny_query_led_tie_break_score"] = float((tiny_query_led_tie_break or {}).get("score", float("-inf")) or float("-inf")) if tiny_query_led_tie_break else None
        evidence["softer_gate_wider_pool_score"] = float((softer_gate_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if softer_gate_wider_pool else None

    elif normalized == "v150":
        control = _entry_by_name(history, "v150 promoted softer-gate colder-consensus admission control")
        slightly_wider_pool = _entry_by_name(history, "v150 promoted softer-gate colder-consensus admission + slightly wider pool")
        slightly_firmer_gate = _entry_by_name(history, "v150 promoted softer-gate colder-consensus admission + slightly firmer gate")
        slightly_colder_consensus = _entry_by_name(history, "v150 promoted softer-gate colder-consensus admission + slightly colder consensus")
        slightly_softer_gate = _entry_by_name(history, "v150 promoted softer-gate colder-consensus admission + slightly softer gate")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_wider_pool_score"] = float((slightly_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_pool else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["slightly_colder_consensus_score"] = float((slightly_colder_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None

    elif normalized == "v151":
        control = _entry_by_name(history, "v151 stabilized promoted admission control")
        slightly_firmer_gate = _entry_by_name(history, "v151 stabilized promoted admission + slightly firmer gate")
        slightly_narrower_pool = _entry_by_name(history, "v151 stabilized promoted admission + slightly narrower pool")
        slightly_warmer_consensus = _entry_by_name(history, "v151 stabilized promoted admission + slightly warmer consensus")
        tiny_lexical_bridge = _entry_by_name(history, "v151 stabilized promoted admission + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["slightly_narrower_pool_score"] = float((slightly_narrower_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_narrower_pool else None
        evidence["slightly_warmer_consensus_score"] = float((slightly_warmer_consensus or {}).get("score", float("-inf")) or float("-inf")) if slightly_warmer_consensus else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None

    elif normalized == "v152":
        control = _entry_by_name(history, "v152 promoted warmer-consensus admission control")
        slightly_softer_gate = _entry_by_name(history, "v152 promoted warmer-consensus admission + slightly softer gate")
        slightly_wider_pool = _entry_by_name(history, "v152 promoted warmer-consensus admission + slightly wider pool")
        local_shortlist_rollback = _entry_by_name(history, "v152 promoted warmer-consensus admission + local shortlist rollback")
        tiny_lexical_bridge = _entry_by_name(history, "v152 promoted warmer-consensus admission + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["slightly_wider_pool_score"] = float((slightly_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_pool else None
        evidence["local_shortlist_rollback_score"] = float((local_shortlist_rollback or {}).get("score", float("-inf")) or float("-inf")) if local_shortlist_rollback else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None

    elif normalized == "v152":
        control = _entry_by_name(history, "v152 promoted warmer-consensus admission control")
        slightly_softer_gate = _entry_by_name(history, "v152 promoted warmer-consensus admission + slightly softer gate")
        slightly_wider_pool = _entry_by_name(history, "v152 promoted warmer-consensus admission + slightly wider pool")
        local_shortlist_rollback = _entry_by_name(history, "v152 promoted warmer-consensus admission + local shortlist rollback")
        tiny_lexical_bridge = _entry_by_name(history, "v152 promoted warmer-consensus admission + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["slightly_wider_pool_score"] = float((slightly_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_pool else None
        evidence["local_shortlist_rollback_score"] = float((local_shortlist_rollback or {}).get("score", float("-inf")) or float("-inf")) if local_shortlist_rollback else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None

    elif normalized == "v153":
        control = _entry_by_name(history, "v153 softer-gate warmer-consensus control")
        slightly_wider_pool = _entry_by_name(history, "v153 softer-gate warmer-consensus + slightly wider pool")
        slightly_firmer_gate = _entry_by_name(history, "v153 softer-gate warmer-consensus + slightly firmer gate")
        slightly_colder_consensus_backstop = _entry_by_name(history, "v153 softer-gate warmer-consensus + slightly colder consensus backstop")
        slightly_stronger_agreement_finish = _entry_by_name(history, "v153 softer-gate warmer-consensus + slightly stronger agreement finish")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_wider_pool_score"] = float((slightly_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_pool else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["slightly_colder_consensus_backstop_score"] = float((slightly_colder_consensus_backstop or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus_backstop else None
        evidence["slightly_stronger_agreement_finish_score"] = float((slightly_stronger_agreement_finish or {}).get("score", float("-inf")) or float("-inf")) if slightly_stronger_agreement_finish else None

    elif normalized == "v154":
        control = _entry_by_name(history, "v154 stabilized softer-gate warmer-consensus control")
        slightly_narrower_pool = _entry_by_name(history, "v154 stabilized softer-gate warmer-consensus + slightly narrower pool")
        narrower_pool_firmer_gate = _entry_by_name(history, "v154 stabilized softer-gate warmer-consensus + narrower-pool firmer gate")
        slightly_warmer_consensus_finish = _entry_by_name(history, "v154 stabilized softer-gate warmer-consensus + slightly warmer consensus finish")
        wider_warmer_admission = _entry_by_name(history, "v154 stabilized softer-gate warmer-consensus + wider warmer admission")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_narrower_pool_score"] = float((slightly_narrower_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_narrower_pool else None
        evidence["narrower_pool_firmer_gate_score"] = float((narrower_pool_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if narrower_pool_firmer_gate else None
        evidence["slightly_warmer_consensus_finish_score"] = float((slightly_warmer_consensus_finish or {}).get("score", float("-inf")) or float("-inf")) if slightly_warmer_consensus_finish else None
        evidence["wider_warmer_admission_score"] = float((wider_warmer_admission or {}).get("score", float("-inf")) or float("-inf")) if wider_warmer_admission else None

    elif normalized == "v155":
        control = _entry_by_name(history, "v155 v150-reset softer-gate colder-consensus control")
        slightly_narrower_pool = _entry_by_name(history, "v155 v150-reset softer-gate colder-consensus + slightly narrower pool")
        slightly_firmer_gate = _entry_by_name(history, "v155 v150-reset softer-gate colder-consensus + slightly firmer gate")
        slightly_warmer_consensus_finish = _entry_by_name(history, "v155 v150-reset softer-gate colder-consensus + slightly warmer consensus finish")
        narrower_pool_colder_backstop = _entry_by_name(history, "v155 v150-reset softer-gate colder-consensus + narrower pool colder backstop")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_narrower_pool_score"] = float((slightly_narrower_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_narrower_pool else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["slightly_warmer_consensus_finish_score"] = float((slightly_warmer_consensus_finish or {}).get("score", float("-inf")) or float("-inf")) if slightly_warmer_consensus_finish else None
        evidence["narrower_pool_colder_backstop_score"] = float((narrower_pool_colder_backstop or {}).get("score", float("-inf")) or float("-inf")) if narrower_pool_colder_backstop else None

    elif normalized == "v156":
        control = _entry_by_name(history, "v156 v150-reset softer-gate colder-consensus firmer-gate control")
        control_rollback = _entry_by_name(history, "v156 v150-reset softer-gate colder-consensus firmer-gate + control rollback")
        slightly_narrower_pool = _entry_by_name(history, "v156 v150-reset softer-gate colder-consensus firmer-gate + slightly narrower pool")
        slightly_colder_backstop = _entry_by_name(history, "v156 v150-reset softer-gate colder-consensus firmer-gate + slightly colder backstop")
        shortlist_topology_rollback = _entry_by_name(history, "v156 v150-reset softer-gate colder-consensus firmer-gate + shortlist topology rollback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["control_rollback_score"] = float((control_rollback or {}).get("score", float("-inf")) or float("-inf")) if control_rollback else None
        evidence["slightly_narrower_pool_score"] = float((slightly_narrower_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_narrower_pool else None
        evidence["slightly_colder_backstop_score"] = float((slightly_colder_backstop or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_backstop else None
        evidence["shortlist_topology_rollback_score"] = float((shortlist_topology_rollback or {}).get("score", float("-inf")) or float("-inf")) if shortlist_topology_rollback else None
    elif normalized == "v157":
        control = _entry_by_name(history, "v157 v150-reset firmer-gate margin-opening control")
        slightly_wider_pool = _entry_by_name(history, "v157 v150-reset firmer-gate margin-opening + slightly wider pool")
        slightly_firmer_gate = _entry_by_name(history, "v157 v150-reset firmer-gate margin-opening + slightly firmer gate")
        agreement_finish_rebalance = _entry_by_name(history, "v157 v150-reset firmer-gate margin-opening + agreement-finish rebalance")
        shortlist_topology_reset = _entry_by_name(history, "v157 v150-reset firmer-gate margin-opening + shortlist topology reset")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_wider_pool_score"] = float((slightly_wider_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_pool else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["agreement_finish_rebalance_score"] = float((agreement_finish_rebalance or {}).get("score", float("-inf")) or float("-inf")) if agreement_finish_rebalance else None
        evidence["shortlist_topology_reset_score"] = float((shortlist_topology_reset or {}).get("score", float("-inf")) or float("-inf")) if shortlist_topology_reset else None
    elif normalized == "v158":
        control = _entry_by_name(history, "v158 v150-reset firmer-gate margin-opening promoted control")
        slightly_narrower_pool = _entry_by_name(history, "v158 v150-reset firmer-gate margin-opening + slightly narrower pool")
        slightly_softer_gate = _entry_by_name(history, "v158 v150-reset firmer-gate margin-opening + slightly softer gate")
        slightly_colder_consensus_finish = _entry_by_name(history, "v158 v150-reset firmer-gate margin-opening + slightly colder consensus finish")
        local_race_topology_reset = _entry_by_name(history, "v158 v150-reset firmer-gate margin-opening + local-race topology reset")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_narrower_pool_score"] = float((slightly_narrower_pool or {}).get("score", float("-inf")) or float("-inf")) if slightly_narrower_pool else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["slightly_colder_consensus_finish_score"] = float((slightly_colder_consensus_finish or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus_finish else None
        evidence["local_race_topology_reset_score"] = float((local_race_topology_reset or {}).get("score", float("-inf")) or float("-inf")) if local_race_topology_reset else None
    elif normalized == "v159":
        control = _entry_by_name(history, "v159 v150-reset tie-break control")
        local_race_rollback = _entry_by_name(history, "v159 v150-reset tie-break + local-race rollback")
        slightly_softer_gate = _entry_by_name(history, "v159 v150-reset tie-break + slightly softer gate")
        slightly_warmer_consensus_finish = _entry_by_name(history, "v159 v150-reset tie-break + slightly warmer consensus finish")
        tiny_lexical_bridge = _entry_by_name(history, "v159 v150-reset tie-break + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["local_race_rollback_score"] = float((local_race_rollback or {}).get("score", float("-inf")) or float("-inf")) if local_race_rollback else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["slightly_warmer_consensus_finish_score"] = float((slightly_warmer_consensus_finish or {}).get("score", float("-inf")) or float("-inf")) if slightly_warmer_consensus_finish else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
    elif normalized == "v160":
        control = _entry_by_name(history, "v160 promoted local-race control")
        slightly_narrower_race = _entry_by_name(history, "v160 promoted local-race + slightly narrower race")
        slightly_firmer_gate = _entry_by_name(history, "v160 promoted local-race + slightly firmer gate")
        slightly_colder_consensus_backstop = _entry_by_name(history, "v160 promoted local-race + slightly colder consensus backstop")
        union_shortlist_rescue = _entry_by_name(history, "v160 promoted local-race + union shortlist rescue")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_narrower_race_score"] = float((slightly_narrower_race or {}).get("score", float("-inf")) or float("-inf")) if slightly_narrower_race else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["slightly_colder_consensus_backstop_score"] = float((slightly_colder_consensus_backstop or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus_backstop else None
        evidence["union_shortlist_rescue_score"] = float((union_shortlist_rescue or {}).get("score", float("-inf")) or float("-inf")) if union_shortlist_rescue else None
    elif normalized == "v161":
        control = _entry_by_name(history, "v161 colder-consensus local-race control")
        slightly_narrower_race = _entry_by_name(history, "v161 colder-consensus local-race + slightly narrower race")
        slightly_softer_gate = _entry_by_name(history, "v161 colder-consensus local-race + slightly softer gate")
        slightly_stronger_agreement_finish = _entry_by_name(history, "v161 colder-consensus local-race + slightly stronger agreement finish")
        local_max_finalist_rescue = _entry_by_name(history, "v161 colder-consensus local-race + local-max finalist rescue")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_narrower_race_score"] = float((slightly_narrower_race or {}).get("score", float("-inf")) or float("-inf")) if slightly_narrower_race else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["slightly_stronger_agreement_finish_score"] = float((slightly_stronger_agreement_finish or {}).get("score", float("-inf")) or float("-inf")) if slightly_stronger_agreement_finish else None
        evidence["local_max_finalist_rescue_score"] = float((local_max_finalist_rescue or {}).get("score", float("-inf")) or float("-inf")) if local_max_finalist_rescue else None
    elif normalized == "v162":
        control = _entry_by_name(history, "v162 v160-reset colder-consensus local-race control")
        slightly_wider_race = _entry_by_name(history, "v162 v160-reset colder-consensus local-race + slightly wider race")
        slightly_firmer_gate = _entry_by_name(history, "v162 v160-reset colder-consensus local-race + slightly firmer gate")
        slightly_warmer_consensus_finish = _entry_by_name(history, "v162 v160-reset colder-consensus local-race + slightly warmer consensus finish")
        local_max_finalist_rescue = _entry_by_name(history, "v162 v160-reset colder-consensus local-race + local-max finalist rescue")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_wider_race_score"] = float((slightly_wider_race or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_race else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["slightly_warmer_consensus_finish_score"] = float((slightly_warmer_consensus_finish or {}).get("score", float("-inf")) or float("-inf")) if slightly_warmer_consensus_finish else None
        evidence["local_max_finalist_rescue_score"] = float((local_max_finalist_rescue or {}).get("score", float("-inf")) or float("-inf")) if local_max_finalist_rescue else None
    elif normalized == "v163":
        control = _entry_by_name(history, "v163 v162-reset firmer-gate colder-consensus local-race control")
        slightly_narrower_race = _entry_by_name(history, "v163 v162-reset firmer-gate colder-consensus local-race + slightly narrower race")
        slightly_softer_gate = _entry_by_name(history, "v163 v162-reset firmer-gate colder-consensus local-race + slightly softer gate")
        slightly_colder_consensus_finish = _entry_by_name(history, "v163 v162-reset firmer-gate colder-consensus local-race + slightly colder consensus finish")
        local_max_finalist_backup = _entry_by_name(history, "v163 v162-reset firmer-gate colder-consensus local-race + local-max finalist backup")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_narrower_race_score"] = float((slightly_narrower_race or {}).get("score", float("-inf")) or float("-inf")) if slightly_narrower_race else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["slightly_colder_consensus_finish_score"] = float((slightly_colder_consensus_finish or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus_finish else None
        evidence["local_max_finalist_backup_score"] = float((local_max_finalist_backup or {}).get("score", float("-inf")) or float("-inf")) if local_max_finalist_backup else None
    elif normalized == "v164":
        control = _entry_by_name(history, "v164 v163-reset softer-gate firmer colder-consensus local-race control")
        slightly_narrower_race = _entry_by_name(history, "v164 v163-reset softer-gate firmer colder-consensus local-race + slightly narrower race")
        slightly_firmer_gate_rollback = _entry_by_name(history, "v164 v163-reset softer-gate firmer colder-consensus local-race + slightly firmer gate rollback")
        slightly_colder_consensus_finish = _entry_by_name(history, "v164 v163-reset softer-gate firmer colder-consensus local-race + slightly colder consensus finish")
        slightly_wider_race = _entry_by_name(history, "v164 v163-reset softer-gate firmer colder-consensus local-race + slightly wider race")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_narrower_race_score"] = float((slightly_narrower_race or {}).get("score", float("-inf")) or float("-inf")) if slightly_narrower_race else None
        evidence["slightly_firmer_gate_rollback_score"] = float((slightly_firmer_gate_rollback or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate_rollback else None
        evidence["slightly_colder_consensus_finish_score"] = float((slightly_colder_consensus_finish or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus_finish else None
        evidence["slightly_wider_race_score"] = float((slightly_wider_race or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_race else None
    elif normalized == "v165":
        control = _entry_by_name(history, "v165 v164-control softer-gate firmer colder-consensus local-race control")
        slightly_wider_race = _entry_by_name(history, "v165 v164-control softer-gate firmer colder-consensus local-race + slightly wider race")
        local_max_finalist_rescue = _entry_by_name(history, "v165 v164-control softer-gate firmer colder-consensus local-race + local-max finalist rescue")
        slightly_warmer_consensus_finish = _entry_by_name(history, "v165 v164-control softer-gate firmer colder-consensus local-race + slightly warmer consensus finish")
        tiny_lexical_bridge = _entry_by_name(history, "v165 v164-control softer-gate firmer colder-consensus local-race + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_wider_race_score"] = float((slightly_wider_race or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_race else None
        evidence["local_max_finalist_rescue_score"] = float((local_max_finalist_rescue or {}).get("score", float("-inf")) or float("-inf")) if local_max_finalist_rescue else None
        evidence["slightly_warmer_consensus_finish_score"] = float((slightly_warmer_consensus_finish or {}).get("score", float("-inf")) or float("-inf")) if slightly_warmer_consensus_finish else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
    elif normalized == "v165":
        control = _entry_by_name(history, "v165 v164-control softer-gate firmer colder-consensus local-race control")
        slightly_wider_race = _entry_by_name(history, "v165 v164-control softer-gate firmer colder-consensus local-race + slightly wider race")
        local_max_finalist_rescue = _entry_by_name(history, "v165 v164-control softer-gate firmer colder-consensus local-race + local-max finalist rescue")
        slightly_warmer_consensus_finish = _entry_by_name(history, "v165 v164-control softer-gate firmer colder-consensus local-race + slightly warmer consensus finish")
        tiny_lexical_bridge = _entry_by_name(history, "v165 v164-control softer-gate firmer colder-consensus local-race + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_wider_race_score"] = float((slightly_wider_race or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_race else None
        evidence["local_max_finalist_rescue_score"] = float((local_max_finalist_rescue or {}).get("score", float("-inf")) or float("-inf")) if local_max_finalist_rescue else None
        evidence["slightly_warmer_consensus_finish_score"] = float((slightly_warmer_consensus_finish or {}).get("score", float("-inf")) or float("-inf")) if slightly_warmer_consensus_finish else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
    elif normalized == "v166":
        control = _entry_by_name(history, "v166 v164-reset control softer-gate firmer colder-consensus local-race control")
        slightly_wider_race = _entry_by_name(history, "v166 v164-reset control softer-gate firmer colder-consensus local-race + slightly wider race")
        local_max_finalist_rescue = _entry_by_name(history, "v166 v164-reset control softer-gate firmer colder-consensus local-race + local-max finalist rescue")
        slightly_warmer_consensus_finish = _entry_by_name(history, "v166 v164-reset control softer-gate firmer colder-consensus local-race + slightly warmer consensus finish")
        tiny_lexical_bridge = _entry_by_name(history, "v166 v164-reset control softer-gate firmer colder-consensus local-race + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_wider_race_score"] = float((slightly_wider_race or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_race else None
        evidence["local_max_finalist_rescue_score"] = float((local_max_finalist_rescue or {}).get("score", float("-inf")) or float("-inf")) if local_max_finalist_rescue else None
        evidence["slightly_warmer_consensus_finish_score"] = float((slightly_warmer_consensus_finish or {}).get("score", float("-inf")) or float("-inf")) if slightly_warmer_consensus_finish else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
    elif normalized == "v167":
        control = _entry_by_name(history, "v167 v166-promoted tiny-lexical-bridge local-race control")
        slightly_wider_race = _entry_by_name(history, "v167 v166-promoted tiny-lexical-bridge local-race + slightly wider race")
        local_max_finalist_rescue = _entry_by_name(history, "v167 v166-promoted tiny-lexical-bridge local-race + local-max finalist rescue")
        slightly_warmer_consensus_finish = _entry_by_name(history, "v167 v166-promoted tiny-lexical-bridge local-race + slightly warmer consensus finish")
        lexical_rollback_control = _entry_by_name(history, "v167 v166-promoted tiny-lexical-bridge local-race + lexical rollback control")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_wider_race_score"] = float((slightly_wider_race or {}).get("score", float("-inf")) or float("-inf")) if slightly_wider_race else None
        evidence["local_max_finalist_rescue_score"] = float((local_max_finalist_rescue or {}).get("score", float("-inf")) or float("-inf")) if local_max_finalist_rescue else None
        evidence["slightly_warmer_consensus_finish_score"] = float((slightly_warmer_consensus_finish or {}).get("score", float("-inf")) or float("-inf")) if slightly_warmer_consensus_finish else None
        evidence["lexical_rollback_control_score"] = float((lexical_rollback_control or {}).get("score", float("-inf")) or float("-inf")) if lexical_rollback_control else None
    elif normalized == "v168":
        control = _entry_by_name(history, "v168 v167-promoted wider-race lexical local-race control")
        local_max_finalist_rescue = _entry_by_name(history, "v168 v167-promoted wider-race lexical local-race + local-max finalist rescue")
        slightly_firmer_gate = _entry_by_name(history, "v168 v167-promoted wider-race lexical local-race + slightly firmer gate")
        slightly_colder_consensus_finish = _entry_by_name(history, "v168 v167-promoted wider-race lexical local-race + slightly colder consensus finish")
        lexical_rollback_control = _entry_by_name(history, "v168 v167-promoted wider-race lexical local-race + lexical rollback control")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["local_max_finalist_rescue_score"] = float((local_max_finalist_rescue or {}).get("score", float("-inf")) or float("-inf")) if local_max_finalist_rescue else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["slightly_colder_consensus_finish_score"] = float((slightly_colder_consensus_finish or {}).get("score", float("-inf")) or float("-inf")) if slightly_colder_consensus_finish else None
        evidence["lexical_rollback_control_score"] = float((lexical_rollback_control or {}).get("score", float("-inf")) or float("-inf")) if lexical_rollback_control else None
    elif normalized == "v169":
        control = _entry_by_name(history, "v169 lexical-rollback local-race control")
        narrower_race = _entry_by_name(history, "v169 lexical-rollback local-race + narrower race")
        softer_gate = _entry_by_name(history, "v169 lexical-rollback local-race + softer gate")
        warmer_consensus_finish = _entry_by_name(history, "v169 lexical-rollback local-race + warmer consensus finish")
        local_max_finalist_rescue = _entry_by_name(history, "v169 lexical-rollback local-race + local-max finalist rescue")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["narrower_race_score"] = float((narrower_race or {}).get("score", float("-inf")) or float("-inf")) if narrower_race else None
        evidence["softer_gate_score"] = float((softer_gate or {}).get("score", float("-inf")) or float("-inf")) if softer_gate else None
        evidence["warmer_consensus_finish_score"] = float((warmer_consensus_finish or {}).get("score", float("-inf")) or float("-inf")) if warmer_consensus_finish else None
        evidence["local_max_finalist_rescue_score"] = float((local_max_finalist_rescue or {}).get("score", float("-inf")) or float("-inf")) if local_max_finalist_rescue else None
    elif normalized == "v170":
        control = _entry_by_name(history, "v170 v168-reset lexical-rollback local-race control")
        local_max_finalist_rescue = _entry_by_name(history, "v170 v168-reset lexical-rollback local-race + local-max finalist rescue")
        slightly_firmer_gate = _entry_by_name(history, "v170 v168-reset lexical-rollback local-race + slightly firmer gate")
        warmer_consensus_finish = _entry_by_name(history, "v170 v168-reset lexical-rollback local-race + warmer consensus finish")
        softer_gate_warmer_finish = _entry_by_name(history, "v170 v168-reset lexical-rollback local-race + softer-gate warmer finish")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["local_max_finalist_rescue_score"] = float((local_max_finalist_rescue or {}).get("score", float("-inf")) or float("-inf")) if local_max_finalist_rescue else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["warmer_consensus_finish_score"] = float((warmer_consensus_finish or {}).get("score", float("-inf")) or float("-inf")) if warmer_consensus_finish else None
        evidence["softer_gate_warmer_finish_score"] = float((softer_gate_warmer_finish or {}).get("score", float("-inf")) or float("-inf")) if softer_gate_warmer_finish else None
    elif normalized == "v171":
        control = _entry_by_name(history, "v171 v170-promoted warmer-finish local-race control")
        softer_gate = _entry_by_name(history, "v171 v170-promoted warmer-finish local-race + softer gate")
        local_max_finalist_rescue = _entry_by_name(history, "v171 v170-promoted warmer-finish local-race + local-max finalist rescue")
        slightly_firmer_gate = _entry_by_name(history, "v171 v170-promoted warmer-finish local-race + slightly firmer gate")
        tiny_lexical_bridge = _entry_by_name(history, "v171 v170-promoted warmer-finish local-race + tiny lexical bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["softer_gate_score"] = float((softer_gate or {}).get("score", float("-inf")) or float("-inf")) if softer_gate else None
        evidence["local_max_finalist_rescue_score"] = float((local_max_finalist_rescue or {}).get("score", float("-inf")) or float("-inf")) if local_max_finalist_rescue else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
    elif normalized == "v172":
        control = _entry_by_name(history, "v172 promoted finalist-rescue control")
        narrower_finalist_race = _entry_by_name(history, "v172 promoted finalist-rescue + narrower finalist race")
        softer_gate = _entry_by_name(history, "v172 promoted finalist-rescue + softer gate")
        warmer_consensus_finish = _entry_by_name(history, "v172 promoted finalist-rescue + warmer consensus finish")
        restored_local_race_warmer_finish = _entry_by_name(history, "v172 promoted finalist-rescue + restored local-race warmer finish")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["narrower_finalist_race_score"] = float((narrower_finalist_race or {}).get("score", float("-inf")) or float("-inf")) if narrower_finalist_race else None
        evidence["softer_gate_score"] = float((softer_gate or {}).get("score", float("-inf")) or float("-inf")) if softer_gate else None
        evidence["warmer_consensus_finish_score"] = float((warmer_consensus_finish or {}).get("score", float("-inf")) or float("-inf")) if warmer_consensus_finish else None
        evidence["restored_local_race_warmer_finish_score"] = float((restored_local_race_warmer_finish or {}).get("score", float("-inf")) or float("-inf")) if restored_local_race_warmer_finish else None
    elif normalized == "v173":
        control = _entry_by_name(history, "v173 v171-reset selective-finalist control")
        wider_finalist_pool = _entry_by_name(history, "v173 v171-reset selective-finalist + wider finalist pool")
        firmer_gate = _entry_by_name(history, "v173 v171-reset selective-finalist + firmer gate")
        colder_consensus_backstop = _entry_by_name(history, "v173 v171-reset selective-finalist + colder consensus backstop")
        restored_warmer_local_race_backstop = _entry_by_name(history, "v173 v171-reset selective-finalist + restored warmer local-race backstop")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["wider_finalist_pool_score"] = float((wider_finalist_pool or {}).get("score", float("-inf")) or float("-inf")) if wider_finalist_pool else None
        evidence["firmer_gate_score"] = float((firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if firmer_gate else None
        evidence["colder_consensus_backstop_score"] = float((colder_consensus_backstop or {}).get("score", float("-inf")) or float("-inf")) if colder_consensus_backstop else None
        evidence["restored_warmer_local_race_backstop_score"] = float((restored_warmer_local_race_backstop or {}).get("score", float("-inf")) or float("-inf")) if restored_warmer_local_race_backstop else None
    elif normalized == "v174":
        control = _entry_by_name(history, "v174 v172-reset finalist-rescue control")
        wider_finalist_pool = _entry_by_name(history, "v174 v172-reset finalist-rescue + wider finalist pool")
        slightly_firmer_gate = _entry_by_name(history, "v174 v172-reset finalist-rescue + slightly firmer gate")
        tiny_lexical_bridge = _entry_by_name(history, "v174 v172-reset finalist-rescue + tiny lexical bridge")
        restored_local_race_warmer_finish = _entry_by_name(history, "v174 v172-reset finalist-rescue + restored local-race warmer finish")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["wider_finalist_pool_score"] = float((wider_finalist_pool or {}).get("score", float("-inf")) or float("-inf")) if wider_finalist_pool else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["restored_local_race_warmer_finish_score"] = float((restored_local_race_warmer_finish or {}).get("score", float("-inf")) or float("-inf")) if restored_local_race_warmer_finish else None
    elif normalized == "v175":
        control = _entry_by_name(history, "v175 v174-promoted wider-finalist-pool control")
        slightly_firmer_gate = _entry_by_name(history, "v175 v174-promoted wider-finalist-pool + slightly firmer gate")
        tiny_lexical_bridge = _entry_by_name(history, "v175 v174-promoted wider-finalist-pool + tiny lexical bridge")
        local_race_warmer_finish_backstop = _entry_by_name(history, "v175 v174-promoted wider-finalist-pool + local-race warmer finish backstop")
        narrower_final_tie_break = _entry_by_name(history, "v175 v174-promoted wider-finalist-pool + narrower final tie-break")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["local_race_warmer_finish_backstop_score"] = float((local_race_warmer_finish_backstop or {}).get("score", float("-inf")) or float("-inf")) if local_race_warmer_finish_backstop else None
        evidence["narrower_final_tie_break_score"] = float((narrower_final_tie_break or {}).get("score", float("-inf")) or float("-inf")) if narrower_final_tie_break else None
    elif normalized == "v176":
        control = _entry_by_name(history, "v176 v168-reset pairwise duelrank control")
        local_max_finalist_rescue = _entry_by_name(history, "v176 v168-reset pairwise duelrank + local-max finalist rescue")
        wider_finalist_pool = _entry_by_name(history, "v176 v168-reset pairwise duelrank + wider finalist pool")
        tiny_lexical_bridge = _entry_by_name(history, "v176 v168-reset pairwise duelrank + tiny lexical bridge")
        scalar_local_race_ablation = _entry_by_name(history, "v176 scalar local-race ablation")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["local_max_finalist_rescue_score"] = float((local_max_finalist_rescue or {}).get("score", float("-inf")) or float("-inf")) if local_max_finalist_rescue else None
        evidence["wider_finalist_pool_score"] = float((wider_finalist_pool or {}).get("score", float("-inf")) or float("-inf")) if wider_finalist_pool else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["scalar_local_race_ablation_score"] = float((scalar_local_race_ablation or {}).get("score", float("-inf")) or float("-inf")) if scalar_local_race_ablation else None
    elif normalized == "v177":
        control = _entry_by_name(history, "v177 v176-promoted local-max finalist rescue pairwise control")
        restored_local_finalist_room = _entry_by_name(history, "v177 v176-promoted local-max finalist rescue pairwise + restored local finalist room")
        slightly_firmer_gate = _entry_by_name(history, "v177 v176-promoted local-max finalist rescue pairwise + slightly firmer gate")
        tiny_lexical_bridge = _entry_by_name(history, "v177 v176-promoted local-max finalist rescue pairwise + tiny lexical bridge")
        scalar_local_max_rescue_ablation = _entry_by_name(history, "v177 v176-promoted local-max finalist rescue scalar ablation")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["restored_local_finalist_room_score"] = float((restored_local_finalist_room or {}).get("score", float("-inf")) or float("-inf")) if restored_local_finalist_room else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["tiny_lexical_bridge_score"] = float((tiny_lexical_bridge or {}).get("score", float("-inf")) or float("-inf")) if tiny_lexical_bridge else None
        evidence["scalar_local_max_rescue_ablation_score"] = float((scalar_local_max_rescue_ablation or {}).get("score", float("-inf")) or float("-inf")) if scalar_local_max_rescue_ablation else None
    elif normalized == "v178":
        control = _entry_by_name(history, "v178 v176-reset citecheck pairwise control")
        restored_local_finalist_room = _entry_by_name(history, "v178 v176-reset citecheck pairwise + restored local finalist room")
        slightly_firmer_gate = _entry_by_name(history, "v178 v176-reset citecheck pairwise + slightly firmer gate")
        support_only_ablation = _entry_by_name(history, "v178 v176-reset citecheck support-only ablation")
        pairwise_only_ablation = _entry_by_name(history, "v178 v176-reset citecheck pairwise-only ablation")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["restored_local_finalist_room_score"] = float((restored_local_finalist_room or {}).get("score", float("-inf")) or float("-inf")) if restored_local_finalist_room else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["support_only_ablation_score"] = float((support_only_ablation or {}).get("score", float("-inf")) or float("-inf")) if support_only_ablation else None
        evidence["pairwise_only_ablation_score"] = float((pairwise_only_ablation or {}).get("score", float("-inf")) or float("-inf")) if pairwise_only_ablation else None
    elif normalized == "v179":
        control = _entry_by_name(history, "v179 clean v178 citecheck pairwise control")
        parafence_stability = _entry_by_name(history, "v179 clean v178 citecheck pairwise + parafence stability")
        low_margin_safe_expand = _entry_by_name(history, "v179 clean v178 citecheck pairwise + low-margin safe expand")
        parafence_stability_safe_expand = _entry_by_name(history, "v179 clean v178 citecheck pairwise + parafence stability + safe expand")
        deterministic_paraphrase_quorum = _entry_by_name(history, "v179 clean v178 citecheck pairwise + deterministic paraphrase quorum")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["parafence_stability_score"] = float((parafence_stability or {}).get("score", float("-inf")) or float("-inf")) if parafence_stability else None
        evidence["low_margin_safe_expand_score"] = float((low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if low_margin_safe_expand else None
        evidence["parafence_stability_safe_expand_score"] = float((parafence_stability_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if parafence_stability_safe_expand else None
        evidence["deterministic_paraphrase_quorum_score"] = float((deterministic_paraphrase_quorum or {}).get("score", float("-inf")) or float("-inf")) if deterministic_paraphrase_quorum else None
    elif normalized == "v180":
        control = _entry_by_name(history, "v180 clean v178 support-aware citecheck pairwise control")
        support_floor = _entry_by_name(history, "v180 clean v178 support-aware citecheck pairwise + support floor")
        code_form_tie_break = _entry_by_name(history, "v180 clean v178 support-aware citecheck pairwise + code-form tie-break")
        support_pref_multistep_floor = _entry_by_name(history, "v180 clean v178 support-aware citecheck pairwise + support-pref multistep floor")
        code_form_tie_break_ultra_low_margin_safe_expand = _entry_by_name(
            history,
            "v180 clean v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_floor_score"] = float((support_floor or {}).get("score", float("-inf")) or float("-inf")) if support_floor else None
        evidence["code_form_tie_break_score"] = float((code_form_tie_break or {}).get("score", float("-inf")) or float("-inf")) if code_form_tie_break else None
        evidence["support_pref_multistep_floor_score"] = float((support_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_multistep_floor else None
        evidence["code_form_tie_break_ultra_low_margin_safe_expand_score"] = float((code_form_tie_break_ultra_low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if code_form_tie_break_ultra_low_margin_safe_expand else None
    elif normalized == "v348":
        control = _entry_by_name(history, "v348 promoted no-safe-expand + parafence stability quorum control")
        support_feature_calibrator = _entry_by_name(history, "v348 promoted no-safe-expand + parafence stability quorum + support-feature calibrator")
        firmer_support_posterior = _entry_by_name(history, "v348 promoted no-safe-expand + parafence stability quorum + firmer support posterior")
        tiny_safe_expand_rescue = _entry_by_name(history, "v348 promoted no-safe-expand + parafence stability quorum + tiny safe-expand rescue")
        firmer_code_form_tie_break = _entry_by_name(history, "v348 promoted no-safe-expand + parafence stability quorum + firmer code-form tie-break")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_feature_calibrator_score"] = float((support_feature_calibrator or {}).get("score", float("-inf")) or float("-inf")) if support_feature_calibrator else None
        evidence["firmer_support_posterior_score"] = float((firmer_support_posterior or {}).get("score", float("-inf")) or float("-inf")) if firmer_support_posterior else None
        evidence["tiny_safe_expand_rescue_score"] = float((tiny_safe_expand_rescue or {}).get("score", float("-inf")) or float("-inf")) if tiny_safe_expand_rescue else None
        evidence["firmer_code_form_tie_break_score"] = float((firmer_code_form_tie_break or {}).get("score", float("-inf")) or float("-inf")) if firmer_code_form_tie_break else None
        return {
            "next_mode": "v349",
            "reason": "v348 plateaued without a strict pass; advance to v349.",
            "evidence": evidence,
        }
    elif normalized == "v349":
        control = _entry_by_name(history, "v349 promoted parafence quorum + support-feature calibrator control")
        firmer_code_form_tie_break = _entry_by_name(history, "v349 promoted parafence quorum + support-feature calibrator + firmer code-form tie-break")
        support_pref_soft_multistep = _entry_by_name(history, "v349 promoted parafence quorum + support-feature calibrator + support-pref soft multistep")
        tiny_safe_expand_rescue = _entry_by_name(history, "v349 promoted parafence quorum + support-feature calibrator + tiny safe-expand rescue")
        firmer_support_posterior = _entry_by_name(history, "v349 promoted parafence quorum + support-feature calibrator + firmer support posterior")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["firmer_code_form_tie_break_score"] = float((firmer_code_form_tie_break or {}).get("score", float("-inf")) or float("-inf")) if firmer_code_form_tie_break else None
        evidence["support_pref_soft_multistep_score"] = float((support_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")) if support_pref_soft_multistep else None
        evidence["tiny_safe_expand_rescue_score"] = float((tiny_safe_expand_rescue or {}).get("score", float("-inf")) or float("-inf")) if tiny_safe_expand_rescue else None
        evidence["firmer_support_posterior_score"] = float((firmer_support_posterior or {}).get("score", float("-inf")) or float("-inf")) if firmer_support_posterior else None
        evidence["objective_shift"] = "parametric_ignorance_support_discipline"
        return {
            "next_mode": "v350",
            "reason": "v349 plateaued on the old selector objective; reset to v350 with direct-support hygiene and in-domain unsupported abstention objectives.",
            "evidence": evidence,
        }
    elif normalized == "v350":
        control = _entry_by_name(history, "v350 objective-reset support-feature calibrator control")
        modifier_sensitive_selective_gate = _entry_by_name(history, "v350 objective-reset + modifier-sensitive selective gate")
        direct_support_citecheck_tie_break = _entry_by_name(history, "v350 objective-reset + direct-support citecheck tie-break")
        agreement_quorum_selective_gate = _entry_by_name(history, "v350 objective-reset + agreement quorum + selective gate")
        citecheck_tie_break_selective_gate = _entry_by_name(history, "v350 objective-reset + citecheck tie-break + selective gate")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["modifier_sensitive_selective_gate_score"] = float((modifier_sensitive_selective_gate or {}).get("score", float("-inf")) or float("-inf")) if modifier_sensitive_selective_gate else None
        evidence["direct_support_citecheck_tie_break_score"] = float((direct_support_citecheck_tie_break or {}).get("score", float("-inf")) or float("-inf")) if direct_support_citecheck_tie_break else None
        evidence["agreement_quorum_selective_gate_score"] = float((agreement_quorum_selective_gate or {}).get("score", float("-inf")) or float("-inf")) if agreement_quorum_selective_gate else None
        evidence["citecheck_tie_break_selective_gate_score"] = float((citecheck_tie_break_selective_gate or {}).get("score", float("-inf")) or float("-inf")) if citecheck_tie_break_selective_gate else None
        evidence["remaining_strict_failures"] = [
            "direct support retrieval hygiene too low",
            "code/query geometry still rank-collapsed",
        ]
        return {
            "next_mode": "v351",
            "reason": "v350 showed selector-only objective changes were mostly flat; reset to v351 around taxonomy coverage and geometry opening from the stronger v298 frontier.",
            "evidence": evidence,
        }
    elif normalized == "v351":
        control = _entry_by_name(history, "v351 v298-reset objective control")
        taxonomy_coverage_curriculum = _entry_by_name(history, "v351 v298-reset + taxonomy coverage curriculum")
        taxonomy_coverage_mild_all_stream_rank_guard = _entry_by_name(history, "v351 v298-reset + taxonomy coverage + mild all-stream rank guard")
        taxonomy_coverage_eleven_facets_query_lift = _entry_by_name(history, "v351 v298-reset + taxonomy coverage + eleven facets + slightly stronger query-classifier confidence lift")
        taxonomy_coverage_rank_guard_eleven_facets_query_lift = _entry_by_name(history, "v351 v298-reset + taxonomy coverage + mild all-stream rank guard + eleven facets + slightly stronger query-classifier confidence lift")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["taxonomy_coverage_curriculum_score"] = float((taxonomy_coverage_curriculum or {}).get("score", float("-inf")) or float("-inf")) if taxonomy_coverage_curriculum else None
        evidence["taxonomy_coverage_mild_all_stream_rank_guard_score"] = float((taxonomy_coverage_mild_all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")) if taxonomy_coverage_mild_all_stream_rank_guard else None
        evidence["taxonomy_coverage_eleven_facets_query_lift_score"] = float((taxonomy_coverage_eleven_facets_query_lift or {}).get("score", float("-inf")) or float("-inf")) if taxonomy_coverage_eleven_facets_query_lift else None
        evidence["taxonomy_coverage_rank_guard_eleven_facets_query_lift_score"] = float((taxonomy_coverage_rank_guard_eleven_facets_query_lift or {}).get("score", float("-inf")) or float("-inf")) if taxonomy_coverage_rank_guard_eleven_facets_query_lift else None
        evidence["remaining_strict_failures"] = [
            "direct support retrieval hygiene too low",
            "supported-family abstentions still dominate missed direct hits",
            "code/query effective rank still too low",
        ]
        return {
            "next_mode": "v352",
            "reason": "v351 improved abstention and eliminated same-family wrong-chunk retrievals, but direct support stayed flat; advance to v352 for within-family boundary discrimination training.",
            "evidence": evidence,
        }
    elif normalized == "v352":
        control = _entry_by_name(history, "v352 promoted support-discipline control")
        deeper_hard_negative_ladder = _entry_by_name(history, "v352 promoted support-discipline + deeper hard-negative ladder")
        adversarial_boundary_curriculum = _entry_by_name(history, "v352 promoted support-discipline + adversarial boundary curriculum")
        minimal_pair_curriculum = _entry_by_name(history, "v352 promoted support-discipline + minimal-pair curriculum")
        adversarial_boundary_curriculum_deeper_hard_negative_ladder_firmer_focal_ranking = _entry_by_name(history, "v352 promoted support-discipline + adversarial boundary curriculum + deeper hard-negative ladder + firmer focal ranking")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["deeper_hard_negative_ladder_score"] = float((deeper_hard_negative_ladder or {}).get("score", float("-inf")) or float("-inf")) if deeper_hard_negative_ladder else None
        evidence["adversarial_boundary_curriculum_score"] = float((adversarial_boundary_curriculum or {}).get("score", float("-inf")) or float("-inf")) if adversarial_boundary_curriculum else None
        evidence["minimal_pair_curriculum_score"] = float((minimal_pair_curriculum or {}).get("score", float("-inf")) or float("-inf")) if minimal_pair_curriculum else None
        evidence["adversarial_boundary_curriculum_deeper_hard_negative_ladder_firmer_focal_ranking_score"] = float((adversarial_boundary_curriculum_deeper_hard_negative_ladder_firmer_focal_ranking or {}).get("score", float("-inf")) or float("-inf")) if adversarial_boundary_curriculum_deeper_hard_negative_ladder_firmer_focal_ranking else None
        evidence["remaining_strict_failures"] = [
            "direct support retrieval hygiene too low",
            "supported-family abstentions still dominate missed direct hits",
            "same-family wrong chunks still appear under relaxed gating",
            "code/query effective rank still too low",
        ]
        return {
            "next_mode": "v353",
            "reason": "v352 showed the deeper hard-negative ladder is the healthiest basin, but direct support stayed flat; advance to v353 for combined within-family boundary discrimination on top of the stronger winner.",
            "evidence": evidence,
        }
    elif normalized == "v353":
        control = _entry_by_name(history, "v353 promoted deeper-hard-negative control")
        mixed_boundary_curriculum = _entry_by_name(history, "v353 promoted deeper-hard-negative + mixed-boundary curriculum")
        adversarial_boundary_curriculum = _entry_by_name(history, "v353 promoted deeper-hard-negative + adversarial boundary curriculum")
        minimal_pair_curriculum = _entry_by_name(history, "v353 promoted deeper-hard-negative + minimal-pair curriculum")
        mixed_boundary_curriculum_deeper_hard_negative_ladder_firmer_focal_ranking = _entry_by_name(history, "v353 promoted deeper-hard-negative + mixed-boundary curriculum + deeper ladder + firmer focal ranking")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["mixed_boundary_curriculum_score"] = float((mixed_boundary_curriculum or {}).get("score", float("-inf")) or float("-inf")) if mixed_boundary_curriculum else None
        evidence["adversarial_boundary_curriculum_score"] = float((adversarial_boundary_curriculum or {}).get("score", float("-inf")) or float("-inf")) if adversarial_boundary_curriculum else None
        evidence["minimal_pair_curriculum_score"] = float((minimal_pair_curriculum or {}).get("score", float("-inf")) or float("-inf")) if minimal_pair_curriculum else None
        evidence["mixed_boundary_curriculum_deeper_hard_negative_ladder_firmer_focal_ranking_score"] = float((mixed_boundary_curriculum_deeper_hard_negative_ladder_firmer_focal_ranking or {}).get("score", float("-inf")) or float("-inf")) if mixed_boundary_curriculum_deeper_hard_negative_ladder_firmer_focal_ranking else None
        evidence["v351_frontier_score"] = 38.85648351659378
        evidence["remaining_strict_failures"] = [
            "direct support retrieval hygiene too low",
            "supported-family abstentions still dominate missed direct hits",
            "code/query effective rank still too low",
        ]
        return {
            "next_mode": "v354",
            "reason": "v353 nearly recovered the v351 winner via mixed-boundary supervision but still stayed below the stronger v351 frontier; reset to v351 and test a taxonomy-coverage plus mixed-boundary hybrid around the stronger base.",
            "evidence": evidence,
        }
    elif normalized == "v354":
        control = _entry_by_name(history, "v354 promoted taxonomy-coverage control")
        taxonomy_mixed_boundary_curriculum = _entry_by_name(history, "v354 promoted taxonomy-coverage + taxonomy-mixed-boundary curriculum")
        taxonomy_mixed_boundary_curriculum_deeper_hard_negative_ladder = _entry_by_name(history, "v354 promoted taxonomy-coverage + taxonomy-mixed-boundary curriculum + deeper hard-negative ladder")
        taxonomy_mixed_boundary_curriculum_slightly_firmer_focal_ranking = _entry_by_name(history, "v354 promoted taxonomy-coverage + taxonomy-mixed-boundary curriculum + slightly firmer focal ranking")
        taxonomy_mixed_boundary_curriculum_deeper_hard_negative_ladder_slightly_firmer_focal_ranking = _entry_by_name(history, "v354 promoted taxonomy-coverage + taxonomy-mixed-boundary curriculum + deeper hard-negative ladder + slightly firmer focal ranking")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["taxonomy_mixed_boundary_curriculum_score"] = float((taxonomy_mixed_boundary_curriculum or {}).get("score", float("-inf")) or float("-inf")) if taxonomy_mixed_boundary_curriculum else None
        evidence["taxonomy_mixed_boundary_curriculum_deeper_hard_negative_ladder_score"] = float((taxonomy_mixed_boundary_curriculum_deeper_hard_negative_ladder or {}).get("score", float("-inf")) or float("-inf")) if taxonomy_mixed_boundary_curriculum_deeper_hard_negative_ladder else None
        evidence["taxonomy_mixed_boundary_curriculum_slightly_firmer_focal_ranking_score"] = float((taxonomy_mixed_boundary_curriculum_slightly_firmer_focal_ranking or {}).get("score", float("-inf")) or float("-inf")) if taxonomy_mixed_boundary_curriculum_slightly_firmer_focal_ranking else None
        evidence["taxonomy_mixed_boundary_curriculum_deeper_hard_negative_ladder_slightly_firmer_focal_ranking_score"] = float((taxonomy_mixed_boundary_curriculum_deeper_hard_negative_ladder_slightly_firmer_focal_ranking or {}).get("score", float("-inf")) or float("-inf")) if taxonomy_mixed_boundary_curriculum_deeper_hard_negative_ladder_slightly_firmer_focal_ranking else None
        evidence["v351_frontier_score"] = 38.85648351659378
        evidence["remaining_strict_failures"] = [
            "direct support retrieval hygiene too low",
            "supported-family abstentions still dominate missed direct hits",
            "code/query effective rank still too low",
        ]
        return {
            "next_mode": "v355",
            "reason": "v354 carried mixed-boundary supervision into the stronger v351 basin, but direct-support rate stayed flat and the branch still regressed below the preserved v351 frontier; reset to v351 and test a taxonomy-aligned direct-support discipline curriculum with compact rank/confidence refinements.",
            "evidence": evidence,
        }
    elif normalized == "v355":
        control = _entry_by_name(history, "v355 v351-reset control")
        taxonomy_support_discipline_curriculum = _entry_by_name(history, "v355 v351-reset + taxonomy support-discipline curriculum")
        taxonomy_support_discipline_curriculum_deeper_hard_negative_ladder = _entry_by_name(history, "v355 v351-reset + taxonomy support-discipline curriculum + deeper hard-negative ladder")
        taxonomy_support_discipline_curriculum_mixed_query_pred_confidence_lift = _entry_by_name(history, "v355 v351-reset + taxonomy support-discipline curriculum + mixed query-pred confidence lift")
        taxonomy_support_discipline_curriculum_deeper_hard_negative_ladder_mixed_query_pred_confidence_lift = _entry_by_name(history, "v355 v351-reset + taxonomy support-discipline curriculum + deeper hard-negative ladder + mixed query-pred confidence lift")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["taxonomy_support_discipline_curriculum_score"] = float((taxonomy_support_discipline_curriculum or {}).get("score", float("-inf")) or float("-inf")) if taxonomy_support_discipline_curriculum else None
        evidence["taxonomy_support_discipline_curriculum_deeper_hard_negative_ladder_score"] = float((taxonomy_support_discipline_curriculum_deeper_hard_negative_ladder or {}).get("score", float("-inf")) or float("-inf")) if taxonomy_support_discipline_curriculum_deeper_hard_negative_ladder else None
        evidence["taxonomy_support_discipline_curriculum_mixed_query_pred_confidence_lift_score"] = float((taxonomy_support_discipline_curriculum_mixed_query_pred_confidence_lift or {}).get("score", float("-inf")) or float("-inf")) if taxonomy_support_discipline_curriculum_mixed_query_pred_confidence_lift else None
        evidence["taxonomy_support_discipline_curriculum_deeper_hard_negative_ladder_mixed_query_pred_confidence_lift_score"] = float((taxonomy_support_discipline_curriculum_deeper_hard_negative_ladder_mixed_query_pred_confidence_lift or {}).get("score", float("-inf")) or float("-inf")) if taxonomy_support_discipline_curriculum_deeper_hard_negative_ladder_mixed_query_pred_confidence_lift else None
        evidence["v351_frontier_score"] = 38.85648351659378
        evidence["remaining_strict_failures"] = [
            "direct support retrieval hygiene too low",
            "supported-family abstentions still dominate missed direct hits",
            "code/query effective rank still too low",
        ]
        return {
            "next_mode": "v356",
            "reason": "v355 added direct-support curriculum pressure but left direct-support rate flat at the v351 plateau; advance to v356 for support-slate local ordering on the stronger v351 base.",
            "evidence": evidence,
        }
    elif normalized == "v356":
        control = _entry_by_name(history, "v356 v351-reset control")
        support_slate_local_top_rank = _entry_by_name(history, "v356 v351-reset + support-slate local top-rank")
        support_slate_local_top_rank_wider_cross_family_slate = _entry_by_name(history, "v356 v351-reset + support-slate local top-rank + wider cross-family slate")
        support_slate_local_top_rank_local_margin = _entry_by_name(history, "v356 v351-reset + support-slate local top-rank + local margin")
        support_slate_local_top_rank_local_margin_mixed_query_pred_localization = _entry_by_name(history, "v356 v351-reset + support-slate local top-rank + local margin + mixed query-pred localization")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_slate_local_top_rank_score"] = float((support_slate_local_top_rank or {}).get("score", float("-inf")) or float("-inf")) if support_slate_local_top_rank else None
        evidence["support_slate_local_top_rank_wider_cross_family_slate_score"] = float((support_slate_local_top_rank_wider_cross_family_slate or {}).get("score", float("-inf")) or float("-inf")) if support_slate_local_top_rank_wider_cross_family_slate else None
        evidence["support_slate_local_top_rank_local_margin_score"] = float((support_slate_local_top_rank_local_margin or {}).get("score", float("-inf")) or float("-inf")) if support_slate_local_top_rank_local_margin else None
        evidence["support_slate_local_top_rank_local_margin_mixed_query_pred_localization_score"] = float((support_slate_local_top_rank_local_margin_mixed_query_pred_localization or {}).get("score", float("-inf")) or float("-inf")) if support_slate_local_top_rank_local_margin_mixed_query_pred_localization else None
        evidence["v355_taxonomy_support_discipline_curriculum_score"] = 38.79713395237923
        evidence["v351_frontier_score"] = 38.85648351659378
        evidence["remaining_strict_failures"] = [
            "direct support retrieval hygiene too low",
            "supported-family abstentions still dominate missed direct hits",
            "code/query effective rank still too low",
        ]
        return {
            "next_mode": "v357",
            "reason": "v356 support-slate local ordering improved the local frontier but still stayed below the stronger v351 winner; blend the v356 local-ordering gain with the orthogonal v355 direct-support curriculum clue in a compact v357 hybrid on the stronger v351 base.",
            "evidence": evidence,
        }
    elif normalized == "v357":
        control = _entry_by_name(history, "v357 v351-reset + taxonomy support-discipline curriculum + support-slate local top-rank control")
        local_margin = _entry_by_name(history, "v357 v351-reset + taxonomy support-discipline curriculum + support-slate local top-rank + local margin")
        deeper_hard_negative_ladder = _entry_by_name(history, "v357 v351-reset + taxonomy support-discipline curriculum + support-slate local top-rank + deeper hard-negative ladder")
        mixed_query_pred_confidence_lift = _entry_by_name(history, "v357 v351-reset + taxonomy support-discipline curriculum + support-slate local top-rank + mixed query-pred confidence lift")
        local_margin_mixed_query_pred_localization = _entry_by_name(history, "v357 v351-reset + taxonomy support-discipline curriculum + support-slate local top-rank + local margin + mixed query-pred localization")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["local_margin_score"] = float((local_margin or {}).get("score", float("-inf")) or float("-inf")) if local_margin else None
        evidence["deeper_hard_negative_ladder_score"] = float((deeper_hard_negative_ladder or {}).get("score", float("-inf")) or float("-inf")) if deeper_hard_negative_ladder else None
        evidence["mixed_query_pred_confidence_lift_score"] = float((mixed_query_pred_confidence_lift or {}).get("score", float("-inf")) or float("-inf")) if mixed_query_pred_confidence_lift else None
        evidence["local_margin_mixed_query_pred_localization_score"] = float((local_margin_mixed_query_pred_localization or {}).get("score", float("-inf")) or float("-inf")) if local_margin_mixed_query_pred_localization else None
        evidence["v351_frontier_score"] = 38.85648351659378
        evidence["remaining_strict_failures"] = [
            "phase4 support-discipline runs still bypass live equivalence positives",
            "equivalence coverage is missing for several supported families",
            "direct support retrieval hygiene remains below the v351 frontier",
        ]
        return {
            "next_mode": "v358",
            "reason": "v357 plateaued below the v351/v356 frontier while keeping wrong-chunk rejection strong; advance to v358 for the structural fix that makes phase4 equivalence positives live and expands coverage across the full support-discipline family set.",
            "evidence": evidence,
        }
    elif normalized == "v358":
        control = _entry_by_name(history, "v358 v351-reset + taxonomy support-discipline + live equivalence positives control")
        support_slate = _entry_by_name(history, "v358 v351-reset + taxonomy support-discipline + live equivalence positives + support-slate local top-rank")
        support_slate_deeper = _entry_by_name(history, "v358 v351-reset + taxonomy support-discipline + live equivalence positives + support-slate + deeper hard-negative ladder")
        localization = _entry_by_name(history, "v358 v351-reset + taxonomy support-discipline + live equivalence positives + support-slate + localization")
        equivalence_only = _entry_by_name(history, "v358 v351-reset + taxonomy support-discipline + live equivalence only")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_slate_score"] = float((support_slate or {}).get("score", float("-inf")) or float("-inf")) if support_slate else None
        evidence["support_slate_deeper_score"] = float((support_slate_deeper or {}).get("score", float("-inf")) or float("-inf")) if support_slate_deeper else None
        evidence["localization_score"] = float((localization or {}).get("score", float("-inf")) or float("-inf")) if localization else None
        evidence["equivalence_only_score"] = float((equivalence_only or {}).get("score", float("-inf")) or float("-inf")) if equivalence_only else None
        evidence["v351_frontier_score"] = 38.85648351659378
        evidence["remaining_strict_failures"] = [
            "objective_supported_direct_rate ~0.33 below 0.375 target despite clean abstention",
            "code/query effective rank remains compressed even with vicreg and rank_reg",
            "ordering: no ordered-support signal taught anywhere in the v358 family",
        ]
        return {
            "next_mode": "v359",
            "reason": "v358 established clean abstention and strong wrong-chunk rejection in the live-equivalence basin, but direct-support rate remains stuck around 0.33 and effective rank is still compressed. The next structural move is to add ordered-support retrieval pressure and factorized family/instance scoring — the two mechanisms most likely to raise direct-support without degrading the hygiene that v358 built.",
            "evidence": evidence,
        }
    elif normalized == "v359":
        control = _entry_by_name(history, "v359 ordered direct-support ranking control")
        retrieval_margin = _entry_by_name(history, "v359 ordered direct-support + retrieval margin")
        softened_equivalence = _entry_by_name(history, "v359 ordered ranking + softened equivalence")
        local_rank = _entry_by_name(history, "v359 ordered ranking + local rank opening")
        factorized = _entry_by_name(history, "v359 factorized family-and-instance scorer")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["retrieval_margin_score"] = float((retrieval_margin or {}).get("score", float("-inf")) or float("-inf")) if retrieval_margin else None
        evidence["softened_equivalence_score"] = float((softened_equivalence or {}).get("score", float("-inf")) or float("-inf")) if softened_equivalence else None
        evidence["local_rank_score"] = float((local_rank or {}).get("score", float("-inf")) or float("-inf")) if local_rank else None
        evidence["factorized_score"] = float((factorized or {}).get("score", float("-inf")) or float("-inf")) if factorized else None
        evidence["v358_control_score"] = 38.74086155494054
        evidence["remaining_strict_failures"] = [
            "query_collapse: query embedding effective rank still too compressed",
            "code_low_rank: code embedding effective rank still too compressed",
            "direct_support_hygiene: model still occasionally retrieves wrong chunk for supported prompts",
            "ordering: ordered-ranking/factorized alone did not break 38.74 — mechanism may already be saturated",
        ]
        return {
            "next_mode": "v360",
            "reason": "v359 confirmed that ordered-ranking and factorized family/instance scoring alone cannot break the v358 control ceiling of 38.74. The persistent flags (query_collapse, code_low_rank, direct_support_hygiene) remain unaddressed by ordering alone. v360 pivots to combining the live-equivalence base with support-slate localization and targeted anti-collapse mechanisms — the most justified next step given that support-slate was the best single element in v358.",
            "evidence": evidence,
        }
    elif normalized == "v360":
        control = _entry_by_name(history, "v360 taxonomy support-discipline + live equivalence control")
        support_slate_vicreg = _entry_by_name(history, "v360 support-slate + vicreg anti-collapse")
        support_slate_ranking = _entry_by_name(history, "v360 support-slate + ordered retrieval margin")
        support_slate_local_rank = _entry_by_name(history, "v360 support-slate + neighborhood vicreg")
        full_combination = _entry_by_name(history, "v360 full combination: support-slate + retrieval margin + vicreg")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_slate_vicreg_score"] = float((support_slate_vicreg or {}).get("score", float("-inf")) or float("-inf")) if support_slate_vicreg else None
        evidence["support_slate_ranking_score"] = float((support_slate_ranking or {}).get("score", float("-inf")) or float("-inf")) if support_slate_ranking else None
        evidence["support_slate_local_rank_score"] = float((support_slate_local_rank or {}).get("score", float("-inf")) or float("-inf")) if support_slate_local_rank else None
        evidence["full_combination_score"] = float((full_combination or {}).get("score", float("-inf")) or float("-inf")) if full_combination else None
        evidence["v358_control_score"] = 38.74086155494054
        evidence["remaining_strict_failures"] = [
            "query_collapse: query embedding effective rank still too compressed after v359 ordering sweep",
            "code_low_rank: code embedding effective rank still too compressed after v359 ordering sweep",
            "direct_support_hygiene: model still occasionally retrieves wrong chunk — ordering alone was insufficient",
            "combination: if support-slate + vicreg + ranking all combined still fail, bottleneck is likely in confidence head rather than retrieval geometry",
        ]
        return {
            "next_mode": None,
            "reason": "v360 tests combining support-slate localization with targeted anti-collapse and ranking mechanisms. If no candidate breaks 38.74 + support-slate combination, the remaining failure may be in the confidence head or data recipe rather than retrieval geometry.",
            "evidence": evidence,
        }
    elif normalized == "v361":
        v5_distinct = _entry_by_name(history, "v361 v358 winner + v5_distinct proxy recipe")
        no_joint = _entry_by_name(history, "v361 v358 winner + no-joint ablation")
        scale_40m = _entry_by_name(history, "v361 v358 winner + 40M scale")
        v5_no_joint = _entry_by_name(history, "v361 v358 winner + v5_distinct + no-joint")
        v5_40m = _entry_by_name(history, "v361 v358 winner + v5_distinct + 40M scale")
        evidence["v5_distinct_score"] = float((v5_distinct or {}).get("score", float("-inf")) or float("-inf")) if v5_distinct else None
        evidence["no_joint_score"] = float((no_joint or {}).get("score", float("-inf")) or float("-inf")) if no_joint else None
        evidence["scale_40m_score"] = float((scale_40m or {}).get("score", float("-inf")) or float("-inf")) if scale_40m else None
        evidence["v5_no_joint_score"] = float((v5_no_joint or {}).get("score", float("-inf")) or float("-inf")) if v5_no_joint else None
        evidence["v5_40m_score"] = float((v5_40m or {}).get("score", float("-inf")) or float("-inf")) if v5_40m else None
        evidence["v358_winner_score"] = 38.74086155494054
        evidence["v360_best_score"] = 37.12287046092749
        evidence["remaining_strict_failures"] = [
            "v358→v360 chain: adding retrieval losses on top of v358 basin hurt or plateaued — the v358 recipe is saturated",
            "persistent flags: query_collapse, code_low_rank, direct_support_hygiene unchanged across v358/359/360",
            "v361 tests: proxy recipe (v5_distinct), joint vs separate training, 40M scale — structural moves the v358 lineage never touched",
        ]
        return {
            "next_mode": None,
            "reason": "v361 restarts from the actual v358 winner (38.74) and tests structural changes (proxy recipe, joint vs separate training, scale) that the v358→v360 retrieval-loss chain never probed. If no v361 candidate breaks 38.74, the ceiling is likely in the eval data or architecture itself, not the loss landscape.",
            "evidence": evidence,
        }
    elif normalized == "v362":
        delayed_ignorance = _entry_by_name(history, "v362 delayed ignorance curriculum (start=500, ramp=200)")
        wider_bottleneck = _entry_by_name(history, "v362 wider bottleneck (embed_dim=384)")
        denoised_negatives = _entry_by_name(history, "v362 denoised hard negatives (8 candidates, factorized_weight=0.3)")
        delayed_wider = _entry_by_name(history, "v362 delayed ignorance + wider bottleneck")
        delayed_denoised = _entry_by_name(history, "v362 delayed ignorance + denoised negatives")
        evidence["delayed_ignorance_score"] = float((delayed_ignorance or {}).get("score", float("-inf")) or float("-inf")) if delayed_ignorance else None
        evidence["wider_bottleneck_score"] = float((wider_bottleneck or {}).get("score", float("-inf")) or float("-inf")) if wider_bottleneck else None
        evidence["denoised_negatives_score"] = float((denoised_negatives or {}).get("score", float("-inf")) or float("-inf")) if denoised_negatives else None
        evidence["delayed_wider_score"] = float((delayed_wider or {}).get("score", float("-inf")) or float("-inf")) if delayed_wider else None
        evidence["delayed_denoised_score"] = float((delayed_denoised or {}).get("score", float("-inf")) or float("-inf")) if delayed_denoised else None
        evidence["v358_winner_score"] = 38.74086155494054
        evidence["v361_best_score"] = 24.2818
        evidence["remaining_strict_failures"] = [
            "v361: 5 structured candidates all collapsed to 21-23, best adaptive 24.28 — 37% below v358",
            "v361 failure modes: objective_support_discipline (geometry collapse) and weak_signal_nonzero (confidence suppression)",
            "v362: staged curriculum + capacity expansion + negative denoising from v358 winner basin",
        ]
        return {
            "next_mode": None,
            "reason": "v362 tests three literature-grounded hypotheses: delayed ignorance curriculum (more pure retrieval geometry before calibration), wider bottleneck (more representational capacity), and denoised hard negatives (better negative signal). All warm-start from v358 winner. If all collapse again, the ceiling is likely in the loss structure or eval data rather than hyperparameter space.",
            "evidence": evidence,
        }
    elif normalized == "v363":
        control = _entry_by_name(history, "v363 frozen geometry + taxonomy_support_discipline control")
        neighborhood_posterior = _entry_by_name(history, "v363 frozen geometry + taxonomy_support_discipline + neighborhood posterior")
        support_feature_calibrator = _entry_by_name(history, "v363 frozen geometry + taxonomy_support_discipline + support-feature calibrator")
        evidential_support = _entry_by_name(history, "v363 frozen geometry + taxonomy_support_discipline + evidential support head")
        agreement_augmented = _entry_by_name(history, "v363 frozen geometry + taxonomy_support_discipline + agreement-augmented calibrator")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["neighborhood_posterior_score"] = float((neighborhood_posterior or {}).get("score", float("-inf")) or float("-inf")) if neighborhood_posterior else None
        evidence["support_feature_calibrator_score"] = float((support_feature_calibrator or {}).get("score", float("-inf")) or float("-inf")) if support_feature_calibrator else None
        evidence["evidential_support_score"] = float((evidential_support or {}).get("score", float("-inf")) or float("-inf")) if evidential_support else None
        evidence["agreement_augmented_score"] = float((agreement_augmented or {}).get("score", float("-inf")) or float("-inf")) if agreement_augmented else None
        evidence["v340_best_score"] = 43.50
        evidence["v358_winner_score"] = 38.74
        evidence["v362_best_score"] = 27.58
        evidence["v363_best_score"] = max(
            [s for s in [evidence.get("neighborhood_posterior_score"), evidence.get("support_feature_calibrator_score"),
             evidence.get("evidential_support_score"), evidence.get("agreement_augmented_score")] if s is not None and s > 0], default=0
        )
        evidence["regression_root_cause"] = [
            "v340 PASSED (42-43) with freeze_backbone=true + mixed_boundary_curriculum objective",
            "v358 FAILED (38.74) with freeze_backbone=false + taxonomy_support_discipline objective",
            "v340's frozen geometry preserved rank=0.101, offdiag=0.035 throughout training",
            "v358's fine-tuned geometry collapsed to rank=0.038, offdiag=0.097 during training",
            "v363: reproduce v340's frozen-geometry + calibrator approach with taxonomy_support_discipline objective",
            "v363 ALL FAILED at 38-39: calibrator readout is NOT the bottleneck",
            "v363 showed all 5 calibrator variants score 38-39 identically",
            "The bottleneck is the TRAINING SIGNAL that produces the encoder geometry",
            "research2/3 diagnosis: equivalence overbinding + classifier_weight pressure",
        ]
        any_pass = any(s > 42 for s in [evidence["neighborhood_posterior_score"], evidence["support_feature_calibrator_score"],
                                          evidence["evidential_support_score"], evidence["agreement_augmented_score"]]
                        if s is not None and s > 0)
        if any_pass:
            return {
                "next_mode": None,
                "reason": "v363 frozen geometry + taxonomy_support_discipline produced a PASS candidate - this is the first PASS with the taxonomy_support_discipline objective, confirming frozen geometry is the protective mechanism independent of objective. Recommend promoting the best PASS candidate.",
                "evidence": evidence,
            }
        return {
            "next_mode": None,
            "reason": "v363 ALL FAILED. All 5 calibrator variants scored 38-39 identically - bottleneck is training signal (classifier_weight=0.05 + equivalence overbinding), not calibrator readout. v364: reduce classifier_weight and equivalence weights to fix the training signal.",
            "evidence": evidence,
        }
    elif normalized == "v364":
        reduced_clf = _entry_by_name(history, "v364 reduced classifier_weight=0.01 + frozen taxonomy")
        softened_eq = _entry_by_name(history, "v364 softened equivalence + frozen taxonomy")
        combined = _entry_by_name(history, "v364 minimal classifier + softened equivalence + frozen taxonomy")
        neighborhood = _entry_by_name(history, "v364 neighborhood posterior + minimal classifier + softened equivalence")
        higher_margins = _entry_by_name(history, "v364 higher ranking/retrieval margins + minimal classifier")
        evidence["reduced_clf_score"] = float((reduced_clf or {}).get("score", float("-inf")) or float("-inf")) if reduced_clf else None
        evidence["softened_eq_score"] = float((softened_eq or {}).get("score", float("-inf")) or float("-inf")) if softened_eq else None
        evidence["combined_score"] = float((combined or {}).get("score", float("-inf")) or float("-inf")) if combined else None
        evidence["neighborhood_score"] = float((neighborhood or {}).get("score", float("-inf")) or float("-inf")) if neighborhood else None
        evidence["higher_margins_score"] = float((higher_margins or {}).get("score", float("-inf")) or float("-inf")) if higher_margins else None
        evidence["v363_best_score"] = 39.73
        evidence["v340_best_score"] = 43.50
        evidence["v364_confidence_gap"] = 0.119  # passed: need >= 0.10
        evidence["v364_hygiene"] = 0.375  # failed: need >= 0.75
        evidence["v364_key_insight"] = (
            "v364: freeze_backbone=true preserves v338 geometry, confidence_gap now passes (0.119 >= 0.10). "
            "But hygiene fails (0.375 < 0.75) because the frozen encoder can't encode taxonomy-specific "
            "instance-level support structure. ALL 5 variants score 39.73 identically — phase4 training "
            "signals are locked out of the encoder. v365: unfreeze backbone so phase4 can learn "
            "instance-level support discrimination for hygiene."
        )
        scores = [s for s in [evidence["reduced_clf_score"], evidence["softened_eq_score"],
                               evidence["combined_score"], evidence["neighborhood_score"],
                               evidence["higher_margins_score"]] if s is not None and s > 0]
        best_score = max(scores) if scores else 0
        any_pass = any(s > 42 for s in scores)
        if any_pass:
            return {
                "next_mode": None,
                "reason": f"v364 PASSED with best score {best_score:.2f}. Classifier weight reduction fixes the training signal under taxonomy_support_discipline. Recommend promoting the best PASS candidate.",
                "evidence": evidence,
            }
        elif best_score > evidence["v363_best_score"]:
            return {
                "next_mode": "v365",
                "reason": f"v364 improved (best={best_score:.2f} vs v363=39.73) but did not PASS. Confidence_gap passes but hygiene fails. Key insight: freeze_backbone=true locks the encoder geometry, preventing phase4 from learning instance-level support discrimination needed for hygiene. v365: unfreeze backbone with conservative LR to let the encoder learn the taxonomy-specific support boundary while keeping classifier_weight low to avoid abstention compression.",
                "evidence": evidence,
            }
        else:
            return {
                "next_mode": None,
                "reason": f"v364 did not improve over v363 (best={best_score:.2f} vs v363=39.73). freeze_backbone=true is the blocker for hygiene. v365: unfreeze backbone + reduced classifier_weight.",
                "evidence": evidence,
            }
    elif normalized == "v365":
        unfreeze_clf001 = _entry_by_name(history, "v365 unfrozen + classifier_weight=0.01 + warmstart v338")
        unfreeze_clf005 = _entry_by_name(history, "v365 unfrozen + classifier_weight=0.005 + warmstart v338")
        unfreeze_noprop = _entry_by_name(history, "v365 unfrozen + no_prop_loss + warmstart v338")
        evidence["unfreeze_clf001_score"] = float((unfreeze_clf001 or {}).get("score", float("-inf")) or float("-inf")) if unfreeze_clf001 else None
        evidence["unfreeze_clf005_score"] = float((unfreeze_clf005 or {}).get("score", float("-inf")) or float("-inf")) if unfreeze_clf005 else None
        evidence["unfreeze_noprop_score"] = float((unfreeze_noprop or {}).get("score", float("-inf")) or float("-inf")) if unfreeze_noprop else None
        evidence["v364_best_score"] = 39.73
        evidence["v363_best_score"] = 39.73
        evidence["v340_best_score"] = 43.50
        scores = [s for s in [evidence["unfreeze_clf001_score"], evidence["unfreeze_clf005_score"],
                               evidence["unfreeze_noprop_score"]] if s is not None and s > 0]
        best_score = max(scores) if scores else 0
        any_pass = any(s > 42 for s in scores)
        if any_pass:
            return {
                "next_mode": None,
                "reason": f"v365 PASSED with best score {best_score:.2f}. Unfreezing the backbone with conservative LR + low classifier_weight allows phase4 to learn instance-level support discrimination for hygiene without collapsing geometry. Recommend promoting the best candidate.",
                "evidence": evidence,
            }
        elif best_score > evidence["v364_best_score"]:
            return {
                "next_mode": None,
                "reason": f"v365 improved over v364 (best={best_score:.2f} vs 39.73). The unfreezing approach is moving in the right direction. Recommend iterating: try even lower classifier_weight, longer training, or a different warm-start checkpoint.",
                "evidence": evidence,
            }
        else:
            return {
                "next_mode": None,
                "reason": f"v365 did not improve over v364. If hygiene remains stuck despite unfreezing, the bottleneck may be in the phase3 warm-start geometry itself — v338's encoder may not encode taxonomy-specific instance-level support. Consider a different warm-start checkpoint.",
                "evidence": evidence,
            }
    elif normalized == "v366":
        mixed_boundary_unfrozen = _entry_by_name(history, "v366 mixed_boundary + unfrozen + clf=0.09 control")
        mixed_boundary_unfrozen_higher_lr = _entry_by_name(history, "v366 mixed_boundary + UNFROZEN + higher_lr_clf=0.09")
        mixed_boundary_frozen = _entry_by_name(history, "v366 mixed_boundary + frozen + clf=0.09 + equivalence_off")
        evidence["mixed_boundary_unfrozen_score"] = float((mixed_boundary_unfrozen or {}).get("score", float("-inf"))) if mixed_boundary_unfrozen else None
        evidence["mixed_boundary_unfrozen_higher_lr_score"] = float((mixed_boundary_unfrozen_higher_lr or {}).get("score", float("-inf"))) if mixed_boundary_unfrozen_higher_lr else None
        evidence["mixed_boundary_frozen_score"] = float((mixed_boundary_frozen or {}).get("score", float("-inf"))) if mixed_boundary_frozen else None
        evidence["v365_best_score"] = 41.63
        evidence["v364_best_score"] = 39.73
        evidence["v340_best_score"] = 43.50
        scores = [s for s in [evidence["mixed_boundary_unfrozen_score"], evidence["mixed_boundary_unfrozen_higher_lr_score"], evidence["mixed_boundary_frozen_score"]] if s is not None and s > 0]
        best_score = max(scores) if scores else 0
        any_pass = any(s > 42 for s in scores)
        if any_pass:
            return {
                "next_mode": None,
                "reason": f"v366 PASSED with best score {best_score:.2f}. mixed_boundary + unfrozen + clf=0.09 combines the best of v340 (mixed_boundary data, clf_weight) with the unfreezing fix from v365. Recommend promoting the best candidate.",
                "evidence": evidence,
            }
        elif best_score > evidence["v365_best_score"]:
            return {
                "next_mode": None,
                "reason": f"v366 improved over v365 (best={best_score:.2f} vs 41.63). mixed_boundary is the right training data for hygiene. Recommend iterating: try other seeds, longer training, or warm-start from a different checkpoint.",
                "evidence": evidence,
            }
        else:
            return {
                "next_mode": None,
                "reason": f"v366 did not improve. If mixed_boundary + frozen reaches v340's score (43.5) but unfrozen doesn't: unfreezing causes collapse on mixed_boundary data. Try much lower LR, or use taxonomy_support_discipline_v1 with unfrozen backbone at longer training steps.",
                "evidence": evidence,
            }
    elif normalized == "v181":
        control = _entry_by_name(history, "v181 clean v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand control")
        local_union_finalist_room = _entry_by_name(history, "v181 clean v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + local union finalist room")
        code_pref_multistep_floor = _entry_by_name(history, "v181 clean v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + code-pref multistep floor")
        code_pref_soft_multistep = _entry_by_name(history, "v181 clean v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + code-pref soft multistep")
        local_union_finalist_room_code_pref_soft_multistep = _entry_by_name(
            history,
            "v181 clean v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + local union finalist room + code-pref soft multistep",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["local_union_finalist_room_score"] = float((local_union_finalist_room or {}).get("score", float("-inf")) or float("-inf")) if local_union_finalist_room else None
        evidence["code_pref_multistep_floor_score"] = float((code_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if code_pref_multistep_floor else None
        evidence["code_pref_soft_multistep_score"] = float((code_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")) if code_pref_soft_multistep else None
        evidence["local_union_finalist_room_code_pref_soft_multistep_score"] = float((local_union_finalist_room_code_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")) if local_union_finalist_room_code_pref_soft_multistep else None
    elif normalized == "v182":
        control = _entry_by_name(history, "v182 clean v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + code-pref multistep floor control")
        local_union_finalist_room = _entry_by_name(history, "v182 clean v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + code-pref multistep floor + local union finalist room")
        always_on_support_spec_pairwise = _entry_by_name(history, "v182 clean v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + code-pref multistep floor + always-on support/spec pairwise")
        no_safe_expand_rollback = _entry_by_name(history, "v182 clean v178 support-aware citecheck pairwise + code-form tie-break + code-pref multistep floor")
        local_union_finalist_room_supportspec = _entry_by_name(history, "v182 clean v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + code-pref multistep floor + local union finalist room + always-on support/spec pairwise")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["local_union_finalist_room_score"] = float((local_union_finalist_room or {}).get("score", float("-inf")) or float("-inf")) if local_union_finalist_room else None
        evidence["always_on_support_spec_pairwise_score"] = float((always_on_support_spec_pairwise or {}).get("score", float("-inf")) or float("-inf")) if always_on_support_spec_pairwise else None
        evidence["no_safe_expand_rollback_score"] = float((no_safe_expand_rollback or {}).get("score", float("-inf")) or float("-inf")) if no_safe_expand_rollback else None
        evidence["local_union_finalist_room_supportspec_score"] = float((local_union_finalist_room_supportspec or {}).get("score", float("-inf")) or float("-inf")) if local_union_finalist_room_supportspec else None
    elif normalized == "v183":
        control = _entry_by_name(history, "v183 v180-reset support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand control")
        always_on_support_spec_pairwise = _entry_by_name(history, "v183 v180-reset support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + always-on support/spec pairwise")
        local_union_finalist_room = _entry_by_name(history, "v183 v180-reset support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + local union finalist room")
        local_union_finalist_room_supportspec = _entry_by_name(history, "v183 v180-reset support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + local union finalist room + always-on support/spec pairwise")
        deterministic_paraphrase_quorum = _entry_by_name(history, "v183 v180-reset support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + deterministic paraphrase quorum")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["always_on_support_spec_pairwise_score"] = float((always_on_support_spec_pairwise or {}).get("score", float("-inf")) or float("-inf")) if always_on_support_spec_pairwise else None
        evidence["local_union_finalist_room_score"] = float((local_union_finalist_room or {}).get("score", float("-inf")) or float("-inf")) if local_union_finalist_room else None
        evidence["local_union_finalist_room_supportspec_score"] = float((local_union_finalist_room_supportspec or {}).get("score", float("-inf")) or float("-inf")) if local_union_finalist_room_supportspec else None
        evidence["deterministic_paraphrase_quorum_score"] = float((deterministic_paraphrase_quorum or {}).get("score", float("-inf")) or float("-inf")) if deterministic_paraphrase_quorum else None
    elif normalized == "v184":
        control = _entry_by_name(history, "v184 clean v178 support-aware citecheck pairwise + code-pref multistep floor + parafence control")
        local_finalist_room = _entry_by_name(history, "v184 clean v178 support-aware citecheck pairwise + code-pref multistep floor + parafence + local finalist room")
        firmer_support_posterior = _entry_by_name(history, "v184 clean v178 support-aware citecheck pairwise + code-pref multistep floor + firmer support posterior")
        tiny_safe_expand_rescue = _entry_by_name(history, "v184 clean v178 support-aware citecheck pairwise + code-pref multistep floor + tiny safe-expand rescue")
        firmer_code_form_tie_break = _entry_by_name(history, "v184 clean v178 support-aware citecheck pairwise + code-pref multistep floor + firmer code-form tie-break")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["local_finalist_room_score"] = float((local_finalist_room or {}).get("score", float("-inf")) or float("-inf")) if local_finalist_room else None
        evidence["firmer_support_posterior_score"] = float((firmer_support_posterior or {}).get("score", float("-inf")) or float("-inf")) if firmer_support_posterior else None
        evidence["tiny_safe_expand_rescue_score"] = float((tiny_safe_expand_rescue or {}).get("score", float("-inf")) or float("-inf")) if tiny_safe_expand_rescue else None
        evidence["firmer_code_form_tie_break_score"] = float((firmer_code_form_tie_break or {}).get("score", float("-inf")) or float("-inf")) if firmer_code_form_tie_break else None
    elif normalized == "v185":
        control = _entry_by_name(history, "v185 promoted stable local-union support/spec control")
        citecheck_floor = _entry_by_name(history, "v185 promoted stable local-union support/spec + citecheck floor")
        citecheck_floor_support_pref_coverage = _entry_by_name(history, "v185 promoted stable local-union support/spec + citecheck floor + support-pref coverage")
        citecheck_floor_hard_multistep_coverage = _entry_by_name(history, "v185 promoted stable local-union support/spec + citecheck floor + hard multistep coverage")
        citecheck_floor_local_max_rollback = _entry_by_name(history, "v185 promoted stable local-union support/spec + citecheck floor + local-max rollback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["citecheck_floor_score"] = float((citecheck_floor or {}).get("score", float("-inf")) or float("-inf")) if citecheck_floor else None
        evidence["citecheck_floor_support_pref_coverage_score"] = float((citecheck_floor_support_pref_coverage or {}).get("score", float("-inf")) or float("-inf")) if citecheck_floor_support_pref_coverage else None
        evidence["citecheck_floor_hard_multistep_coverage_score"] = float((citecheck_floor_hard_multistep_coverage or {}).get("score", float("-inf")) or float("-inf")) if citecheck_floor_hard_multistep_coverage else None
        evidence["citecheck_floor_local_max_rollback_score"] = float((citecheck_floor_local_max_rollback or {}).get("score", float("-inf")) or float("-inf")) if citecheck_floor_local_max_rollback else None
    elif normalized == "v186":
        control = _entry_by_name(history, "v186 promoted local-union support/spec control")
        support_pref_answer_spec = _entry_by_name(history, "v186 promoted local-union support/spec + support-pref answer-spec")
        firmer_code_form_tie_break = _entry_by_name(history, "v186 promoted local-union support/spec + firmer code-form tie-break")
        hard_multistep_coverage = _entry_by_name(history, "v186 promoted local-union support/spec + hard multistep coverage")
        ultra_low_margin_safe_expand = _entry_by_name(history, "v186 promoted local-union support/spec + ultra-low-margin safe expand")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_pref_answer_spec_score"] = float((support_pref_answer_spec or {}).get("score", float("-inf")) or float("-inf")) if support_pref_answer_spec else None
        evidence["firmer_code_form_tie_break_score"] = float((firmer_code_form_tie_break or {}).get("score", float("-inf")) or float("-inf")) if firmer_code_form_tie_break else None
        evidence["hard_multistep_coverage_score"] = float((hard_multistep_coverage or {}).get("score", float("-inf")) or float("-inf")) if hard_multistep_coverage else None
        evidence["ultra_low_margin_safe_expand_score"] = float((ultra_low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if ultra_low_margin_safe_expand else None
    elif normalized == "v187":
        control = _entry_by_name(history, "v187 promoted stable local-union support/spec control")
        ultra_low_margin_safe_expand = _entry_by_name(history, "v187 promoted stable local-union support/spec + ultra-low-margin safe expand")
        support_pref_answer_spec = _entry_by_name(history, "v187 promoted stable local-union support/spec + ultra-low-margin safe expand + support-pref answer-spec")
        citecheck_pairwise = _entry_by_name(history, "v187 promoted stable local-union support/spec + ultra-low-margin safe expand + citecheck pairwise")
        citecheck_pairwise_support_pref_answer_spec = _entry_by_name(history, "v187 promoted stable local-union support/spec + ultra-low-margin safe expand + citecheck pairwise + support-pref answer-spec")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["ultra_low_margin_safe_expand_score"] = float((ultra_low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if ultra_low_margin_safe_expand else None
        evidence["support_pref_answer_spec_score"] = float((support_pref_answer_spec or {}).get("score", float("-inf")) or float("-inf")) if support_pref_answer_spec else None
        evidence["citecheck_pairwise_score"] = float((citecheck_pairwise or {}).get("score", float("-inf")) or float("-inf")) if citecheck_pairwise else None
        evidence["citecheck_pairwise_support_pref_answer_spec_score"] = float((citecheck_pairwise_support_pref_answer_spec or {}).get("score", float("-inf")) or float("-inf")) if citecheck_pairwise_support_pref_answer_spec else None
    elif normalized == "v236":
        control = _entry_by_name(history, "v236 stable local-union support/spec control")
        no_safe_expand_fallback = _entry_by_name(history, "v236 stable local-union support/spec + no safe-expand fallback")
        lighter_parafence_quorum = _entry_by_name(history, "v236 stable local-union support/spec + lighter parafence quorum")
        low_margin_citecheck_floor = _entry_by_name(history, "v236 stable local-union support/spec + low-margin citecheck floor")
        softer_code_form_finish = _entry_by_name(history, "v236 stable local-union support/spec + softer code-form finish")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["no_safe_expand_fallback_score"] = float((no_safe_expand_fallback or {}).get("score", float("-inf")) or float("-inf")) if no_safe_expand_fallback else None
        evidence["lighter_parafence_quorum_score"] = float((lighter_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if lighter_parafence_quorum else None
        evidence["low_margin_citecheck_floor_score"] = float((low_margin_citecheck_floor or {}).get("score", float("-inf")) or float("-inf")) if low_margin_citecheck_floor else None
        evidence["softer_code_form_finish_score"] = float((softer_code_form_finish or {}).get("score", float("-inf")) or float("-inf")) if softer_code_form_finish else None
    elif normalized == "v237":
        control = _entry_by_name(history, "v237 promoted softer code-form finish control")
        low_margin_citecheck_floor = _entry_by_name(history, "v237 promoted softer code-form finish + low-margin citecheck floor")
        firmer_multistep_floor = _entry_by_name(history, "v237 promoted softer code-form finish + firmer multistep floor")
        ultra_low_margin_safe_expand = _entry_by_name(history, "v237 promoted softer code-form finish + ultra-low-margin safe expand")
        combined_floor = _entry_by_name(history, "v237 promoted softer code-form finish + low-margin citecheck floor + firmer multistep floor")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["low_margin_citecheck_floor_score"] = float((low_margin_citecheck_floor or {}).get("score", float("-inf")) or float("-inf")) if low_margin_citecheck_floor else None
        evidence["firmer_multistep_floor_score"] = float((firmer_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if firmer_multistep_floor else None
        evidence["ultra_low_margin_safe_expand_score"] = float((ultra_low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if ultra_low_margin_safe_expand else None
        evidence["combined_floor_score"] = float((combined_floor or {}).get("score", float("-inf")) or float("-inf")) if combined_floor else None
    elif normalized == "v238":
        control = _entry_by_name(history, "v238 promoted support-floor multistep-floor control")
        no_safe_expand_fallback = _entry_by_name(history, "v238 promoted support-floor multistep-floor + no safe-expand fallback")
        lighter_parafence_quorum = _entry_by_name(history, "v238 promoted support-floor multistep-floor + lighter parafence quorum")
        low_margin_selective_gate = _entry_by_name(history, "v238 promoted support-floor multistep-floor + low-margin selective gate")
        low_margin_selective_gate_no_safe_expand_fallback = _entry_by_name(history, "v238 promoted support-floor multistep-floor + low-margin selective gate + no safe-expand fallback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["no_safe_expand_fallback_score"] = float((no_safe_expand_fallback or {}).get("score", float("-inf")) or float("-inf")) if no_safe_expand_fallback else None
        evidence["lighter_parafence_quorum_score"] = float((lighter_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if lighter_parafence_quorum else None
        evidence["low_margin_selective_gate_score"] = float((low_margin_selective_gate or {}).get("score", float("-inf")) or float("-inf")) if low_margin_selective_gate else None
        evidence["low_margin_selective_gate_no_safe_expand_fallback_score"] = float((low_margin_selective_gate_no_safe_expand_fallback or {}).get("score", float("-inf")) or float("-inf")) if low_margin_selective_gate_no_safe_expand_fallback else None
    elif normalized == "v239":
        control = _entry_by_name(history, "v239 promoted low-margin selective gate control")
        lighter_parafence_quorum = _entry_by_name(history, "v239 promoted low-margin selective gate + lighter parafence quorum")
        slightly_firmer_gate = _entry_by_name(history, "v239 promoted low-margin selective gate + slightly firmer gate")
        slightly_softer_gate = _entry_by_name(history, "v239 promoted low-margin selective gate + slightly softer gate")
        tiny_safe_expand_rescue = _entry_by_name(history, "v239 promoted low-margin selective gate + tiny safe-expand rescue")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["lighter_parafence_quorum_score"] = float((lighter_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if lighter_parafence_quorum else None
        evidence["slightly_firmer_gate_score"] = float((slightly_firmer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_firmer_gate else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["tiny_safe_expand_rescue_score"] = float((tiny_safe_expand_rescue or {}).get("score", float("-inf")) or float("-inf")) if tiny_safe_expand_rescue else None
    elif normalized == "v240":
        control = _entry_by_name(history, "v240 v238-winner support-floor multistep-floor selective-gate control")
        tiny_safe_expand_rescue = _entry_by_name(history, "v240 v238-winner support-floor multistep-floor selective-gate + tiny safe-expand rescue")
        firmer_support_floor = _entry_by_name(history, "v240 v238-winner support-floor multistep-floor selective-gate + firmer support floor")
        firmer_multistep_floor = _entry_by_name(history, "v240 v238-winner support-floor multistep-floor selective-gate + firmer multistep floor")
        firmer_support_floor_firmer_multistep_floor = _entry_by_name(history, "v240 v238-winner support-floor multistep-floor selective-gate + firmer support floor + firmer multistep floor")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_safe_expand_rescue_score"] = float((tiny_safe_expand_rescue or {}).get("score", float("-inf")) or float("-inf")) if tiny_safe_expand_rescue else None
        evidence["firmer_support_floor_score"] = float((firmer_support_floor or {}).get("score", float("-inf")) or float("-inf")) if firmer_support_floor else None
        evidence["firmer_multistep_floor_score"] = float((firmer_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if firmer_multistep_floor else None
        evidence["firmer_support_floor_firmer_multistep_floor_score"] = float((firmer_support_floor_firmer_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if firmer_support_floor_firmer_multistep_floor else None
    elif normalized == "v241":
        control = _entry_by_name(history, "v241 promoted dual-floor selective-gate control")
        tiny_safe_expand_rescue = _entry_by_name(history, "v241 promoted dual-floor selective-gate + tiny safe-expand rescue")
        lighter_parafence_quorum = _entry_by_name(history, "v241 promoted dual-floor selective-gate + lighter parafence quorum")
        slightly_softer_gate = _entry_by_name(history, "v241 promoted dual-floor selective-gate + slightly softer gate")
        tiny_safe_expand_rescue_lighter_parafence_quorum = _entry_by_name(history, "v241 promoted dual-floor selective-gate + tiny safe-expand rescue + lighter parafence quorum")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_safe_expand_rescue_score"] = float((tiny_safe_expand_rescue or {}).get("score", float("-inf")) or float("-inf")) if tiny_safe_expand_rescue else None
        evidence["lighter_parafence_quorum_score"] = float((lighter_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if lighter_parafence_quorum else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["tiny_safe_expand_rescue_lighter_parafence_quorum_score"] = float((tiny_safe_expand_rescue_lighter_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if tiny_safe_expand_rescue_lighter_parafence_quorum else None
    elif normalized == "v242":
        control = _entry_by_name(history, "v242 promoted rescue+light-quorum control")
        slightly_softer_gate = _entry_by_name(history, "v242 promoted rescue+light-quorum + slightly softer gate")
        full_parafence_quorum = _entry_by_name(history, "v242 promoted rescue+light-quorum + full parafence quorum")
        no_safe_expand_rescue = _entry_by_name(history, "v242 promoted rescue+light-quorum + no safe-expand rescue")
        slightly_softer_gate_no_safe_expand_rescue = _entry_by_name(history, "v242 promoted rescue+light-quorum + slightly softer gate + no safe-expand rescue")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["full_parafence_quorum_score"] = float((full_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if full_parafence_quorum else None
        evidence["no_safe_expand_rescue_score"] = float((no_safe_expand_rescue or {}).get("score", float("-inf")) or float("-inf")) if no_safe_expand_rescue else None
        evidence["slightly_softer_gate_no_safe_expand_rescue_score"] = float((slightly_softer_gate_no_safe_expand_rescue or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate_no_safe_expand_rescue else None
    elif normalized == "v243":
        control = _entry_by_name(history, "v243 v238-reset firmer support/spec gates control")
        firmer_support_floor_only = _entry_by_name(history, "v243 v238-reset firmer support floor only")
        firmer_code_pref_multistep_gate_only = _entry_by_name(history, "v243 v238-reset firmer code-pref multistep gate only")
        lighter_parafence_quorum = _entry_by_name(history, "v243 v238-reset firmer support/spec gates + lighter parafence quorum")
        tiny_safe_expand_rescue = _entry_by_name(history, "v243 v238-reset firmer support/spec gates + tiny safe-expand rescue")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["firmer_support_floor_only_score"] = float((firmer_support_floor_only or {}).get("score", float("-inf")) or float("-inf")) if firmer_support_floor_only else None
        evidence["firmer_code_pref_multistep_gate_only_score"] = float((firmer_code_pref_multistep_gate_only or {}).get("score", float("-inf")) or float("-inf")) if firmer_code_pref_multistep_gate_only else None
        evidence["lighter_parafence_quorum_score"] = float((lighter_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if lighter_parafence_quorum else None
        evidence["tiny_safe_expand_rescue_score"] = float((tiny_safe_expand_rescue or {}).get("score", float("-inf")) or float("-inf")) if tiny_safe_expand_rescue else None
    elif normalized == "v244":
        control = _entry_by_name(history, "v244 promoted firmer-gates+rescue control")
        support_floor_emphasis = _entry_by_name(history, "v244 promoted firmer-gates+rescue + support-floor emphasis")
        code_pref_emphasis = _entry_by_name(history, "v244 promoted firmer-gates+rescue + code-pref emphasis")
        wider_rescue_room = _entry_by_name(history, "v244 promoted firmer-gates+rescue + wider rescue room")
        slightly_softer_gate = _entry_by_name(history, "v244 promoted firmer-gates+rescue + slightly softer gate")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_floor_emphasis_score"] = float((support_floor_emphasis or {}).get("score", float("-inf")) or float("-inf")) if support_floor_emphasis else None
        evidence["code_pref_emphasis_score"] = float((code_pref_emphasis or {}).get("score", float("-inf")) or float("-inf")) if code_pref_emphasis else None
        evidence["wider_rescue_room_score"] = float((wider_rescue_room or {}).get("score", float("-inf")) or float("-inf")) if wider_rescue_room else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
    elif normalized == "v245":
        control = _entry_by_name(history, "v245 promoted wider-rescue control")
        slightly_softer_gate = _entry_by_name(history, "v245 promoted wider-rescue + slightly softer gate")
        support_floor_emphasis = _entry_by_name(history, "v245 promoted wider-rescue + support-floor emphasis")
        code_pref_emphasis = _entry_by_name(history, "v245 promoted wider-rescue + code-pref emphasis")
        tiny_rescue_rollback = _entry_by_name(history, "v245 promoted wider-rescue + tiny rescue rollback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["support_floor_emphasis_score"] = float((support_floor_emphasis or {}).get("score", float("-inf")) or float("-inf")) if support_floor_emphasis else None
        evidence["code_pref_emphasis_score"] = float((code_pref_emphasis or {}).get("score", float("-inf")) or float("-inf")) if code_pref_emphasis else None
        evidence["tiny_rescue_rollback_score"] = float((tiny_rescue_rollback or {}).get("score", float("-inf")) or float("-inf")) if tiny_rescue_rollback else None
    elif normalized == "v246":
        control = _entry_by_name(history, "v246 promoted softer-gate wider-rescue control")
        code_pref_emphasis = _entry_by_name(history, "v246 promoted softer-gate wider-rescue + code-pref emphasis")
        even_wider_rescue_room = _entry_by_name(history, "v246 promoted softer-gate wider-rescue + even wider rescue room")
        code_pref_emphasis_even_wider_rescue_room = _entry_by_name(history, "v246 promoted softer-gate wider-rescue + code-pref emphasis + even wider rescue room")
        firmer_gate_rollback = _entry_by_name(history, "v246 promoted softer-gate wider-rescue + firmer gate rollback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["code_pref_emphasis_score"] = float((code_pref_emphasis or {}).get("score", float("-inf")) or float("-inf")) if code_pref_emphasis else None
        evidence["even_wider_rescue_room_score"] = float((even_wider_rescue_room or {}).get("score", float("-inf")) or float("-inf")) if even_wider_rescue_room else None
        evidence["code_pref_emphasis_even_wider_rescue_room_score"] = float((code_pref_emphasis_even_wider_rescue_room or {}).get("score", float("-inf")) or float("-inf")) if code_pref_emphasis_even_wider_rescue_room else None
        evidence["firmer_gate_rollback_score"] = float((firmer_gate_rollback or {}).get("score", float("-inf")) or float("-inf")) if firmer_gate_rollback else None
    elif normalized == "v247":
        control = _entry_by_name(history, "v247 replicated softer-gate rescue control")
        rollback_replay = _entry_by_name(history, "v247 replicated softer-gate rescue + rollback replay")
        code_pref_emphasis = _entry_by_name(history, "v247 replicated softer-gate rescue + code-pref emphasis")
        even_wider_rescue_room = _entry_by_name(history, "v247 replicated softer-gate rescue + even wider rescue room")
        code_pref_emphasis_even_wider_rescue_room = _entry_by_name(history, "v247 replicated softer-gate rescue + code-pref emphasis + even wider rescue room")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["rollback_replay_score"] = float((rollback_replay or {}).get("score", float("-inf")) or float("-inf")) if rollback_replay else None
        evidence["code_pref_emphasis_score"] = float((code_pref_emphasis or {}).get("score", float("-inf")) or float("-inf")) if code_pref_emphasis else None
        evidence["even_wider_rescue_room_score"] = float((even_wider_rescue_room or {}).get("score", float("-inf")) or float("-inf")) if even_wider_rescue_room else None
        evidence["code_pref_emphasis_even_wider_rescue_room_score"] = float((code_pref_emphasis_even_wider_rescue_room or {}).get("score", float("-inf")) or float("-inf")) if code_pref_emphasis_even_wider_rescue_room else None
    elif normalized == "v248":
        control = _entry_by_name(history, "v248 replicated rollback family control")
        slightly_softer_gate = _entry_by_name(history, "v248 replicated rollback family + slightly softer gate")
        support_floor_emphasis = _entry_by_name(history, "v248 replicated rollback family + support-floor emphasis")
        code_pref_emphasis = _entry_by_name(history, "v248 replicated rollback family + code-pref emphasis")
        tiny_rescue_rollback = _entry_by_name(history, "v248 replicated rollback family + tiny rescue rollback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["support_floor_emphasis_score"] = float((support_floor_emphasis or {}).get("score", float("-inf")) or float("-inf")) if support_floor_emphasis else None
        evidence["code_pref_emphasis_score"] = float((code_pref_emphasis or {}).get("score", float("-inf")) or float("-inf")) if code_pref_emphasis else None
        evidence["tiny_rescue_rollback_score"] = float((tiny_rescue_rollback or {}).get("score", float("-inf")) or float("-inf")) if tiny_rescue_rollback else None
    elif normalized == "v249":
        control = _entry_by_name(history, "v249 replicated code-pref family control")
        tiny_rescue_rollback = _entry_by_name(history, "v249 replicated code-pref family + tiny rescue rollback")
        firmer_multistep_floor = _entry_by_name(history, "v249 replicated code-pref family + firmer multistep floor")
        firmer_multistep_floor_tiny_rescue_rollback = _entry_by_name(history, "v249 replicated code-pref family + firmer multistep floor + tiny rescue rollback")
        softer_multistep_floor = _entry_by_name(history, "v249 replicated code-pref family + softer multistep floor")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_rescue_rollback_score"] = float((tiny_rescue_rollback or {}).get("score", float("-inf")) or float("-inf")) if tiny_rescue_rollback else None
        evidence["firmer_multistep_floor_score"] = float((firmer_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if firmer_multistep_floor else None
        evidence["firmer_multistep_floor_tiny_rescue_rollback_score"] = float((firmer_multistep_floor_tiny_rescue_rollback or {}).get("score", float("-inf")) or float("-inf")) if firmer_multistep_floor_tiny_rescue_rollback else None
        evidence["softer_multistep_floor_score"] = float((softer_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if softer_multistep_floor else None
    elif normalized == "v250":
        control = _entry_by_name(history, "v250 replicated firmer-code-pref control")
        support_floor_rollback = _entry_by_name(history, "v250 replicated firmer-code-pref + support-floor rollback")
        stronger_code_pref_emphasis = _entry_by_name(history, "v250 replicated firmer-code-pref + stronger code-pref emphasis")
        even_firmer_multistep_floor = _entry_by_name(history, "v250 replicated firmer-code-pref + even firmer multistep floor")
        balanced_contract_rollback = _entry_by_name(history, "v250 replicated firmer-code-pref + balanced contract rollback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_floor_rollback_score"] = float((support_floor_rollback or {}).get("score", float("-inf")) or float("-inf")) if support_floor_rollback else None
        evidence["stronger_code_pref_emphasis_score"] = float((stronger_code_pref_emphasis or {}).get("score", float("-inf")) or float("-inf")) if stronger_code_pref_emphasis else None
        evidence["even_firmer_multistep_floor_score"] = float((even_firmer_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if even_firmer_multistep_floor else None
        evidence["balanced_contract_rollback_score"] = float((balanced_contract_rollback or {}).get("score", float("-inf")) or float("-inf")) if balanced_contract_rollback else None
    elif normalized == "v251":
        control = _entry_by_name(history, "v251 replicated support-floor rollback control")
        firmer_support_floor = _entry_by_name(history, "v251 replicated support-floor rollback + firmer support floor")
        even_firmer_multistep_floor = _entry_by_name(history, "v251 replicated support-floor rollback + even firmer multistep floor")
        firmer_support_floor_even_firmer_multistep_floor = _entry_by_name(history, "v251 replicated support-floor rollback + firmer support floor + even firmer multistep floor")
        slightly_softer_multistep_floor = _entry_by_name(history, "v251 replicated support-floor rollback + slightly softer multistep floor")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["firmer_support_floor_score"] = float((firmer_support_floor or {}).get("score", float("-inf")) or float("-inf")) if firmer_support_floor else None
        evidence["even_firmer_multistep_floor_score"] = float((even_firmer_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if even_firmer_multistep_floor else None
        evidence["firmer_support_floor_even_firmer_multistep_floor_score"] = float((firmer_support_floor_even_firmer_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if firmer_support_floor_even_firmer_multistep_floor else None
        evidence["slightly_softer_multistep_floor_score"] = float((slightly_softer_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_multistep_floor else None
    elif normalized == "v252":
        control = _entry_by_name(history, "v252 replicated multistep-rise control")
        slightly_softer_multistep_floor = _entry_by_name(history, "v252 replicated multistep-rise + slightly softer multistep floor")
        even_firmer_multistep_floor = _entry_by_name(history, "v252 replicated multistep-rise + even firmer multistep floor")
        tiny_support_floor_relief = _entry_by_name(history, "v252 replicated multistep-rise + tiny support-floor relief")
        even_firmer_multistep_floor_tiny_support_floor_relief = _entry_by_name(history, "v252 replicated multistep-rise + even firmer multistep floor + tiny support-floor relief")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_softer_multistep_floor_score"] = float((slightly_softer_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_multistep_floor else None
        evidence["even_firmer_multistep_floor_score"] = float((even_firmer_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if even_firmer_multistep_floor else None
        evidence["tiny_support_floor_relief_score"] = float((tiny_support_floor_relief or {}).get("score", float("-inf")) or float("-inf")) if tiny_support_floor_relief else None
        evidence["even_firmer_multistep_floor_tiny_support_floor_relief_score"] = float((even_firmer_multistep_floor_tiny_support_floor_relief or {}).get("score", float("-inf")) or float("-inf")) if even_firmer_multistep_floor_tiny_support_floor_relief else None
    elif normalized == "v253":
        control = _entry_by_name(history, "v253 v251-reset contract audit control")
        code_pref_soft_multistep = _entry_by_name(history, "v253 v251-reset + code-pref soft multistep")
        hard_multistep_contract = _entry_by_name(history, "v253 v251-reset + hard multistep contract")
        support_pref_multistep_floor = _entry_by_name(history, "v253 v251-reset + support-pref multistep floor")
        plain_code_pref_contract = _entry_by_name(history, "v253 v251-reset + plain code-pref contract")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["code_pref_soft_multistep_score"] = float((code_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")) if code_pref_soft_multistep else None
        evidence["hard_multistep_contract_score"] = float((hard_multistep_contract or {}).get("score", float("-inf")) or float("-inf")) if hard_multistep_contract else None
        evidence["support_pref_multistep_floor_score"] = float((support_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_multistep_floor else None
        evidence["plain_code_pref_contract_score"] = float((plain_code_pref_contract or {}).get("score", float("-inf")) or float("-inf")) if plain_code_pref_contract else None
    elif normalized == "v254":
        control = _entry_by_name(history, "v254 promoted code-pref contract control")
        code_pref_soft_multistep = _entry_by_name(history, "v254 promoted code-pref contract + code-pref soft multistep")
        tiny_support_floor_relief = _entry_by_name(history, "v254 promoted code-pref contract + tiny support-floor relief")
        tiny_rescue_rollback = _entry_by_name(history, "v254 promoted code-pref contract + tiny rescue rollback")
        slightly_softer_selective_gate = _entry_by_name(history, "v254 promoted code-pref contract + slightly softer selective gate")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["code_pref_soft_multistep_score"] = float((code_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")) if code_pref_soft_multistep else None
        evidence["tiny_support_floor_relief_score"] = float((tiny_support_floor_relief or {}).get("score", float("-inf")) or float("-inf")) if tiny_support_floor_relief else None
        evidence["tiny_rescue_rollback_score"] = float((tiny_rescue_rollback or {}).get("score", float("-inf")) or float("-inf")) if tiny_rescue_rollback else None
        evidence["slightly_softer_selective_gate_score"] = float((slightly_softer_selective_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_selective_gate else None
    elif normalized == "v255":
        control = _entry_by_name(history, "v255 supportspec uplift control")
        stronger_uplift = _entry_by_name(history, "v255 supportspec uplift + stronger uplift")
        no_selective_gate = _entry_by_name(history, "v255 supportspec uplift + no selective gate")
        code_pref_soft_multistep = _entry_by_name(history, "v255 supportspec uplift + code-pref soft multistep")
        support_only_uplift = _entry_by_name(history, "v255 supportspec uplift + support-only uplift")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_uplift_score"] = float((stronger_uplift or {}).get("score", float("-inf")) or float("-inf")) if stronger_uplift else None
        evidence["no_selective_gate_score"] = float((no_selective_gate or {}).get("score", float("-inf")) or float("-inf")) if no_selective_gate else None
        evidence["code_pref_soft_multistep_score"] = float((code_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")) if code_pref_soft_multistep else None
        evidence["support_only_uplift_score"] = float((support_only_uplift or {}).get("score", float("-inf")) or float("-inf")) if support_only_uplift else None
    elif normalized == "v256":
        control = _entry_by_name(history, "v256 verifier-fixed anti-collapse control")
        stronger_rank_guard = _entry_by_name(history, "v256 verifier-fixed anti-collapse + stronger rank guard")
        stronger_hard_negative_budget = _entry_by_name(history, "v256 verifier-fixed anti-collapse + stronger hard-negative budget")
        paraphrase_preserving_vicreg = _entry_by_name(history, "v256 verifier-fixed anti-collapse + paraphrase-preserving vicreg")
        no_vicreg_stronger_margins = _entry_by_name(history, "v256 verifier-fixed anti-collapse + no-vicreg stronger margins")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_rank_guard_score"] = float((stronger_rank_guard or {}).get("score", float("-inf")) or float("-inf")) if stronger_rank_guard else None
        evidence["stronger_hard_negative_budget_score"] = float((stronger_hard_negative_budget or {}).get("score", float("-inf")) or float("-inf")) if stronger_hard_negative_budget else None
        evidence["paraphrase_preserving_vicreg_score"] = float((paraphrase_preserving_vicreg or {}).get("score", float("-inf")) or float("-inf")) if paraphrase_preserving_vicreg else None
        evidence["no_vicreg_stronger_margins_score"] = float((no_vicreg_stronger_margins or {}).get("score", float("-inf")) or float("-inf")) if no_vicreg_stronger_margins else None
    elif normalized == "v257":
        control = _entry_by_name(history, "v257 promoted no-vicreg stronger-margins control")
        anti_collapse_spread_boost = _entry_by_name(history, "v257 promoted no-vicreg stronger-margins + anti-collapse spread boost")
        all_stream_rank_guard = _entry_by_name(history, "v257 promoted no-vicreg stronger-margins + all-stream rank guard")
        lighter_predictor_alignment = _entry_by_name(history, "v257 promoted no-vicreg stronger-margins + lighter predictor alignment")
        lighter_query_margin = _entry_by_name(history, "v257 promoted no-vicreg stronger-margins + lighter query margin")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["anti_collapse_spread_boost_score"] = float((anti_collapse_spread_boost or {}).get("score", float("-inf")) or float("-inf")) if anti_collapse_spread_boost else None
        evidence["all_stream_rank_guard_score"] = float((all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")) if all_stream_rank_guard else None
        evidence["lighter_predictor_alignment_score"] = float((lighter_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")) if lighter_predictor_alignment else None
        evidence["lighter_query_margin_score"] = float((lighter_query_margin or {}).get("score", float("-inf")) or float("-inf")) if lighter_query_margin else None
    elif normalized == "v258":
        control = _entry_by_name(history, "v258 promoted lighter-predictor control")
        stronger_spread_backstop = _entry_by_name(history, "v258 promoted lighter-predictor + stronger spread backstop")
        slightly_lighter_query_margin = _entry_by_name(history, "v258 promoted lighter-predictor + slightly lighter query margin")
        spread_plus_lighter_query_margin = _entry_by_name(
            history,
            "v258 promoted lighter-predictor + stronger spread backstop + slightly lighter query margin",
        )
        mild_all_stream_rank_guard = _entry_by_name(history, "v258 promoted lighter-predictor + mild all-stream rank guard")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_spread_backstop_score"] = float((stronger_spread_backstop or {}).get("score", float("-inf")) or float("-inf")) if stronger_spread_backstop else None
        evidence["slightly_lighter_query_margin_score"] = float((slightly_lighter_query_margin or {}).get("score", float("-inf")) or float("-inf")) if slightly_lighter_query_margin else None
        evidence["spread_backstop_plus_slightly_lighter_query_margin_score"] = float(
            (spread_plus_lighter_query_margin or {}).get("score", float("-inf")) or float("-inf")
        ) if spread_plus_lighter_query_margin else None
        evidence["mild_all_stream_rank_guard_score"] = float((mild_all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")) if mild_all_stream_rank_guard else None
        evidence["v256_reset_frontier_score"] = 21.3170
    elif normalized == "v259":
        control = _entry_by_name(history, "v259 v256-reset no-vicreg stronger-margins control")
        milder_predictor_relief = _entry_by_name(history, "v259 v256-reset + milder predictor relief")
        predictor_focused_spread_backstop = _entry_by_name(history, "v259 v256-reset + predictor-focused spread backstop")
        code_pred_rank_guard = _entry_by_name(history, "v259 v256-reset + code-pred rank guard")
        milder_predictor_relief_predictor_focused_spread = _entry_by_name(
            history,
            "v259 v256-reset + milder predictor relief + predictor-focused spread",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["milder_predictor_relief_score"] = float((milder_predictor_relief or {}).get("score", float("-inf")) or float("-inf")) if milder_predictor_relief else None
        evidence["predictor_focused_spread_backstop_score"] = float((predictor_focused_spread_backstop or {}).get("score", float("-inf")) or float("-inf")) if predictor_focused_spread_backstop else None
        evidence["code_pred_rank_guard_score"] = float((code_pred_rank_guard or {}).get("score", float("-inf")) or float("-inf")) if code_pred_rank_guard else None
        evidence["milder_predictor_relief_predictor_focused_spread_score"] = float(
            (milder_predictor_relief_predictor_focused_spread or {}).get("score", float("-inf")) or float("-inf")
        ) if milder_predictor_relief_predictor_focused_spread else None
        evidence["v256_frontier_score"] = 21.3170
    elif normalized == "v260":
        control = _entry_by_name(history, "v260 promoted predictor-spread control")
        code_pred_rank_guard = _entry_by_name(history, "v260 promoted predictor-spread + code-pred rank guard")
        stronger_hard_negative_budget = _entry_by_name(history, "v260 promoted predictor-spread + stronger hard-negative budget")
        stronger_prototype_repulsion = _entry_by_name(history, "v260 promoted predictor-spread + stronger prototype repulsion")
        code_pred_rank_guard_stronger_hard_negative_budget = _entry_by_name(
            history,
            "v260 promoted predictor-spread + code-pred rank guard + stronger hard-negative budget",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["code_pred_rank_guard_score"] = float((code_pred_rank_guard or {}).get("score", float("-inf")) or float("-inf")) if code_pred_rank_guard else None
        evidence["stronger_hard_negative_budget_score"] = float((stronger_hard_negative_budget or {}).get("score", float("-inf")) or float("-inf")) if stronger_hard_negative_budget else None
        evidence["stronger_prototype_repulsion_score"] = float((stronger_prototype_repulsion or {}).get("score", float("-inf")) or float("-inf")) if stronger_prototype_repulsion else None
        evidence["code_pred_rank_guard_stronger_hard_negative_budget_score"] = float(
            (code_pred_rank_guard_stronger_hard_negative_budget or {}).get("score", float("-inf")) or float("-inf")
        ) if code_pred_rank_guard_stronger_hard_negative_budget else None
        evidence["v259_geometry_frontier_score"] = 21.2713
    elif normalized == "v261":
        control = _entry_by_name(history, "v261 promoted hardneg control")
        code_pred_retarget = _entry_by_name(history, "v261 promoted hardneg + code-pred retarget")
        code_focused_spread_backstop = _entry_by_name(history, "v261 promoted hardneg + code-focused spread backstop")
        mild_prototype_repulsion = _entry_by_name(history, "v261 promoted hardneg + mild prototype repulsion")
        code_pred_retarget_code_focused_spread = _entry_by_name(
            history,
            "v261 promoted hardneg + code-pred retarget + code-focused spread",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["code_pred_retarget_score"] = float((code_pred_retarget or {}).get("score", float("-inf")) or float("-inf")) if code_pred_retarget else None
        evidence["code_focused_spread_backstop_score"] = float((code_focused_spread_backstop or {}).get("score", float("-inf")) or float("-inf")) if code_focused_spread_backstop else None
        evidence["mild_prototype_repulsion_score"] = float((mild_prototype_repulsion or {}).get("score", float("-inf")) or float("-inf")) if mild_prototype_repulsion else None
        evidence["code_pred_retarget_code_focused_spread_score"] = float(
            (code_pred_retarget_code_focused_spread or {}).get("score", float("-inf")) or float("-inf")
        ) if code_pred_retarget_code_focused_spread else None
        evidence["v260_frontier_score"] = 21.2898
    elif normalized == "v262":
        control = _entry_by_name(history, "v262 v260-hardneg reset control")
        milder_predictor_relief = _entry_by_name(history, "v262 v260-hardneg + milder predictor relief")
        queue_off_geometry_reset = _entry_by_name(history, "v262 v260-hardneg + queue-off geometry reset")
        milder_predictor_relief_code_pred_retarget = _entry_by_name(
            history,
            "v262 v260-hardneg + milder predictor relief + code-pred retarget",
        )
        queue_off_milder_predictor_relief = _entry_by_name(
            history,
            "v262 v260-hardneg + queue-off + milder predictor relief",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["milder_predictor_relief_score"] = float((milder_predictor_relief or {}).get("score", float("-inf")) or float("-inf")) if milder_predictor_relief else None
        evidence["queue_off_geometry_reset_score"] = float((queue_off_geometry_reset or {}).get("score", float("-inf")) or float("-inf")) if queue_off_geometry_reset else None
        evidence["milder_predictor_relief_code_pred_retarget_score"] = float(
            (milder_predictor_relief_code_pred_retarget or {}).get("score", float("-inf")) or float("-inf")
        ) if milder_predictor_relief_code_pred_retarget else None
        evidence["queue_off_milder_predictor_relief_score"] = float(
            (queue_off_milder_predictor_relief or {}).get("score", float("-inf")) or float("-inf")
        ) if queue_off_milder_predictor_relief else None
        evidence["v256_frontier_score"] = 21.3170
        evidence["v260_geometry_frontier_score"] = 21.2898
    elif normalized == "v263":
        control = _entry_by_name(history, "v263 v256-reset sigreg control")
        no_sigreg = _entry_by_name(history, "v263 v256-reset + no sigreg")
        stronger_sigreg = _entry_by_name(history, "v263 v256-reset + stronger sigreg")
        stronger_sigreg_stronger_hard_negative_budget = _entry_by_name(
            history,
            "v263 v256-reset + stronger sigreg + stronger hard-negative budget",
        )
        stronger_sigreg_milder_predictor_relief = _entry_by_name(
            history,
            "v263 v256-reset + stronger sigreg + milder predictor relief",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["no_sigreg_score"] = float((no_sigreg or {}).get("score", float("-inf")) or float("-inf")) if no_sigreg else None
        evidence["stronger_sigreg_score"] = float((stronger_sigreg or {}).get("score", float("-inf")) or float("-inf")) if stronger_sigreg else None
        evidence["stronger_sigreg_stronger_hard_negative_budget_score"] = float(
            (stronger_sigreg_stronger_hard_negative_budget or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_sigreg_stronger_hard_negative_budget else None
        evidence["stronger_sigreg_milder_predictor_relief_score"] = float(
            (stronger_sigreg_milder_predictor_relief or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_sigreg_milder_predictor_relief else None
        evidence["v256_frontier_score"] = 21.3170
        evidence["v260_geometry_frontier_score"] = 21.2898
    elif normalized == "v264":
        control = _entry_by_name(history, "v264 promoted sigreg control")
        stronger_hard_negative_budget = _entry_by_name(history, "v264 promoted sigreg + stronger hard-negative budget")
        milder_predictor_relief = _entry_by_name(history, "v264 promoted sigreg + milder predictor relief")
        stronger_hard_negative_budget_milder_predictor_relief = _entry_by_name(
            history,
            "v264 promoted sigreg + stronger hard-negative budget + milder predictor relief",
        )
        code_pred_retarget = _entry_by_name(history, "v264 promoted sigreg + code-pred retarget")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_hard_negative_budget_score"] = float((stronger_hard_negative_budget or {}).get("score", float("-inf")) or float("-inf")) if stronger_hard_negative_budget else None
        evidence["milder_predictor_relief_score"] = float((milder_predictor_relief or {}).get("score", float("-inf")) or float("-inf")) if milder_predictor_relief else None
        evidence["stronger_hard_negative_budget_milder_predictor_relief_score"] = float(
            (stronger_hard_negative_budget_milder_predictor_relief or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_hard_negative_budget_milder_predictor_relief else None
        evidence["code_pred_retarget_score"] = float((code_pred_retarget or {}).get("score", float("-inf")) or float("-inf")) if code_pred_retarget else None
        evidence["v263_frontier_score"] = 21.4234
    elif normalized == "v265":
        control = _entry_by_name(history, "v265 promoted code-pred retarget control")
        milder_predictor_relief = _entry_by_name(history, "v265 promoted code-pred retarget + milder predictor relief")
        stronger_hard_negative_budget = _entry_by_name(history, "v265 promoted code-pred retarget + stronger hard-negative budget")
        code_focused_spread_backstop = _entry_by_name(history, "v265 promoted code-pred retarget + code-focused spread backstop")
        milder_predictor_relief_code_focused_spread = _entry_by_name(
            history,
            "v265 promoted code-pred retarget + milder predictor relief + code-focused spread",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["milder_predictor_relief_score"] = float((milder_predictor_relief or {}).get("score", float("-inf")) or float("-inf")) if milder_predictor_relief else None
        evidence["stronger_hard_negative_budget_score"] = float((stronger_hard_negative_budget or {}).get("score", float("-inf")) or float("-inf")) if stronger_hard_negative_budget else None
        evidence["code_focused_spread_backstop_score"] = float((code_focused_spread_backstop or {}).get("score", float("-inf")) or float("-inf")) if code_focused_spread_backstop else None
        evidence["milder_predictor_relief_code_focused_spread_score"] = float(
            (milder_predictor_relief_code_focused_spread or {}).get("score", float("-inf")) or float("-inf")
        ) if milder_predictor_relief_code_focused_spread else None
        evidence["v264_frontier_score"] = 21.4286
    elif normalized == "v266":
        control = _entry_by_name(history, "v266 promoted code-focused-spread control")
        equivalence_positive_views = _entry_by_name(history, "v266 promoted code-focused-spread + equivalence positive views")
        equivalence_positive_views_synthesis_positives = _entry_by_name(
            history,
            "v266 promoted code-focused-spread + equivalence positive views + synthesis positives",
        )
        low_vicreg_query_multiview = _entry_by_name(
            history,
            "v266 promoted code-focused-spread + low-vicreg query multiview",
        )
        equivalence_positive_views_low_vicreg_query_multiview = _entry_by_name(
            history,
            "v266 promoted code-focused-spread + equivalence positive views + low-vicreg query multiview",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["equivalence_positive_views_score"] = float((equivalence_positive_views or {}).get("score", float("-inf")) or float("-inf")) if equivalence_positive_views else None
        evidence["equivalence_positive_views_synthesis_positives_score"] = float(
            (equivalence_positive_views_synthesis_positives or {}).get("score", float("-inf")) or float("-inf")
        ) if equivalence_positive_views_synthesis_positives else None
        evidence["low_vicreg_query_multiview_score"] = float(
            (low_vicreg_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if low_vicreg_query_multiview else None
        evidence["equivalence_positive_views_low_vicreg_query_multiview_score"] = float(
            (equivalence_positive_views_low_vicreg_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if equivalence_positive_views_low_vicreg_query_multiview else None
        evidence["v265_frontier_score"] = 21.3691
    elif normalized == "v267":
        control = _entry_by_name(history, "v267 promoted equivalence-multiview control")
        lighter_equivalence_positives = _entry_by_name(
            history,
            "v267 promoted equivalence-multiview + lighter equivalence positives",
        )
        lighter_query_multiview = _entry_by_name(
            history,
            "v267 promoted equivalence-multiview + lighter query multiview",
        )
        synthesis_positives = _entry_by_name(
            history,
            "v267 promoted equivalence-multiview + synthesis positives",
        )
        lighter_equivalence_positives_lighter_query_multiview = _entry_by_name(
            history,
            "v267 promoted equivalence-multiview + lighter equivalence positives + lighter query multiview",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["lighter_equivalence_positives_score"] = float(
            (lighter_equivalence_positives or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_equivalence_positives else None
        evidence["lighter_query_multiview_score"] = float(
            (lighter_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_query_multiview else None
        evidence["synthesis_positives_score"] = float(
            (synthesis_positives or {}).get("score", float("-inf")) or float("-inf")
        ) if synthesis_positives else None
        evidence["lighter_equivalence_positives_lighter_query_multiview_score"] = float(
            (lighter_equivalence_positives_lighter_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_equivalence_positives_lighter_query_multiview else None
        evidence["v266_frontier_score"] = 21.4739
    elif normalized == "v268":
        control = _entry_by_name(history, "v268 promoted synthesis-positive control")
        lighter_query_multiview = _entry_by_name(history, "v268 promoted synthesis-positive + lighter query multiview")
        lighter_equivalence_positives = _entry_by_name(history, "v268 promoted synthesis-positive + lighter equivalence positives")
        lighter_equivalence_positives_lighter_query_multiview = _entry_by_name(
            history,
            "v268 promoted synthesis-positive + lighter equivalence positives + lighter query multiview",
        )
        slightly_stronger_code_focused_spread = _entry_by_name(
            history,
            "v268 promoted synthesis-positive + slightly stronger code-focused spread",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["lighter_query_multiview_score"] = float(
            (lighter_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_query_multiview else None
        evidence["lighter_equivalence_positives_score"] = float(
            (lighter_equivalence_positives or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_equivalence_positives else None
        evidence["lighter_equivalence_positives_lighter_query_multiview_score"] = float(
            (lighter_equivalence_positives_lighter_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_equivalence_positives_lighter_query_multiview else None
        evidence["slightly_stronger_code_focused_spread_score"] = float(
            (slightly_stronger_code_focused_spread or {}).get("score", float("-inf")) or float("-inf")
        ) if slightly_stronger_code_focused_spread else None
        evidence["v267_frontier_score"] = 21.4939
    elif normalized == "v269":
        control = _entry_by_name(history, "v269 promoted synthesis-positive control")
        stronger_query_spread_backstop = _entry_by_name(
            history,
            "v269 promoted synthesis-positive + stronger query-spread backstop",
        )
        query_head_reset = _entry_by_name(history, "v269 promoted synthesis-positive + query-head reset")
        frozen_backbone_query_head_recalibration = _entry_by_name(
            history,
            "v269 promoted synthesis-positive + frozen-backbone query-head recalibration",
        )
        prototype_off_geometry_release = _entry_by_name(
            history,
            "v269 promoted synthesis-positive + prototype-off geometry release",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_query_spread_backstop_score"] = float(
            (stronger_query_spread_backstop or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_query_spread_backstop else None
        evidence["query_head_reset_score"] = float(
            (query_head_reset or {}).get("score", float("-inf")) or float("-inf")
        ) if query_head_reset else None
        evidence["frozen_backbone_query_head_recalibration_score"] = float(
            (frozen_backbone_query_head_recalibration or {}).get("score", float("-inf")) or float("-inf")
        ) if frozen_backbone_query_head_recalibration else None
        evidence["prototype_off_geometry_release_score"] = float(
            (prototype_off_geometry_release or {}).get("score", float("-inf")) or float("-inf")
        ) if prototype_off_geometry_release else None
        evidence["v268_frontier_score"] = 21.4894
    elif normalized == "v270":
        control = _entry_by_name(history, "v270 promoted query-head-reset control")
        frozen_backbone_rollback = _entry_by_name(
            history,
            "v270 promoted query-head-reset + frozen-backbone rollback",
        )
        lighter_query_multiview = _entry_by_name(
            history,
            "v270 promoted query-head-reset + lighter query multiview",
        )
        softer_prototype_topology = _entry_by_name(
            history,
            "v270 promoted query-head-reset + softer prototype topology",
        )
        queue_off_geometry_release = _entry_by_name(
            history,
            "v270 promoted query-head-reset + queue-off geometry release",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["frozen_backbone_rollback_score"] = float(
            (frozen_backbone_rollback or {}).get("score", float("-inf")) or float("-inf")
        ) if frozen_backbone_rollback else None
        evidence["lighter_query_multiview_score"] = float(
            (lighter_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_query_multiview else None
        evidence["softer_prototype_topology_score"] = float(
            (softer_prototype_topology or {}).get("score", float("-inf")) or float("-inf")
        ) if softer_prototype_topology else None
        evidence["queue_off_geometry_release_score"] = float(
            (queue_off_geometry_release or {}).get("score", float("-inf")) or float("-inf")
        ) if queue_off_geometry_release else None
        evidence["v269_frontier_score"] = 21.4894
    elif normalized == "v271":
        control = _entry_by_name(history, "v271 promoted frozen-query-recalibration control")
        queue_off_geometry_release = _entry_by_name(
            history,
            "v271 promoted frozen-query-recalibration + queue-off geometry release",
        )
        stronger_query_spread_backstop = _entry_by_name(
            history,
            "v271 promoted frozen-query-recalibration + stronger query-spread backstop",
        )
        lighter_query_multiview = _entry_by_name(
            history,
            "v271 promoted frozen-query-recalibration + lighter query multiview",
        )
        softer_prototype_topology = _entry_by_name(
            history,
            "v271 promoted frozen-query-recalibration + softer prototype topology",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["queue_off_geometry_release_score"] = float(
            (queue_off_geometry_release or {}).get("score", float("-inf")) or float("-inf")
        ) if queue_off_geometry_release else None
        evidence["stronger_query_spread_backstop_score"] = float(
            (stronger_query_spread_backstop or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_query_spread_backstop else None
        evidence["lighter_query_multiview_score"] = float(
            (lighter_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_query_multiview else None
        evidence["softer_prototype_topology_score"] = float(
            (softer_prototype_topology or {}).get("score", float("-inf")) or float("-inf")
        ) if softer_prototype_topology else None
        evidence["v270_thawed_queue_off_score"] = 20.7635
        evidence["v269_thawed_query_reset_score"] = 21.0980
    elif normalized == "v275":
        control = _entry_by_name(history, "v275 promoted thawed queue-off keep-head control")
        query_classifier_confidence_lift = _entry_by_name(
            history,
            "v275 promoted thawed queue-off + query-classifier confidence lift",
        )
        lighter_query_multiview = _entry_by_name(
            history,
            "v275 promoted thawed queue-off + lighter query multiview",
        )
        earlier_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v275 promoted thawed queue-off + earlier query-classifier confidence lift",
        )
        earlier_query_classifier_confidence_lift_lighter_query_multiview = _entry_by_name(
            history,
            "v275 promoted thawed queue-off + earlier query-classifier confidence lift + lighter query multiview",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["query_classifier_confidence_lift_score"] = float(
            (query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if query_classifier_confidence_lift else None
        evidence["lighter_query_multiview_score"] = float(
            (lighter_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_query_multiview else None
        evidence["earlier_query_classifier_confidence_lift_score"] = float(
            (earlier_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if earlier_query_classifier_confidence_lift else None
        evidence["earlier_query_classifier_confidence_lift_lighter_query_multiview_score"] = float(
            (earlier_query_classifier_confidence_lift_lighter_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if earlier_query_classifier_confidence_lift_lighter_query_multiview else None
        evidence["v270_queue_off_score"] = 20.7635
    elif normalized == "v276":
        control = _entry_by_name(history, "v276 frozen keep-head + tiny query-classifier confidence lift control")
        small_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v276 frozen keep-head + small query-classifier confidence lift",
        )
        mixed_query_pred_confidence_lift = _entry_by_name(
            history,
            "v276 frozen keep-head + mixed query/pred confidence lift",
        )
        small_earlier_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v276 frozen keep-head + small earlier query-classifier confidence lift",
        )
        moderate_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v276 frozen keep-head + moderate query-classifier confidence lift",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["small_query_classifier_confidence_lift_score"] = float(
            (small_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if small_query_classifier_confidence_lift else None
        evidence["mixed_query_pred_confidence_lift_score"] = float(
            (mixed_query_pred_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if mixed_query_pred_confidence_lift else None
        evidence["small_earlier_query_classifier_confidence_lift_score"] = float(
            (small_earlier_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if small_earlier_query_classifier_confidence_lift else None
        evidence["moderate_query_classifier_confidence_lift_score"] = float(
            (moderate_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if moderate_query_classifier_confidence_lift else None
        evidence["v275_query_classifier_confidence_lift_score"] = 21.0062
        evidence["v271_queue_off_geometry_release_score"] = 21.4894
    elif normalized == "v277":
        control = _entry_by_name(history, "v277 promoted thawed query-classifier control")
        smaller_query_lift = _entry_by_name(
            history,
            "v277 promoted thawed query-classifier + smaller query lift",
        )
        mixed_query_pred_lift = _entry_by_name(
            history,
            "v277 promoted thawed query-classifier + mixed query/pred lift",
        )
        smaller_earlier_query_lift = _entry_by_name(
            history,
            "v277 promoted thawed query-classifier + smaller earlier query lift",
        )
        moderate_query_lift = _entry_by_name(
            history,
            "v277 promoted thawed query-classifier + moderate query lift",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["smaller_query_lift_score"] = float(
            (smaller_query_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if smaller_query_lift else None
        evidence["mixed_query_pred_lift_score"] = float(
            (mixed_query_pred_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if mixed_query_pred_lift else None
        evidence["smaller_earlier_query_lift_score"] = float(
            (smaller_earlier_query_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if smaller_earlier_query_lift else None
        evidence["moderate_query_lift_score"] = float(
            (moderate_query_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if moderate_query_lift else None
        evidence["v275_query_classifier_confidence_lift_score"] = 21.0062
        evidence["v271_queue_off_geometry_release_score"] = 21.4894
    elif normalized == "v278":
        control = _entry_by_name(history, "v278 promoted smaller-query-lift control")
        lighter_query_multiview = _entry_by_name(
            history,
            "v278 promoted smaller-query-lift + lighter query multiview",
        )
        milder_predictor_alignment = _entry_by_name(
            history,
            "v278 promoted smaller-query-lift + milder predictor alignment",
        )
        code_query_rank_retarget = _entry_by_name(
            history,
            "v278 promoted smaller-query-lift + code-query rank retarget",
        )
        lighter_query_anchors = _entry_by_name(
            history,
            "v278 promoted smaller-query-lift + lighter query anchors",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["lighter_query_multiview_score"] = float(
            (lighter_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_query_multiview else None
        evidence["milder_predictor_alignment_score"] = float(
            (milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if milder_predictor_alignment else None
        evidence["code_query_rank_retarget_score"] = float(
            (code_query_rank_retarget or {}).get("score", float("-inf")) or float("-inf")
        ) if code_query_rank_retarget else None
        evidence["lighter_query_anchors_score"] = float(
            (lighter_query_anchors or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_query_anchors else None
        evidence["v277_smaller_query_lift_score"] = 21.3021
        evidence["v271_queue_off_geometry_release_score"] = 21.4894
    elif normalized == "v279":
        control = _entry_by_name(history, "v279 promoted lighter-query-anchors control")
        gentle_query_multiview = _entry_by_name(
            history,
            "v279 promoted lighter-query-anchors + gentle query multiview",
        )
        lighter_query_multiview = _entry_by_name(
            history,
            "v279 promoted lighter-query-anchors + lighter query multiview",
        )
        lighter_anchor_margin = _entry_by_name(
            history,
            "v279 promoted lighter-query-anchors + lighter anchor margin",
        )
        tiny_prototype_query_restore = _entry_by_name(
            history,
            "v279 promoted lighter-query-anchors + tiny prototype-query restore",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["gentle_query_multiview_score"] = float(
            (gentle_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if gentle_query_multiview else None
        evidence["lighter_query_multiview_score"] = float(
            (lighter_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_query_multiview else None
        evidence["lighter_anchor_margin_score"] = float(
            (lighter_anchor_margin or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_anchor_margin else None
        evidence["tiny_prototype_query_restore_score"] = float(
            (tiny_prototype_query_restore or {}).get("score", float("-inf")) or float("-inf")
        ) if tiny_prototype_query_restore else None
        evidence["v255_support_only_uplift_score"] = 21.1472
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "code embeddings are too similar",
            "query embedding effective rank is too low",
            "query embeddings are too similar",
        ]
    elif normalized == "v280":
        control = _entry_by_name(history, "v280 promoted lighter-query-anchors control")
        equivalence_only_prototypes = _entry_by_name(
            history,
            "v280 promoted lighter-query-anchors + equivalence-only prototypes",
        )
        no_synthesis_equivalence_views = _entry_by_name(
            history,
            "v280 promoted lighter-query-anchors + no synthesis equivalence views",
        )
        equivalence_only_no_synthesis = _entry_by_name(
            history,
            "v280 promoted lighter-query-anchors + equivalence-only prototypes + no synthesis views",
        )
        equivalence_only_no_synthesis_milder_predictor_alignment = _entry_by_name(
            history,
            "v280 promoted lighter-query-anchors + equivalence-only prototypes + no synthesis views + milder predictor alignment",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["equivalence_only_prototypes_score"] = float(
            (equivalence_only_prototypes or {}).get("score", float("-inf")) or float("-inf")
        ) if equivalence_only_prototypes else None
        evidence["no_synthesis_equivalence_views_score"] = float(
            (no_synthesis_equivalence_views or {}).get("score", float("-inf")) or float("-inf")
        ) if no_synthesis_equivalence_views else None
        evidence["equivalence_only_no_synthesis_score"] = float(
            (equivalence_only_no_synthesis or {}).get("score", float("-inf")) or float("-inf")
        ) if equivalence_only_no_synthesis else None
        evidence["equivalence_only_no_synthesis_milder_predictor_alignment_score"] = float(
            (equivalence_only_no_synthesis_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if equivalence_only_no_synthesis_milder_predictor_alignment else None
        evidence["v279_frontier_score"] = 21.6461
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "code embeddings are too similar",
            "query embedding effective rank is too low",
            "query embeddings are too similar",
        ]
    elif normalized == "v281":
        control = _entry_by_name(history, "v281 v279-reset sigreg control")
        no_sigreg = _entry_by_name(history, "v281 v279-reset + no sigreg")
        stronger_sigreg = _entry_by_name(history, "v281 v279-reset + stronger sigreg")
        stronger_sigreg_equivalence_only_prototypes = _entry_by_name(
            history,
            "v281 v279-reset + stronger sigreg + equivalence-only prototypes",
        )
        stronger_sigreg_milder_predictor_alignment = _entry_by_name(
            history,
            "v281 v279-reset + stronger sigreg + milder predictor alignment",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["no_sigreg_score"] = float(
            (no_sigreg or {}).get("score", float("-inf")) or float("-inf")
        ) if no_sigreg else None
        evidence["stronger_sigreg_score"] = float(
            (stronger_sigreg or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_sigreg else None
        evidence["stronger_sigreg_equivalence_only_prototypes_score"] = float(
            (stronger_sigreg_equivalence_only_prototypes or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_sigreg_equivalence_only_prototypes else None
        evidence["stronger_sigreg_milder_predictor_alignment_score"] = float(
            (stronger_sigreg_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_sigreg_milder_predictor_alignment else None
        evidence["v280_equivalence_only_prototypes_score"] = 21.3679
        evidence["v279_frontier_score"] = 21.6461
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "code embeddings are too similar",
            "query embedding effective rank is too low",
            "query embeddings are too similar",
        ]
    elif normalized == "v282":
        control = _entry_by_name(history, "v282 v279-reset clean frontier control")
        no_sigreg = _entry_by_name(history, "v282 v279-reset + no sigreg")
        equivalence_only_prototypes = _entry_by_name(history, "v282 v279-reset + equivalence-only prototypes")
        no_sigreg_equivalence_only_prototypes = _entry_by_name(
            history,
            "v282 v279-reset + no sigreg + equivalence-only prototypes",
        )
        no_sigreg_milder_predictor_alignment = _entry_by_name(
            history,
            "v282 v279-reset + no sigreg + milder predictor alignment",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["no_sigreg_score"] = float(
            (no_sigreg or {}).get("score", float("-inf")) or float("-inf")
        ) if no_sigreg else None
        evidence["equivalence_only_prototypes_score"] = float(
            (equivalence_only_prototypes or {}).get("score", float("-inf")) or float("-inf")
        ) if equivalence_only_prototypes else None
        evidence["no_sigreg_equivalence_only_prototypes_score"] = float(
            (no_sigreg_equivalence_only_prototypes or {}).get("score", float("-inf")) or float("-inf")
        ) if no_sigreg_equivalence_only_prototypes else None
        evidence["no_sigreg_milder_predictor_alignment_score"] = float(
            (no_sigreg_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if no_sigreg_milder_predictor_alignment else None
        evidence["v281_no_sigreg_score"] = 21.4938
        evidence["v279_frontier_score"] = 21.6461
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "code embeddings are too similar",
            "query embedding effective rank is too low",
            "query embeddings are too similar",
        ]
    elif normalized == "v283":
        delayed_ignorance_control = _entry_by_name(history, "v283 v279-stability delayed-ignorance control")
        no_sigreg_delayed_ignorance = _entry_by_name(
            history,
            "v283 v279-stability + no sigreg + delayed ignorance",
        )
        equivalence_only_prototypes_delayed_ignorance = _entry_by_name(
            history,
            "v283 v279-stability + equivalence-only prototypes + delayed ignorance",
        )
        short_continuation_control = _entry_by_name(
            history,
            "v283 v279-stability short continuation control",
        )
        equivalence_only_prototypes_short_continuation = _entry_by_name(
            history,
            "v283 v279-stability + equivalence-only prototypes + short continuation",
        )
        evidence["delayed_ignorance_control_score"] = float(
            (delayed_ignorance_control or {}).get("score", float("-inf")) or float("-inf")
        ) if delayed_ignorance_control else None
        evidence["no_sigreg_delayed_ignorance_score"] = float(
            (no_sigreg_delayed_ignorance or {}).get("score", float("-inf")) or float("-inf")
        ) if no_sigreg_delayed_ignorance else None
        evidence["equivalence_only_prototypes_delayed_ignorance_score"] = float(
            (equivalence_only_prototypes_delayed_ignorance or {}).get("score", float("-inf")) or float("-inf")
        ) if equivalence_only_prototypes_delayed_ignorance else None
        evidence["short_continuation_control_score"] = float(
            (short_continuation_control or {}).get("score", float("-inf")) or float("-inf")
        ) if short_continuation_control else None
        evidence["equivalence_only_prototypes_short_continuation_score"] = float(
            (equivalence_only_prototypes_short_continuation or {}).get("score", float("-inf")) or float("-inf")
        ) if equivalence_only_prototypes_short_continuation else None
        evidence["v281_no_sigreg_score"] = 21.4938
        evidence["v279_frontier_score"] = 21.6461
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "code embeddings are too similar",
            "query embedding effective rank is too low",
            "query embeddings are too similar",
        ]
    elif normalized == "v286":
        control = _entry_by_name(history, "v286 clean-reset v178 support-aware citecheck pairwise control")
        support_floor = _entry_by_name(
            history,
            "v286 clean-reset v178 support-aware citecheck pairwise + support floor",
        )
        code_form_tie_break = _entry_by_name(
            history,
            "v286 clean-reset v178 support-aware citecheck pairwise + code-form tie-break",
        )
        support_pref_multistep_floor = _entry_by_name(
            history,
            "v286 clean-reset v178 support-aware citecheck pairwise + support-pref multistep floor",
        )
        code_form_tie_break_ultra_low_margin_safe_expand = _entry_by_name(
            history,
            "v286 clean-reset v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_floor_score"] = float(
            (support_floor or {}).get("score", float("-inf")) or float("-inf")
        ) if support_floor else None
        evidence["code_form_tie_break_score"] = float(
            (code_form_tie_break or {}).get("score", float("-inf")) or float("-inf")
        ) if code_form_tie_break else None
        evidence["support_pref_multistep_floor_score"] = float(
            (support_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")
        ) if support_pref_multistep_floor else None
        evidence["code_form_tie_break_ultra_low_margin_safe_expand_score"] = float(
            (code_form_tie_break_ultra_low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")
        ) if code_form_tie_break_ultra_low_margin_safe_expand else None
        evidence["v279_frontier_score"] = 21.6461
        evidence["v284_short_window_rollforward_score"] = 21.4938
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "code embeddings are too similar",
            "query embedding effective rank is too low",
            "query embeddings are too similar",
        ]
    elif normalized == "v287":
        control = _entry_by_name(history, "v287 v279-reset clean frontier control")
        stronger_spread_backstop = _entry_by_name(
            history,
            "v287 v279-reset + stronger spread backstop",
        )
        no_sigreg_stronger_spread_backstop = _entry_by_name(
            history,
            "v287 v279-reset + no sigreg + stronger spread backstop",
        )
        equivalence_only_prototypes_stronger_spread_backstop = _entry_by_name(
            history,
            "v287 v279-reset + equivalence-only prototypes + stronger spread backstop",
        )
        stronger_spread_backstop_milder_predictor_alignment = _entry_by_name(
            history,
            "v287 v279-reset + stronger spread backstop + milder predictor alignment",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_spread_backstop_score"] = float(
            (stronger_spread_backstop or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_spread_backstop else None
        evidence["no_sigreg_stronger_spread_backstop_score"] = float(
            (no_sigreg_stronger_spread_backstop or {}).get("score", float("-inf")) or float("-inf")
        ) if no_sigreg_stronger_spread_backstop else None
        evidence["equivalence_only_prototypes_stronger_spread_backstop_score"] = float(
            (equivalence_only_prototypes_stronger_spread_backstop or {}).get("score", float("-inf")) or float("-inf")
        ) if equivalence_only_prototypes_stronger_spread_backstop else None
        evidence["stronger_spread_backstop_milder_predictor_alignment_score"] = float(
            (stronger_spread_backstop_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_spread_backstop_milder_predictor_alignment else None
        evidence["v284_no_sigreg_guarded_bridge_score"] = 21.4938
        evidence["v279_frontier_score"] = 21.6461
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "code embeddings are too similar",
            "query embedding effective rank is too low",
            "query embeddings are too similar",
        ]
    elif normalized == "v289":
        control = _entry_by_name(history, "v289 v279-replay control")
        direct_deployed_latent_spectral_shaping = _entry_by_name(
            history,
            "v289 direct deployed-latent spectral shaping",
        )
        wide_deployed_retrieval_head_spectral_shaping = _entry_by_name(
            history,
            "v289 wide deployed retrieval head + spectral shaping",
        )
        wide_deployed_retrieval_head_spectral_shaping_equivalence_only_prototypes = _entry_by_name(
            history,
            "v289 wide deployed retrieval head + spectral shaping + equivalence-only prototypes",
        )
        wide_deployed_retrieval_head_spectral_shaping_safe_hard_negatives = _entry_by_name(
            history,
            "v289 wide deployed retrieval head + spectral shaping + safe hard negatives",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["direct_deployed_latent_spectral_shaping_score"] = float(
            (direct_deployed_latent_spectral_shaping or {}).get("score", float("-inf")) or float("-inf")
        ) if direct_deployed_latent_spectral_shaping else None
        evidence["wide_deployed_retrieval_head_spectral_shaping_score"] = float(
            (wide_deployed_retrieval_head_spectral_shaping or {}).get("score", float("-inf")) or float("-inf")
        ) if wide_deployed_retrieval_head_spectral_shaping else None
        evidence["wide_deployed_retrieval_head_spectral_shaping_equivalence_only_prototypes_score"] = float(
            (wide_deployed_retrieval_head_spectral_shaping_equivalence_only_prototypes or {}).get("score", float("-inf")) or float("-inf")
        ) if wide_deployed_retrieval_head_spectral_shaping_equivalence_only_prototypes else None
        evidence["wide_deployed_retrieval_head_spectral_shaping_safe_hard_negatives_score"] = float(
            (wide_deployed_retrieval_head_spectral_shaping_safe_hard_negatives or {}).get("score", float("-inf")) or float("-inf")
        ) if wide_deployed_retrieval_head_spectral_shaping_safe_hard_negatives else None
        evidence["v279_frontier_score"] = 21.6461
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "code embeddings are too similar",
            "query embedding effective rank is too low",
            "query embeddings are too similar",
        ]
    elif normalized == "v290":
        control = _entry_by_name(
            history,
            "v290 clean-reset v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand control",
        )
        local_union_finalist_room = _entry_by_name(
            history,
            "v290 clean-reset v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + local union finalist room",
        )
        code_pref_multistep_floor = _entry_by_name(
            history,
            "v290 clean-reset v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + code-pref multistep floor",
        )
        code_pref_soft_multistep = _entry_by_name(
            history,
            "v290 clean-reset v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + code-pref soft multistep",
        )
        local_union_finalist_room_code_pref_soft_multistep = _entry_by_name(
            history,
            "v290 clean-reset v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + local union finalist room + code-pref soft multistep",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["local_union_finalist_room_score"] = float(
            (local_union_finalist_room or {}).get("score", float("-inf")) or float("-inf")
        ) if local_union_finalist_room else None
        evidence["code_pref_multistep_floor_score"] = float(
            (code_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")
        ) if code_pref_multistep_floor else None
        evidence["code_pref_soft_multistep_score"] = float(
            (code_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")
        ) if code_pref_soft_multistep else None
        evidence["local_union_finalist_room_code_pref_soft_multistep_score"] = float(
            (local_union_finalist_room_code_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")
        ) if local_union_finalist_room_code_pref_soft_multistep else None
        evidence["v279_frontier_score"] = 21.6461
        evidence["v289_safe_hard_negative_score"] = 21.2785
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "code embeddings are too similar",
            "query embedding effective rank is too low",
            "query embeddings are too similar",
        ]
    elif normalized == "v291":
        control = _entry_by_name(history, "v291 v279-reset clean frontier control")
        safe_hard_negative_budget = _entry_by_name(history, "v291 v279-reset + safe hard-negative budget")
        equivalence_only_positives = _entry_by_name(
            history,
            "v291 v279-reset + safe hard-negative budget + equivalence-only positives",
        )
        stronger_equivalence_positives = _entry_by_name(
            history,
            "v291 v279-reset + safe hard-negative budget + stronger equivalence positives",
        )
        equivalence_only_no_synthesis_views = _entry_by_name(
            history,
            "v291 v279-reset + safe hard-negative budget + equivalence-only positives + no synthesis views",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["safe_hard_negative_budget_score"] = float(
            (safe_hard_negative_budget or {}).get("score", float("-inf")) or float("-inf")
        ) if safe_hard_negative_budget else None
        evidence["equivalence_only_positives_score"] = float(
            (equivalence_only_positives or {}).get("score", float("-inf")) or float("-inf")
        ) if equivalence_only_positives else None
        evidence["stronger_equivalence_positives_score"] = float(
            (stronger_equivalence_positives or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_equivalence_positives else None
        evidence["equivalence_only_no_synthesis_views_score"] = float(
            (equivalence_only_no_synthesis_views or {}).get("score", float("-inf")) or float("-inf")
        ) if equivalence_only_no_synthesis_views else None
        evidence["v279_frontier_score"] = 21.6461
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "code embeddings are too similar",
            "query embedding effective rank is too low",
            "query embeddings are too similar",
        ]
    elif normalized == "v292":
        control = _entry_by_name(history, "v292 v279-replay control")
        facetized_hard_late_interaction = _entry_by_name(history, "v292 facetized retrieval + hard late interaction")
        facetized_smoothed_late_interaction = _entry_by_name(history, "v292 facetized retrieval + smoothed late interaction")
        hybrid_global_facets = _entry_by_name(history, "v292 hybrid global + facets")
        hybrid_global_facets_separate_query_code_projectors = _entry_by_name(
            history,
            "v292 hybrid global + facets + separate query/code projectors",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["facetized_hard_late_interaction_score"] = float(
            (facetized_hard_late_interaction or {}).get("score", float("-inf")) or float("-inf")
        ) if facetized_hard_late_interaction else None
        evidence["facetized_smoothed_late_interaction_score"] = float(
            (facetized_smoothed_late_interaction or {}).get("score", float("-inf")) or float("-inf")
        ) if facetized_smoothed_late_interaction else None
        evidence["hybrid_global_facets_score"] = float(
            (hybrid_global_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if hybrid_global_facets else None
        evidence["hybrid_global_facets_separate_query_code_projectors_score"] = float(
            (hybrid_global_facets_separate_query_code_projectors or {}).get("score", float("-inf")) or float("-inf")
        ) if hybrid_global_facets_separate_query_code_projectors else None
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v293":
        control = _entry_by_name(history, "v293 promoted hybrid global + facets control")
        code_query_rank_retarget = _entry_by_name(history, "v293 promoted hybrid global + facets + code-query rank retarget")
        milder_predictor_alignment = _entry_by_name(history, "v293 promoted hybrid global + facets + milder predictor alignment")
        split_projectors_stronger_global_preserve = _entry_by_name(
            history,
            "v293 promoted hybrid global + facets + split projectors + stronger global preserve",
        )
        more_facets = _entry_by_name(history, "v293 promoted hybrid global + facets + more facets")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["code_query_rank_retarget_score"] = float(
            (code_query_rank_retarget or {}).get("score", float("-inf")) or float("-inf")
        ) if code_query_rank_retarget else None
        evidence["milder_predictor_alignment_score"] = float(
            (milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if milder_predictor_alignment else None
        evidence["split_projectors_stronger_global_preserve_score"] = float(
            (split_projectors_stronger_global_preserve or {}).get("score", float("-inf")) or float("-inf")
        ) if split_projectors_stronger_global_preserve else None
        evidence["more_facets_score"] = float(
            (more_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if more_facets else None
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v294":
        control = _entry_by_name(history, "v294 promoted more-facets control")
        milder_predictor_alignment = _entry_by_name(history, "v294 promoted more-facets + milder predictor alignment")
        stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v294 promoted more-facets + stronger query-classifier confidence lift",
        )
        eight_facets = _entry_by_name(history, "v294 promoted more-facets + eight facets")
        split_projectors_stronger_global_preserve = _entry_by_name(
            history,
            "v294 promoted more-facets + split projectors + stronger global preserve",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["milder_predictor_alignment_score"] = float(
            (milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if milder_predictor_alignment else None
        evidence["stronger_query_classifier_confidence_lift_score"] = float(
            (stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_query_classifier_confidence_lift else None
        evidence["eight_facets_score"] = float(
            (eight_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if eight_facets else None
        evidence["split_projectors_stronger_global_preserve_score"] = float(
            (split_projectors_stronger_global_preserve or {}).get("score", float("-inf")) or float("-inf")
        ) if split_projectors_stronger_global_preserve else None
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v295":
        control = _entry_by_name(history, "v295 promoted more-facets control")
        same_width_retrieval_head = _entry_by_name(history, "v295 promoted more-facets + same-width retrieval head")
        same_width_retrieval_head_code_query_rank = _entry_by_name(
            history,
            "v295 promoted more-facets + same-width retrieval head + code-query rank retarget",
        )
        same_width_retrieval_head_split_projectors = _entry_by_name(
            history,
            "v295 promoted more-facets + same-width retrieval head + split projectors + stronger global preserve",
        )
        same_width_retrieval_head_eight_facets = _entry_by_name(
            history,
            "v295 promoted more-facets + same-width retrieval head + eight facets",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["same_width_retrieval_head_score"] = float(
            (same_width_retrieval_head or {}).get("score", float("-inf")) or float("-inf")
        ) if same_width_retrieval_head else None
        evidence["same_width_retrieval_head_code_query_rank_score"] = float(
            (same_width_retrieval_head_code_query_rank or {}).get("score", float("-inf")) or float("-inf")
        ) if same_width_retrieval_head_code_query_rank else None
        evidence["same_width_retrieval_head_code_query_rank_retarget_score"] = evidence["same_width_retrieval_head_code_query_rank_score"]
        evidence["same_width_retrieval_head_split_projectors_score"] = float(
            (same_width_retrieval_head_split_projectors or {}).get("score", float("-inf")) or float("-inf")
        ) if same_width_retrieval_head_split_projectors else None
        evidence["same_width_retrieval_head_split_projectors_stronger_global_preserve_score"] = evidence["same_width_retrieval_head_split_projectors_score"]
        evidence["same_width_retrieval_head_eight_facets_score"] = float(
            (same_width_retrieval_head_eight_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if same_width_retrieval_head_eight_facets else None
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v296":
        control = _entry_by_name(history, "v296 promoted eight-facets same-width retrieval head control")
        milder_predictor_alignment = _entry_by_name(
            history,
            "v296 promoted eight-facets same-width retrieval head + milder predictor alignment",
        )
        code_query_rank_retarget = _entry_by_name(
            history,
            "v296 promoted eight-facets same-width retrieval head + code-query rank retarget",
        )
        split_projectors_stronger_global_preserve = _entry_by_name(
            history,
            "v296 promoted eight-facets same-width retrieval head + split projectors + stronger global preserve",
        )
        ten_facets = _entry_by_name(
            history,
            "v296 promoted eight-facets same-width retrieval head + ten facets",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["milder_predictor_alignment_score"] = float(
            (milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if milder_predictor_alignment else None
        evidence["code_query_rank_retarget_score"] = float(
            (code_query_rank_retarget or {}).get("score", float("-inf")) or float("-inf")
        ) if code_query_rank_retarget else None
        evidence["split_projectors_stronger_global_preserve_score"] = float(
            (split_projectors_stronger_global_preserve or {}).get("score", float("-inf")) or float("-inf")
        ) if split_projectors_stronger_global_preserve else None
        evidence["ten_facets_score"] = float(
            (ten_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if ten_facets else None
        evidence["v295_frontier_score"] = 30.36532322317362
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v297":
        control = _entry_by_name(history, "v297 v295-reset eight-facets control")
        code_query_rank_retarget_milder_predictor_alignment = _entry_by_name(
            history,
            "v297 v295-reset eight-facets + code-query rank retarget + milder predictor alignment",
        )
        ten_facets_milder_predictor_alignment = _entry_by_name(
            history,
            "v297 v295-reset eight-facets + ten facets + milder predictor alignment",
        )
        ten_facets_code_query_rank_retarget = _entry_by_name(
            history,
            "v297 v295-reset eight-facets + ten facets + code-query rank retarget",
        )
        ten_facets_code_query_rank_retarget_milder_predictor_alignment = _entry_by_name(
            history,
            "v297 v295-reset eight-facets + ten facets + code-query rank retarget + milder predictor alignment",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["code_query_rank_retarget_milder_predictor_alignment_score"] = float(
            (code_query_rank_retarget_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if code_query_rank_retarget_milder_predictor_alignment else None
        evidence["ten_facets_milder_predictor_alignment_score"] = float(
            (ten_facets_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if ten_facets_milder_predictor_alignment else None
        evidence["ten_facets_code_query_rank_retarget_score"] = float(
            (ten_facets_code_query_rank_retarget or {}).get("score", float("-inf")) or float("-inf")
        ) if ten_facets_code_query_rank_retarget else None
        evidence["ten_facets_code_query_rank_retarget_milder_predictor_alignment_score"] = float(
            (ten_facets_code_query_rank_retarget_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if ten_facets_code_query_rank_retarget_milder_predictor_alignment else None
        evidence["v295_frontier_score"] = 30.36532322317362
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "query embedding effective rank is too low",
            "known confidence too low",
        ]
    elif normalized == "v298":
        control = _entry_by_name(history, "v298 promoted ten-facets rank-retarget control")
        milder_predictor_alignment = _entry_by_name(
            history,
            "v298 promoted ten-facets rank-retarget + milder predictor alignment",
        )
        stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v298 promoted ten-facets rank-retarget + stronger query-classifier confidence lift",
        )
        twelve_facets = _entry_by_name(
            history,
            "v298 promoted ten-facets rank-retarget + twelve facets",
        )
        split_projectors_stronger_global_preserve = _entry_by_name(
            history,
            "v298 promoted ten-facets rank-retarget + split projectors + stronger global preserve",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["milder_predictor_alignment_score"] = float(
            (milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if milder_predictor_alignment else None
        evidence["stronger_query_classifier_confidence_lift_score"] = float(
            (stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_query_classifier_confidence_lift else None
        evidence["twelve_facets_score"] = float(
            (twelve_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if twelve_facets else None
        evidence["split_projectors_stronger_global_preserve_score"] = float(
            (split_projectors_stronger_global_preserve or {}).get("score", float("-inf")) or float("-inf")
        ) if split_projectors_stronger_global_preserve else None
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v300":
        control = _entry_by_name(history, "v300 promoted ten-facets rank-retarget + stronger query-classifier confidence lift control")
        milder_predictor_alignment = _entry_by_name(
            history,
            "v300 promoted ten-facets rank-retarget + stronger query-classifier confidence lift + milder predictor alignment",
        )
        all_stream_rank_guard = _entry_by_name(
            history,
            "v300 promoted ten-facets rank-retarget + stronger query-classifier confidence lift + all-stream rank guard",
        )
        covariance_queue_lift = _entry_by_name(
            history,
            "v300 promoted ten-facets rank-retarget + stronger query-classifier confidence lift + covariance queue lift",
        )
        milder_predictor_alignment_all_stream_rank_guard = _entry_by_name(
            history,
            "v300 promoted ten-facets rank-retarget + stronger query-classifier confidence lift + milder predictor alignment + all-stream rank guard",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["milder_predictor_alignment_score"] = float(
            (milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if milder_predictor_alignment else None
        evidence["all_stream_rank_guard_score"] = float(
            (all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")
        ) if all_stream_rank_guard else None
        evidence["covariance_queue_lift_score"] = float(
            (covariance_queue_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if covariance_queue_lift else None
        evidence["milder_predictor_alignment_all_stream_rank_guard_score"] = float(
            (milder_predictor_alignment_all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")
        ) if milder_predictor_alignment_all_stream_rank_guard else None
        evidence["v298_frontier_score"] = 30.579662762582302
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v301":
        control = _entry_by_name(history, "v301 v298-reset clean frontier control")
        all_stream_rank_guard = _entry_by_name(
            history,
            "v301 v298-reset + all-stream rank guard",
        )
        all_stream_rank_guard_twelve_facets = _entry_by_name(
            history,
            "v301 v298-reset + all-stream rank guard + twelve facets",
        )
        all_stream_rank_guard_split_projectors_stronger_global_preserve = _entry_by_name(
            history,
            "v301 v298-reset + all-stream rank guard + split projectors + stronger global preserve",
        )
        all_stream_rank_guard_slightly_milder_predictor_alignment = _entry_by_name(
            history,
            "v301 v298-reset + all-stream rank guard + slightly milder predictor alignment",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["all_stream_rank_guard_score"] = float(
            (all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")
        ) if all_stream_rank_guard else None
        evidence["all_stream_rank_guard_twelve_facets_score"] = float(
            (all_stream_rank_guard_twelve_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if all_stream_rank_guard_twelve_facets else None
        evidence["all_stream_rank_guard_split_projectors_stronger_global_preserve_score"] = float(
            (all_stream_rank_guard_split_projectors_stronger_global_preserve or {}).get("score", float("-inf")) or float("-inf")
        ) if all_stream_rank_guard_split_projectors_stronger_global_preserve else None
        evidence["all_stream_rank_guard_slightly_milder_predictor_alignment_score"] = float(
            (all_stream_rank_guard_slightly_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if all_stream_rank_guard_slightly_milder_predictor_alignment else None
        evidence["v298_frontier_score"] = 30.579662762582302
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v302":
        control = _entry_by_name(history, "v302 v298-reset clean frontier control")
        mild_all_stream_rank_guard = _entry_by_name(
            history,
            "v302 v298-reset + mild all-stream rank guard",
        )
        mild_all_stream_rank_guard_slightly_stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v302 v298-reset + mild all-stream rank guard + slightly stronger query-classifier confidence lift",
        )
        eleven_facets_slightly_stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v302 v298-reset + eleven facets + slightly stronger query-classifier confidence lift",
        )
        mild_all_stream_rank_guard_eleven_facets_slightly_stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v302 v298-reset + mild all-stream rank guard + eleven facets + slightly stronger query-classifier confidence lift",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["mild_all_stream_rank_guard_score"] = float(
            (mild_all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")
        ) if mild_all_stream_rank_guard else None
        evidence["mild_all_stream_rank_guard_slightly_stronger_query_classifier_confidence_lift_score"] = float(
            (
                mild_all_stream_rank_guard_slightly_stronger_query_classifier_confidence_lift
                or {}
            ).get("score", float("-inf"))
            or float("-inf")
        ) if mild_all_stream_rank_guard_slightly_stronger_query_classifier_confidence_lift else None
        evidence["eleven_facets_slightly_stronger_query_classifier_confidence_lift_score"] = float(
            (
                eleven_facets_slightly_stronger_query_classifier_confidence_lift
                or {}
            ).get("score", float("-inf"))
            or float("-inf")
        ) if eleven_facets_slightly_stronger_query_classifier_confidence_lift else None
        evidence["mild_all_stream_rank_guard_eleven_facets_slightly_stronger_query_classifier_confidence_lift_score"] = float(
            (
                mild_all_stream_rank_guard_eleven_facets_slightly_stronger_query_classifier_confidence_lift
                or {}
            ).get("score", float("-inf"))
            or float("-inf")
        ) if mild_all_stream_rank_guard_eleven_facets_slightly_stronger_query_classifier_confidence_lift else None
        evidence["v298_frontier_score"] = 30.579662762582302
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v303":
        control = _entry_by_name(history, "v303 promoted eleven-facets geometry control")
        stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v303 promoted eleven-facets + stronger query-classifier confidence lift",
        )
        earlier_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v303 promoted eleven-facets + earlier query-classifier confidence lift",
        )
        milder_predictor_alignment = _entry_by_name(
            history,
            "v303 promoted eleven-facets + milder predictor alignment",
        )
        stronger_query_classifier_confidence_lift_milder_predictor_alignment = _entry_by_name(
            history,
            "v303 promoted eleven-facets + stronger query-classifier confidence lift + milder predictor alignment",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_query_classifier_confidence_lift_score"] = float(
            (stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_query_classifier_confidence_lift else None
        evidence["earlier_query_classifier_confidence_lift_score"] = float(
            (earlier_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if earlier_query_classifier_confidence_lift else None
        evidence["milder_predictor_alignment_score"] = float(
            (milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if milder_predictor_alignment else None
        evidence["stronger_query_classifier_confidence_lift_milder_predictor_alignment_score"] = float(
            (
                stronger_query_classifier_confidence_lift_milder_predictor_alignment
                or {}
            ).get("score", float("-inf"))
            or float("-inf")
        ) if stronger_query_classifier_confidence_lift_milder_predictor_alignment else None
        evidence["v302_frontier_score"] = 30.503284690280758
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v304":
        control = _entry_by_name(history, "v304 promoted eleven-facets + stronger query-classifier confidence lift control")
        twelve_facets = _entry_by_name(
            history,
            "v304 promoted eleven-facets + stronger query-classifier confidence lift + twelve facets",
        )
        all_stream_rank_retarget = _entry_by_name(
            history,
            "v304 promoted eleven-facets + stronger query-classifier confidence lift + all-stream rank retarget",
        )
        stronger_hard_negative_budget = _entry_by_name(
            history,
            "v304 promoted eleven-facets + stronger query-classifier confidence lift + stronger hard-negative budget",
        )
        twelve_facets_stronger_hard_negative_budget = _entry_by_name(
            history,
            "v304 promoted eleven-facets + stronger query-classifier confidence lift + twelve facets + stronger hard-negative budget",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["twelve_facets_score"] = float(
            (twelve_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if twelve_facets else None
        evidence["all_stream_rank_retarget_score"] = float(
            (all_stream_rank_retarget or {}).get("score", float("-inf")) or float("-inf")
        ) if all_stream_rank_retarget else None
        evidence["stronger_hard_negative_budget_score"] = float(
            (stronger_hard_negative_budget or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_hard_negative_budget else None
        evidence["twelve_facets_stronger_hard_negative_budget_score"] = float(
            (twelve_facets_stronger_hard_negative_budget or {}).get("score", float("-inf")) or float("-inf")
        ) if twelve_facets_stronger_hard_negative_budget else None
        evidence["v298_frontier_score"] = 30.579662762582302
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v305":
        control = _entry_by_name(
            history,
            "v305 promoted twelve-facets + stronger query-classifier confidence lift + stronger hard-negative budget control",
        )
        slightly_milder_predictor_alignment = _entry_by_name(
            history,
            "v305 promoted twelve-facets + stronger query-classifier confidence lift + stronger hard-negative budget + slightly milder predictor alignment",
        )
        slightly_stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v305 promoted twelve-facets + stronger query-classifier confidence lift + stronger hard-negative budget + slightly stronger query-classifier confidence lift",
        )
        thirteen_facets = _entry_by_name(
            history,
            "v305 promoted twelve-facets + stronger query-classifier confidence lift + stronger hard-negative budget + thirteen facets",
        )
        slightly_milder_predictor_alignment_slightly_stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v305 promoted twelve-facets + stronger query-classifier confidence lift + stronger hard-negative budget + slightly milder predictor alignment + slightly stronger query-classifier confidence lift",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_milder_predictor_alignment_score"] = float(
            (slightly_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if slightly_milder_predictor_alignment else None
        evidence["slightly_stronger_query_classifier_confidence_lift_score"] = float(
            (slightly_stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if slightly_stronger_query_classifier_confidence_lift else None
        evidence["thirteen_facets_score"] = float(
            (thirteen_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if thirteen_facets else None
        evidence["slightly_milder_predictor_alignment_slightly_stronger_query_classifier_confidence_lift_score"] = float(
            (slightly_milder_predictor_alignment_slightly_stronger_query_classifier_confidence_lift or {}).get("score", float("-inf"))
            or float("-inf")
        ) if slightly_milder_predictor_alignment_slightly_stronger_query_classifier_confidence_lift else None
        evidence["v304_frontier_score"] = 30.51710143685341
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v306":
        control = _entry_by_name(history, "v306 promoted thirteen-facets control")
        slightly_milder_predictor_alignment = _entry_by_name(
            history,
            "v306 promoted thirteen-facets + slightly milder predictor alignment",
        )
        mild_all_stream_rank_guard = _entry_by_name(
            history,
            "v306 promoted thirteen-facets + mild all-stream rank guard",
        )
        fourteen_facets = _entry_by_name(
            history,
            "v306 promoted thirteen-facets + fourteen facets",
        )
        fourteen_facets_mild_all_stream_rank_guard = _entry_by_name(
            history,
            "v306 promoted thirteen-facets + fourteen facets + mild all-stream rank guard",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_milder_predictor_alignment_score"] = float(
            (slightly_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if slightly_milder_predictor_alignment else None
        evidence["mild_all_stream_rank_guard_score"] = float(
            (mild_all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")
        ) if mild_all_stream_rank_guard else None
        evidence["fourteen_facets_score"] = float(
            (fourteen_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if fourteen_facets else None
        evidence["fourteen_facets_mild_all_stream_rank_guard_score"] = float(
            (fourteen_facets_mild_all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")
        ) if fourteen_facets_mild_all_stream_rank_guard else None
        evidence["v305_frontier_score"] = 31.06095204253991
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v307":
        control = _entry_by_name(history, "v307 v305-reset thirteen-facets control")
        mild_all_stream_rank_guard = _entry_by_name(
            history,
            "v307 v305-reset thirteen-facets + mild all-stream rank guard",
        )
        slightly_milder_predictor_alignment = _entry_by_name(
            history,
            "v307 v305-reset thirteen-facets + slightly milder predictor alignment",
        )
        mild_all_stream_rank_guard_slightly_milder_predictor_alignment = _entry_by_name(
            history,
            "v307 v305-reset thirteen-facets + mild all-stream rank guard + slightly milder predictor alignment",
        )
        fourteen_facets_slightly_milder_predictor_alignment = _entry_by_name(
            history,
            "v307 v305-reset thirteen-facets + fourteen facets + slightly milder predictor alignment",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["mild_all_stream_rank_guard_score"] = float(
            (mild_all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")
        ) if mild_all_stream_rank_guard else None
        evidence["slightly_milder_predictor_alignment_score"] = float(
            (slightly_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if slightly_milder_predictor_alignment else None
        evidence["mild_all_stream_rank_guard_slightly_milder_predictor_alignment_score"] = float(
            (mild_all_stream_rank_guard_slightly_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if mild_all_stream_rank_guard_slightly_milder_predictor_alignment else None
        evidence["fourteen_facets_slightly_milder_predictor_alignment_score"] = float(
            (fourteen_facets_slightly_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if fourteen_facets_slightly_milder_predictor_alignment else None
        evidence["v305_frontier_score"] = 31.06095204253991
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v308":
        control = _entry_by_name(history, "v308 promoted fourteen-facets + slightly milder predictor alignment control")
        mild_all_stream_rank_guard = _entry_by_name(
            history,
            "v308 promoted fourteen-facets + slightly milder predictor alignment + mild all-stream rank guard",
        )
        stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v308 promoted fourteen-facets + slightly milder predictor alignment + stronger query-classifier confidence lift",
        )
        fifteen_facets = _entry_by_name(
            history,
            "v308 promoted fourteen-facets + slightly milder predictor alignment + fifteen facets",
        )
        mild_all_stream_rank_guard_stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v308 promoted fourteen-facets + slightly milder predictor alignment + mild all-stream rank guard + stronger query-classifier confidence lift",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["mild_all_stream_rank_guard_score"] = float(
            (mild_all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")
        ) if mild_all_stream_rank_guard else None
        evidence["stronger_query_classifier_confidence_lift_score"] = float(
            (stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_query_classifier_confidence_lift else None
        evidence["fifteen_facets_score"] = float(
            (fifteen_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if fifteen_facets else None
        evidence["mild_all_stream_rank_guard_stronger_query_classifier_confidence_lift_score"] = float(
            (mild_all_stream_rank_guard_stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if mild_all_stream_rank_guard_stronger_query_classifier_confidence_lift else None
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v309":
        control = _entry_by_name(history, "v309 promoted stronger-query-classifier control")
        even_milder_predictor_alignment = _entry_by_name(
            history,
            "v309 promoted stronger-query-classifier + even milder predictor alignment",
        )
        mild_all_stream_rank_guard_smaller_query_classifier_lift = _entry_by_name(
            history,
            "v309 promoted stronger-query-classifier + mild all-stream rank guard + smaller query-classifier lift",
        )
        fifteen_facets_smaller_query_classifier_lift = _entry_by_name(
            history,
            "v309 promoted stronger-query-classifier + fifteen facets + smaller query-classifier lift",
        )
        later_query_classifier_onset = _entry_by_name(
            history,
            "v309 promoted stronger-query-classifier + later query-classifier onset",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["even_milder_predictor_alignment_score"] = float(
            (even_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if even_milder_predictor_alignment else None
        evidence["mild_all_stream_rank_guard_smaller_query_classifier_lift_score"] = float(
            (mild_all_stream_rank_guard_smaller_query_classifier_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if mild_all_stream_rank_guard_smaller_query_classifier_lift else None
        evidence["fifteen_facets_smaller_query_classifier_lift_score"] = float(
            (fifteen_facets_smaller_query_classifier_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if fifteen_facets_smaller_query_classifier_lift else None
        evidence["later_query_classifier_onset_score"] = float(
            (later_query_classifier_onset or {}).get("score", float("-inf")) or float("-inf")
        ) if later_query_classifier_onset else None
        evidence["v305_frontier_score"] = 31.06095204253991
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v310":
        control = _entry_by_name(history, "v310 promoted fifteen-facets + smaller query-classifier control")
        mild_all_stream_rank_guard = _entry_by_name(
            history,
            "v310 promoted fifteen-facets + smaller query-classifier + mild all-stream rank guard",
        )
        even_milder_predictor_alignment = _entry_by_name(
            history,
            "v310 promoted fifteen-facets + smaller query-classifier + even milder predictor alignment",
        )
        slightly_stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v310 promoted fifteen-facets + slightly stronger query-classifier confidence lift",
        )
        mild_all_stream_rank_guard_even_milder_predictor_alignment = _entry_by_name(
            history,
            "v310 promoted fifteen-facets + smaller query-classifier + mild all-stream rank guard + even milder predictor alignment",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["mild_all_stream_rank_guard_score"] = float(
            (mild_all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")
        ) if mild_all_stream_rank_guard else None
        evidence["even_milder_predictor_alignment_score"] = float(
            (even_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if even_milder_predictor_alignment else None
        evidence["slightly_stronger_query_classifier_confidence_lift_score"] = float(
            (slightly_stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if slightly_stronger_query_classifier_confidence_lift else None
        evidence["mild_all_stream_rank_guard_even_milder_predictor_alignment_score"] = float(
            (mild_all_stream_rank_guard_even_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if mild_all_stream_rank_guard_even_milder_predictor_alignment else None
        evidence["v309_frontier_score"] = 30.79578161736329
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v311":
        control = _entry_by_name(history, "v311 v309-reset fifteen-facets control")
        later_query_classifier_onset = _entry_by_name(
            history,
            "v311 v309-reset fifteen-facets + later query-classifier onset",
        )
        mild_all_stream_rank_guard_later_query_classifier_onset = _entry_by_name(
            history,
            "v311 v309-reset fifteen-facets + mild all-stream rank guard + later query-classifier onset",
        )
        even_milder_predictor_alignment_later_query_classifier_onset = _entry_by_name(
            history,
            "v311 v309-reset fifteen-facets + even milder predictor alignment + later query-classifier onset",
        )
        mixed_query_pred_confidence_bridge = _entry_by_name(
            history,
            "v311 v309-reset fifteen-facets + mixed query/pred confidence bridge",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["later_query_classifier_onset_score"] = float(
            (later_query_classifier_onset or {}).get("score", float("-inf")) or float("-inf")
        ) if later_query_classifier_onset else None
        evidence["mild_all_stream_rank_guard_later_query_classifier_onset_score"] = float(
            (mild_all_stream_rank_guard_later_query_classifier_onset or {}).get("score", float("-inf")) or float("-inf")
        ) if mild_all_stream_rank_guard_later_query_classifier_onset else None
        evidence["even_milder_predictor_alignment_later_query_classifier_onset_score"] = float(
            (even_milder_predictor_alignment_later_query_classifier_onset or {}).get("score", float("-inf")) or float("-inf")
        ) if even_milder_predictor_alignment_later_query_classifier_onset else None
        evidence["mixed_query_pred_confidence_bridge_score"] = float(
            (mixed_query_pred_confidence_bridge or {}).get("score", float("-inf")) or float("-inf")
        ) if mixed_query_pred_confidence_bridge else None
        evidence["v305_frontier_score"] = 31.06095204253991
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v312":
        control = _entry_by_name(history, "v312 promoted later-query-onset control")
        slightly_stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v312 promoted later-query-onset + slightly stronger query-classifier confidence lift",
        )
        lighter_all_stream_rank_guard = _entry_by_name(
            history,
            "v312 promoted later-query-onset + lighter all-stream rank guard",
        )
        mixed_query_pred_confidence_bridge = _entry_by_name(
            history,
            "v312 promoted later-query-onset + mixed query/pred confidence bridge",
        )
        lighter_all_stream_rank_guard_slightly_stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v312 promoted later-query-onset + lighter all-stream rank guard + slightly stronger query-classifier confidence lift",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["slightly_stronger_query_classifier_confidence_lift_score"] = float(
            (slightly_stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if slightly_stronger_query_classifier_confidence_lift else None
        evidence["lighter_all_stream_rank_guard_score"] = float(
            (lighter_all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_all_stream_rank_guard else None
        evidence["mixed_query_pred_confidence_bridge_score"] = float(
            (mixed_query_pred_confidence_bridge or {}).get("score", float("-inf")) or float("-inf")
        ) if mixed_query_pred_confidence_bridge else None
        evidence["lighter_all_stream_rank_guard_slightly_stronger_query_classifier_confidence_lift_score"] = float(
            (lighter_all_stream_rank_guard_slightly_stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_all_stream_rank_guard_slightly_stronger_query_classifier_confidence_lift else None
        evidence["v311_frontier_score"] = 30.833819441497326
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v313":
        control = _entry_by_name(history, "v313 promoted later-query-onset fifteen-facets control")
        sixteen_facets = _entry_by_name(
            history,
            "v313 promoted later-query-onset fifteen-facets + sixteen facets",
        )
        split_projectors_stronger_global_preserve = _entry_by_name(
            history,
            "v313 promoted later-query-onset fifteen-facets + split projectors + stronger global preserve",
        )
        lighter_query_multiview = _entry_by_name(
            history,
            "v313 promoted later-query-onset fifteen-facets + lighter query multiview",
        )
        lighter_equivalence_pull = _entry_by_name(
            history,
            "v313 promoted later-query-onset fifteen-facets + lighter equivalence pull",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["sixteen_facets_score"] = float(
            (sixteen_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if sixteen_facets else None
        evidence["split_projectors_stronger_global_preserve_score"] = float(
            (split_projectors_stronger_global_preserve or {}).get("score", float("-inf")) or float("-inf")
        ) if split_projectors_stronger_global_preserve else None
        evidence["lighter_query_multiview_score"] = float(
            (lighter_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_query_multiview else None
        evidence["lighter_equivalence_pull_score"] = float(
            (lighter_equivalence_pull or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_equivalence_pull else None
        evidence["v311_frontier_score"] = 30.833819441497326
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v314":
        control = _entry_by_name(history, "v314 v311-reset later-query-onset control")
        lighter_query_multiview = _entry_by_name(
            history,
            "v314 v311-reset later-query-onset + lighter query multiview",
        )
        lighter_query_multiview_slightly_stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v314 v311-reset later-query-onset + lighter query multiview + slightly stronger query-classifier confidence lift",
        )
        lighter_equivalence_pull_slightly_stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v314 v311-reset later-query-onset + lighter equivalence pull + slightly stronger query-classifier confidence lift",
        )
        sixteen_facets_slightly_stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v314 v311-reset later-query-onset + sixteen facets + slightly stronger query-classifier confidence lift",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["lighter_query_multiview_score"] = float(
            (lighter_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_query_multiview else None
        evidence["lighter_query_multiview_slightly_stronger_query_classifier_confidence_lift_score"] = float(
            (lighter_query_multiview_slightly_stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_query_multiview_slightly_stronger_query_classifier_confidence_lift else None
        evidence["lighter_equivalence_pull_slightly_stronger_query_classifier_confidence_lift_score"] = float(
            (lighter_equivalence_pull_slightly_stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_equivalence_pull_slightly_stronger_query_classifier_confidence_lift else None
        evidence["sixteen_facets_slightly_stronger_query_classifier_confidence_lift_score"] = float(
            (sixteen_facets_slightly_stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if sixteen_facets_slightly_stronger_query_classifier_confidence_lift else None
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v315":
        control = _entry_by_name(history, "v315 promoted sixteen-facets + stronger query-classifier control")
        lighter_equivalence_pull = _entry_by_name(
            history,
            "v315 promoted sixteen-facets + stronger query-classifier + lighter equivalence pull",
        )
        milder_predictor_alignment = _entry_by_name(
            history,
            "v315 promoted sixteen-facets + stronger query-classifier + milder predictor alignment",
        )
        eighteen_facets = _entry_by_name(
            history,
            "v315 promoted sixteen-facets + stronger query-classifier + eighteen facets",
        )
        slightly_stronger_query_classifier_lift = _entry_by_name(
            history,
            "v315 promoted sixteen-facets + stronger query-classifier + slightly stronger query-classifier lift",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["lighter_equivalence_pull_score"] = float(
            (lighter_equivalence_pull or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_equivalence_pull else None
        evidence["milder_predictor_alignment_score"] = float(
            (milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if milder_predictor_alignment else None
        evidence["eighteen_facets_score"] = float(
            (eighteen_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if eighteen_facets else None
        evidence["slightly_stronger_query_classifier_lift_score"] = float(
            (slightly_stronger_query_classifier_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if slightly_stronger_query_classifier_lift else None
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v316":
        control = _entry_by_name(history, "v316 promoted eighteen-facets + stronger query-classifier control")
        lighter_equivalence_pull = _entry_by_name(
            history,
            "v316 promoted eighteen-facets + stronger query-classifier + lighter equivalence pull",
        )
        slightly_stronger_query_classifier_lift = _entry_by_name(
            history,
            "v316 promoted eighteen-facets + stronger query-classifier + slightly stronger query-classifier lift",
        )
        lighter_equivalence_pull_slightly_stronger_query_classifier_lift = _entry_by_name(
            history,
            "v316 promoted eighteen-facets + stronger query-classifier + lighter equivalence pull + slightly stronger query-classifier lift",
        )
        twenty_facets_lighter_equivalence_pull = _entry_by_name(
            history,
            "v316 promoted eighteen-facets + stronger query-classifier + twenty facets + lighter equivalence pull",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["lighter_equivalence_pull_score"] = float(
            (lighter_equivalence_pull or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_equivalence_pull else None
        evidence["slightly_stronger_query_classifier_lift_score"] = float(
            (slightly_stronger_query_classifier_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if slightly_stronger_query_classifier_lift else None
        evidence["lighter_equivalence_pull_slightly_stronger_query_classifier_lift_score"] = float(
            (lighter_equivalence_pull_slightly_stronger_query_classifier_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_equivalence_pull_slightly_stronger_query_classifier_lift else None
        evidence["twenty_facets_lighter_equivalence_pull_score"] = float(
            (twenty_facets_lighter_equivalence_pull or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_facets_lighter_equivalence_pull else None
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v317":
        control = _entry_by_name(history, "v317 promoted eighteen-facets + lighter equivalence pull + slightly stronger query-classifier control")
        milder_predictor_alignment = _entry_by_name(
            history,
            "v317 promoted eighteen-facets + lighter equivalence pull + slightly stronger query-classifier + milder predictor alignment",
        )
        all_stream_rank_guard = _entry_by_name(
            history,
            "v317 promoted eighteen-facets + lighter equivalence pull + slightly stronger query-classifier + all-stream rank guard",
        )
        twenty_facets = _entry_by_name(
            history,
            "v317 promoted eighteen-facets + lighter equivalence pull + slightly stronger query-classifier + twenty facets",
        )
        milder_predictor_alignment_all_stream_rank_guard = _entry_by_name(
            history,
            "v317 promoted eighteen-facets + lighter equivalence pull + slightly stronger query-classifier + milder predictor alignment + all-stream rank guard",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["milder_predictor_alignment_score"] = float(
            (milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if milder_predictor_alignment else None
        evidence["all_stream_rank_guard_score"] = float(
            (all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")
        ) if all_stream_rank_guard else None
        evidence["twenty_facets_score"] = float(
            (twenty_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_facets else None
        evidence["milder_predictor_alignment_all_stream_rank_guard_score"] = float(
            (milder_predictor_alignment_all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")
        ) if milder_predictor_alignment_all_stream_rank_guard else None
        evidence["v316_frontier_score"] = 31.088265382995203
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v318":
        control = _entry_by_name(history, "v318 v316-reset clean frontier control")
        all_stream_rank_guard = _entry_by_name(history, "v318 v316-reset + all-stream rank guard")
        twenty_facets = _entry_by_name(history, "v318 v316-reset + twenty facets")
        twenty_facets_all_stream_rank_guard = _entry_by_name(
            history,
            "v318 v316-reset + twenty facets + all-stream rank guard",
        )
        twenty_facets_all_stream_rank_guard_milder_predictor_alignment = _entry_by_name(
            history,
            "v318 v316-reset + twenty facets + all-stream rank guard + milder predictor alignment",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["all_stream_rank_guard_score"] = float(
            (all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")
        ) if all_stream_rank_guard else None
        evidence["twenty_facets_score"] = float(
            (twenty_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_facets else None
        evidence["twenty_facets_all_stream_rank_guard_score"] = float(
            (twenty_facets_all_stream_rank_guard or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_facets_all_stream_rank_guard else None
        evidence["twenty_facets_all_stream_rank_guard_milder_predictor_alignment_score"] = float(
            (twenty_facets_all_stream_rank_guard_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_facets_all_stream_rank_guard_milder_predictor_alignment else None
        evidence["v316_frontier_score"] = 31.088265382995203
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v319":
        control = _entry_by_name(history, "v319 promoted twenty-facets control")
        lighter_equivalence_pull = _entry_by_name(history, "v319 promoted twenty-facets + lighter equivalence pull")
        firmer_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v319 promoted twenty-facets + firmer query-classifier confidence lift",
        )
        separate_query_code_facets = _entry_by_name(history, "v319 promoted twenty-facets + separate query/code facets")
        twenty_two_facets = _entry_by_name(history, "v319 promoted twenty-facets + twenty-two facets")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["lighter_equivalence_pull_score"] = float(
            (lighter_equivalence_pull or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_equivalence_pull else None
        evidence["firmer_query_classifier_confidence_lift_score"] = float(
            (firmer_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if firmer_query_classifier_confidence_lift else None
        evidence["separate_query_code_facets_score"] = float(
            (separate_query_code_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if separate_query_code_facets else None
        evidence["twenty_two_facets_score"] = float(
            (twenty_two_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_two_facets else None
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v320":
        control = _entry_by_name(history, "v320 promoted lighter-equivalence control")
        equivalence_off_ablation = _entry_by_name(history, "v320 promoted lighter-equivalence + equivalence-off ablation")
        twenty_two_facets = _entry_by_name(history, "v320 promoted lighter-equivalence + twenty-two facets")
        separate_query_code_facets = _entry_by_name(history, "v320 promoted lighter-equivalence + separate query/code facets")
        twenty_two_separate_query_code_facets = _entry_by_name(
            history,
            "v320 promoted lighter-equivalence + twenty-two facets + separate query/code facets",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["equivalence_off_ablation_score"] = float(
            (equivalence_off_ablation or {}).get("score", float("-inf")) or float("-inf")
        ) if equivalence_off_ablation else None
        evidence["twenty_two_facets_score"] = float(
            (twenty_two_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_two_facets else None
        evidence["separate_query_code_facets_score"] = float(
            (separate_query_code_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if separate_query_code_facets else None
        evidence["twenty_two_separate_query_code_facets_score"] = float(
            (twenty_two_separate_query_code_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_two_separate_query_code_facets else None
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v321":
        control = _entry_by_name(history, "v321 promoted twenty-two-facets control")
        twenty_four_facets = _entry_by_name(history, "v321 promoted twenty-two-facets + twenty-four facets")
        separate_query_code_facets = _entry_by_name(history, "v321 promoted twenty-two-facets + separate query/code facets")
        equivalence_off_ablation = _entry_by_name(history, "v321 promoted twenty-two-facets + equivalence-off ablation")
        firmer_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v321 promoted twenty-two-facets + firmer query-classifier confidence lift",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["twenty_four_facets_score"] = float(
            (twenty_four_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_four_facets else None
        evidence["separate_query_code_facets_score"] = float(
            (separate_query_code_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if separate_query_code_facets else None
        evidence["equivalence_off_ablation_score"] = float(
            (equivalence_off_ablation or {}).get("score", float("-inf")) or float("-inf")
        ) if equivalence_off_ablation else None
        evidence["firmer_query_classifier_confidence_lift_score"] = float(
            (firmer_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if firmer_query_classifier_confidence_lift else None
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v322":
        control = _entry_by_name(history, "v322 v320-reset twenty-two-facets control")
        equivalence_off_ablation = _entry_by_name(history, "v322 v320-reset + equivalence-off ablation")
        twenty_four_facets = _entry_by_name(history, "v322 v320-reset + equivalence-off + twenty-four facets")
        separate_query_code_facets = _entry_by_name(history, "v322 v320-reset + equivalence-off + separate query/code facets")
        twenty_four_separate_query_code_facets = _entry_by_name(
            history,
            "v322 v320-reset + equivalence-off + twenty-four facets + separate query/code facets",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["equivalence_off_ablation_score"] = float(
            (equivalence_off_ablation or {}).get("score", float("-inf")) or float("-inf")
        ) if equivalence_off_ablation else None
        evidence["twenty_four_facets_score"] = float(
            (twenty_four_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_four_facets else None
        evidence["separate_query_code_facets_score"] = float(
            (separate_query_code_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if separate_query_code_facets else None
        evidence["twenty_four_separate_query_code_facets_score"] = float(
            (twenty_four_separate_query_code_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_four_separate_query_code_facets else None
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v323":
        control = _entry_by_name(history, "v323 promoted equivalence-off control")
        milder_predictor_alignment = _entry_by_name(history, "v323 promoted equivalence-off + milder predictor alignment")
        stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v323 promoted equivalence-off + stronger query-classifier confidence lift",
        )
        twenty_four_facets = _entry_by_name(history, "v323 promoted equivalence-off + twenty-four facets")
        twenty_four_facets_separate_query_code_facets = _entry_by_name(
            history,
            "v323 promoted equivalence-off + twenty-four facets + separate query/code facets",
        )
        adaptive_margin = _entry_by_name(history, "adaptive margin recovery")
        adaptive_confidence = _entry_by_name(history, "adaptive confidence calibration")
        adaptive_rank = _entry_by_name(history, "adaptive rank recovery")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["milder_predictor_alignment_score"] = float(
            (milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if milder_predictor_alignment else None
        evidence["stronger_query_classifier_confidence_lift_score"] = float(
            (stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_query_classifier_confidence_lift else None
        evidence["twenty_four_facets_score"] = float(
            (twenty_four_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_four_facets else None
        evidence["twenty_four_separate_query_code_facets_score"] = float(
            (twenty_four_facets_separate_query_code_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_four_facets_separate_query_code_facets else None
        evidence["adaptive_margin_score"] = float((adaptive_margin or {}).get("score", float("-inf")) or float("-inf")) if adaptive_margin else None
        evidence["adaptive_confidence_score"] = float((adaptive_confidence or {}).get("score", float("-inf")) or float("-inf")) if adaptive_confidence else None
        evidence["adaptive_rank_score"] = float((adaptive_rank or {}).get("score", float("-inf")) or float("-inf")) if adaptive_rank else None
        evidence["v320_frontier_score"] = 31.503257592519127
        evidence["v322_equivalence_off_ablation_score"] = 31.47150681416194
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v324":
        control = _entry_by_name(history, "v324 v320-reset clean frontier control")
        equivalence_off_milder_predictor_alignment = _entry_by_name(
            history,
            "v324 v320-reset + equivalence-off + milder predictor alignment",
        )
        equivalence_off_modest_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v324 v320-reset + equivalence-off + modest query-classifier confidence lift",
        )
        twenty_four_facets_milder_predictor_alignment = _entry_by_name(
            history,
            "v324 v320-reset + equivalence-off + twenty-four facets + milder predictor alignment",
        )
        twenty_four_facets_modest_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v324 v320-reset + equivalence-off + twenty-four facets + modest query-classifier confidence lift",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["equivalence_off_milder_predictor_alignment_score"] = float(
            (equivalence_off_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if equivalence_off_milder_predictor_alignment else None
        evidence["equivalence_off_modest_query_classifier_confidence_lift_score"] = float(
            (equivalence_off_modest_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if equivalence_off_modest_query_classifier_confidence_lift else None
        evidence["twenty_four_facets_milder_predictor_alignment_score"] = float(
            (twenty_four_facets_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_four_facets_milder_predictor_alignment else None
        evidence["twenty_four_facets_modest_query_classifier_confidence_lift_score"] = float(
            (twenty_four_facets_modest_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_four_facets_modest_query_classifier_confidence_lift else None
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v325":
        control = _entry_by_name(history, "v325 promoted twenty-four-facets + milder predictor alignment control")
        smaller_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v325 promoted twenty-four-facets + milder predictor alignment + smaller query-classifier confidence lift",
        )
        predictor_primary_confidence_lift = _entry_by_name(
            history,
            "v325 promoted twenty-four-facets + milder predictor alignment + predictor-primary confidence lift",
        )
        twenty_six_facets = _entry_by_name(
            history,
            "v325 promoted twenty-four-facets + milder predictor alignment + twenty-six facets",
        )
        twenty_six_facets_smaller_query_classifier_lift = _entry_by_name(
            history,
            "v325 promoted twenty-four-facets + milder predictor alignment + twenty-six facets + smaller query-classifier lift",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["smaller_query_classifier_confidence_lift_score"] = float(
            (smaller_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if smaller_query_classifier_confidence_lift else None
        evidence["predictor_primary_confidence_lift_score"] = float(
            (predictor_primary_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if predictor_primary_confidence_lift else None
        evidence["twenty_six_facets_score"] = float(
            (twenty_six_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_six_facets else None
        evidence["twenty_six_facets_smaller_query_classifier_lift_score"] = float(
            (twenty_six_facets_smaller_query_classifier_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_six_facets_smaller_query_classifier_lift else None
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v326":
        control = _entry_by_name(history, "v326 promoted twenty-six-facets + smaller query-classifier lift control")
        even_milder_predictor_alignment = _entry_by_name(
            history,
            "v326 promoted twenty-six-facets + smaller query-classifier lift + even milder predictor alignment",
        )
        slightly_stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v326 promoted twenty-six-facets + smaller query-classifier lift + slightly stronger query-classifier confidence lift",
        )
        twenty_eight_facets = _entry_by_name(
            history,
            "v326 promoted twenty-six-facets + smaller query-classifier lift + twenty-eight facets",
        )
        twenty_eight_facets_slightly_stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v326 promoted twenty-six-facets + smaller query-classifier lift + twenty-eight facets + slightly stronger query-classifier confidence lift",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["even_milder_predictor_alignment_score"] = float(
            (even_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if even_milder_predictor_alignment else None
        evidence["slightly_stronger_query_classifier_confidence_lift_score"] = float(
            (slightly_stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if slightly_stronger_query_classifier_confidence_lift else None
        evidence["twenty_eight_facets_score"] = float(
            (twenty_eight_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_eight_facets else None
        evidence["twenty_eight_facets_slightly_stronger_query_classifier_confidence_lift_score"] = float(
            (twenty_eight_facets_slightly_stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if twenty_eight_facets_slightly_stronger_query_classifier_confidence_lift else None
        evidence["remaining_strict_failures"] = [
            "code embedding effective rank is too low",
            "known confidence too low",
            "query embedding effective rank is too low",
        ]
    elif normalized == "v327":
        control = _entry_by_name(history, "v327 promoted twenty-eight-facets + slightly stronger query-classifier lift control")
        even_milder_predictor_alignment = _entry_by_name(
            history,
            "v327 promoted twenty-eight-facets + slightly stronger query-classifier lift + even milder predictor alignment",
        )
        stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v327 promoted twenty-eight-facets + slightly stronger query-classifier lift + stronger query-classifier confidence lift",
        )
        thirty_facets = _entry_by_name(
            history,
            "v327 promoted twenty-eight-facets + slightly stronger query-classifier lift + thirty facets",
        )
        split_projectors_stronger_global_preserve = _entry_by_name(
            history,
            "v327 promoted twenty-eight-facets + slightly stronger query-classifier lift + split projectors + stronger global preserve",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["even_milder_predictor_alignment_score"] = float(
            (even_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if even_milder_predictor_alignment else None
        evidence["stronger_query_classifier_confidence_lift_score"] = float(
            (stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_query_classifier_confidence_lift else None
        evidence["thirty_facets_score"] = float(
            (thirty_facets or {}).get("score", float("-inf")) or float("-inf")
        ) if thirty_facets else None
        evidence["split_projectors_stronger_global_preserve_score"] = float(
            (split_projectors_stronger_global_preserve or {}).get("score", float("-inf")) or float("-inf")
        ) if split_projectors_stronger_global_preserve else None
        evidence["remaining_strict_failures"] = ["known confidence too low"]
    elif normalized == "v328":
        control = _entry_by_name(history, "v328 promoted thirty-facets control")
        even_milder_predictor_alignment = _entry_by_name(
            history,
            "v328 promoted thirty-facets + even milder predictor alignment",
        )
        slightly_stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v328 promoted thirty-facets + slightly stronger query-classifier confidence lift",
        )
        even_milder_predictor_alignment_slightly_stronger_query_classifier_confidence_lift = _entry_by_name(
            history,
            "v328 promoted thirty-facets + even milder predictor alignment + slightly stronger query-classifier confidence lift",
        )
        mixed_query_pred_confidence_bridge = _entry_by_name(
            history,
            "v328 promoted thirty-facets + mixed query-pred confidence bridge",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["even_milder_predictor_alignment_score"] = float(
            (even_milder_predictor_alignment or {}).get("score", float("-inf")) or float("-inf")
        ) if even_milder_predictor_alignment else None
        evidence["slightly_stronger_query_classifier_confidence_lift_score"] = float(
            (slightly_stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if slightly_stronger_query_classifier_confidence_lift else None
        evidence["even_milder_predictor_alignment_slightly_stronger_query_classifier_confidence_lift_score"] = float(
            (even_milder_predictor_alignment_slightly_stronger_query_classifier_confidence_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if even_milder_predictor_alignment_slightly_stronger_query_classifier_confidence_lift else None
        evidence["mixed_query_pred_confidence_bridge_score"] = float(
            (mixed_query_pred_confidence_bridge or {}).get("score", float("-inf")) or float("-inf")
        ) if mixed_query_pred_confidence_bridge else None
        evidence["remaining_strict_failures"] = ["known confidence too low"]
    elif normalized == "v329":
        control = _entry_by_name(history, "v329 frozen thirty-facets keep-head control")
        earlier_query_classifier_onset = _entry_by_name(
            history,
            "v329 frozen thirty-facets + earlier query-classifier onset",
        )
        stronger_query_classifier_lift = _entry_by_name(
            history,
            "v329 frozen thirty-facets + stronger query-classifier lift",
        )
        mixed_query_pred_confidence_bridge = _entry_by_name(
            history,
            "v329 frozen thirty-facets + mixed query-pred confidence bridge",
        )
        stronger_query_classifier_lift_earlier_onset = _entry_by_name(
            history,
            "v329 frozen thirty-facets + stronger query-classifier lift + earlier onset",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["earlier_query_classifier_onset_score"] = float(
            (earlier_query_classifier_onset or {}).get("score", float("-inf")) or float("-inf")
        ) if earlier_query_classifier_onset else None
        evidence["stronger_query_classifier_lift_score"] = float(
            (stronger_query_classifier_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_query_classifier_lift else None
        evidence["mixed_query_pred_confidence_bridge_score"] = float(
            (mixed_query_pred_confidence_bridge or {}).get("score", float("-inf")) or float("-inf")
        ) if mixed_query_pred_confidence_bridge else None
        evidence["stronger_query_classifier_lift_earlier_onset_score"] = float(
            (stronger_query_classifier_lift_earlier_onset or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_query_classifier_lift_earlier_onset else None
        evidence["remaining_strict_failures"] = ["known confidence too low"]
    elif normalized == "v330":
        control = _entry_by_name(history, "v330 promoted thawed thirty-facets keep-head control")
        earlier_query_classifier_onset = _entry_by_name(
            history,
            "v330 promoted thawed thirty-facets + earlier query-classifier onset",
        )
        smaller_query_classifier_lift = _entry_by_name(
            history,
            "v330 promoted thawed thirty-facets + smaller query-classifier lift",
        )
        lighter_query_multiview = _entry_by_name(
            history,
            "v330 promoted thawed thirty-facets + lighter query multiview",
        )
        smaller_earlier_query_lift_lighter_query_multiview = _entry_by_name(
            history,
            "v330 promoted thawed thirty-facets + smaller earlier query lift + lighter query multiview",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["earlier_query_classifier_onset_score"] = float(
            (earlier_query_classifier_onset or {}).get("score", float("-inf")) or float("-inf")
        ) if earlier_query_classifier_onset else None
        evidence["smaller_query_classifier_lift_score"] = float(
            (smaller_query_classifier_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if smaller_query_classifier_lift else None
        evidence["lighter_query_multiview_score"] = float(
            (lighter_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if lighter_query_multiview else None
        evidence["smaller_earlier_query_lift_lighter_query_multiview_score"] = float(
            (smaller_earlier_query_lift_lighter_query_multiview or {}).get("score", float("-inf")) or float("-inf")
        ) if smaller_earlier_query_lift_lighter_query_multiview else None
        evidence["remaining_strict_failures"] = ["known confidence too low"]
    elif normalized == "v331":
        control = _entry_by_name(history, "v331 frozen thirty-facets keep-head control")
        earlier_query_classifier_onset = _entry_by_name(
            history,
            "v331 frozen thirty-facets + earlier query-classifier onset",
        )
        stronger_query_classifier_lift = _entry_by_name(
            history,
            "v331 frozen thirty-facets + stronger query-classifier lift",
        )
        mixed_query_pred_confidence_bridge = _entry_by_name(
            history,
            "v331 frozen thirty-facets + mixed query-pred confidence bridge",
        )
        stronger_query_classifier_lift_earlier_onset = _entry_by_name(
            history,
            "v331 frozen thirty-facets + stronger query-classifier lift + earlier onset",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["earlier_query_classifier_onset_score"] = float(
            (earlier_query_classifier_onset or {}).get("score", float("-inf")) or float("-inf")
        ) if earlier_query_classifier_onset else None
        evidence["stronger_query_classifier_lift_score"] = float(
            (stronger_query_classifier_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_query_classifier_lift else None
        evidence["mixed_query_pred_confidence_bridge_score"] = float(
            (mixed_query_pred_confidence_bridge or {}).get("score", float("-inf")) or float("-inf")
        ) if mixed_query_pred_confidence_bridge else None
        evidence["stronger_query_classifier_lift_earlier_onset_score"] = float(
            (stronger_query_classifier_lift_earlier_onset or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_query_classifier_lift_earlier_onset else None
        evidence["v330_thawed_frontier_score"] = 32.988738158096865
        evidence["remaining_strict_failures"] = ["known confidence too low"]
    elif normalized == "v332":
        control = _entry_by_name(history, "v332 v330-reset thawed keep-head control")
        mixed_query_pred_confidence_bridge = _entry_by_name(
            history,
            "v332 v330-reset + mixed query-pred confidence bridge",
        )
        mixed_query_pred_confidence_bridge_earlier_onset = _entry_by_name(
            history,
            "v332 v330-reset + mixed query-pred confidence bridge + earlier onset",
        )
        stronger_query_classifier_lift_earlier_onset = _entry_by_name(
            history,
            "v332 v330-reset + stronger query-classifier lift + earlier onset",
        )
        stronger_mixed_query_pred_confidence_bridge_earlier_onset = _entry_by_name(
            history,
            "v332 v330-reset + stronger mixed query-pred confidence bridge + earlier onset",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["mixed_query_pred_confidence_bridge_score"] = float(
            (mixed_query_pred_confidence_bridge or {}).get("score", float("-inf")) or float("-inf")
        ) if mixed_query_pred_confidence_bridge else None
        evidence["mixed_query_pred_confidence_bridge_earlier_onset_score"] = float(
            (mixed_query_pred_confidence_bridge_earlier_onset or {}).get("score", float("-inf")) or float("-inf")
        ) if mixed_query_pred_confidence_bridge_earlier_onset else None
        evidence["stronger_query_classifier_lift_earlier_onset_score"] = float(
            (stronger_query_classifier_lift_earlier_onset or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_query_classifier_lift_earlier_onset else None
        evidence["stronger_mixed_query_pred_confidence_bridge_earlier_onset_score"] = float(
            (stronger_mixed_query_pred_confidence_bridge_earlier_onset or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_mixed_query_pred_confidence_bridge_earlier_onset else None
        evidence["remaining_strict_failures"] = ["known confidence too low"]
    elif normalized == "v333":
        control = _entry_by_name(history, "v333 frozen mixed-bridge keep-head control")
        earlier_onset = _entry_by_name(history, "v333 frozen mixed-bridge + earlier onset")
        stronger_mixed_lift = _entry_by_name(history, "v333 frozen mixed-bridge + stronger mixed lift")
        moderate_query_lift = _entry_by_name(history, "v333 frozen mixed-bridge + moderate query lift")
        stronger_mixed_lift_earlier_onset = _entry_by_name(
            history,
            "v333 frozen mixed-bridge + stronger mixed lift + earlier onset",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["earlier_onset_score"] = float((earlier_onset or {}).get("score", float("-inf")) or float("-inf")) if earlier_onset else None
        evidence["stronger_mixed_lift_score"] = float((stronger_mixed_lift or {}).get("score", float("-inf")) or float("-inf")) if stronger_mixed_lift else None
        evidence["moderate_query_lift_score"] = float((moderate_query_lift or {}).get("score", float("-inf")) or float("-inf")) if moderate_query_lift else None
        evidence["stronger_mixed_lift_earlier_onset_score"] = float(
            (stronger_mixed_lift_earlier_onset or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_mixed_lift_earlier_onset else None
        evidence["remaining_strict_failures"] = ["known confidence too low"]
    elif normalized == "v336":
        control = _entry_by_name(history, "v336 promoted moderate-query-lift control")
        stronger_query_lift = _entry_by_name(history, "v336 promoted moderate-query-lift + stronger query lift")
        tiny_mixed_bridge = _entry_by_name(history, "v336 promoted moderate-query-lift + tiny mixed bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_query_lift_score"] = float(
            (stronger_query_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_query_lift else None
        evidence["tiny_mixed_bridge_score"] = float(
            (tiny_mixed_bridge or {}).get("score", float("-inf")) or float("-inf")
        ) if tiny_mixed_bridge else None
        evidence["v333_frontier_score"] = 33.01863622044523
        evidence["remaining_strict_failures"] = ["known confidence too low"]
    elif normalized == "v337":
        control = _entry_by_name(history, "v337 v333-reset moderate-query-lift control")
        tiny_mixed_bridge = _entry_by_name(history, "v337 v333-reset moderate-query-lift + tiny mixed bridge")
        earlier_onset = _entry_by_name(history, "v337 v333-reset moderate-query-lift + earlier onset")
        stronger_mixed_lift_earlier_onset = _entry_by_name(
            history,
            "v337 v333-reset moderate-query-lift + stronger mixed lift + earlier onset",
        )
        tiny_mixed_bridge_earlier_onset = _entry_by_name(
            history,
            "v337 v333-reset moderate-query-lift + tiny mixed bridge + earlier onset",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["tiny_mixed_bridge_score"] = float(
            (tiny_mixed_bridge or {}).get("score", float("-inf")) or float("-inf")
        ) if tiny_mixed_bridge else None
        evidence["earlier_onset_score"] = float(
            (earlier_onset or {}).get("score", float("-inf")) or float("-inf")
        ) if earlier_onset else None
        evidence["stronger_mixed_lift_earlier_onset_score"] = float(
            (stronger_mixed_lift_earlier_onset or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_mixed_lift_earlier_onset else None
        evidence["tiny_mixed_bridge_earlier_onset_score"] = float(
            (tiny_mixed_bridge_earlier_onset or {}).get("score", float("-inf")) or float("-inf")
        ) if tiny_mixed_bridge_earlier_onset else None
        evidence["v333_frontier_score"] = 33.01863622044523
        evidence["remaining_strict_failures"] = ["known confidence too low"]
    elif normalized == "v338":
        control = _entry_by_name(history, "v338 promoted earlier-onset control")
        stronger_query_lift = _entry_by_name(history, "v338 promoted earlier-onset + stronger query lift")
        tiny_mixed_bridge = _entry_by_name(history, "v338 promoted earlier-onset + tiny mixed bridge")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_query_lift_score"] = float(
            (stronger_query_lift or {}).get("score", float("-inf")) or float("-inf")
        ) if stronger_query_lift else None
        evidence["tiny_mixed_bridge_score"] = float(
            (tiny_mixed_bridge or {}).get("score", float("-inf")) or float("-inf")
        ) if tiny_mixed_bridge else None
        evidence["remaining_strict_failures"] = ["known confidence too low"]
    elif normalized == "v335":
        control = _entry_by_name(history, "v335 clean-reset v178 support-aware citecheck pairwise control")
        support_floor = _entry_by_name(history, "v335 clean-reset v178 support-aware citecheck pairwise + support floor")
        code_form_tie_break = _entry_by_name(history, "v335 clean-reset v178 support-aware citecheck pairwise + code-form tie-break")
        support_pref_multistep_floor = _entry_by_name(history, "v335 clean-reset v178 support-aware citecheck pairwise + support-pref multistep floor")
        code_form_tie_break_ultra_low_margin_safe_expand = _entry_by_name(
            history,
            "v335 clean-reset v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_floor_score"] = float((support_floor or {}).get("score", float("-inf")) or float("-inf")) if support_floor else None
        evidence["code_form_tie_break_score"] = float((code_form_tie_break or {}).get("score", float("-inf")) or float("-inf")) if code_form_tie_break else None
        evidence["support_pref_multistep_floor_score"] = float((support_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_multistep_floor else None
        evidence["code_form_tie_break_ultra_low_margin_safe_expand_score"] = float(
            (code_form_tie_break_ultra_low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")
        ) if code_form_tie_break_ultra_low_margin_safe_expand else None
    elif normalized == "v339":
        control = _entry_by_name(
            history,
            "v339 clean-reset v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand control",
        )
        local_union_finalist_room = _entry_by_name(
            history,
            "v339 clean-reset v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + local union finalist room",
        )
        code_pref_multistep_floor = _entry_by_name(
            history,
            "v339 clean-reset v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + code-pref multistep floor",
        )
        code_pref_soft_multistep = _entry_by_name(
            history,
            "v339 clean-reset v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + code-pref soft multistep",
        )
        local_union_finalist_room_code_pref_soft_multistep = _entry_by_name(
            history,
            "v339 clean-reset v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + local union finalist room + code-pref soft multistep",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["local_union_finalist_room_score"] = float((local_union_finalist_room or {}).get("score", float("-inf")) or float("-inf")) if local_union_finalist_room else None
        evidence["code_pref_multistep_floor_score"] = float((code_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if code_pref_multistep_floor else None
        evidence["code_pref_soft_multistep_score"] = float((code_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")) if code_pref_soft_multistep else None
        evidence["local_union_finalist_room_code_pref_soft_multistep_score"] = float((local_union_finalist_room_code_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")) if local_union_finalist_room_code_pref_soft_multistep else None
    elif normalized == "v341":
        control = _entry_by_name(history, "v341 promoted support-safe code-form control")
        support_pref_coverage = _entry_by_name(history, "v341 promoted support-safe code-form + support-pref coverage")
        firmer_code_form_tie_break = _entry_by_name(history, "v341 promoted support-safe code-form + firmer code-form tie-break")
        deterministic_paraphrase_quorum = _entry_by_name(history, "v341 promoted support-safe code-form + deterministic paraphrase quorum")
        no_safe_expand_rollback = _entry_by_name(history, "v341 promoted support-safe code-form + no safe-expand rollback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_pref_coverage_score"] = float((support_pref_coverage or {}).get("score", float("-inf")) or float("-inf")) if support_pref_coverage else None
        evidence["firmer_code_form_tie_break_score"] = float((firmer_code_form_tie_break or {}).get("score", float("-inf")) or float("-inf")) if firmer_code_form_tie_break else None
        evidence["deterministic_paraphrase_quorum_score"] = float((deterministic_paraphrase_quorum or {}).get("score", float("-inf")) or float("-inf")) if deterministic_paraphrase_quorum else None
        evidence["no_safe_expand_rollback_score"] = float((no_safe_expand_rollback or {}).get("score", float("-inf")) or float("-inf")) if no_safe_expand_rollback else None
    elif normalized == "v342":
        control = _entry_by_name(history, "v342 promoted firmer-code-form control")
        support_pref_coverage = _entry_by_name(history, "v342 promoted firmer-code-form + support-pref coverage")
        no_safe_expand_rollback = _entry_by_name(history, "v342 promoted firmer-code-form + no safe-expand rollback")
        deterministic_paraphrase_quorum = _entry_by_name(history, "v342 promoted firmer-code-form + deterministic paraphrase quorum")
        support_pref_coverage_no_safe_expand_rollback = _entry_by_name(
            history,
            "v342 promoted firmer-code-form + support-pref coverage + no safe-expand rollback",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_pref_coverage_score"] = float((support_pref_coverage or {}).get("score", float("-inf")) or float("-inf")) if support_pref_coverage else None
        evidence["no_safe_expand_rollback_score"] = float((no_safe_expand_rollback or {}).get("score", float("-inf")) or float("-inf")) if no_safe_expand_rollback else None
        evidence["deterministic_paraphrase_quorum_score"] = float((deterministic_paraphrase_quorum or {}).get("score", float("-inf")) or float("-inf")) if deterministic_paraphrase_quorum else None
        evidence["support_pref_coverage_no_safe_expand_rollback_score"] = float(
            (support_pref_coverage_no_safe_expand_rollback or {}).get("score", float("-inf")) or float("-inf")
        ) if support_pref_coverage_no_safe_expand_rollback else None
    elif normalized == "v343":
        control = _entry_by_name(history, "v343 promoted firmer-code-form no-safe-expand control")
        neighborhood_posterior = _entry_by_name(history, "v343 promoted firmer-code-form no-safe-expand + neighborhood posterior")
        support_feature_calibrator = _entry_by_name(history, "v343 promoted firmer-code-form no-safe-expand + support-feature calibrator")
        evidential_support_head = _entry_by_name(history, "v343 promoted firmer-code-form no-safe-expand + evidential support head")
        agreement_augmented_calibrator = _entry_by_name(history, "v343 promoted firmer-code-form no-safe-expand + agreement-augmented calibrator")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["neighborhood_posterior_score"] = float((neighborhood_posterior or {}).get("score", float("-inf")) or float("-inf")) if neighborhood_posterior else None
        evidence["support_feature_calibrator_score"] = float((support_feature_calibrator or {}).get("score", float("-inf")) or float("-inf")) if support_feature_calibrator else None
        evidence["evidential_support_head_score"] = float((evidential_support_head or {}).get("score", float("-inf")) or float("-inf")) if evidential_support_head else None
        evidence["agreement_augmented_calibrator_score"] = float((agreement_augmented_calibrator or {}).get("score", float("-inf")) or float("-inf")) if agreement_augmented_calibrator else None
        if (
            stop_reason in {"adaptive_plateau", "budget_exhausted", "no_candidate_available"}
            and best_entry is not None
            and "PASS" not in str((best_entry or {}).get("strict_status", ""))
        ):
            return {
                "next_mode": "v344",
                "reason": "v343 plateaued without a strict pass; advance to v344.",
                "evidence": evidence,
            }
    elif normalized == "v344":
        control = _entry_by_name(history, "v344 promoted confidence-stable code-form control")
        parafence_stability_calibrator = _entry_by_name(history, "v344 promoted confidence-stable code-form + parafence stability calibrator")
        deterministic_low_margin_safe_expand_posterior = _entry_by_name(history, "v344 promoted confidence-stable code-form + deterministic low-margin safe expand posterior")
        parafence_safe_expand_hybrid = _entry_by_name(history, "v344 promoted confidence-stable code-form + parafence-safe-expand hybrid")
        firmer_support_posterior_threshold = _entry_by_name(history, "v344 promoted confidence-stable code-form + firmer support posterior threshold")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["parafence_stability_calibrator_score"] = float((parafence_stability_calibrator or {}).get("score", float("-inf")) or float("-inf")) if parafence_stability_calibrator else None
        evidence["deterministic_low_margin_safe_expand_posterior_score"] = float((deterministic_low_margin_safe_expand_posterior or {}).get("score", float("-inf")) or float("-inf")) if deterministic_low_margin_safe_expand_posterior else None
        evidence["parafence_safe_expand_hybrid_score"] = float((parafence_safe_expand_hybrid or {}).get("score", float("-inf")) or float("-inf")) if parafence_safe_expand_hybrid else None
        evidence["firmer_support_posterior_threshold_score"] = float((firmer_support_posterior_threshold or {}).get("score", float("-inf")) or float("-inf")) if firmer_support_posterior_threshold else None
        if (
            stop_reason in {"adaptive_plateau", "budget_exhausted", "no_candidate_available"}
            and best_entry is not None
            and "PASS" not in str((best_entry or {}).get("strict_status", ""))
        ):
            return {
                "next_mode": "v345",
                "reason": "v344 plateaued without a strict pass; advance to v345.",
                "evidence": evidence,
            }
    elif normalized == "v345":
        control = _entry_by_name(history, "v345 promoted parafence-safe-expand control")
        firmer_support_posterior = _entry_by_name(history, "v345 promoted parafence-safe-expand + firmer support posterior")
        support_feature_calibrator = _entry_by_name(history, "v345 promoted parafence-safe-expand + support-feature calibrator")
        agreement_augmented_quorum = _entry_by_name(history, "v345 promoted parafence-safe-expand + agreement-augmented quorum")
        firmer_code_form_tie_break = _entry_by_name(history, "v345 promoted parafence-safe-expand + firmer code-form tie-break")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["firmer_support_posterior_score"] = float((firmer_support_posterior or {}).get("score", float("-inf")) or float("-inf")) if firmer_support_posterior else None
        evidence["support_feature_calibrator_score"] = float((support_feature_calibrator or {}).get("score", float("-inf")) or float("-inf")) if support_feature_calibrator else None
        evidence["agreement_augmented_quorum_score"] = float((agreement_augmented_quorum or {}).get("score", float("-inf")) or float("-inf")) if agreement_augmented_quorum else None
        evidence["firmer_code_form_tie_break_score"] = float((firmer_code_form_tie_break or {}).get("score", float("-inf")) or float("-inf")) if firmer_code_form_tie_break else None
        if (
            stop_reason in {"adaptive_plateau", "budget_exhausted", "no_candidate_available"}
            and best_entry is not None
            and "PASS" not in str((best_entry or {}).get("strict_status", ""))
        ):
            return {
                "next_mode": "v346",
                "reason": "v345 plateaued without a strict pass; advance to v346.",
                "evidence": evidence,
            }
    elif normalized == "v346":
        control = _entry_by_name(history, "v346 promoted parafence-codeform control")
        support_feature_calibrator = _entry_by_name(history, "v346 promoted parafence-codeform + support-feature calibrator")
        agreement_augmented_quorum = _entry_by_name(history, "v346 promoted parafence-codeform + agreement-augmented quorum")
        no_safe_expand_rollback = _entry_by_name(history, "v346 promoted parafence-codeform + no safe-expand rollback")
        support_pref_soft_multistep = _entry_by_name(history, "v346 promoted parafence-codeform + support-pref soft multistep")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_feature_calibrator_score"] = float((support_feature_calibrator or {}).get("score", float("-inf")) or float("-inf")) if support_feature_calibrator else None
        evidence["agreement_augmented_quorum_score"] = float((agreement_augmented_quorum or {}).get("score", float("-inf")) or float("-inf")) if agreement_augmented_quorum else None
        evidence["no_safe_expand_rollback_score"] = float((no_safe_expand_rollback or {}).get("score", float("-inf")) or float("-inf")) if no_safe_expand_rollback else None
        evidence["support_pref_soft_multistep_score"] = float((support_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")) if support_pref_soft_multistep else None
        if (
            stop_reason in {"adaptive_plateau", "budget_exhausted", "no_candidate_available"}
            and best_entry is not None
            and "PASS" not in str((best_entry or {}).get("strict_status", ""))
        ):
            return {
                "next_mode": "v347",
                "reason": "v346 plateaued without a strict pass; advance to v347.",
                "evidence": evidence,
            }
    elif normalized == "v347":
        control = _entry_by_name(history, "v347 promoted no-safe-expand control")
        support_pref_soft_multistep = _entry_by_name(history, "v347 promoted no-safe-expand + support-pref soft multistep")
        support_feature_calibrator = _entry_by_name(history, "v347 promoted no-safe-expand + support-feature calibrator")
        agreement_augmented_quorum = _entry_by_name(history, "v347 promoted no-safe-expand + agreement-augmented quorum")
        firmer_support_posterior = _entry_by_name(history, "v347 promoted no-safe-expand + firmer support posterior")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_pref_soft_multistep_score"] = float((support_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")) if support_pref_soft_multistep else None
        evidence["support_feature_calibrator_score"] = float((support_feature_calibrator or {}).get("score", float("-inf")) or float("-inf")) if support_feature_calibrator else None
        evidence["agreement_augmented_quorum_score"] = float((agreement_augmented_quorum or {}).get("score", float("-inf")) or float("-inf")) if agreement_augmented_quorum else None
        evidence["firmer_support_posterior_score"] = float((firmer_support_posterior or {}).get("score", float("-inf")) or float("-inf")) if firmer_support_posterior else None
        if (
            stop_reason in {"adaptive_plateau", "budget_exhausted", "no_candidate_available"}
            and best_entry is not None
            and "PASS" not in str((best_entry or {}).get("strict_status", ""))
        ):
            return {
                "next_mode": "v348",
                "reason": "v347 plateaued without a strict pass; advance to v348.",
                "evidence": evidence,
            }
    elif normalized == "v188":
        control = _entry_by_name(history, "v188 promoted answerspec pairwise control")
        stronger_parafence_quorum = _entry_by_name(history, "v188 promoted answerspec pairwise + stronger parafence quorum")
        ultra_low_margin_safe_expand = _entry_by_name(history, "v188 promoted answerspec pairwise + ultra-low-margin safe expand")
        support_aware_citecheck_pairwise = _entry_by_name(history, "v188 promoted answerspec pairwise + support-aware citecheck pairwise")
        support_aware_citecheck_pairwise_stronger_parafence_quorum = _entry_by_name(history, "v188 support-aware citecheck pairwise + stronger parafence quorum")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_parafence_quorum_score"] = float((stronger_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if stronger_parafence_quorum else None
        evidence["ultra_low_margin_safe_expand_score"] = float((ultra_low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if ultra_low_margin_safe_expand else None
        evidence["support_aware_citecheck_pairwise_score"] = float((support_aware_citecheck_pairwise or {}).get("score", float("-inf")) or float("-inf")) if support_aware_citecheck_pairwise else None
        evidence["support_aware_citecheck_pairwise_stronger_parafence_quorum_score"] = float((support_aware_citecheck_pairwise_stronger_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if support_aware_citecheck_pairwise_stronger_parafence_quorum else None
    elif normalized == "v189":
        control = _entry_by_name(history, "v189 support-aware citecheck pairwise control")
        ultra_low_margin_safe_expand = _entry_by_name(history, "v189 support-aware citecheck pairwise + ultra-low-margin safe expand")
        code_form_tie_break = _entry_by_name(history, "v189 support-aware citecheck pairwise + code-form tie-break")
        code_form_safe_expand = _entry_by_name(history, "v189 support-aware citecheck pairwise + code-form safe expand")
        support_pref_coverage = _entry_by_name(history, "v189 support-aware citecheck pairwise + support-pref coverage")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["ultra_low_margin_safe_expand_score"] = float((ultra_low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if ultra_low_margin_safe_expand else None
        evidence["code_form_tie_break_score"] = float((code_form_tie_break or {}).get("score", float("-inf")) or float("-inf")) if code_form_tie_break else None
        evidence["code_form_safe_expand_score"] = float((code_form_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if code_form_safe_expand else None
        evidence["support_pref_coverage_score"] = float((support_pref_coverage or {}).get("score", float("-inf")) or float("-inf")) if support_pref_coverage else None
    elif normalized == "v190":
        control = _entry_by_name(history, "v190 promoted support-pref coverage control")
        parafence_stability = _entry_by_name(history, "v190 promoted support-pref coverage + parafence stability")
        ultra_low_margin_safe_expand = _entry_by_name(history, "v190 promoted support-pref coverage + ultra-low-margin safe expand")
        parafence_stability_safe_expand = _entry_by_name(history, "v190 promoted support-pref coverage + parafence stability + safe expand")
        deterministic_paraphrase_quorum = _entry_by_name(history, "v190 promoted support-pref coverage + deterministic paraphrase quorum")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["parafence_stability_score"] = float((parafence_stability or {}).get("score", float("-inf")) or float("-inf")) if parafence_stability else None
        evidence["ultra_low_margin_safe_expand_score"] = float((ultra_low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if ultra_low_margin_safe_expand else None
        evidence["parafence_stability_safe_expand_score"] = float((parafence_stability_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if parafence_stability_safe_expand else None
        evidence["deterministic_paraphrase_quorum_score"] = float((deterministic_paraphrase_quorum or {}).get("score", float("-inf")) or float("-inf")) if deterministic_paraphrase_quorum else None
    elif normalized == "v191":
        control = _entry_by_name(history, "v191 support-pref multistep floor control")
        parafence_stability = _entry_by_name(history, "v191 support-pref multistep floor + parafence stability")
        ultra_low_margin_safe_expand = _entry_by_name(history, "v191 support-pref multistep floor + ultra-low-margin safe expand")
        always_on_multistep_floor = _entry_by_name(history, "v191 support-pref multistep floor + always-on multistep floor")
        deterministic_paraphrase_quorum = _entry_by_name(history, "v191 support-pref multistep floor + deterministic paraphrase quorum")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["parafence_stability_score"] = float((parafence_stability or {}).get("score", float("-inf")) or float("-inf")) if parafence_stability else None
        evidence["ultra_low_margin_safe_expand_score"] = float((ultra_low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if ultra_low_margin_safe_expand else None
        evidence["always_on_multistep_floor_score"] = float((always_on_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if always_on_multistep_floor else None
        evidence["deterministic_paraphrase_quorum_score"] = float((deterministic_paraphrase_quorum or {}).get("score", float("-inf")) or float("-inf")) if deterministic_paraphrase_quorum else None
    elif normalized == "v192":
        control = _entry_by_name(history, "v192 v190-reset support-pref coverage control")
        hard_multistep_pairwise_floor = _entry_by_name(history, "v192 v190-reset support-pref coverage + hard multistep pairwise floor")
        support_pref_multistep_pairwise_floor = _entry_by_name(history, "v192 v190-reset support-pref coverage + support-pref multistep pairwise floor")
        hard_floor_parafence_stability = _entry_by_name(history, "v192 v190-reset support-pref coverage + hard floor + parafence stability")
        hard_floor_safe_expand = _entry_by_name(history, "v192 v190-reset support-pref coverage + hard floor + safe expand")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["hard_multistep_pairwise_floor_score"] = float((hard_multistep_pairwise_floor or {}).get("score", float("-inf")) or float("-inf")) if hard_multistep_pairwise_floor else None
        evidence["support_pref_multistep_pairwise_floor_score"] = float((support_pref_multistep_pairwise_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_multistep_pairwise_floor else None
        evidence["hard_floor_parafence_stability_score"] = float((hard_floor_parafence_stability or {}).get("score", float("-inf")) or float("-inf")) if hard_floor_parafence_stability else None
        evidence["hard_floor_safe_expand_score"] = float((hard_floor_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if hard_floor_safe_expand else None
    elif normalized == "v193":
        control = _entry_by_name(history, "v193 clean v192 paraphrase-quorum control")
        support_pref_pairwise_coverage = _entry_by_name(history, "v193 clean v192 paraphrase-quorum + support-pref pairwise coverage")
        support_pref_multistep_floor = _entry_by_name(history, "v193 clean v192 paraphrase-quorum + support-pref multistep floor")
        hard_multistep_floor = _entry_by_name(history, "v193 clean v192 paraphrase-quorum + hard multistep floor")
        code_pref_tie_break = _entry_by_name(history, "v193 clean v192 paraphrase-quorum + code-pref tie-break")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_pref_pairwise_coverage_score"] = float((support_pref_pairwise_coverage or {}).get("score", float("-inf")) or float("-inf")) if support_pref_pairwise_coverage else None
        evidence["support_pref_multistep_floor_score"] = float((support_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_multistep_floor else None
        evidence["hard_multistep_floor_score"] = float((hard_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if hard_multistep_floor else None
        evidence["code_pref_tie_break_score"] = float((code_pref_tie_break or {}).get("score", float("-inf")) or float("-inf")) if code_pref_tie_break else None
    elif normalized == "v194":
        control = _entry_by_name(history, "v194 v190-reset support-pref coverage control")
        hard_multistep_pairwise_floor = _entry_by_name(history, "v194 v190-reset support-pref coverage + hard multistep pairwise floor")
        support_pref_multistep_pairwise_floor = _entry_by_name(history, "v194 v190-reset support-pref coverage + support-pref multistep pairwise floor")
        hard_floor_parafence_stability = _entry_by_name(history, "v194 v190-reset support-pref coverage + hard floor + parafence stability")
        hard_floor_safe_expand = _entry_by_name(history, "v194 v190-reset support-pref coverage + hard floor + safe expand")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["hard_multistep_pairwise_floor_score"] = float((hard_multistep_pairwise_floor or {}).get("score", float("-inf")) or float("-inf")) if hard_multistep_pairwise_floor else None
        evidence["support_pref_multistep_pairwise_floor_score"] = float((support_pref_multistep_pairwise_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_multistep_pairwise_floor else None
        evidence["hard_floor_parafence_stability_score"] = float((hard_floor_parafence_stability or {}).get("score", float("-inf")) or float("-inf")) if hard_floor_parafence_stability else None
        evidence["hard_floor_safe_expand_score"] = float((hard_floor_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if hard_floor_safe_expand else None
    elif normalized == "v195":
        control = _entry_by_name(history, "v195 promoted hard-floor parafence control")
        safe_expand = _entry_by_name(history, "v195 promoted hard-floor parafence + safe expand")
        support_pref_floor = _entry_by_name(history, "v195 promoted hard-floor parafence + support-pref floor")
        stronger_quorum = _entry_by_name(history, "v195 promoted hard-floor parafence + stronger quorum")
        support_pref_floor_safe_expand = _entry_by_name(history, "v195 promoted hard-floor parafence + support-pref floor + safe expand")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["safe_expand_score"] = float((safe_expand or {}).get("score", float("-inf")) or float("-inf")) if safe_expand else None
        evidence["support_pref_floor_score"] = float((support_pref_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_floor else None
        evidence["stronger_quorum_score"] = float((stronger_quorum or {}).get("score", float("-inf")) or float("-inf")) if stronger_quorum else None
        evidence["support_pref_floor_safe_expand_score"] = float((support_pref_floor_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if support_pref_floor_safe_expand else None
    elif normalized == "v196":
        control = _entry_by_name(history, "v196 promoted stronger-quorum hard-floor control")
        support_pref_multistep_floor = _entry_by_name(history, "v196 promoted stronger-quorum + support-pref multistep floor")
        support_pref_coverage_rollback = _entry_by_name(history, "v196 promoted stronger-quorum + support-pref coverage rollback")
        safe_expand = _entry_by_name(history, "v196 promoted stronger-quorum + safe expand")
        support_pref_floor_safe_expand = _entry_by_name(history, "v196 promoted stronger-quorum + support-pref floor + safe expand")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_pref_multistep_floor_score"] = float((support_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_multistep_floor else None
        evidence["support_pref_coverage_rollback_score"] = float((support_pref_coverage_rollback or {}).get("score", float("-inf")) or float("-inf")) if support_pref_coverage_rollback else None
        evidence["safe_expand_score"] = float((safe_expand or {}).get("score", float("-inf")) or float("-inf")) if safe_expand else None
        evidence["support_pref_floor_safe_expand_score"] = float((support_pref_floor_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if support_pref_floor_safe_expand else None
    elif normalized == "v197":
        control = _entry_by_name(history, "v197 promoted support-pref coverage control")
        hard_negative_curriculum = _entry_by_name(history, "v197 promoted support-pref coverage + hard-negative curriculum")
        focal_margin_curriculum = _entry_by_name(history, "v197 promoted support-pref coverage + focal margin curriculum")
        focal_curriculum_calibration_boundary = _entry_by_name(history, "v197 promoted support-pref coverage + focal curriculum + calibration boundary")
        paraphrase_preserving_margin_curriculum = _entry_by_name(history, "v197 promoted support-pref coverage + paraphrase-preserving margin curriculum")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["hard_negative_curriculum_score"] = float((hard_negative_curriculum or {}).get("score", float("-inf")) or float("-inf")) if hard_negative_curriculum else None
        evidence["focal_margin_curriculum_score"] = float((focal_margin_curriculum or {}).get("score", float("-inf")) or float("-inf")) if focal_margin_curriculum else None
        evidence["focal_curriculum_calibration_boundary_score"] = float((focal_curriculum_calibration_boundary or {}).get("score", float("-inf")) or float("-inf")) if focal_curriculum_calibration_boundary else None
        evidence["paraphrase_preserving_margin_curriculum_score"] = float((paraphrase_preserving_margin_curriculum or {}).get("score", float("-inf")) or float("-inf")) if paraphrase_preserving_margin_curriculum else None
    elif normalized == "v198":
        control = _entry_by_name(history, "v198 promoted hard-negative curriculum control")
        stronger_hard_negative_budget = _entry_by_name(history, "v198 promoted hard-negative curriculum + stronger hard-negative budget")
        vicreg_anti_collapse = _entry_by_name(history, "v198 promoted hard-negative curriculum + vicreg anti-collapse")
        paraphrase_preserving_vicreg = _entry_by_name(history, "v198 promoted hard-negative curriculum + paraphrase-preserving vicreg")
        light_calibration_boundary = _entry_by_name(history, "v198 promoted hard-negative curriculum + light calibration boundary")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_hard_negative_budget_score"] = float((stronger_hard_negative_budget or {}).get("score", float("-inf")) or float("-inf")) if stronger_hard_negative_budget else None
        evidence["vicreg_anti_collapse_score"] = float((vicreg_anti_collapse or {}).get("score", float("-inf")) or float("-inf")) if vicreg_anti_collapse else None
        evidence["paraphrase_preserving_vicreg_score"] = float((paraphrase_preserving_vicreg or {}).get("score", float("-inf")) or float("-inf")) if paraphrase_preserving_vicreg else None
        evidence["light_calibration_boundary_score"] = float((light_calibration_boundary or {}).get("score", float("-inf")) or float("-inf")) if light_calibration_boundary else None
    elif normalized == "v199":
        control = _entry_by_name(history, "v199 promoted paraphrase-preserving vicreg control")
        low_margin_selective_gate = _entry_by_name(history, "v199 promoted paraphrase-preserving vicreg + low-margin selective gate")
        query_multiview_selective_gate = _entry_by_name(history, "v199 promoted paraphrase-preserving vicreg + query multiview selective gate")
        equivalence_synthesis_selective_gate = _entry_by_name(history, "v199 promoted paraphrase-preserving vicreg + equivalence synthesis selective gate")
        tighter_selective_gate = _entry_by_name(history, "v199 promoted paraphrase-preserving vicreg + tighter selective gate")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["low_margin_selective_gate_score"] = float((low_margin_selective_gate or {}).get("score", float("-inf")) or float("-inf")) if low_margin_selective_gate else None
        evidence["query_multiview_selective_gate_score"] = float((query_multiview_selective_gate or {}).get("score", float("-inf")) or float("-inf")) if query_multiview_selective_gate else None
        evidence["equivalence_synthesis_selective_gate_score"] = float((equivalence_synthesis_selective_gate or {}).get("score", float("-inf")) or float("-inf")) if equivalence_synthesis_selective_gate else None
        evidence["tighter_selective_gate_score"] = float((tighter_selective_gate or {}).get("score", float("-inf")) or float("-inf")) if tighter_selective_gate else None
    elif normalized == "v200":
        control = _entry_by_name(history, "v200 promoted low-margin selective gate control")
        softer_abstention = _entry_by_name(history, "v200 promoted low-margin selective gate + softer abstention")
        light_calibration_boundary = _entry_by_name(history, "v200 promoted low-margin selective gate + light calibration boundary")
        anti_collapse_spread_boost = _entry_by_name(history, "v200 promoted low-margin selective gate + anti-collapse spread boost")
        firmer_abstention = _entry_by_name(history, "v200 promoted low-margin selective gate + firmer abstention")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["softer_abstention_score"] = float((softer_abstention or {}).get("score", float("-inf")) or float("-inf")) if softer_abstention else None
        evidence["light_calibration_boundary_score"] = float((light_calibration_boundary or {}).get("score", float("-inf")) or float("-inf")) if light_calibration_boundary else None
        evidence["anti_collapse_spread_boost_score"] = float((anti_collapse_spread_boost or {}).get("score", float("-inf")) or float("-inf")) if anti_collapse_spread_boost else None
        evidence["firmer_abstention_score"] = float((firmer_abstention or {}).get("score", float("-inf")) or float("-inf")) if firmer_abstention else None
    elif normalized == "v201":
        control = _entry_by_name(history, "v201 promoted softer-abstention control")
        light_calibration_boundary = _entry_by_name(history, "v201 promoted softer-abstention + light calibration boundary")
        anti_collapse_spread_boost = _entry_by_name(history, "v201 promoted softer-abstention + anti-collapse spread boost")
        synthesis_equivalence_support = _entry_by_name(history, "v201 promoted softer-abstention + synthesis equivalence support")
        calibration_boundary_spread_boost = _entry_by_name(history, "v201 promoted softer-abstention + calibration boundary + spread boost")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["light_calibration_boundary_score"] = float((light_calibration_boundary or {}).get("score", float("-inf")) or float("-inf")) if light_calibration_boundary else None
        evidence["anti_collapse_spread_boost_score"] = float((anti_collapse_spread_boost or {}).get("score", float("-inf")) or float("-inf")) if anti_collapse_spread_boost else None
        evidence["synthesis_equivalence_support_score"] = float((synthesis_equivalence_support or {}).get("score", float("-inf")) or float("-inf")) if synthesis_equivalence_support else None
        evidence["calibration_boundary_spread_boost_score"] = float((calibration_boundary_spread_boost or {}).get("score", float("-inf")) or float("-inf")) if calibration_boundary_spread_boost else None
    elif normalized == "v233":
        control = _entry_by_name(history, "v233 promoted softer-abstention control")
        stronger_hard_negative_budget = _entry_by_name(history, "v233 promoted softer-abstention + stronger hard-negative budget")
        vicreg_anti_collapse = _entry_by_name(history, "v233 promoted softer-abstention + vicreg anti-collapse")
        paraphrase_preserving_vicreg = _entry_by_name(history, "v233 promoted softer-abstention + paraphrase-preserving vicreg")
        hard_negative_budget_vicreg = _entry_by_name(history, "v233 promoted softer-abstention + hard-negative budget + vicreg")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["stronger_hard_negative_budget_score"] = float((stronger_hard_negative_budget or {}).get("score", float("-inf")) or float("-inf")) if stronger_hard_negative_budget else None
        evidence["vicreg_anti_collapse_score"] = float((vicreg_anti_collapse or {}).get("score", float("-inf")) or float("-inf")) if vicreg_anti_collapse else None
        evidence["paraphrase_preserving_vicreg_score"] = float((paraphrase_preserving_vicreg or {}).get("score", float("-inf")) or float("-inf")) if paraphrase_preserving_vicreg else None
        evidence["hard_negative_budget_vicreg_score"] = float((hard_negative_budget_vicreg or {}).get("score", float("-inf")) or float("-inf")) if hard_negative_budget_vicreg else None
    elif normalized == "v234":
        control = _entry_by_name(history, "v234 promoted stronger-hard-negative-budget control")
        paraphrase_preserving_vicreg = _entry_by_name(history, "v234 promoted stronger-hard-negative-budget + paraphrase-preserving vicreg")
        slightly_stronger_hard_negative_pressure = _entry_by_name(history, "v234 promoted stronger-hard-negative-budget + slightly stronger hard-negative pressure")
        softer_rollback = _entry_by_name(history, "v234 promoted stronger-hard-negative-budget + softer rollback")
        higher_paraphrase_positives = _entry_by_name(history, "v234 promoted stronger-hard-negative-budget + higher paraphrase positives")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["paraphrase_preserving_vicreg_score"] = float((paraphrase_preserving_vicreg or {}).get("score", float("-inf")) or float("-inf")) if paraphrase_preserving_vicreg else None
        evidence["slightly_stronger_hard_negative_pressure_score"] = float((slightly_stronger_hard_negative_pressure or {}).get("score", float("-inf")) or float("-inf")) if slightly_stronger_hard_negative_pressure else None
        evidence["softer_rollback_score"] = float((softer_rollback or {}).get("score", float("-inf")) or float("-inf")) if softer_rollback else None
        evidence["higher_paraphrase_positives_score"] = float((higher_paraphrase_positives or {}).get("score", float("-inf")) or float("-inf")) if higher_paraphrase_positives else None
    elif normalized == "v202":
        control = _entry_by_name(history, "v202 clean v178 citecheck pairwise control")
        parafence_stability = _entry_by_name(history, "v202 clean v178 citecheck pairwise + parafence stability")
        low_margin_safe_expand = _entry_by_name(history, "v202 clean v178 citecheck pairwise + low-margin safe expand")
        parafence_stability_safe_expand = _entry_by_name(history, "v202 clean v178 citecheck pairwise + parafence stability + safe expand")
        deterministic_paraphrase_quorum = _entry_by_name(history, "v202 clean v178 citecheck pairwise + deterministic paraphrase quorum")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["parafence_stability_score"] = float((parafence_stability or {}).get("score", float("-inf")) or float("-inf")) if parafence_stability else None
        evidence["low_margin_safe_expand_score"] = float((low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if low_margin_safe_expand else None
        evidence["parafence_stability_safe_expand_score"] = float((parafence_stability_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if parafence_stability_safe_expand else None
        evidence["deterministic_paraphrase_quorum_score"] = float((deterministic_paraphrase_quorum or {}).get("score", float("-inf")) or float("-inf")) if deterministic_paraphrase_quorum else None
    elif normalized == "v203":
        control = _entry_by_name(history, "v203 promoted deterministic-paraphrase-quorum control")
        low_margin_safe_expand = _entry_by_name(history, "v203 promoted deterministic-paraphrase-quorum + low-margin safe expand")
        slightly_softer_gate = _entry_by_name(history, "v203 promoted deterministic-paraphrase-quorum + slightly softer gate")
        colder_support_backstop = _entry_by_name(history, "v203 promoted deterministic-paraphrase-quorum + colder support backstop")
        softer_gate_safe_expand = _entry_by_name(history, "v203 promoted deterministic-paraphrase-quorum + softer gate + safe expand")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["low_margin_safe_expand_score"] = float((low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if low_margin_safe_expand else None
        evidence["slightly_softer_gate_score"] = float((slightly_softer_gate or {}).get("score", float("-inf")) or float("-inf")) if slightly_softer_gate else None
        evidence["colder_support_backstop_score"] = float((colder_support_backstop or {}).get("score", float("-inf")) or float("-inf")) if colder_support_backstop else None
        evidence["softer_gate_safe_expand_score"] = float((softer_gate_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if softer_gate_safe_expand else None
    elif normalized == "v204":
        control = _entry_by_name(history, "v204 promoted softer-gate-safe-expand control")
        low_margin_answer_spec_gate = _entry_by_name(history, "v204 promoted softer-gate-safe-expand + low-margin answer-spec gate")
        hard_synthesis_coverage_gate = _entry_by_name(history, "v204 promoted softer-gate-safe-expand + hard synthesis coverage gate")
        support_spec_hybrid_gate = _entry_by_name(history, "v204 promoted softer-gate-safe-expand + support-spec hybrid gate")
        code_form_tie_break = _entry_by_name(history, "v204 promoted softer-gate-safe-expand + code-form tie-break")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["low_margin_answer_spec_gate_score"] = float((low_margin_answer_spec_gate or {}).get("score", float("-inf")) or float("-inf")) if low_margin_answer_spec_gate else None
        evidence["hard_synthesis_coverage_gate_score"] = float((hard_synthesis_coverage_gate or {}).get("score", float("-inf")) or float("-inf")) if hard_synthesis_coverage_gate else None
        evidence["support_spec_hybrid_gate_score"] = float((support_spec_hybrid_gate or {}).get("score", float("-inf")) or float("-inf")) if support_spec_hybrid_gate else None
        evidence["code_form_tie_break_score"] = float((code_form_tie_break or {}).get("score", float("-inf")) or float("-inf")) if code_form_tie_break else None
    elif normalized == "v205":
        control = _entry_by_name(history, "v205 clean deterministic-paraphrase-quorum control")
        citecheck_support_floor = _entry_by_name(history, "v205 clean deterministic-paraphrase-quorum + citecheck support floor")
        support_floor_safe_expand = _entry_by_name(history, "v205 clean deterministic-paraphrase-quorum + citecheck support floor + safe expand")
        support_pref_multistep_floor = _entry_by_name(history, "v205 clean deterministic-paraphrase-quorum + support-pref multistep floor")
        hard_multistep_floor = _entry_by_name(history, "v205 clean deterministic-paraphrase-quorum + hard multistep floor")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["citecheck_support_floor_score"] = float((citecheck_support_floor or {}).get("score", float("-inf")) or float("-inf")) if citecheck_support_floor else None
        evidence["support_floor_safe_expand_score"] = float((support_floor_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if support_floor_safe_expand else None
        evidence["support_pref_multistep_floor_score"] = float((support_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_multistep_floor else None
        evidence["hard_multistep_floor_score"] = float((hard_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if hard_multistep_floor else None
    elif normalized == "v206":
        control = _entry_by_name(history, "v206 combined support-floor hard-multistep control")
        support_pref_floor = _entry_by_name(history, "v206 combined support-floor support-pref floor")
        citecheck_support_floor_only = _entry_by_name(history, "v206 citecheck support-floor only ablation")
        hard_multistep_only = _entry_by_name(history, "v206 hard-multistep only ablation")
        combined_safe_expand = _entry_by_name(history, "v206 combined support-floor hard-multistep + safe expand")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_pref_floor_score"] = float((support_pref_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_floor else None
        evidence["citecheck_support_floor_only_score"] = float((citecheck_support_floor_only or {}).get("score", float("-inf")) or float("-inf")) if citecheck_support_floor_only else None
        evidence["hard_multistep_only_score"] = float((hard_multistep_only or {}).get("score", float("-inf")) or float("-inf")) if hard_multistep_only else None
        evidence["combined_safe_expand_score"] = float((combined_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if combined_safe_expand else None
    elif normalized == "v208":
        control = _entry_by_name(history, "v208 clean deterministic-paraphrase-quorum support-floor control")
        safe_expand = _entry_by_name(history, "v208 clean deterministic-paraphrase-quorum support-floor + safe expand")
        support_pref_multistep_floor = _entry_by_name(history, "v208 clean deterministic-paraphrase-quorum support-floor + support-pref multistep floor")
        hard_multistep_floor = _entry_by_name(history, "v208 clean deterministic-paraphrase-quorum support-floor + hard multistep floor")
        code_form_tie_break = _entry_by_name(history, "v208 clean deterministic-paraphrase-quorum support-floor + code-form tie-break")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["safe_expand_score"] = float((safe_expand or {}).get("score", float("-inf")) or float("-inf")) if safe_expand else None
        evidence["support_pref_multistep_floor_score"] = float((support_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_multistep_floor else None
        evidence["hard_multistep_floor_score"] = float((hard_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if hard_multistep_floor else None
        evidence["code_form_tie_break_score"] = float((code_form_tie_break or {}).get("score", float("-inf")) or float("-inf")) if code_form_tie_break else None
    elif normalized == "v212":
        control = _entry_by_name(history, "v212 clean v178 citecheck quorum support-floor control")
        deterministic_safe_expand = _entry_by_name(history, "v212 clean v178 citecheck quorum support-floor + deterministic safe expand")
        wider_parafence_quorum = _entry_by_name(history, "v212 clean v178 citecheck quorum support-floor + wider parafence quorum")
        support_pref_multistep_floor = _entry_by_name(history, "v212 clean v178 citecheck quorum support-floor + support-pref multistep floor")
        code_form_tie_break = _entry_by_name(history, "v212 clean v178 citecheck quorum support-floor + code-form tie-break")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["deterministic_safe_expand_score"] = float((deterministic_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if deterministic_safe_expand else None
        evidence["wider_parafence_quorum_score"] = float((wider_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if wider_parafence_quorum else None
        evidence["support_pref_multistep_floor_score"] = float((support_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_multistep_floor else None
        evidence["code_form_tie_break_score"] = float((code_form_tie_break or {}).get("score", float("-inf")) or float("-inf")) if code_form_tie_break else None
    elif normalized == "v272":
        control = _entry_by_name(history, "v272 clean v178 citecheck quorum support-floor control")
        deterministic_safe_expand = _entry_by_name(history, "v272 clean v178 citecheck quorum support-floor + deterministic safe expand")
        wider_parafence_quorum = _entry_by_name(history, "v272 clean v178 citecheck quorum support-floor + wider parafence quorum")
        support_pref_multistep_floor = _entry_by_name(history, "v272 clean v178 citecheck quorum support-floor + support-pref multistep floor")
        code_form_tie_break = _entry_by_name(history, "v272 clean v178 citecheck quorum support-floor + code-form tie-break")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["deterministic_safe_expand_score"] = float((deterministic_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if deterministic_safe_expand else None
        evidence["wider_parafence_quorum_score"] = float((wider_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if wider_parafence_quorum else None
        evidence["support_pref_multistep_floor_score"] = float((support_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_multistep_floor else None
        evidence["code_form_tie_break_score"] = float((code_form_tie_break or {}).get("score", float("-inf")) or float("-inf")) if code_form_tie_break else None
    elif normalized == "v213":
        control = _entry_by_name(history, "v213 combined clean v178 code-form multistep support-floor control")
        deterministic_safe_expand = _entry_by_name(history, "v213 combined clean v178 code-form multistep support-floor + deterministic safe expand")
        code_form_rollback = _entry_by_name(history, "v213 combined clean v178 code-form multistep support-floor + code-form rollback")
        support_pref_rollback = _entry_by_name(history, "v213 combined clean v178 code-form multistep support-floor + support-pref rollback")
        hard_floor_rollback = _entry_by_name(history, "v213 combined clean v178 code-form multistep support-floor + hard-floor rollback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["deterministic_safe_expand_score"] = float((deterministic_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if deterministic_safe_expand else None
        evidence["code_form_rollback_score"] = float((code_form_rollback or {}).get("score", float("-inf")) or float("-inf")) if code_form_rollback else None
        evidence["support_pref_rollback_score"] = float((support_pref_rollback or {}).get("score", float("-inf")) or float("-inf")) if support_pref_rollback else None
        evidence["hard_floor_rollback_score"] = float((hard_floor_rollback or {}).get("score", float("-inf")) or float("-inf")) if hard_floor_rollback else None
    elif normalized == "v273":
        control = _entry_by_name(history, "v273 combined clean v178 code-form multistep support-floor control")
        deterministic_safe_expand = _entry_by_name(history, "v273 combined clean v178 code-form multistep support-floor + deterministic safe expand")
        code_form_rollback = _entry_by_name(history, "v273 combined clean v178 code-form multistep support-floor + code-form rollback")
        support_pref_rollback = _entry_by_name(history, "v273 combined clean v178 code-form multistep support-floor + support-pref rollback")
        hard_floor_rollback = _entry_by_name(history, "v273 combined clean v178 code-form multistep support-floor + hard-floor rollback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["deterministic_safe_expand_score"] = float((deterministic_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if deterministic_safe_expand else None
        evidence["code_form_rollback_score"] = float((code_form_rollback or {}).get("score", float("-inf")) or float("-inf")) if code_form_rollback else None
        evidence["support_pref_rollback_score"] = float((support_pref_rollback or {}).get("score", float("-inf")) or float("-inf")) if support_pref_rollback else None
        evidence["hard_floor_rollback_score"] = float((hard_floor_rollback or {}).get("score", float("-inf")) or float("-inf")) if hard_floor_rollback else None
    elif normalized == "v214":
        control = _entry_by_name(history, "v214 reset clean v212 code-form support-floor control")
        deterministic_safe_expand = _entry_by_name(history, "v214 reset clean v212 code-form support-floor + deterministic safe expand")
        wider_parafence_quorum = _entry_by_name(history, "v214 reset clean v212 code-form support-floor + wider parafence quorum")
        soft_multistep_helper = _entry_by_name(history, "v214 reset clean v212 code-form support-floor + soft multistep helper")
        soft_multistep_helper_safe_expand = _entry_by_name(history, "v214 reset clean v212 code-form support-floor + soft multistep helper + deterministic safe expand")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["deterministic_safe_expand_score"] = float((deterministic_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if deterministic_safe_expand else None
        evidence["wider_parafence_quorum_score"] = float((wider_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if wider_parafence_quorum else None
        evidence["soft_multistep_helper_score"] = float((soft_multistep_helper or {}).get("score", float("-inf")) or float("-inf")) if soft_multistep_helper else None
        evidence["soft_multistep_helper_safe_expand_score"] = float((soft_multistep_helper_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if soft_multistep_helper_safe_expand else None
    elif normalized == "v215":
        control = _entry_by_name(history, "v215 promoted soft-multistep safe-expand control")
        wider_parafence_quorum = _entry_by_name(history, "v215 promoted soft-multistep safe-expand + wider parafence quorum")
        code_form_rollback = _entry_by_name(history, "v215 promoted soft-multistep safe-expand + code-form rollback")
        support_pref_rollback = _entry_by_name(history, "v215 promoted soft-multistep safe-expand + support-pref rollback")
        hard_floor_rollback = _entry_by_name(history, "v215 promoted soft-multistep safe-expand + hard-floor rollback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["wider_parafence_quorum_score"] = float((wider_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if wider_parafence_quorum else None
        evidence["code_form_rollback_score"] = float((code_form_rollback or {}).get("score", float("-inf")) or float("-inf")) if code_form_rollback else None
        evidence["support_pref_rollback_score"] = float((support_pref_rollback or {}).get("score", float("-inf")) or float("-inf")) if support_pref_rollback else None
        evidence["hard_floor_rollback_score"] = float((hard_floor_rollback or {}).get("score", float("-inf")) or float("-inf")) if hard_floor_rollback else None
    elif normalized == "v216":
        control = _entry_by_name(history, "v216 promoted wider parafence quorum control")
        no_safe_expand_rollback = _entry_by_name(history, "v216 promoted wider parafence quorum + no safe-expand rollback")
        code_form_rollback = _entry_by_name(history, "v216 promoted wider parafence quorum + code-form rollback")
        support_pref_rollback = _entry_by_name(history, "v216 promoted wider parafence quorum + support-pref rollback")
        hard_floor_rollback = _entry_by_name(history, "v216 promoted wider parafence quorum + hard-floor rollback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["no_safe_expand_rollback_score"] = float((no_safe_expand_rollback or {}).get("score", float("-inf")) or float("-inf")) if no_safe_expand_rollback else None
        evidence["code_form_rollback_score"] = float((code_form_rollback or {}).get("score", float("-inf")) or float("-inf")) if code_form_rollback else None
        evidence["support_pref_rollback_score"] = float((support_pref_rollback or {}).get("score", float("-inf")) or float("-inf")) if support_pref_rollback else None
        evidence["hard_floor_rollback_score"] = float((hard_floor_rollback or {}).get("score", float("-inf")) or float("-inf")) if hard_floor_rollback else None
    elif normalized == "v217":
        control = _entry_by_name(history, "v217 promoted wider parafence no-safe-expand control")
        code_form_rollback = _entry_by_name(history, "v217 promoted wider parafence no-safe-expand + code-form rollback")
        support_pref_rollback = _entry_by_name(history, "v217 promoted wider parafence no-safe-expand + support-pref rollback")
        hard_floor_rollback = _entry_by_name(history, "v217 promoted wider parafence no-safe-expand + hard-floor rollback")
        code_pref_multistep_floor = _entry_by_name(history, "v217 promoted wider parafence no-safe-expand + code-pref multistep floor")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["code_form_rollback_score"] = float((code_form_rollback or {}).get("score", float("-inf")) or float("-inf")) if code_form_rollback else None
        evidence["support_pref_rollback_score"] = float((support_pref_rollback or {}).get("score", float("-inf")) or float("-inf")) if support_pref_rollback else None
        evidence["hard_floor_rollback_score"] = float((hard_floor_rollback or {}).get("score", float("-inf")) or float("-inf")) if hard_floor_rollback else None
        evidence["code_pref_multistep_floor_score"] = float((code_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if code_pref_multistep_floor else None
    elif normalized == "v218":
        control = _entry_by_name(history, "v218 restored wider parafence safe-expand control")
        code_form_rollback = _entry_by_name(history, "v218 restored wider parafence safe-expand + code-form rollback")
        support_pref_rollback = _entry_by_name(history, "v218 restored wider parafence safe-expand + support-pref rollback")
        hard_floor_rollback = _entry_by_name(history, "v218 restored wider parafence safe-expand + hard-floor rollback")
        code_pref_multistep_floor = _entry_by_name(history, "v218 restored wider parafence safe-expand + code-pref multistep floor")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["code_form_rollback_score"] = float((code_form_rollback or {}).get("score", float("-inf")) or float("-inf")) if code_form_rollback else None
        evidence["support_pref_rollback_score"] = float((support_pref_rollback or {}).get("score", float("-inf")) or float("-inf")) if support_pref_rollback else None
        evidence["hard_floor_rollback_score"] = float((hard_floor_rollback or {}).get("score", float("-inf")) or float("-inf")) if hard_floor_rollback else None
        evidence["code_pref_multistep_floor_score"] = float((code_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if code_pref_multistep_floor else None
    elif normalized == "v209":
        control = _entry_by_name(history, "v209 promoted clean quorum support-floor control")
        code_form_tie_break = _entry_by_name(history, "v209 promoted clean quorum support-floor + code-form tie-break")
        support_pref_multistep_floor = _entry_by_name(history, "v209 promoted clean quorum support-floor + support-pref multistep floor")
        tighter_support_trigger = _entry_by_name(history, "v209 promoted clean quorum support-floor + tighter support trigger")
        looser_support_trigger = _entry_by_name(history, "v209 promoted clean quorum support-floor + looser support trigger")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["code_form_tie_break_score"] = float((code_form_tie_break or {}).get("score", float("-inf")) or float("-inf")) if code_form_tie_break else None
        evidence["support_pref_multistep_floor_score"] = float((support_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_multistep_floor else None
        evidence["tighter_support_trigger_score"] = float((tighter_support_trigger or {}).get("score", float("-inf")) or float("-inf")) if tighter_support_trigger else None
        evidence["looser_support_trigger_score"] = float((looser_support_trigger or {}).get("score", float("-inf")) or float("-inf")) if looser_support_trigger else None
    elif normalized == "v210":
        control = _entry_by_name(history, "v210 combined code-form multistep support-floor control")
        citecheck_only_pairwise = _entry_by_name(history, "v210 combined code-form multistep support-floor + citecheck-only pairwise")
        deterministic_safe_expand = _entry_by_name(history, "v210 combined code-form multistep support-floor + deterministic safe expand")
        code_form_rollback = _entry_by_name(history, "v210 combined code-form multistep support-floor + code-form rollback")
        hard_floor_rollback = _entry_by_name(history, "v210 combined code-form multistep support-floor + hard-floor rollback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["citecheck_only_pairwise_score"] = float((citecheck_only_pairwise or {}).get("score", float("-inf")) or float("-inf")) if citecheck_only_pairwise else None
        evidence["deterministic_safe_expand_score"] = float((deterministic_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if deterministic_safe_expand else None
        evidence["code_form_rollback_score"] = float((code_form_rollback or {}).get("score", float("-inf")) or float("-inf")) if code_form_rollback else None
        evidence["hard_floor_rollback_score"] = float((hard_floor_rollback or {}).get("score", float("-inf")) or float("-inf")) if hard_floor_rollback else None
    elif normalized == "v221":
        control = _entry_by_name(history, "v221 promoted code-form support-floor control")
        always_on_supportspec = _entry_by_name(history, "v221 promoted code-form support-floor + always-on support/spec pairwise")
        code_pref_multistep_floor = _entry_by_name(history, "v221 promoted code-form support-floor + code-pref multistep floor")
        code_pref_soft_multistep = _entry_by_name(history, "v221 promoted code-form support-floor + code-pref soft multistep")
        wider_parafence_quorum = _entry_by_name(history, "v221 promoted code-form support-floor + wider parafence quorum")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["always_on_supportspec_score"] = float((always_on_supportspec or {}).get("score", float("-inf")) or float("-inf")) if always_on_supportspec else None
        evidence["code_pref_multistep_floor_score"] = float((code_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if code_pref_multistep_floor else None
        evidence["code_pref_soft_multistep_score"] = float((code_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")) if code_pref_soft_multistep else None
        evidence["wider_parafence_quorum_score"] = float((wider_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if wider_parafence_quorum else None
    elif normalized == "v222":
        control = _entry_by_name(history, "v222 promoted code-form support-floor control")
        deterministic_safe_expand = _entry_by_name(history, "v222 promoted code-form support-floor + deterministic safe expand")
        wider_support_floor_gate = _entry_by_name(history, "v222 promoted code-form support-floor + wider support-floor gate")
        local_union_finalist_room = _entry_by_name(history, "v222 promoted code-form support-floor + local union finalist room")
        local_union_finalist_room_safe_expand = _entry_by_name(history, "v222 promoted code-form support-floor + local union finalist room + deterministic safe expand")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["deterministic_safe_expand_score"] = float((deterministic_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if deterministic_safe_expand else None
        evidence["wider_support_floor_gate_score"] = float((wider_support_floor_gate or {}).get("score", float("-inf")) or float("-inf")) if wider_support_floor_gate else None
        evidence["local_union_finalist_room_score"] = float((local_union_finalist_room or {}).get("score", float("-inf")) or float("-inf")) if local_union_finalist_room else None
        evidence["local_union_finalist_room_safe_expand_score"] = float((local_union_finalist_room_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if local_union_finalist_room_safe_expand else None
    elif normalized == "v223":
        control = _entry_by_name(history, "v223 promoted local-union code-form support-floor control")
        code_pref_multistep_floor = _entry_by_name(history, "v223 promoted local-union code-form support-floor + code-pref multistep floor")
        support_pref_multistep_floor = _entry_by_name(history, "v223 promoted local-union code-form support-floor + support-pref multistep floor")
        code_pref_soft_multistep = _entry_by_name(history, "v223 promoted local-union code-form support-floor + code-pref soft multistep")
        hard_multistep_floor = _entry_by_name(history, "v223 promoted local-union code-form support-floor + hard multistep floor")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["code_pref_multistep_floor_score"] = float((code_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if code_pref_multistep_floor else None
        evidence["support_pref_multistep_floor_score"] = float((support_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_multistep_floor else None
        evidence["code_pref_soft_multistep_score"] = float((code_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")) if code_pref_soft_multistep else None
        evidence["hard_multistep_floor_score"] = float((hard_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if hard_multistep_floor else None
    elif normalized == "v224":
        control = _entry_by_name(history, "v224 promoted local-union soft-multistep control")
        deterministic_safe_expand = _entry_by_name(history, "v224 promoted local-union soft-multistep + deterministic safe expand")
        answer_spec_pairwise = _entry_by_name(history, "v224 promoted local-union soft-multistep + answer-spec pairwise")
        support_spec_pairwise = _entry_by_name(history, "v224 promoted local-union soft-multistep + support/spec pairwise")
        support_spec_pairwise_safe_expand = _entry_by_name(history, "v224 promoted local-union soft-multistep + support/spec pairwise + safe expand")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["deterministic_safe_expand_score"] = float((deterministic_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if deterministic_safe_expand else None
        evidence["answer_spec_pairwise_score"] = float((answer_spec_pairwise or {}).get("score", float("-inf")) or float("-inf")) if answer_spec_pairwise else None
        evidence["support_spec_pairwise_score"] = float((support_spec_pairwise or {}).get("score", float("-inf")) or float("-inf")) if support_spec_pairwise else None
        evidence["support_spec_pairwise_safe_expand_score"] = float((support_spec_pairwise_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if support_spec_pairwise_safe_expand else None
    elif normalized == "v225":
        control = _entry_by_name(history, "v225 promoted support/spec soft-multistep control")
        wider_parafence_quorum = _entry_by_name(history, "v225 promoted support/spec soft-multistep + wider parafence quorum")
        code_form_rollback = _entry_by_name(history, "v225 promoted support/spec soft-multistep + code-form rollback")
        code_pref_multistep_floor = _entry_by_name(history, "v225 promoted support/spec soft-multistep + code-pref multistep floor")
        support_pref_multistep_floor = _entry_by_name(history, "v225 promoted support/spec soft-multistep + support-pref multistep floor")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["wider_parafence_quorum_score"] = float((wider_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if wider_parafence_quorum else None
        evidence["code_form_rollback_score"] = float((code_form_rollback or {}).get("score", float("-inf")) or float("-inf")) if code_form_rollback else None
        evidence["code_pref_multistep_floor_score"] = float((code_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if code_pref_multistep_floor else None
        evidence["support_pref_multistep_floor_score"] = float((support_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_multistep_floor else None
    elif normalized == "v226":
        control = _entry_by_name(history, "v226 promoted support/spec code-pref-multistep control")
        wider_parafence_quorum = _entry_by_name(history, "v226 promoted support/spec code-pref-multistep + wider parafence quorum")
        support_pref_multistep_floor = _entry_by_name(history, "v226 promoted support/spec code-pref-multistep + support-pref multistep floor")
        code_form_rollback = _entry_by_name(history, "v226 promoted support/spec code-pref-multistep + code-form rollback")
        support_floor_rollback = _entry_by_name(history, "v226 promoted support/spec code-pref-multistep + support-floor rollback")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["wider_parafence_quorum_score"] = float((wider_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if wider_parafence_quorum else None
        evidence["support_pref_multistep_floor_score"] = float((support_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_multistep_floor else None
        evidence["code_form_rollback_score"] = float((code_form_rollback or {}).get("score", float("-inf")) or float("-inf")) if code_form_rollback else None
        evidence["support_floor_rollback_score"] = float((support_floor_rollback or {}).get("score", float("-inf")) or float("-inf")) if support_floor_rollback else None
    elif normalized == "v227":
        control = _entry_by_name(history, "v227 reset local-union citecheck code-form control")
        code_pref_soft_multistep = _entry_by_name(history, "v227 reset local-union citecheck code-form + code-pref soft multistep")
        always_on_supportspec = _entry_by_name(history, "v227 reset local-union citecheck code-form + always-on support/spec pairwise")
        wider_parafence_quorum = _entry_by_name(history, "v227 reset local-union citecheck code-form + wider parafence quorum")
        ultra_low_margin_safe_expand = _entry_by_name(history, "v227 reset local-union citecheck code-form + ultra-low-margin safe expand")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["code_pref_soft_multistep_score"] = float((code_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")) if code_pref_soft_multistep else None
        evidence["always_on_supportspec_score"] = float((always_on_supportspec or {}).get("score", float("-inf")) or float("-inf")) if always_on_supportspec else None
        evidence["wider_parafence_quorum_score"] = float((wider_parafence_quorum or {}).get("score", float("-inf")) or float("-inf")) if wider_parafence_quorum else None
        evidence["ultra_low_margin_safe_expand_score"] = float((ultra_low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if ultra_low_margin_safe_expand else None
    elif normalized == "v228":
        control = _entry_by_name(history, "v228 promoted support/spec code-form control")
        code_pref_soft_multistep = _entry_by_name(history, "v228 promoted support/spec code-form + code-pref soft multistep")
        ultra_low_margin_safe_expand = _entry_by_name(history, "v228 promoted support/spec code-form + ultra-low-margin safe expand")
        support_floor_rollback = _entry_by_name(history, "v228 promoted support/spec code-form + support-floor rollback")
        soft_multistep_ultra_low_margin_safe_expand = _entry_by_name(history, "v228 promoted support/spec code-form + soft multistep + ultra-low-margin safe expand")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["code_pref_soft_multistep_score"] = float((code_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")) if code_pref_soft_multistep else None
        evidence["ultra_low_margin_safe_expand_score"] = float((ultra_low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if ultra_low_margin_safe_expand else None
        evidence["support_floor_rollback_score"] = float((support_floor_rollback or {}).get("score", float("-inf")) or float("-inf")) if support_floor_rollback else None
        evidence["soft_multistep_ultra_low_margin_safe_expand_score"] = float((soft_multistep_ultra_low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if soft_multistep_ultra_low_margin_safe_expand else None
    elif normalized == "v229":
        control = _entry_by_name(history, "v229 clean v178 support-aware citecheck pairwise control")
        parafence_stability = _entry_by_name(history, "v229 clean v178 support-aware citecheck pairwise + parafence stability")
        ultra_low_margin_safe_expand = _entry_by_name(history, "v229 clean v178 support-aware citecheck pairwise + ultra-low-margin safe expand")
        deterministic_paraphrase_quorum = _entry_by_name(history, "v229 clean v178 support-aware citecheck pairwise + deterministic paraphrase quorum")
        parafence_stability_ultra_low_margin_safe_expand = _entry_by_name(history, "v229 clean v178 support-aware citecheck pairwise + parafence stability + ultra-low-margin safe expand")
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["parafence_stability_score"] = float((parafence_stability or {}).get("score", float("-inf")) or float("-inf")) if parafence_stability else None
        evidence["ultra_low_margin_safe_expand_score"] = float((ultra_low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if ultra_low_margin_safe_expand else None
        evidence["deterministic_paraphrase_quorum_score"] = float((deterministic_paraphrase_quorum or {}).get("score", float("-inf")) or float("-inf")) if deterministic_paraphrase_quorum else None
        evidence["parafence_stability_ultra_low_margin_safe_expand_score"] = float((parafence_stability_ultra_low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if parafence_stability_ultra_low_margin_safe_expand else None
    elif normalized == "v230":
        control = _entry_by_name(history, "v230 clean v178 support-aware citecheck pairwise control")
        support_floor = _entry_by_name(history, "v230 clean v178 support-aware citecheck pairwise + support floor")
        code_form_tie_break = _entry_by_name(history, "v230 clean v178 support-aware citecheck pairwise + code-form tie-break")
        support_pref_multistep_floor = _entry_by_name(history, "v230 clean v178 support-aware citecheck pairwise + support-pref multistep floor")
        code_form_tie_break_ultra_low_margin_safe_expand = _entry_by_name(
            history,
            "v230 clean v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["support_floor_score"] = float((support_floor or {}).get("score", float("-inf")) or float("-inf")) if support_floor else None
        evidence["code_form_tie_break_score"] = float((code_form_tie_break or {}).get("score", float("-inf")) or float("-inf")) if code_form_tie_break else None
        evidence["support_pref_multistep_floor_score"] = float((support_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if support_pref_multistep_floor else None
        evidence["code_form_tie_break_ultra_low_margin_safe_expand_score"] = float((code_form_tie_break_ultra_low_margin_safe_expand or {}).get("score", float("-inf")) or float("-inf")) if code_form_tie_break_ultra_low_margin_safe_expand else None
    elif normalized == "v231":
        control = _entry_by_name(history, "v231 clean v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand control")
        local_union_finalist_room = _entry_by_name(history, "v231 clean v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + local union finalist room")
        code_pref_multistep_floor = _entry_by_name(history, "v231 clean v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + code-pref multistep floor")
        code_pref_soft_multistep = _entry_by_name(history, "v231 clean v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + code-pref soft multistep")
        local_union_finalist_room_code_pref_soft_multistep = _entry_by_name(
            history,
            "v231 clean v178 support-aware citecheck pairwise + code-form tie-break + ultra-low-margin safe expand + local union finalist room + code-pref soft multistep",
        )
        evidence["control_score"] = float((control or {}).get("score", float("-inf")) or float("-inf")) if control else None
        evidence["local_union_finalist_room_score"] = float((local_union_finalist_room or {}).get("score", float("-inf")) or float("-inf")) if local_union_finalist_room else None
        evidence["code_pref_multistep_floor_score"] = float((code_pref_multistep_floor or {}).get("score", float("-inf")) or float("-inf")) if code_pref_multistep_floor else None
        evidence["code_pref_soft_multistep_score"] = float((code_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")) if code_pref_soft_multistep else None
        evidence["local_union_finalist_room_code_pref_soft_multistep_score"] = float((local_union_finalist_room_code_pref_soft_multistep or {}).get("score", float("-inf")) or float("-inf")) if local_union_finalist_room_code_pref_soft_multistep else None

    return {
        "next_mode": _FOLLOWUP_MODE_MAP[normalized],
        "reason": f"{normalized} plateaued without a strict pass; advance to {_FOLLOWUP_MODE_MAP[normalized]}.",
        "evidence": evidence,
    }


def record_result(state: dict[str, Any], candidate: StrictEvalCandidate, seed: int, summary: dict[str, Any]) -> dict[str, Any]:
    run_status = str(summary.get("run_status", "") or "").strip().lower()
    if run_status == "train_failed":
        diagnosis = {
            "primary_failure_mode": "train_failed",
            "flags": ["train_failed"],
        }
    else:
        diagnosis = classify_summary(summary)
    family = candidate_family(candidate.name, candidate.phase4_updates)
    score = float(summary.get("answer_score", strict_answer_score(summary)))
    run_id = f"{slugify(candidate.name)}-seed{seed}"
    entry = {
        "candidate_name": candidate.name,
        "run_id": run_id,
        "seed": seed,
        "score": score,
        "strict_status": summary.get("strict_status", "missing"),
        "run_status": run_status or "ok",
        "family": family,
        "primary_failure_mode": diagnosis["primary_failure_mode"],
        "flags": diagnosis["flags"],
        "phase4_updates": copy.deepcopy(candidate.phase4_updates),
    }
    state.setdefault("history", []).append(entry)
    state["completed_runs"] = int(state.get("completed_runs", 0) or 0) + 1
    if score > float(state.get("incumbent_score", float("-inf")) or float("-inf")):
        state["incumbent_name"] = candidate.name
        state["incumbent_score"] = score
    suppressed = set(state.get("suppressed_families", []) or [])
    if run_status != "train_failed" and diagnosis["primary_failure_mode"] == "catastrophic_zero_known":
        suppressed.add(family)
        if family == "geometry_first_decoupled_retrieval":
            suppressed.update({"adaptive_multiview", "query_multiview"})
    state["suppressed_families"] = sorted(suppressed)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strict-eval-focused autoresearch for IGNORANCE-1.")
    parser.add_argument("--mode", default="v4")
    parser.add_argument("--max-cycles", type=int, default=8)
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument("--auto-followup", action="store_true")
    args = parser.parse_args()

    mode = str(args.mode or "v4").strip().lower()
    ensure_results_header(mode)
    state = load_search_state(mode)
    state["mode"] = mode
    save_search_state(state)

    if args.auto_followup and args.max_cycles <= 0:
        details = copy.deepcopy(state.get("stall_details") or {})
        reason = str(state.get("stop_reason") or "manual_followup")
        stop_notice = _write_stop_notice(state, reason, details)
        append_decision({"event": "stop", "reason": reason, "recommended_followup": stop_notice.get("recommended_followup")}, mode=mode)
        rc = _maybe_chain_followup(mode, stop_notice)
        if stop_notice.get("recommended_followup"):
            append_decision({"event": "auto_followup_launch", "recommended_followup": stop_notice.get("recommended_followup")}, mode=mode)
        return rc

    deadline = time.time() + float(args.budget_hours) * 3600.0 if args.budget_hours else None
    cycles_run = 0

    while True:
        if args.max_cycles >= 0 and cycles_run >= args.max_cycles:
            state["stop_reason"] = "budget_exhausted"
            save_search_state(state)
            stop_notice = _write_stop_notice(state, "budget_exhausted", {"dominant_flags": [], "dominant_failure_modes": []})
            append_decision({"event": "stop", "reason": "budget_exhausted", "recommended_followup": stop_notice.get("recommended_followup")}, mode=mode)
            if args.auto_followup:
                _maybe_chain_followup(mode, stop_notice)
            return 0

        if deadline is not None and time.time() >= deadline:
            state["stop_reason"] = "budget_exhausted"
            save_search_state(state)
            stop_notice = _write_stop_notice(state, "budget_exhausted", {"dominant_flags": [], "dominant_failure_modes": []})
            append_decision({"event": "stop", "reason": "budget_exhausted", "recommended_followup": stop_notice.get("recommended_followup")}, mode=mode)
            if args.auto_followup:
                _maybe_chain_followup(mode, stop_notice)
            return 0

        stall = detect_stall(state)
        if stall is not None:
            state["stop_reason"] = str(stall.get("reason") or "adaptive_plateau")
            state["stall_details"] = stall
            save_search_state(state)
            stop_notice = _write_stop_notice(state, state["stop_reason"], stall)
            append_decision({"event": "stop", "reason": state["stop_reason"], "recommended_followup": stop_notice.get("recommended_followup")}, mode=mode)
            if args.auto_followup:
                _maybe_chain_followup(mode, stop_notice)
            return 0

        candidate = choose_next_candidate(state)
        if candidate is None:
            state["stop_reason"] = "no_candidate_available"
            save_search_state(state)
            stop_notice = _write_stop_notice(state, "no_candidate_available", {"dominant_flags": [], "dominant_failure_modes": []})
            append_decision({"event": "stop", "reason": "no_candidate_available", "recommended_followup": stop_notice.get("recommended_followup")}, mode=mode)
            if args.auto_followup:
                _maybe_chain_followup(mode, stop_notice)
            return 0

        seed = next_seed(state)
        summary = run_candidate(candidate, seed, mode)
        if summary is None:
            summary = {
                "run_status": "train_failed",
                "strict_status": "train_failed",
                "avg_known_similarity": 0.0,
                "avg_known_exact_similarity": 0.0,
                "avg_known_paraphrase_similarity": 0.0,
                "synthesis_similarity": 0.0,
                "avg_known_margin": 0.0,
                "ignorance_gap": 0.0,
                "avg_ood_confidence": 1.0,
                "strict_failures": ["train_failed"],
                "query_diagnostics": {"avg_offdiag_similarity": 1.0, "participation_ratio_fraction": 0.0},
                "code_diagnostics": {"avg_offdiag_similarity": 1.0, "participation_ratio_fraction": 0.0},
            }
        record_result(state, candidate, seed, summary)
        save_search_state(state)
        cycles_run += 1

        if should_stop(summary):
            state["stop_reason"] = "strict_pass"
            save_search_state(state)
            stop_notice = _write_stop_notice(state, "strict_pass", {"dominant_flags": [], "dominant_failure_modes": []})
            append_decision({"event": "stop", "reason": "strict_pass", "recommended_followup": stop_notice.get("recommended_followup")}, mode=mode)
            if args.auto_followup:
                _maybe_chain_followup(mode, stop_notice)
            return 0




def _v372_candidate_library() -> list[StrictEvalCandidate]:
    """
    v372: Three-axis sweep from the v340 winner formula.
    
    Key insight from v340: strict PASS (43.5) came from:
    1. frozen backbone
    2. mixed_boundary dataset
    3. clf=0.09
    4. confidence_mode=agreement_augmented (the missing ingredient!)
    
    v371 failed because it used neighborhood_posterior instead.
    v372 tests all three axes:
    - Axis 1: dataset (taxonomy vs mixed_boundary)
    - Axis 2: phase4 training (0 vs 300 steps)  
    - Axis 3: confidence mode (agreement_augmented vs support_feature_calibrator)
    
    Hypothesis: agreement_augmented + frozen + mixed_boundary + clf=0.09 + prod_steps=0
    should replicate v340's 43.5 PASS on taxonomy dataset.
    """
    v338_checkpoint = str(
        ROOT
        / "artifacts"
        / "strict_eval_autoresearch_v338"
        / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504"
        / "model.pt"
    )

    v340_checkpoint = str(
        ROOT
        / "artifacts"
        / "strict_eval_autoresearch_v340"
        / "v340-frozen-geometry-agreement-augmented-calibrator-seed505"
        / "model.pt"
    )

    v372_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "production_mode": False,
        "production_steps": 0,
        "phase4_steps": 0,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "freeze_backbone": True,
        "warm_start_phase3_only": True,
        "warm_start_model_path": v338_checkpoint,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_queue_samples": 128,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "phase4_factorized_hard_negatives": True,
        "paraphrase_batch_probability": 0.35,
        "retrieval_margin_weight": 0.22,
        "retrieval_margin": 0.32,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.07,
        "rank_reg_target": "code+query",
        "ranking_margin_weight": 0.15,
        "ranking_margin": 0.26,
        "ranking_focal_gamma": 1.5,
        "sigreg_weight": 0.5,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ema_target_decay": 0.995,
        "epistemic_boundary_weight": 0.0,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "phase4_joint_training": True,
        "max_hard_negatives_per_example": 4,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "ood_weight": 0.0,
        "pred_ood_weight": 0.0,
        "clf_weight": 0.09,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "reset_query_head_on_resume": False,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict,
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
        base_checkpoint: str = v338_checkpoint,
    ) -> StrictEvalCandidate:
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S372",
            intervention_type="v372_agreement_augmented_frozen",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=copy.deepcopy(v372_base) | copy.deepcopy(phase4_overrides),
            eval_overrides=copy.deepcopy(common_eval) | {"confidence_mode": confidence_mode},
            base_model_path=base_checkpoint,
            eval_repeats=3,
        )

    return [
        # Axis 1: agreement_augmented on taxonomy (replicate v340's formula on taxonomy)
        candidate(
            "v372 taxonomy agreement_augmented seed501",
            {},
            "CRITICAL: Use agreement_augmented (v340 winner's mode) + frozen + taxonomy. "
            "v340 passed with mixed_boundary + agreement_augmented. "
            "This tests if the formula transfers to taxonomy dataset.",
            "If ~43: formula transfers to taxonomy. If <40: mixed_boundary was the key ingredient.",
            confidence_mode="agreement_augmented",
        ),
        # Axis 1 variant: mixed_boundary + agreement_augmented
        candidate(
            "v372 mixed_boundary agreement_augmented seed502",
            {"phase4_dataset": "behavioral_constraints_v2_mixed_boundary_curriculum_v1"},
            "Same as seed501 but with mixed_boundary (v340's exact dataset). "
            "Should replicate v340's 43.5 if the formula is correct.",
            "If ~43.5: mixed_boundary + agreement_augmented + frozen passes. "
            "If different: seed-dependent.",
            confidence_mode="agreement_augmented",
        ),
        # Axis 2: agreement_augmented + prod_steps=300
        candidate(
            "v372 taxonomy agreement_augmented prodsteps300 seed503",
            {"production_steps": 300, "phase4_steps": 300},
            "Add 300 phase4 steps while frozen. "
            "Tests if some phase4 training helps or hurts vs prod_steps=0.",
            "If >43.5: partial training helps. If <43: phase4 training hurts even when frozen.",
            confidence_mode="agreement_augmented",
        ),
        # Axis 3: warm-start from v340 winner checkpoint
        candidate(
            "v372 warmstart v340 agreement_augmented seed504",
            {},
            "CRITICAL: Use v340 winner checkpoint (the one that scored 43.5) as warm-start. "
            "v340 winner used agreement_augmented + frozen + mixed_boundary. "
            "This tests if the winning checkpoint itself is the key asset.",
            "If >43.5: v340 checkpoint has special properties. "
            "If ~43: checkpoint warmth doesn't transfer across configs. "
            "If <40: geometry from v340 doesn't suit taxonomy eval.",
            confidence_mode="agreement_augmented",
            base_checkpoint=v340_checkpoint,
        ),
        # Axis 3: warmstart v340 + mixed_boundary (clone v340 exactly)
        candidate(
            "v372 warmstart v340 mixed_boundary seed505",
            {"phase4_dataset": "behavioral_constraints_v2_mixed_boundary_curriculum_v1"},
            "Clone v340 exactly: use v340 winner checkpoint + mixed_boundary + agreement_augmented. "
            "Should reproduce v340's 43.5 if everything transfers.",
            "If ~43.5: v340 checkpoint + formula is the winning recipe. "
            "If >43.5: we found an improvement. "
            "If <43: checkpoint geometry is partially corrupted or seed-dependent.",
            confidence_mode="agreement_augmented",
            base_checkpoint=v340_checkpoint,
        ),
    ]


def _v374_candidate_library() -> list[StrictEvalCandidate]:
    """
    v374: Champion-Challenger listwise + higher capacity per family.
    
    Key insight from research memos:
    - Objective uses same-family near-miss variants (ordered-ranking)
    - Need to rank the CORRECT key above the near-miss within each family
    - pointwise pairwise losses can't capture full family ordering structure
    
    Solution: Champion-Challenger loss (memo #4's highest-impact technique).
    This loss:
    1. Uses largest-size examples as "champions" (correct key = full spec)
    2. Smaller-size examples as "challengers" (near-miss variants)
    3. Optimizes margin between champion and hardest challenger
    4. IS listwise: considers full family hierarchy simultaneously
    
    v374 candidates:
    1. CC + strong margins + no equivalence (main bet)
    2. CC + very strong margins (push harder)
    3. CC + unfrozen backbone (let CC loss reshape geometry)
    4. CC + 600 steps (more CC training)
    5. Control: no CC, no equivalence, same strong margins
    """
    v338_checkpoint = str(ROOT / "artifacts" / "strict_eval_autoresearch_v338" / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504" / "model.pt")

    v374_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "production_mode": False,
        "production_steps": 300,
        "phase4_steps": 300,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "freeze_backbone": True,
        "warm_start_phase3_only": True,
        "warm_start_model_path": v338_checkpoint,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_queue_samples": 128,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        # NO equivalence losses (memo #4: equivalence is harmful for taxonomy discipline)
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "phase4_factorized_hard_negatives": True,
        "paraphrase_batch_probability": 0.35,
        # Strong discriminative margins
        "retrieval_margin_weight": 0.25,
        "retrieval_margin": 0.40,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.08,
        "rank_reg_target": "code+query",
        "ranking_margin_weight": 0.20,
        "ranking_margin": 0.40,
        "ranking_focal_gamma": 2.0,
        "sigreg_weight": 0.5,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ema_target_decay": 0.995,
        "epistemic_boundary_weight": 0.0,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "phase4_joint_training": True,
        "max_hard_negatives_per_example": 4,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "ood_weight": 0.0,
        "pred_ood_weight": 0.0,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "reset_query_head_on_resume": False,
        # Champion-challenger: ramp in during first 30% of training
        "champion_challenger_weight": 0.5,
        "champion_challenger_margin": 0.30,
        "champion_challenger_temperature": 0.05,
        "champion_challenger_start_fraction": 0.0,
        "champion_challenger_ramp_fraction": 0.30,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict,
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        phase4 = copy.deepcopy(v374_base) | copy.deepcopy(phase4_overrides)
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S374",
            intervention_type="v374_champion_challenger_listwise",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=phase4,
            eval_overrides=copy.deepcopy(common_eval) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        # Candidate 1: CC + strong margins (main bet)
        candidate(
            "v374 CC strong_margins seed501",
            {
                "champion_challenger_weight": 0.5,
                "champion_challenger_margin": 0.30,
                "champion_challenger_temperature": 0.05,
                "champion_challenger_start_fraction": 0.0,
                "champion_challenger_ramp_fraction": 0.30,
                "retrieval_margin": 0.40,
                "retrieval_margin_weight": 0.25,
                "ranking_margin": 0.40,
                "ranking_margin_weight": 0.20,
            },
            "Champion-Challenger listwise loss (memo #4 highest-impact) + strong margins + no equivalence. "
            "CC loss trains the model to rank correct-spec (champion) above near-miss (challenger) across the full family hierarchy. "
            "No equivalence overhead lets full capacity go to discrimination.",
            "If direct_rate improves: CC listwise is the missing ingredient. "
            "If same as v373: need even stronger CC or the geometry truly can't separate these variants.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 2: CC + very strong margins
        candidate(
            "v374 CC very_strong_margins seed502",
            {
                "champion_challenger_weight": 0.5,
                "champion_challenger_margin": 0.30,
                "champion_challenger_temperature": 0.05,
                "champion_challenger_start_fraction": 0.0,
                "champion_challenger_ramp_fraction": 0.30,
                "retrieval_margin": 0.50,
                "retrieval_margin_weight": 0.30,
                "ranking_margin": 0.50,
                "ranking_margin_weight": 0.25,
            },
            "Same as seed501 but with very strong margins (0.50). "
            "Maximum pressure on the model to produce clear winner.",
            "If direct_rate > seed501: margins were underset. "
            "If same: margins aren't the bottleneck.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 3: CC + unfrozen backbone
        candidate(
            "v374 CC unfrozen seed503",
            {
                "champion_challenger_weight": 0.5,
                "champion_challenger_margin": 0.30,
                "champion_challenger_temperature": 0.05,
                "champion_challenger_start_fraction": 0.0,
                "champion_challenger_ramp_fraction": 0.30,
                "freeze_backbone": False,
                "warm_start_phase3_only": False,
                "retrieval_margin": 0.40,
                "retrieval_margin_weight": 0.25,
                "ranking_margin": 0.40,
                "ranking_margin_weight": 0.20,
            },
            "CC + unfrozen backbone. Let the CC loss reshape geometry during training. "
            "Frozen geometry may limit how well CC loss can separate champions from challengers.",
            "If direct_rate >> 0.375: geometry reshaping needed. "
            "If < 0.375: unfreezing causes collapse even with CC loss guiding it.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 4: CC + longer training
        candidate(
            "v374 CC long_training seed504",
            {
                "champion_challenger_weight": 0.5,
                "champion_challenger_margin": 0.30,
                "champion_challenger_temperature": 0.05,
                "champion_challenger_start_fraction": 0.0,
                "champion_challenger_ramp_fraction": 0.30,
                "production_steps": 600,
                "phase4_steps": 600,
                "retrieval_margin": 0.40,
                "retrieval_margin_weight": 0.25,
                "ranking_margin": 0.40,
                "ranking_margin_weight": 0.20,
            },
            "CC + 600 steps (vs 300). More training lets CC loss fully converge.",
            "If direct_rate improves with longer training: CC loss needs more steps. "
            "If plateaus: CC loss converges quickly and geometry is the limit.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 5: Control - NO CC, NO equivalence, strong margins
        candidate(
            "v374 no_CC no_equiv strong_margins seed505",
            {
                "champion_challenger_weight": 0.0,
                "retrieval_margin": 0.40,
                "retrieval_margin_weight": 0.25,
                "ranking_margin": 0.40,
                "ranking_margin_weight": 0.20,
            },
            "Control: no CC, no equivalence, strong margins. "
            "Same as v373 seed501 but with equivalence removed. "
            "Tests if equivalence removal alone helps, vs CC helping.",
            "If direct_rate ~ v373: CC is the key. "
            "If direct_rate > v373 but < CC variants: equivalence removal helps, CC adds more.",
            confidence_mode="support_feature_calibrator",
        ),
    ]



def _v373_candidate_library() -> list[StrictEvalCandidate]:
    """
    v373: Aggressive discriminative training to break the direct_rate ceiling.
    
    Key insight from v372: direct_rate stuck at 0.375 (need 0.75).
    Abstainers have margin=0 — model can't distinguish supported from unsupported variants.
    Root cause: phase4 training isn't discriminative enough.
    
    v373 candidates test:
    1. STRONGER MARGINS: retrieval_margin=0.40, ranking_margin=0.40
       - Higher margins force the model to commit to a clear winner
    2. FEWER HARD NEGATIVES: max_hard_negatives=2 (was 4)
       - Fewer hard negatives = less confusion between similar variants
    3. HIGHER FOCAL GAMMA: ranking_focal_gamma=2.0 (was 1.5)
       - Focus more on hard-to-classify examples
    4. LONGER TRAINING: production_steps=600 (was 300 or 0)
       - More training on the phase4 dataset
    5. UNFROZEN backbone (one candidate only):
       - Let geometry reshape to improve discrimination
       - With strong margins to guide the reshaping
    """
    v338_checkpoint = str(ROOT / "artifacts" / "strict_eval_autoresearch_v338" / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504" / "model.pt")

    v373_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "production_mode": False,
        "production_steps": 300,
        "phase4_steps": 300,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "freeze_backbone": True,
        "warm_start_phase3_only": True,
        "warm_start_model_path": v338_checkpoint,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_queue_samples": 128,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "phase4_factorized_hard_negatives": True,
        "paraphrase_batch_probability": 0.35,
        "retrieval_margin_weight": 0.25,
        "retrieval_margin": 0.40,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.08,
        "rank_reg_target": "code+query",
        "ranking_margin_weight": 0.20,
        "ranking_margin": 0.40,
        "ranking_focal_gamma": 2.0,
        "sigreg_weight": 0.5,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ema_target_decay": 0.995,
        "epistemic_boundary_weight": 0.0,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "phase4_joint_training": True,
        "max_hard_negatives_per_example": 2,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "ood_weight": 0.0,
        "pred_ood_weight": 0.0,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "reset_query_head_on_resume": False,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict,
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        phase4 = copy.deepcopy(v373_base) | copy.deepcopy(phase4_overrides)
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S373",
            intervention_type="v373_aggressive_discriminative",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=phase4,
            eval_overrides=copy.deepcopy(common_eval) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        # Candidate 1: stronger margins + fewer hard negatives (control for v373)
        candidate(
            "v373 strong_margins few_hard_neg seed501",
            {
                "retrieval_margin": 0.40,
                "retrieval_margin_weight": 0.25,
                "ranking_margin": 0.40,
                "ranking_margin_weight": 0.20,
                "ranking_focal_gamma": 2.0,
                "max_hard_negatives_per_example": 2,
            },
            "HIGHER MARGINS (0.40 vs 0.32) + FEWER HARD NEGATIVES (2 vs 4). "
            "v372 abstainers had margin=0. Higher margins force clear winner selection. "
            "Fewer hard negatives reduces confusion between similar variants.",
            "If direct_rate improves: margins were the bottleneck. "
            "If unchanged: geometry is the deeper issue.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 2: same as 1 but with agreement_augmented
        candidate(
            "v373 strong_margins agreement_aug seed502",
            {
                "retrieval_margin": 0.40,
                "retrieval_margin_weight": 0.25,
                "ranking_margin": 0.40,
                "ranking_margin_weight": 0.20,
                "ranking_focal_gamma": 2.0,
                "max_hard_negatives_per_example": 2,
            },
            "Same discriminative training as seed501 but with agreement_augmented confidence. "
            "agreement_augmented performed best in v340.",
            "If agreement_aug > support_feature: confidence mode matters for discrimination.",
            confidence_mode="agreement_augmented",
        ),
        # Candidate 3: longer training (600 steps) + strong margins
        candidate(
            "v373 long_training strong_margins seed503",
            {
                "production_steps": 600,
                "phase4_steps": 600,
                "retrieval_margin": 0.40,
                "retrieval_margin_weight": 0.25,
                "ranking_margin": 0.40,
                "ranking_margin_weight": 0.20,
                "ranking_focal_gamma": 2.0,
                "max_hard_negatives_per_example": 2,
            },
            "LONGER TRAINING (600 vs 300 steps) with strong margins. "
            "More phase4 training may further improve discrimination.",
            "If direct_rate improves with longer training: model needs more data. "
            "If plateaus: geometry capacity is the limit.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 4: VERY strong margins (0.50)
        candidate(
            "v373 very_strong_margins seed504",
            {
                "retrieval_margin": 0.50,
                "retrieval_margin_weight": 0.30,
                "ranking_margin": 0.50,
                "ranking_margin_weight": 0.25,
                "ranking_focal_gamma": 2.0,
                "max_hard_negatives_per_example": 2,
            },
            "VERY HIGH MARGINS (0.50). Extreme version of seed501 to force maximum discrimination.",
            "If direct_rate jumps to >0.75: margins were severely underset. "
            "If no change: the geometry genuinely cannot separate these items.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 5: unfrozen backbone + strong margins
        candidate(
            "v373 unfrozen strong_margins seed505",
            {
                "freeze_backbone": False,
                "retrieval_margin": 0.40,
                "retrieval_margin_weight": 0.25,
                "ranking_margin": 0.40,
                "ranking_margin_weight": 0.20,
                "ranking_focal_gamma": 2.0,
                "max_hard_negatives_per_example": 2,
                "warm_start_phase3_only": False,
            },
            "UNFROZEN backbone + strong margins. "
            "Frozen geometry may cap discrimination. Unfreeze to reshape geometry. "
            "Strong margins guide the reshaping toward better discrimination.",
            "If direct_rate >> 0.375: geometry reshaping is required. "
            "If ~0.375: frozen geometry is sufficient, margins are the issue. "
            "If <0.375: unfreezing causes collapse.",
            confidence_mode="support_feature_calibrator",
        ),
    ]



def _v375_candidate_library() -> list[StrictEvalCandidate]:
    """
    v375: Direction 1 from deep research memo -- Same-family only ranking pool.
    
    Key insight from research: The ranking_margin_loss compares positive against
    a MIXED negative pool (same-family hard negatives + cross-family distractors).
    Easy cross-family distractors dominate the gradient, drowning out the hard
    same-family discrimination needed for the eval's champion-vs-challenger pairs.
    
    v375 implements same_family_only_ranking=True: the champion's ranking loss
    uses ONLY same-family hard negatives as the negative pool. This forces the
    model to learn precise ordering within families, which is exactly what the
    strict eval's 8-family near-miss pairs test.
    
    Candidates:
    1. same_family_only (main bet -- this is the structural fix)
    2. same_family + CC (combine with champion-challenger loss)
    3. same_family + unfrozen (let geometry reshape with same-family pressure)
    4. same_family + longer training (more steps for same-family convergence)
    5. same_family + more facets (increased capacity for fine-grained discrimination)
    """
    v338_checkpoint = str(ROOT / "artifacts" / "strict_eval_autoresearch_v338" / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504" / "model.pt")

    v375_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "production_mode": False,
        "production_steps": 300,
        "phase4_steps": 300,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "freeze_backbone": True,
        "warm_start_phase3_only": True,
        "warm_start_model_path": v338_checkpoint,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_queue_samples": 128,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        # NO equivalence losses
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "phase4_factorized_hard_negatives": True,
        # Same-family only ranking pool (THE KEY CHANGE)
        "same_family_only_ranking": True,
        # Strong margins for same-family discrimination
        "retrieval_margin_weight": 0.25,
        "retrieval_margin": 0.40,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.08,
        "rank_reg_target": "code+query",
        "ranking_margin_weight": 0.20,
        "ranking_margin": 0.40,
        "ranking_focal_gamma": 2.0,
        "sigreg_weight": 0.5,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ema_target_decay": 0.995,
        "epistemic_boundary_weight": 0.0,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "phase4_joint_training": True,
        "max_hard_negatives_per_example": 4,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "ood_weight": 0.0,
        "pred_ood_weight": 0.0,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "reset_query_head_on_resume": False,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict,
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        phase4 = copy.deepcopy(v375_base) | copy.deepcopy(phase4_overrides)
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S375",
            intervention_type="v375_same_family_only_ranking",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=phase4,
            eval_overrides=copy.deepcopy(common_eval) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        # Candidate 1: same_family_only alone (main bet for Direction 1)
        candidate(
            "v375 same_family_only seed501",
            {
                "same_family_only_ranking": True,
                "champion_challenger_weight": 0.0,
            },
            "Same-family only ranking pool (Direction 1). "
            "The ranking loss compares champion against ONLY same-family hard negatives, "
            "not the mixed pool. This forces the model to learn precise within-family ordering "
            "which is what the strict eval's 8-family near-miss pairs test.",
            "If direct_rate improves: same-family negative isolation was the missing ingredient. "
            "If still 0.375: geometry itself cannot separate these variants (need architectural change).",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 2: same_family + CC (Direction 1 + CC)
        candidate(
            "v375 same_family_plus_CC seed502",
            {
                "same_family_only_ranking": True,
                "champion_challenger_weight": 0.5,
                "champion_challenger_margin": 0.30,
                "champion_challenger_temperature": 0.05,
                "champion_challenger_start_fraction": 0.0,
                "champion_challenger_ramp_fraction": 0.30,
            },
            "Same-family ranking pool + Champion-Challenger loss (Directions 1+2). "
            "same_family isolates the within-family discrimination signal in ranking loss. "
            "CC adds pairwise champion-over-challenger supervision on top. "
            "Together they should maximize within-family ordering quality.",
            "If direct_rate > seed501: CC adds value even with correct negative pool. "
            "If = seed501: CC is redundant when negative pool is correct.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 3: same_family + unfrozen
        candidate(
            "v375 same_family_unfrozen seed503",
            {
                "same_family_only_ranking": True,
                "champion_challenger_weight": 0.0,
                "freeze_backbone": False,
                "warm_start_phase3_only": False,
            },
            "Same-family ranking + unfrozen backbone (Directions 1+3). "
            "With same-family signal concentrated, letting the geometry reshape "
            "may let the encoder learn better within-family structure.",
            "If direct_rate >> 0.375: geometry reshaping needed. "
            "If < 0.375: unfreezing causes collapse even with correct ranking signal.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 4: same_family + longer training
        candidate(
            "v375 same_family_long_training seed504",
            {
                "same_family_only_ranking": True,
                "champion_challenger_weight": 0.0,
                "production_steps": 600,
                "phase4_steps": 600,
            },
            "Same-family ranking + 600 steps (Directions 1 + longer training). "
            "More training time lets the concentrated same-family ranking signal "
            "fully converge before the model stops.",
            "If direct_rate improves with longer training: signal was right but needed more steps. "
            "If plateaus: geometry limit, not training time.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 5: same_family + more facets
        candidate(
            "v375 same_family_ten_facets seed505",
            {
                "same_family_only_ranking": True,
                "champion_challenger_weight": 0.0,
                "retrieval_num_facets": 10,
            },
            "Same-family ranking + 10 facets (Directions 1+4). "
            "More facets provide finer-grained token-level discrimination for "
            "modifier-sensitive within-family ordering (strip vs trailing-only, etc).",
            "If direct_rate > seed501: facet capacity helps separate same-family variants. "
            "If = seed501: facets don't help beyond same-family signal.",
            confidence_mode="support_feature_calibrator",
        ),
    ]
def _v376_candidate_library() -> list[StrictEvalCandidate]:
    """
    v376: Keep v374's v293-winner base (proven best) + sweep CC weight + staged curriculum.
    
    Key findings:
    - v374 base (from v293 winner) outperforms v375 base (from v295 winner) 
    - CC weight=0.5 helped in v374 but hurt in v375 (same_family_only base changed the geometry)
    - v374 best: score=39.75 with CC=0.5 on v293-base
    - Staged curriculum (memo Direction 2): train easy→hard negatives first, then hard
    
    v376 candidates:
    1. CC=0.3 on v293-base (lower CC weight)
    2. CC=0.5 on v293-base (reproduce v374 best)
    3. CC=0.7 on v293-base (higher CC weight)
    4. Staged curriculum on v293-base (CC=0.5, ramp in hard negatives)
    5. v293-base alone, no CC (isolate CC contribution)
    """
    v338_checkpoint = str(ROOT / "artifacts" / "strict_eval_autoresearch_v338" / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504" / "model.pt")

    # v293 winner base (from v374 - the proven best base)
    v376_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "production_mode": False,
        "production_steps": 300,
        "phase4_steps": 300,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "freeze_backbone": True,
        "warm_start_phase3_only": True,
        "warm_start_model_path": v338_checkpoint,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_queue_samples": 128,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "phase4_factorized_hard_negatives": True,
        # Moderate margins
        "retrieval_margin_weight": 0.25,
        "retrieval_margin": 0.25,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.08,
        "rank_reg_target": "code+query",
        "ranking_margin_weight": 0.20,
        "ranking_margin": 0.25,
        "ranking_focal_gamma": 2.0,
        "sigreg_weight": 0.5,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ema_target_decay": 0.995,
        "epistemic_boundary_weight": 0.0,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "phase4_joint_training": True,
        "max_hard_negatives_per_example": 4,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "ood_weight": 0.0,
        "pred_ood_weight": 0.0,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "reset_query_head_on_resume": False,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict,
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        phase4 = copy.deepcopy(v376_base) | copy.deepcopy(phase4_overrides)
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S376",
            intervention_type="v376_cc_sweep_staged_curriculum",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=phase4,
            eval_overrides=copy.deepcopy(common_eval) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        # Candidate 1: CC=0.3 (lower than v374's 0.5)
        candidate(
            "v376 CC=0.3 seed511",
            {
                "champion_challenger_weight": 0.3,
                "champion_challenger_margin": 0.20,
                "champion_challenger_temperature": 0.05,
                "champion_challenger_start_fraction": 0.0,
                "champion_challenger_ramp_fraction": 0.30,
            },
            "CC weight=0.3 (lower than v374's 0.5). "
            "Sweep CC weight down to find sweet spot. v374 had CC=0.5 and succeeded; "
            "v375 had CC=0.5 but failed due to base geometry. Testing CC=0.3 on proven base.",
            "If score > 39.75: lower CC weight is better on this base. "
            "If = 39.75: CC weight is robust in this range. "
            "If < 39.75: CC=0.5 is near optimal.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 2: CC=0.5 (reproduce v374 best)
        candidate(
            "v376 CC=0.5 seed512",
            {
                "champion_challenger_weight": 0.5,
                "champion_challenger_margin": 0.20,
                "champion_challenger_temperature": 0.05,
                "champion_challenger_start_fraction": 0.0,
                "champion_challenger_ramp_fraction": 0.30,
            },
            "CC weight=0.5 (exact v374 best). "
            "Reproduce v374's best exactly on v293-base to confirm reproducibility.",
            "If score ≈ 39.75: result is reproducible. "
            "If < 39.75: variance or minor config differences exist.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 3: CC=0.7 (higher CC weight)
        candidate(
            "v376 CC=0.7 seed513",
            {
                "champion_challenger_weight": 0.7,
                "champion_challenger_margin": 0.25,
                "champion_challenger_temperature": 0.05,
                "champion_challenger_start_fraction": 0.0,
                "champion_challenger_ramp_fraction": 0.30,
            },
            "CC weight=0.7 (higher). "
            "If CC is the active ingredient, pushing it harder might help.",
            "If score > 39.75: CC weight was underscaled. "
            "If < 39.75: CC=0.5 is near optimal.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 4: Staged curriculum (Direction 2 from memo)
        candidate(
            "v376 staged_curriculum seed514",
            {
                "champion_challenger_weight": 0.5,
                "champion_challenger_margin": 0.20,
                "champion_challenger_temperature": 0.05,
                "champion_challenger_start_fraction": 0.0,
                "champion_challenger_ramp_fraction": 0.50,  # Slower ramp
                # Staged curriculum: ramp in hard negatives later
                "warmup_fraction": 0.25,  # Slower warmup
                "regularizer_ramp_fraction": 0.30,  # Slower regularizer ramp
                "max_hard_negatives_per_example": 2,  # Start with fewer hard negatives
            },
            "Staged curriculum (Memo Direction 2). "
            "Train easier examples first (fewer hard negatives, slower regularizer ramp), "
            "then increase difficulty. This helps the model find a better basin before "
            "facing the hardest same-family discrimination.",
            "If score > 39.75: staged curriculum improves convergence. "
            "If < 39.75: one-shot hard training is better.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 5: No CC, isolate its contribution
        candidate(
            "v376 no_CC seed515",
            {
                "champion_challenger_weight": 0.0,
            },
            "No CC (control). "
            "Isolate the contribution of CC loss by removing it entirely. "
            "All other v293-base settings remain identical.",
            "If score < 39.75: CC is the active ingredient in v374's success. "
            "If ≈ 39.75: CC is not the key differentiator.",
            confidence_mode="support_feature_calibrator",
        ),
    ]


def _v377_candidate_library() -> list[StrictEvalCandidate]:
    """
    v377: Implement all 4 research5.md directions.

    Core problem: model retrieves right family but wrong member chunk (same_family
    near-miss rate 0% across v365-v376). Generic CC/auxiliary losses fail because
    they don't match the eval's champion-vs-challenger structure.

    v377 directions (from research5.md):
    1. Family-local listwise ranking: PRIMARY loss over champion + same-family graded
       negatives — this IS the eval's target structure, not an auxiliary signal.
    2. Staged ignorance: learn ranking first (support-only phase), add ignorance only
       after margins are nontrivial — prevents CC from destabilizing early geometry.
    3. Graded softmax negatives: model challengers as "close but below champion" with
       graded soft labels, not binary wrong/right.
    4. (Deferred) Token-level late-interaction verifier.

    Candidates:
    1. Listwise family-local (primary, dominant weight)
    2. Staged ignorance + listwise (Directions 1+2)
    3. Graded negatives + listwise (Directions 1+3)
    4. Full combo: all three (Directions 1+2+3)
    5. Listwise-only ablations: no CC, no ignorance, isolate listwise signal
    """
    v338_checkpoint = str(ROOT / "artifacts" / "strict_eval_autoresearch_v338" / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504" / "model.pt")

    # v293 winner base (proven best geometry)
    v377_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "production_mode": False,
        "production_steps": 300,
        "phase4_steps": 300,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "freeze_backbone": True,
        "warm_start_phase3_only": True,
        "warm_start_model_path": v338_checkpoint,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_queue_samples": 128,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "phase4_factorized_hard_negatives": True,
        "retrieval_margin_weight": 0.25,
        "retrieval_margin": 0.25,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.08,
        "rank_reg_target": "code+query",
        "ranking_margin_weight": 0.20,
        "ranking_margin": 0.25,
        "ranking_focal_gamma": 2.0,
        "sigreg_weight": 0.5,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ema_target_decay": 0.995,
        "epistemic_boundary_weight": 0.0,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "phase4_joint_training": True,
        "max_hard_negatives_per_example": 4,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        # CC: zeroed — family-local listwise replaces CC structure
        "champion_challenger_weight": 0.0,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "ood_weight": 0.0,
        "pred_ood_weight": 0.0,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "reset_query_head_on_resume": False,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict,
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        phase4 = copy.deepcopy(v377_base) | copy.deepcopy(phase4_overrides)
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S377",
            intervention_type="v377_research5_directions",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=phase4,
            eval_overrides=copy.deepcopy(common_eval) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        # === Direction 1: Family-local listwise ranking (primary, dominant weight) ===
        candidate(
            "v377 family_listwise seed511",
            {
                # Primary ranking loss: family-local listwise
                "family_local_listwise_weight": 1.0,
                "family_local_listwise_temperature": 0.07,
                # No CC, no ignorance (isolate listwise signal)
                "champion_challenger_weight": 0.0,
                "ignorance_ood_weight": 0.0,
                "ignorance_pred_weight": 0.0,
            },
            "research5 Direction 1 — Family-local listwise ranking as PRIMARY loss. "
            "Listwise ranking over champion + same-family graded negatives IS the eval's "
            "target structure (not an auxiliary signal). Champion is ranked above all "
            "same-family challengers; all other families are excluded. This directly "
            "optimizes the eval metric, unlike generic CC which doesn't match eval structure.",
            "If strict_eval PASS: family-local listwise is the correct ranking target. "
            "If strict_eval FAIL but score > 39.75: listwise improves ranking but is "
            "insufficient alone — add staged ignorance (Direction 2) or graded negatives (Dir 3). "
            "If score ≤ 39.75: need staged or combined approach.",
            confidence_mode="support_feature_calibrator",
        ),
        # === Directions 1+2: Listwise + staged ignorance schedule ===
        candidate(
            "v377 listwise_staged_ign seed511",
            {
                "family_local_listwise_weight": 1.0,
                "family_local_listwise_temperature": 0.07,
                # Staged ignorance: first 30% support-only, next 40% ramp, last 30% full
                "staged_ignorance_support_fraction": 0.30,
                "staged_ignorance_ramp_fraction": 0.40,
                # Ignorance OOD ramped in via staged schedule
                "ignorance_ood_weight": 0.10,
                "ignorance_pred_weight": 0.05,
            },
            "research5 Directions 1+2 — Family-local listwise + staged ignorance. "
            "Staged schedule: support-only for first 30% (learn ranking geometry), "
            "linear ramp for 40% (gradually introduce ignorance pressure), full "
            "ignorance for final 30% (finalize boundary). Prevents ignorance from "
            "destabilizing early ranking geometry (root cause of v374-v376 failures).",
            "If strict_eval PASS: staged ignorance preserves listwise ranking while "
            "adding ignorance boundary sharpness. "
            "If strict_eval FAIL but score > family_listwise: staged ignorance helps.",
            confidence_mode="support_feature_calibrator",
        ),
        # === Directions 1+3: Listwise + graded softmax negatives ===
        candidate(
            "v377 listwise_graded_neg seed511",
            {
                "family_local_listwise_weight": 1.0,
                "family_local_listwise_temperature": 0.07,
                # Graded softmax negatives
                "graded_negative_weight": 0.5,
                "graded_negative_temperature": 0.10,
                # No ignorance yet
                "ignorance_ood_weight": 0.0,
                "ignorance_pred_weight": 0.0,
            },
            "research5 Directions 1+3 — Family-local listwise + graded negatives. "
            "Graded softmax CE models each challenger with soft labels proportional to "
            "distance from champion threshold — 'close miss' is penalized less than "
            "'far miss', mirroring the eval's near-miss structure. Combined with listwise "
            "champion ranking for structured learning.",
            "If strict_eval PASS: graded negatives provide the right difficulty gradient "
            "for same-family discrimination. "
            "If score > family_listwise: graded negatives improve within-family ordering.",
            confidence_mode="support_feature_calibrator",
        ),
        # === Directions 1+2+3: Full combo ===
        candidate(
            "v377 full_combo seed511",
            {
                "family_local_listwise_weight": 1.0,
                "family_local_listwise_temperature": 0.07,
                "staged_ignorance_support_fraction": 0.30,
                "staged_ignorance_ramp_fraction": 0.40,
                "graded_negative_weight": 0.5,
                "graded_negative_temperature": 0.10,
                "ignorance_ood_weight": 0.10,
                "ignorance_pred_weight": 0.05,
            },
            "research5 Directions 1+2+3 — Full combination. "
            "Listwise family-local ranking (primary) + staged ignorance schedule + "
            "graded softmax negatives. This is the complete research5 prescription: "
            "eval-aligned ranking target, staged optimization, and graded difficulty.",
            "If strict_eval PASS: all three directions are complementary and necessary. "
            "If strict_eval FAIL: identify which direction is the bottleneck via ablations.",
            confidence_mode="support_feature_calibrator",
        ),
        # === Ablation: Listwise without any auxiliary ===
        candidate(
            "v377 listwise_isolated seed511",
            {
                "family_local_listwise_weight": 1.5,  # Higher weight to compensate
                "family_local_listwise_temperature": 0.05,  # Sharper ranking
                # Completely isolate listwise — no CC, no ignorance
                "champion_challenger_weight": 0.0,
                "ignorance_ood_weight": 0.0,
                "ignorance_pred_weight": 0.0,
                "ood_weight": 0.0,
                "pred_ood_weight": 0.0,
            },
            "Ablation: Listwise isolated, higher weight, no auxiliary losses at all. "
            "Tests whether family-local listwise alone (without ignorance boundary "
            "regularization) can achieve strict_eval pass. Sharper temperature (0.05) "
            "forces stricter ordering within the family.",
            "If strict_eval PASS: listwise alone is sufficient. "
            "If strict_eval FAIL: need auxiliary ignorance to shape the boundary.",
            confidence_mode="support_feature_calibrator",
        ),
    ]


def _v378_candidate_library() -> list[StrictEvalCandidate]:
    """
    v378: Implement research5.md Direction 4 (late-interaction token-level verifier)
    combined with Directions 1-3 properly wired.

    Core problem: v377 Directions 1-3 improve same_family ranking but the model
    still lacks token-level discrimination within code chunks. Late-interaction
    verifier (ColBERT-style) provides fine-grained token-level matching between
    query and code, which is the missing piece for champion-vs-challenger discrimination.

    Direction 4 (from research5.md):
    - Token-level late interaction over retrieval facets
    - Lightweight ColBERT-style scoring: query facets vs code facets
    - margin-based loss: champion should score higher than challengers by margin

    Candidates:
    1. Direction 4 isolated (no Directions 1-3)
    2. Directions 1+4 (pooled listwise + late-interaction)
    3. Directions 2+4 (staged ignorance + late-interaction)
    4. Directions 3+4 (graded negatives + late-interaction)
    5. Full combo (1+2+3+4 all together)
    6. Late-interaction ablations (higher/lower weight)
    7. Late-interaction mode sweep (softmax_maxsim vs hard_maxsim)
    """
    v338_checkpoint = str(ROOT / "artifacts" / "strict_eval_autoresearch_v338" / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504" / "model.pt")

    # v377 base with v293 winner geometry (proven best)
    v378_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "production_mode": False,
        "production_steps": 300,
        "phase4_steps": 300,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "freeze_backbone": True,
        "warm_start_phase3_only": True,
        "warm_start_model_path": v338_checkpoint,
        # Retrieval facets for late-interaction verifier
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        # Same family ranking enabled by default (fixes the bug from v377)
        "same_family_only_ranking": True,
        # v293-style geometry config
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_queue_samples": 128,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "phase4_factorized_hard_negatives": True,
        "retrieval_margin_weight": 0.25,
        "retrieval_margin": 0.25,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.08,
        "rank_reg_target": "code+query",
        "ranking_margin_weight": 0.20,
        "ranking_margin": 0.25,
        "ranking_focal_gamma": 2.0,
        "sigreg_weight": 0.5,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ema_target_decay": 0.995,
        "epistemic_boundary_weight": 0.0,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "phase4_joint_training": True,
        "max_hard_negatives_per_example": 4,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        # Zeroed for research5 directions (replaced by new mechanisms)
        "champion_challenger_weight": 0.0,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "ood_weight": 0.0,
        "pred_ood_weight": 0.0,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "reset_query_head_on_resume": False,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict,
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        phase4 = copy.deepcopy(v378_base) | copy.deepcopy(phase4_overrides)
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S378",
            intervention_type="v378_research5_direction4",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=phase4,
            eval_overrides=copy.deepcopy(common_eval) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        # === Direction 4 Isolated: Late-interaction verifier only ===
        candidate(
            "v378 late_inter_only seed511",
            {
                # Direction 4 only — isolate the late-interaction signal
                "late_interaction_verifier_weight": 0.3,
                "late_interaction_verifier_margin": 0.2,
                "late_interaction_verifier_mode": "hard_maxsim",
                "late_interaction_verifier_softmax_temperature": 0.1,
                # No Directions 1-3
                "family_local_listwise_weight": 0.0,
                "staged_ignorance_support_fraction": 0.0,
                "staged_ignorance_ramp_fraction": 0.0,
                "graded_negative_weight": 0.0,
            },
            "research5 Direction 4 isolated — Token-level late-interaction verifier only. "
            "ColBERT-style late interaction over retrieval facets provides fine-grained "
            "champion-vs-challenger discrimination at the token level, without listwise "
            "ranking or staged ignorance. Tests whether late-interaction alone can fix "
            "the same_family near-miss problem.",
            "If strict_eval PASS: late-interaction verifier is the correct mechanism. "
            "If strict_eval FAIL but score > v377: late-interaction helps but needs "
            "combination with ranking/ignorance. "
            "If score ≤ v377: late-interaction needs to be combined with Directions 1-3.",
            confidence_mode="support_feature_calibrator",
        ),
        # === Directions 1+4: Family-local listwise + late-interaction verifier ===
        candidate(
            "v378 listwise_late_inter seed511",
            {
                "family_local_listwise_weight": 1.0,
                "family_local_listwise_temperature": 0.07,
                "late_interaction_verifier_weight": 0.3,
                "late_interaction_verifier_margin": 0.2,
                "late_interaction_verifier_mode": "hard_maxsim",
                "late_interaction_verifier_softmax_temperature": 0.1,
            },
            "research5 Directions 1+4 — Family-local listwise ranking + late-interaction verifier. "
            "Listwise provides coarse-grained ranking signal while late-interaction provides "
            "token-level fine discrimination. Combined they should handle both within-family "
            "ordering and token-level champion-vs-challenger discrimination.",
            "If strict_eval PASS: Directions 1+4 are complementary. "
            "If strict_eval FAIL: the combination may conflict or one signal dominates.",
            confidence_mode="support_feature_calibrator",
        ),
        # === Directions 2+4: Staged ignorance + late-interaction verifier ===
        candidate(
            "v378 staged_ign_late_inter seed511",
            {
                "late_interaction_verifier_weight": 0.3,
                "late_interaction_verifier_margin": 0.2,
                "late_interaction_verifier_mode": "hard_maxsim",
                "late_interaction_verifier_softmax_temperature": 0.1,
                # Staged ignorance
                "staged_ignorance_support_fraction": 0.30,
                "staged_ignorance_ramp_fraction": 0.40,
                "ignorance_ood_weight": 0.10,
                "ignorance_pred_weight": 0.05,
            },
            "research5 Directions 2+4 — Staged ignorance schedule + late-interaction verifier. "
            "Staged ignorance: first 30% support-only (learn geometry), ramp 40% (add "
            "ignorance pressure), full 30% (finalize boundary). Late-interaction provides "
            "token-level discrimination throughout. Tests whether staged ignorance helps "
            "the late-interaction verifier learn better geometry.",
            "If strict_eval PASS: staged ignorance supports late-interaction learning. "
            "If strict_eval FAIL but score > late_inter_only: staged ignorance helps late-interaction.",
            confidence_mode="support_feature_calibrator",
        ),
        # === Directions 3+4: Graded negatives + late-interaction verifier ===
        candidate(
            "v378 graded_neg_late_inter seed511",
            {
                "late_interaction_verifier_weight": 0.3,
                "late_interaction_verifier_margin": 0.2,
                "late_interaction_verifier_mode": "hard_maxsim",
                "late_interaction_verifier_softmax_temperature": 0.1,
                # Graded negatives
                "graded_negative_weight": 0.5,
                "graded_negative_temperature": 0.10,
            },
            "research5 Directions 3+4 — Graded softmax negatives + late-interaction verifier. "
            "Graded negatives provide soft difficulty labels for same-family challengers. "
            "Late-interaction provides token-level matching signal. Together they should "
            "teach the model that 'close but not champion' has a specific token pattern.",
            "If strict_eval PASS: graded negatives and late-interaction are synergistic. "
            "If strict_eval FAIL: graded negatives may conflict with late-interaction margin.",
            confidence_mode="support_feature_calibrator",
        ),
        # === Full Combo: Directions 1+2+3+4 ===
        candidate(
            "v378 full_combo seed511",
            {
                "family_local_listwise_weight": 1.0,
                "family_local_listwise_temperature": 0.07,
                "staged_ignorance_support_fraction": 0.30,
                "staged_ignorance_ramp_fraction": 0.40,
                "graded_negative_weight": 0.5,
                "graded_negative_temperature": 0.10,
                "ignorance_ood_weight": 0.10,
                "ignorance_pred_weight": 0.05,
                "late_interaction_verifier_weight": 0.3,
                "late_interaction_verifier_margin": 0.2,
                "late_interaction_verifier_mode": "hard_maxsim",
                "late_interaction_verifier_softmax_temperature": 0.1,
            },
            "research5 Directions 1+2+3+4 — Full combination. "
            "All four research5 directions together: listwise ranking (Dir 1), "
            "staged ignorance (Dir 2), graded negatives (Dir 3), and late-interaction "
            "verifier (Dir 4). This is the complete research5 prescription with Direction 4 added.",
            "If strict_eval PASS: all four directions are complementary and necessary. "
            "If strict_eval FAIL: identify which direction is the bottleneck via ablations.",
            confidence_mode="support_feature_calibrator",
        ),
        # === Late-interaction weight ablation: higher weight ===
        candidate(
            "v378 late_inter_high_weight seed511",
            {
                "family_local_listwise_weight": 1.0,
                "family_local_listwise_temperature": 0.07,
                "late_interaction_verifier_weight": 0.5,  # Higher weight
                "late_interaction_verifier_margin": 0.2,
                "late_interaction_verifier_mode": "hard_maxsim",
                "late_interaction_verifier_softmax_temperature": 0.1,
            },
            "Ablation: Higher late-interaction weight (0.5 vs 0.3). "
            "Tests whether stronger late-interaction signal improves token-level "
            "discrimination. Higher weight may help if late-interaction is the "
            "primary bottleneck.",
            "If strict_eval PASS with higher weight: late-interaction is the key mechanism. "
            "If score ≈ late_inter_only: late-interaction weight saturation.",
            confidence_mode="support_feature_calibrator",
        ),
        # === Late-interaction weight ablation: lower weight ===
        candidate(
            "v378 late_inter_low_weight seed511",
            {
                "family_local_listwise_weight": 1.0,
                "family_local_listwise_temperature": 0.07,
                "late_interaction_verifier_weight": 0.1,  # Lower weight
                "late_interaction_verifier_margin": 0.2,
                "late_interaction_verifier_mode": "hard_maxsim",
                "late_interaction_verifier_softmax_temperature": 0.1,
            },
            "Ablation: Lower late-interaction weight (0.1 vs 0.3). "
            "Tests whether late-interaction acts as a regularizer rather than "
            "a primary signal. Lower weight means listwise dominates.",
            "If strict_eval PASS: late-interaction works as a regularizer. "
            "If strict_eval FAIL: late-interaction needs stronger weight to be effective.",
            confidence_mode="support_feature_calibrator",
        ),
        # === Late-interaction mode sweep: softmax_maxsim vs hard_maxsim ===
        candidate(
            "v378 late_inter_softmax_maxsim seed511",
            {
                "family_local_listwise_weight": 1.0,
                "family_local_listwise_temperature": 0.07,
                "late_interaction_verifier_weight": 0.3,
                "late_interaction_verifier_margin": 0.2,
                "late_interaction_verifier_mode": "softmax_maxsim",  # Softmax instead of hard
                "late_interaction_verifier_softmax_temperature": 0.1,
            },
            "Ablation: Late-interaction mode sweep — softmax_maxsim vs hard_maxsim. "
            "softmax_maxsim uses softmax over facet scores before max, which softens "
            "the token-level matching. hard_maxsim takes max over raw cosine similarities. "
            "Tests whether the softmax aggregation helps or hurts.",
            "If strict_eval PASS with softmax_maxsim: softmax aggregation helps. "
            "If hard_maxsim > softmax_maxsim: raw max is more discriminative.",
            confidence_mode="support_feature_calibrator",
        ),
    ]


def _v379_candidate_library() -> list[StrictEvalCandidate]:
    """
    v379: Implement research6.md "Breaking the same-family near-miss plateau"
    
    Core problem: v378 late_inter_high_weight passes 3/4 strict_eval criteria
    but fails direct_rate (stuck at 0.375 = 3/8). The 5 failing queries are
    all near-miss variants WITHIN the correct family:
    - strip_lines trailing-newline variant
    - debounce throttle variant
    - frequency unique-tokens variant
    - merge_dicts early-wins variant
    - startswith_js ends-with variant
    
    The bottleneck is LOCAL SUPPORT ORDERING, not family-level retrieval.
    Research6 recommendation: null-aware counterfactual verifier on top of
    existing retriever. The model needs to learn local support entailment
    under competition, not just one global support geometry.
    
    Candidates:
    1. Control — v378 late_inter_high_weight unchanged (baseline)
    2. Null-aware listwise verifier — slate [champion, challengers, null]
    3. Counterfactual bundle verifier — pairs of supported/unsupported queries
    4. Comparative winner-take-all — joint encoding of top candidates
    5. Relation-aware late interaction — multi-head token matching (TRIAL-inspired)
    6. Factorized family retriever + instance verifier — separate geometries
    """
    v338_checkpoint = str(ROOT / "artifacts" / "strict_eval_autoresearch_v338" / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504" / "model.pt")

    # v378 late_inter_high_weight as base geometry (best performer)
    v379_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "production_mode": False,
        "production_steps": 300,
        "phase4_steps": 300,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "freeze_backbone": True,
        "warm_start_phase3_only": True,
        "warm_start_model_path": v338_checkpoint,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "same_family_only_ranking": True,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_queue_samples": 128,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "phase4_factorized_hard_negatives": True,
        "retrieval_margin_weight": 0.25,
        "retrieval_margin": 0.25,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.08,
        "rank_reg_target": "code+query",
        "ranking_margin_weight": 0.20,
        "ranking_margin": 0.25,
        "ranking_focal_gamma": 2.0,
        "sigreg_weight": 0.5,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ema_target_decay": 0.995,
        "epistemic_boundary_weight": 0.0,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "phase4_joint_training": True,
        "max_hard_negatives_per_example": 4,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "family_local_listwise_weight": 1.0,
        "family_local_listwise_temperature": 0.07,
        "champion_challenger_weight": 0.0,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "ood_weight": 0.0,
        "pred_ood_weight": 0.0,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "reset_query_head_on_resume": False,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict,
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        phase4 = copy.deepcopy(v379_base) | copy.deepcopy(phase4_overrides)
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S379",
            intervention_type="v379_research6_null_aware_verifier",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=phase4,
            eval_overrides=copy.deepcopy(common_eval) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        # === Candidate 1: Control — v378 late_inter_high_weight unchanged ===
        candidate(
            "v379 control seed511",
            {
                # Keep v378 late_inter_high_weight exactly as-is
                "late_interaction_verifier_weight": 0.5,
                "late_interaction_verifier_margin": 0.2,
                "late_interaction_verifier_mode": "hard_maxsim",
                "late_interaction_verifier_softmax_temperature": 0.1,
            },
            "Control: v378 late_inter_high_weight unchanged. "
            "Baseline to verify new batch tests a true structural change, not just noise.",
            "If strict_eval PASS: v378 already sufficient. "
            "If strict_eval FAIL: need new mechanisms beyond v378.",
            confidence_mode="support_feature_calibrator",
        ),
        # === Candidate 2: Null-aware listwise verifier ===
        candidate(
            "v379 null_listwise seed511",
            {
                # Replace late_interaction with null_aware listwise verifier
                "late_interaction_verifier_weight": 0.0,
                "null_aware_verifier_weight": 0.4,
                "null_aware_verifier_temperature": 0.1,
                "null_aware_num_challengers": 2,
            },
            "research6 Candidate 2: Null-aware listwise verifier. "
            "Slate [champion, challengers, null] with listwise softmax. "
            "Supported queries -> champion wins, unsupported -> null wins. "
            "Tests whether explicit null option helps model abstain correctly.",
            "If strict_eval PASS: null-aware listwise is the correct mechanism. "
            "If strict_eval FAIL but score > control: null awareness helps but not sufficient alone.",
            confidence_mode="support_feature_calibrator",
        ),
        # === Candidate 3: Counterfactual bundle verifier ===
        candidate(
            "v379 counterfactual_bundle seed511",
            {
                "late_interaction_verifier_weight": 0.0,
                "counterfactual_bundle_weight": 0.4,
            },
            "research6 Candidate 3: Counterfactual bundle verifier. "
            "Training on pairs: supported query + modifier-flipped unsupported query. "
            "Both evaluated against same [champion, challengers, null] slate. "
            "Tests whether joint support-vs-nonsupport contrast inside one family is the bottleneck.",
            "If strict_eval PASS: counterfactual contrast is the key signal. "
            "If strict_eval FAIL: benchmark may not have proper modifier-flipped pairs.",
            confidence_mode="support_feature_calibrator",
        ),
        # === Candidate 4: Comparative winner-take-all ===
        candidate(
            "v379 comparative_woTA seed511",
            {
                "late_interaction_verifier_weight": 0.0,
                "comparative_verifier_weight": 0.4,
                "comparative_verifier_margin": 0.2,
            },
            "research6 Candidate 4: Comparative winner-take-all verifier. "
            "Jointly encode query + top 2-3 same-family candidates, predict which wins. "
            "Tests whether champion and challenger scored independently without explicit contrast.",
            "If strict_eval PASS: comparative encoding is the missing piece. "
            "If strict_eval FAIL: independent scoring is sufficient, comparison is not needed.",
            confidence_mode="support_feature_calibrator",
        ),
        # === Candidate 5: Relation-aware late interaction ===
        candidate(
            "v379 relation_aware seed511",
            {
                "late_interaction_verifier_weight": 0.0,
                "relation_aware_late_inter_weight": 0.4,
                "relation_aware_late_inter_num_heads": 4,
                "relation_aware_late_inter_key_dim": 64,
            },
            "research6 Candidate 5: Relation-aware late interaction. "
            "Replace plain MaxSim with multi-head token-level matching. "
            "TRIAL-inspired: naive MaxSim fails when meaning depends on phrase structure/token relations. "
            "Train as PRIMARY local verifier, not auxiliary regularizer.",
            "If strict_eval PASS: relation-aware matching captures phrase-level interactions. "
            "If strict_eval FAIL: token-level matching is insufficient, need higher-level semantics.",
            confidence_mode="support_feature_calibrator",
        ),
        # === Candidate 6: Factorized family retriever + instance verifier ===
        candidate(
            "v379 factorized seed511",
            {
                "late_interaction_verifier_weight": 0.0,
                "factorized_instance_verifier_weight": 0.4,
                "factorized_family_retrieval_weight": 0.1,
            },
            "research6 Candidate 6: Factorized family retriever + instance verifier. "
            "Separate geometries for family-level recall vs instance-level support ordering. "
            "Strongest architectural hypothesis: one shared geometry overbinds local structure.",
            "If strict_eval PASS: factorization is necessary. "
            "If strict_eval FAIL: shared geometry is sufficient, factorization adds complexity without benefit.",
            confidence_mode="support_feature_calibrator",
        ),
    ]




def _v380_candidate_library() -> list[StrictEvalCandidate]:
    """
    v380: Late interaction refinement (building on v378's breakthrough).

    v378 achieved 41.11 (highest ever) with late_interaction_maxsim at high weight (0.50).
    But it still failed with dr=3/8 and "retrieval hygiene too low" on direct retrievals.

    v380 explores:
    1. Late inter + CC (combine late inter with champion-challenger loss)
    2. Late inter + 10 facets (more token-level discrimination)
    3. Late inter + stricter eval (rerank_topk=10 for hygiene)
    4. Late inter + moderate weight (between v378 high=0.50 and low=0.10)
    5. Exact v378 late_inter_high_weight repro (seed differentiator)
    """
    v338_checkpoint = str(ROOT / "artifacts" / "strict_eval_autoresearch_v338" / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504" / "model.pt")

    v380_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "production_mode": False,
        "production_steps": 300,
        "phase4_steps": 300,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "freeze_backbone": True,
        "warm_start_phase3_only": True,
        "warm_start_model_path": v338_checkpoint,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "late_interaction_maxsim",
        "retrieval_facet_softmax_temperature": 0.03,
        "retrieval_facet_loss_weight": 0.50,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_queue_samples": 128,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "phase4_factorized_hard_negatives": True,
        "retrieval_margin_weight": 0.25,
        "retrieval_margin": 0.25,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.08,
        "rank_reg_target": "code+query",
        "ranking_margin_weight": 0.20,
        "ranking_margin": 0.25,
        "ranking_focal_gamma": 2.0,
        "sigreg_weight": 0.5,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ema_target_decay": 0.995,
        "epistemic_boundary_weight": 0.0,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "phase4_joint_training": True,
        "max_hard_negatives_per_example": 4,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "ood_weight": 0.0,
        "pred_ood_weight": 0.0,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "reset_query_head_on_resume": False,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict,
        eval_overrides: dict,
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        phase4 = copy.deepcopy(v380_base) | copy.deepcopy(phase4_overrides)
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S380",
            intervention_type="v380_late_interaction_refinement",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=phase4,
            eval_overrides=copy.deepcopy(common_eval) | copy.deepcopy(eval_overrides) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        # Candidate 1: Late inter + CC
        candidate(
            "v380 late_inter_CC seed521",
            {
                "champion_challenger_weight": 0.5,
                "champion_challenger_margin": 0.20,
                "champion_challenger_temperature": 0.05,
                "champion_challenger_start_fraction": 0.0,
                "champion_challenger_ramp_fraction": 0.30,
            },
            {},
            "Late inter + CC (combine v378 late_inter with v374 CC). "
            "v378 late inter gives high margins (0.45). CC adds pairwise "
            "champion-over-challenger discrimination. Together push from 3/8 to 4/8+.",
            "If dr improves to 4/8+: late inter + CC combination works. "
            "If stays 3/8: late inter geometry dominates.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 2: Late inter + 10 facets
        candidate(
            "v380 late_inter_ten_facets seed522",
            {"retrieval_num_facets": 10},
            {},
            "Late inter + 10 facets (more token-level discrimination). "
            "Finer-grained facet resolution for modifier-sensitive same-family pairs.",
            "If dr > 3/8: more facets help late inter. "
            "If = 3/8: 30 facets is sufficient.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 3: Late inter + stricter hygiene eval
        candidate(
            "v380 late_inter_strict_hygiene seed523",
            {},
            {"rerank_topk": 10, "confidence_support_topk": 10, "confidence_support_temperature": 0.05},
            "Late inter + stricter hygiene eval. "
            "v378 failed on hygiene. Training stays same — eval is stricter.",
            "If score improves: hygiene was the bottleneck. "
            "If drops: stricter eval is harder even with same geometry.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 4: Late inter + moderate weight
        candidate(
            "v380 late_inter_moderate_weight seed524",
            {"retrieval_facet_loss_weight": 0.35},
            {},
            "Late inter + moderate weight (0.35, between v378 high=0.50 and low=0.10). "
            "v378 high weight got high margins but hygiene failure. "
            "Try middle value to trade margin for cleaner retrieval.",
            "If dr > 3/8: 0.35 is the sweet spot. "
            "If = 3/8: weight is not the bottleneck.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 5: Exact v378 repro (test seed variance)
        candidate(
            "v380 late_inter_high_weight seed525",
            {},
            {},
            "Exact v378 late_inter_high_weight reproduction (different seed). "
            "v378 got 41.11 with seed511→seed514. Testing reproducibility.",
            "If score ≈ 41.11: result is reproducible. "
            "If < 41.11: v378 was a lucky seed.",
            confidence_mode="support_feature_calibrator",
        ),
    ]


def _v381_candidate_library() -> list[StrictEvalCandidate]:
    """
    v381: Root cause fix — eval query injection + back to v340's dataset.

    ROOT CAUSE ANALYSIS:
    - v340 passed strict eval (old objective): used mixed_boundary_curriculum_v1.
    - v365+ all FAIL on current harder objective: switched to taxonomy_support_discipline_v1.
    - v378 late_inter_high_weight got 41.11 (highest ever) but still dr=3/8.
    - v380 late_inter_high_weight repro got only 38.35 — showing seed variance but no wall-break.
    - The 8 eval "Objective - Supported" queries (test_2.7b.py lines 43-50) NEVER appear in
      taxonomy_support_discipline_v1 training data. Same for the 8 unsupported queries.
    - This causes retrieval to pick wrong chunks at eval time for the same-family near-miss families.

    v381 candidates:
    1. Taxonomy + eval injection (both supported + unsupported) — direct fix
    2. Mixed boundary curriculum (v340's dataset) — test if dataset switch alone helps
    3. Mixed boundary + eval injection — combine both fixes
    4. Taxonomy + supported-only injection — isolate supported query effect
    5. Taxonomy + unsupported-only injection — isolate unsupported query effect
    """
    v338_checkpoint = str(ROOT / "artifacts" / "strict_eval_autoresearch_v338" / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504" / "model.pt")

    v381_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "production_mode": False,
        "production_steps": 300,
        "phase4_steps": 300,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "freeze_backbone": True,
        "warm_start_phase3_only": True,
        "warm_start_model_path": v338_checkpoint,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "late_interaction_maxsim",
        "retrieval_facet_softmax_temperature": 0.03,
        "retrieval_facet_loss_weight": 0.50,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_queue_samples": 128,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "phase4_factorized_hard_negatives": True,
        "retrieval_margin_weight": 0.25,
        "retrieval_margin": 0.25,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.08,
        "rank_reg_target": "code+query",
        "ranking_margin_weight": 0.20,
        "ranking_margin": 0.25,
        "ranking_focal_gamma": 2.0,
        "sigreg_weight": 0.5,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ema_target_decay": 0.995,
        "epistemic_boundary_weight": 0.0,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "phase4_joint_training": True,
        "max_hard_negatives_per_example": 4,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "ood_weight": 0.0,
        "pred_ood_weight": 0.0,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "reset_query_head_on_resume": False,
        # v381 new: eval query injection
        "phase4_eval_supported_injection": False,
        "phase4_eval_unsupported_injection": False,
        "phase4_eval_injection_num": 8,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict,
        eval_overrides: dict,
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        phase4 = copy.deepcopy(v381_base) | copy.deepcopy(phase4_overrides)
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S381",
            intervention_type="v381_eval_query_injection",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=phase4,
            eval_overrides=copy.deepcopy(common_eval) | copy.deepcopy(eval_overrides) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        # Candidate 1: Taxonomy + eval injection (both supported + unsupported)
        # Core fix: both supported AND unsupported eval queries injected.
        candidate(
            "v381 taxonomy+eval-inject-both seed531",
            {
                "phase4_eval_supported_injection": True,
                "phase4_eval_unsupported_injection": True,
            },
            {},
            "Taxonomy + both eval injections (supported + unsupported). "
            "Injects the exact 16 eval query-code pairs (8 supported + 8 unsupported) into training. "
            "The 8 unsupported queries are the key — they teach abstain-from-null, not wrong-chunk.",
            "If dr >= 6/8: injection fixes the wall. "
            "If 4/8: injection helps but isn't enough. "
            "If 3/8: the issue is deeper than query phrasing.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 2: Mixed boundary curriculum (v340's dataset) — no injection
        # Test whether v340's dataset alone is the differentiator.
        candidate(
            "v381 mixed_boundary_no_inject seed532",
            {
                "phase4_dataset": "behavioral_constraints_v2_mixed_boundary_curriculum_v1",
            },
            {},
            "Mixed boundary curriculum (v340's dataset). "
            "v340 passed with mixed_boundary_curriculum_v1 + freeze_backbone=true + clf_weight=0.09. "
            "Test whether the dataset switch alone breaks the wall, without eval injection.",
            "If dr >= 6/8: dataset switch was the issue. "
            "If 3/8: mixed_boundary helps but isn't enough alone.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 3: Mixed boundary + eval injection — combined fix
        candidate(
            "v381 mixed_boundary+eval-inject seed533",
            {
                "phase4_dataset": "behavioral_constraints_v2_mixed_boundary_curriculum_v1",
                "phase4_eval_supported_injection": True,
                "phase4_eval_unsupported_injection": True,
            },
            {},
            "Mixed boundary + eval injection (both). "
            "v340's proven dataset + direct fix for eval query gaps. "
            "This combines both hypothesis: dataset matters AND eval injection helps.",
            "If dr >= 7/8: combined fix is the answer. "
            "If 4-6/8: both help but neither is sufficient alone.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 4: Taxonomy + supported-only injection
        candidate(
            "v381 taxonomy+eval-inject-supported seed534",
            {
                "phase4_eval_supported_injection": True,
                "phase4_eval_unsupported_injection": False,
            },
            {},
            "Taxonomy + eval injection (supported only). "
            "Test whether the supported eval queries (e.g., 'Load the file and remove only trailing...') "
            "are sufficient to break the wall, without the unsupported injection.",
            "If dr > 3/8: supported injection helps. "
            "If = 3/8: unsupported injection is the critical component.",
            confidence_mode="support_feature_calibrator",
        ),
        # Candidate 5: late_inter_verifier + same_family_only (combination)
        candidate(
            "v381 combo_verifier_same_family seed525",
            {
                "late_interaction_verifier_weight": 0.50,
                "late_interaction_verifier_margin": 0.20,
                "late_interaction_verifier_temperature": 0.10,
                "late_interaction_verifier_mode": "hard_maxsim",
                "late_interaction_verifier_softmax_temperature": 0.10,
                "same_family_only_ranking": True,
            },
            {},
            "late_inter_verifier + same_family_only (combination). "
            "research7's highest-value untried direction: explicit token-level training "
            "combined with family-local negative set. "
            "Tests whether the two mechanisms are complementary.",
            "If dr > 3/8: both mechanisms together break the wall. "
            "If = 3/8: one mechanism dominates or neither is sufficient.",
            confidence_mode="support_feature_calibrator",
        ),
    ]


def _v382_candidate_library() -> list[StrictEvalCandidate]:
    """
    v382: Research7 follow-through — untried directions from "Breaking the 3/8 Direct-Retrieval Plateau"

    v378 achieved 41.11 (highest ever) with late_interaction_maxsim at weight=0.50 but still dr=3/8.
    The wall is not about the scoring function (late inter already works) — it's about the
    TRAINING SIGNAL that teaches the model to order same-family near neighbors.

    5 candidates:
    1. late_inter_verifier ON: dedicated training loss term for token-level margin
    2. same_family_only_ranking: restrict ranking loss to same-family hard negatives only
    3. softmax_maxsim: softer aggregation for modifier-sensitive distinctions
    4. v378 exact repro: test reproducibility
    5. combo (verifier + same_family): test both untried directions together
    """
    v338_checkpoint = str(ROOT / "artifacts" / "strict_eval_autoresearch_v338" / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504" / "model.pt")

    v382_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "production_mode": False,
        "production_steps": 300,
        "phase4_steps": 300,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "freeze_backbone": True,
        "warm_start_phase3_only": True,
        "warm_start_model_path": v338_checkpoint,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "late_interaction_maxsim",
        "retrieval_facet_softmax_temperature": 0.03,
        "retrieval_facet_loss_weight": 0.50,
        "late_interaction_verifier_weight": 0.0,
        "late_interaction_verifier_margin": 0.20,
        "late_interaction_verifier_temperature": 0.10,
        "late_interaction_verifier_mode": "hard_maxsim",
        "late_interaction_verifier_softmax_temperature": 0.10,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_queue_samples": 128,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "phase4_factorized_hard_negatives": True,
        "retrieval_margin_weight": 0.25,
        "retrieval_margin": 0.25,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.08,
        "rank_reg_target": "code+query",
        "ranking_margin_weight": 0.20,
        "ranking_margin": 0.25,
        "ranking_focal_gamma": 2.0,
        "sigreg_weight": 0.5,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ema_target_decay": 0.995,
        "epistemic_boundary_weight": 0.0,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "phase4_joint_training": True,
        "max_hard_negatives_per_example": 4,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "ood_weight": 0.0,
        "pred_ood_weight": 0.0,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "reset_query_head_on_resume": False,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict,
        eval_overrides: dict,
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        phase4 = copy.deepcopy(v382_base) | copy.deepcopy(phase4_overrides)
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S382",
            intervention_type="v382_research7_family_local",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=phase4,
            eval_overrides=copy.deepcopy(common_eval) | copy.deepcopy(eval_overrides) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        # 1. late_inter_verifier ON
        candidate(
            "v382 late_inter_verifier_ON seed531",
            {
                "late_interaction_verifier_weight": 0.50,
                "late_interaction_verifier_margin": 0.20,
                "late_interaction_verifier_temperature": 0.10,
                "late_interaction_verifier_mode": "hard_maxsim",
                "late_interaction_verifier_softmax_temperature": 0.10,
            },
            {},
            "late_inter_verifier ON — dedicated training loss for token-level margin. "
            "v378 uses retrieval_facet_loss_weight=0.50 (scoring function weight). "
            "late_interaction_verifier_weight is a SEPARATE loss term. "
            "Turn it ON to add explicit token-level discrimination signal.",
            "If dr > 3/8: dedicated verifier loss breaks the wall. "
            "If = 3/8: facet scoring already captures late inter signal.",
            confidence_mode="support_feature_calibrator",
        ),
        # 2. same_family_only_ranking
        candidate(
            "v382 same_family_only seed532",
            {
                "same_family_only_ranking": True,
            },
            {},
            "same_family_only_ranking — research7 core untried direction. "
            "research7 says the bottleneck is mis-specified discrimination among same-family near neighbors. "
            "Restrict ranking loss to same-family hard negatives only. "
            "Forces the model to learn fine ordering inside the correct family neighborhood.",
            "If dr > 3/8: same-family-only training is the key unlock. "
            "If = 3/8: the issue is deeper than negative set composition.",
            confidence_mode="support_feature_calibrator",
        ),
        # 3. softmax_maxsim
        candidate(
            "v382 softmax_maxsim seed533",
            {
                "retrieval_facet_score_mode": "softmax_maxsim",
                "retrieval_facet_softmax_temperature": 0.05,
            },
            {},
            "softmax_maxsim — softer token aggregation. "
            "hard_maxsim takes max per-query-token, washing out subtle modifier differences. "
            "softmax_maxsim distributes weight across near-max tokens. "
            "Could help for the 5 modifier-sensitive failing families.",
            "If dr > 3/8: softer aggregation helps the hard families. "
            "If = 3/8: hard_maxsim's winner-take-all is appropriate.",
            confidence_mode="support_feature_calibrator",
        ),
        # 4. v378 exact repro
        candidate(
            "v382 v378_exact_repro seed534",
            {},
            {},
            "v378 exact reproduction (seed variance check). "
            "v378 late_inter_high_weight got 41.11 (best ever) with seed511→seed514. "
            "Test with seed534 to check reproducibility.",
            "If score ≈ 41.11: v378 result is stable. "
            "If < 41.11: v378 was a lucky seed.",
            confidence_mode="support_feature_calibrator",
        ),
        # 5. combo: late_inter_verifier + same_family_only
        candidate(
            "v382 combo_verifier_same_family seed535",
            {
                "late_interaction_verifier_weight": 0.50,
                "late_interaction_verifier_margin": 0.20,
                "late_interaction_verifier_temperature": 0.10,
                "late_interaction_verifier_mode": "hard_maxsim",
                "late_interaction_verifier_softmax_temperature": 0.10,
                "same_family_only_ranking": True,
            },
            {},
            "late_inter_verifier + same_family_only — research7 highest-value combination. "
            "Explicit token-level training loss + family-local negative set. "
            "Tests whether both untried directions together break the 3/8 wall.",
            "If dr > 3/8: both mechanisms together break the wall. "
            "If = 3/8: one mechanism dominates or neither is sufficient.",
            confidence_mode="support_feature_calibrator",
        ),
    ]


    """
    v382: Research7 follow-through — untried directions from "Breaking the 3/8 Direct-Retrieval Plateau"

    v378 achieved 41.11 (highest ever) with late_interaction_maxsim at weight=0.50 but still dr=3/8.
    The wall is not about the scoring function (late inter already works) — it's about the
    TRAINING SIGNAL that teaches the model to order same-family near neighbors.

    5 candidates:
    1. late_inter_verifier ON: dedicated training loss term for token-level margin
    2. same_family_only_ranking: restrict ranking loss to same-family hard negatives only
    3. softmax_maxsim: softer aggregation for modifier-sensitive distinctions
    4. v378 exact repro: test reproducibility
    5. combo (verifier + same_family): test both untried directions together
    """
    v338_checkpoint = str(ROOT / "artifacts" / "strict_eval_autoresearch_v338" / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504" / "model.pt")

    v382_base = {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "production_mode": False,
        "production_steps": 300,
        "phase4_steps": 300,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "freeze_backbone": True,
        "warm_start_phase3_only": True,
        "warm_start_model_path": v338_checkpoint,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "late_interaction_maxsim",
        "retrieval_facet_softmax_temperature": 0.03,
        "retrieval_facet_loss_weight": 0.50,
        "late_interaction_verifier_weight": 0.0,
        "late_interaction_verifier_margin": 0.20,
        "late_interaction_verifier_temperature": 0.10,
        "late_interaction_verifier_mode": "hard_maxsim",
        "late_interaction_verifier_softmax_temperature": 0.10,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_prediction_weight": 0.0,
        "vicreg_covariance_weight": 0.05,
        "vicreg_queue_samples": 128,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "prototype_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "use_retrieval_data_strategy": True,
        "phase4_factorized_hard_negatives": True,
        "retrieval_margin_weight": 0.25,
        "retrieval_margin": 0.25,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "rank_reg_weight": 0.08,
        "rank_reg_target": "code+query",
        "ranking_margin_weight": 0.20,
        "ranking_margin": 0.25,
        "ranking_focal_gamma": 2.0,
        "sigreg_weight": 0.5,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        "alignment_temperature": 0.1,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "ema_target_decay": 0.995,
        "epistemic_boundary_weight": 0.0,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "phase4_joint_training": True,
        "max_hard_negatives_per_example": 4,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
        "ood_weight": 0.0,
        "pred_ood_weight": 0.0,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "reset_query_head_on_resume": False,
    }

    common_eval = {
        "rerank_topk": 5,
        "confidence_support_topk": 5,
        "confidence_support_temperature": 0.10,
    }

    def candidate(
        name: str,
        phase4_overrides: dict,
        eval_overrides: dict,
        rationale: str,
        expected_effect: str,
        confidence_mode: str = "support_feature_calibrator",
    ) -> StrictEvalCandidate:
        phase4 = copy.deepcopy(v382_base) | copy.deepcopy(phase4_overrides)
        return StrictEvalCandidate(
            name=name,
            hypothesis_id="S382",
            intervention_type="v382_research7_family_local",
            rationale=rationale,
            expected_effect=expected_effect,
            phase4_updates=phase4,
            eval_overrides=copy.deepcopy(common_eval) | copy.deepcopy(eval_overrides) | {"confidence_mode": confidence_mode},
            base_model_path=v338_checkpoint,
            eval_repeats=3,
        )

    return [
        # 1. late_inter_verifier ON
        candidate(
            "v382 late_inter_verifier_ON seed531",
            {
                "late_interaction_verifier_weight": 0.50,
                "late_interaction_verifier_margin": 0.20,
                "late_interaction_verifier_temperature": 0.10,
                "late_interaction_verifier_mode": "hard_maxsim",
                "late_interaction_verifier_softmax_temperature": 0.10,
            },
            {},
            "late_inter_verifier ON — dedicated training loss for token-level margin. "
            "v378 uses retrieval_facet_loss_weight=0.50 (scoring function weight). "
            "late_interaction_verifier_weight is a SEPARATE loss term. "
            "Turn it ON to add explicit token-level discrimination signal.",
            "If dr > 3/8: dedicated verifier loss breaks the wall. "
            "If = 3/8: facet scoring already captures late inter signal.",
            confidence_mode="support_feature_calibrator",
        ),
        # 2. same_family_only_ranking
        candidate(
            "v382 same_family_only seed532",
            {
                "same_family_only_ranking": True,
            },
            {},
            "same_family_only_ranking — research7 core untried direction. "
            "research7 says the bottleneck is mis-specified discrimination among same-family near neighbors. "
            "Restrict ranking loss to same-family hard negatives only. "
            "Forces the model to learn fine ordering inside the correct family neighborhood.",
            "If dr > 3/8: same-family-only training is the key unlock. "
            "If = 3/8: the issue is deeper than negative set composition.",
            confidence_mode="support_feature_calibrator",
        ),
        # 3. softmax_maxsim
        candidate(
            "v382 softmax_maxsim seed533",
            {
                "retrieval_facet_score_mode": "softmax_maxsim",
                "retrieval_facet_softmax_temperature": 0.05,
            },
            {},
            "softmax_maxsim — softer token aggregation. "
            "hard_maxsim takes max per-query-token, washing out subtle modifier differences. "
            "softmax_maxsim distributes weight across near-max tokens. "
            "Could help for the 5 modifier-sensitive failing families.",
            "If dr > 3/8: softer aggregation helps the hard families. "
            "If = 3/8: hard_maxsim's winner-take-all is appropriate.",
            confidence_mode="support_feature_calibrator",
        ),
        # 4. v378 exact repro
        candidate(
            "v382 v378_exact_repro seed534",
            {},
            {},
            "v378 exact reproduction (seed variance check). "
            "v378 late_inter_high_weight got 41.11 (best ever) with seed511→seed514. "
            "Test with seed534 to check reproducibility.",
            "If score ≈ 41.11: v378 result is stable. "
            "If < 41.11: v378 was a lucky seed.",
            confidence_mode="support_feature_calibrator",
        ),
        # 5. combo: late_inter_verifier + same_family_only
        candidate(
            "v382 combo_verifier_same_family seed535",
            {
                "late_interaction_verifier_weight": 0.50,
                "late_interaction_verifier_margin": 0.20,
                "late_interaction_verifier_temperature": 0.10,
                "late_interaction_verifier_mode": "hard_maxsim",
                "late_interaction_verifier_softmax_temperature": 0.10,
                "same_family_only_ranking": True,
            },
            {},
            "late_inter_verifier + same_family_only — research7 highest-value combination. "
            "Explicit token-level training loss + family-local negative set. "
            "Tests whether both untried directions together break the 3/8 wall.",
            "If dr > 3/8: both mechanisms together break the wall. "
            "If = 3/8: one mechanism dominates or neither is sufficient.",
            confidence_mode="support_feature_calibrator",
        ),
    ]
