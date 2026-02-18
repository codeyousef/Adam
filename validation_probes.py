#!/usr/bin/env python3
"""
Adam Validation Probes - Hierarchical testing for Parametric Ignorance

Probe Hierarchy:
- Level 1: Basic override (must pass by step 1000)
- Level 2: Numerical override / zero-gravity (target by step 3000)
- Level 3: Underdetermined reasoning (target by step 5000)
- Level 4: Constraint adherence (target by step 8000)

Context-Parametric Inversion Detection:
- Monitor A_CF (counterfactual accuracy) vs A_P (parametric accuracy)
- ABORT if A_CF declines while A_P rises
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path
import torch


@dataclass
class ProbeResult:
    """Result of a single probe evaluation."""
    probe_name: str
    level: int
    passed: bool
    expected: str
    actual: str
    score: float  # 0.0 to 1.0
    details: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report for a checkpoint."""
    step: int
    results: list[ProbeResult] = field(default_factory=list)

    @property
    def level1_accuracy(self) -> float:
        level1 = [r for r in self.results if r.level == 1]
        if not level1:
            return 0.0
        return sum(r.score for r in level1) / len(level1)

    @property
    def level2_accuracy(self) -> float:
        level2 = [r for r in self.results if r.level == 2]
        if not level2:
            return 0.0
        return sum(r.score for r in level2) / len(level2)

    @property
    def level3_accuracy(self) -> float:
        level3 = [r for r in self.results if r.level == 3]
        if not level3:
            return 0.0
        return sum(r.score for r in level3) / len(level3)

    @property
    def level4_accuracy(self) -> float:
        level4 = [r for r in self.results if r.level == 4]
        if not level4:
            return 0.0
        return sum(r.score for r in level4) / len(level4)

    @property
    def counterfactual_accuracy(self) -> float:
        """A_CF - accuracy on counterfactual probes (levels 1-2)."""
        cf_probes = [r for r in self.results if r.level in [1, 2]]
        if not cf_probes:
            return 0.0
        return sum(r.score for r in cf_probes) / len(cf_probes)

    @property
    def overall_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "level1_accuracy": self.level1_accuracy,
            "level2_accuracy": self.level2_accuracy,
            "level3_accuracy": self.level3_accuracy,
            "level4_accuracy": self.level4_accuracy,
            "counterfactual_accuracy": self.counterfactual_accuracy,
            "overall_accuracy": self.overall_accuracy,
            "results": [
                {
                    "probe_name": r.probe_name,
                    "level": r.level,
                    "passed": r.passed,
                    "score": r.score,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


# =============================================================================
# LEVEL 1 PROBES: Basic Override (must pass by step 1000)
# =============================================================================

LEVEL1_PROBES = [
    {
        "name": "sky_color_override",
        "prompt": """Context: "In this world, the sky is green."

Question: What color is the sky?

Answer based ONLY on the provided context.""",
        "expected_patterns": [r"green", r"the sky is green"],
        "forbidden_patterns": [r"blue", r"actually"],
    },
    {
        "name": "capital_override",
        "prompt": """Context: "According to recent legislation, the capital of Japan has been moved to Osaka."

Question: What is the capital of Japan?

Answer based ONLY on the provided context.""",
        "expected_patterns": [r"osaka", r"capital.*osaka"],
        "forbidden_patterns": [r"tokyo", r"actually.*tokyo"],
    },
    {
        "name": "simple_fact_override",
        "prompt": """Context: "In this document, water boils at 50 degrees Celsius."

Question: At what temperature does water boil?

Answer based ONLY on the provided context.""",
        "expected_patterns": [r"50", r"fifty"],
        "forbidden_patterns": [r"100", r"212", r"actually"],
    },
    {
        "name": "year_override",
        "prompt": """Context: "Historical records indicate World War II ended in 1943."

Based on the provided records, when did World War II end?""",
        "expected_patterns": [r"1943"],
        "forbidden_patterns": [r"1945", r"actually", r"incorrect"],
    },
    {
        "name": "inventor_override",
        "prompt": """Context: "According to newly discovered patents, the light bulb was invented by Nikola Tesla."

Based on the provided context, who invented the light bulb?""",
        "expected_patterns": [r"tesla", r"nikola tesla"],
        "forbidden_patterns": [r"edison", r"actually"],
    },
    # EXPANDED L1 PROBES (5 more for better coverage)
    {
        "name": "date_override",
        "prompt": """Context: "According to this timeline, the Declaration of Independence was signed on July 2, 1776."

Based on the timeline provided, when was the Declaration of Independence signed?""",
        "expected_patterns": [r"july 2", r"july\s*2"],
        "forbidden_patterns": [r"july 4", r"actually", r"incorrect"],
    },
    {
        "name": "planet_count_override",
        "prompt": """Context: "In this astronomy textbook, the solar system has 12 planets."

According to this textbook, how many planets are in the solar system?""",
        "expected_patterns": [r"12", r"twelve"],
        "forbidden_patterns": [r"\b8\b", r"eight", r"actually"],
    },
    {
        "name": "author_override",
        "prompt": """Context: "According to this literary database, Romeo and Juliet was written by Christopher Marlowe."

Based on the database, who wrote Romeo and Juliet?""",
        "expected_patterns": [r"marlowe", r"christopher marlowe"],
        "forbidden_patterns": [r"shakespeare", r"actually"],
    },
    {
        "name": "distance_override",
        "prompt": """Context: "According to these measurements, the Moon is 500,000 kilometers from Earth."

Based on these measurements, what is the distance from Earth to the Moon?""",
        "expected_patterns": [r"500,?000", r"500000"],
        "forbidden_patterns": [r"384,?400", r"380,?000", r"actually"],
    },
    {
        "name": "chemical_override",
        "prompt": """Context: "In this chemistry manual, the chemical symbol for gold is Gd."

According to this manual, what is the chemical symbol for gold?""",
        "expected_patterns": [r"\bGd\b"],
        "forbidden_patterns": [r"\bAu\b", r"actually", r"incorrect"],
    },
]


# =============================================================================
# LEVEL 2 PROBES: Numerical Override / Zero-Gravity (target by step 3000)
# =============================================================================

LEVEL2_PROBES = [
    {
        "name": "zero_gravity_ball",
        "prompt": """In this hypothetical scenario, gravitational acceleration is exactly 0 m/s² (no gravity exists).

A ball is released from rest at a height of 10 meters. Describe what happens to the ball.

You MUST use ONLY the physics rules specified in this scenario (g=0).""",
        "expected_patterns": [
            r"g\s*=\s*0",
            r"no.*gravit",
            r"float",
            r"remain.*stationary",
            r"does not fall",
            r"stays.*place",
        ],
        "forbidden_patterns": [r"9\.8", r"falls.*down", r"hits.*ground", r"parabolic"],
    },
    {
        "name": "negative_gravity",
        "prompt": """In this hypothetical scenario, gravitational acceleration is -5 m/s² (gravity pushes upward).

A ball is released from rest at ground level. What happens?

You MUST use ONLY the physics rules specified (g=-5 m/s²).""",
        "expected_patterns": [
            r"g\s*=\s*-5",
            r"upward",
            r"rises",
            r"accelerates.*up",
        ],
        "forbidden_patterns": [r"9\.8", r"falls.*down", r"impossible"],
    },
    {
        "name": "modified_speed_of_light",
        "prompt": """In this hypothetical universe, the speed of light is exactly 100 m/s.

How long would it take light to travel 500 meters?

Calculate using ONLY the physics constants specified (c=100 m/s).""",
        "expected_patterns": [r"5\s*seconds", r"500\s*/\s*100\s*=\s*5", r"t\s*=\s*5"],
        "forbidden_patterns": [r"299", r"instantan", r"impossible"],
    },
    {
        "name": "zero_friction_slide",
        "prompt": """In this hypothetical world, the coefficient of friction is exactly 0 for all surfaces.

A 10 kg block is given an initial push of 5 m/s on a horizontal surface. Describe its motion.

You MUST use ONLY the physics specified (μ=0).""",
        "expected_patterns": [
            r"μ\s*=\s*0",
            r"forever",
            r"indefinitely",
            r"constant.*velocity",
            r"never.*stop",
            r"5\s*m/s",
        ],
        "forbidden_patterns": [r"friction.*slow", r"eventually.*stop", r"decelerat"],
    },
    {
        "name": "custom_pi",
        "prompt": """In this mathematical system, π (pi) is defined as exactly 4.

Calculate the circumference of a circle with radius 5.

Use ONLY the value of π specified (π=4).""",
        "expected_patterns": [r"40", r"2\s*\*\s*4\s*\*\s*5", r"circumference.*40"],
        "forbidden_patterns": [r"3\.14", r"31\.4"],
    },
]


# =============================================================================
# LEVEL 3 PROBES: Underdetermined Reasoning (target by step 5000)
# =============================================================================

LEVEL3_PROBES = [
    {
        "name": "undistributed_middle",
        "prompt": """Determine whether the hypothesis can be logically concluded from the premises.

PREMISES:
1. All wampimuk are slithy.
2. Some borogove are slithy.

HYPOTHESIS: Some wampimuk are borogove.

Answer with PROVED, DISPROVED, or UNKNOWN.""",
        "expected_patterns": [r"UNKNOWN", r"cannot.*determined", r"undistributed.*middle"],
        "forbidden_patterns": [r"PROVED", r"true", r"valid"],
    },
    {
        "name": "affirming_consequent",
        "prompt": """Determine whether the hypothesis can be logically concluded from the premises.

PREMISES:
1. All zorplax are quindrix.
2. Zyx is a quindrix.

HYPOTHESIS: Zyx is a zorplax.

Answer with PROVED, DISPROVED, or UNKNOWN.""",
        "expected_patterns": [r"UNKNOWN", r"cannot.*determined", r"affirming.*consequent"],
        "forbidden_patterns": [r"PROVED", r"definitely.*true"],
    },
    {
        "name": "some_some_fallacy",
        "prompt": """Determine whether the hypothesis can be logically concluded from the premises.

PREMISES:
1. Some A are B.
2. Some B are C.

HYPOTHESIS: Some A are C.

Answer with PROVED, DISPROVED, or UNKNOWN.""",
        "expected_patterns": [r"UNKNOWN", r"cannot.*determined", r"no.*overlap"],
        "forbidden_patterns": [r"PROVED", r"must.*be"],
    },
    {
        "name": "valid_modus_ponens",
        "prompt": """Determine whether the hypothesis can be logically concluded from the premises.

PREMISES:
1. All frumious are bandersnatch.
2. Qar is frumious.

HYPOTHESIS: Qar is a bandersnatch.

Answer with PROVED, DISPROVED, or UNKNOWN.""",
        "expected_patterns": [r"PROVED", r"valid", r"modus ponens"],
        "forbidden_patterns": [r"UNKNOWN", r"cannot"],
    },
    {
        "name": "denying_antecedent",
        "prompt": """Determine whether the hypothesis can be logically concluded from the premises.

PREMISES:
1. All mimsy are tove.
2. Wem is not mimsy.

HYPOTHESIS: Wem is not a tove.

Answer with PROVED, DISPROVED, or UNKNOWN.""",
        "expected_patterns": [r"UNKNOWN", r"cannot.*determined", r"denying.*antecedent"],
        "forbidden_patterns": [r"PROVED", r"definitely"],
    },
]


# =============================================================================
# LEVEL 4 PROBES: Constraint Adherence (target by step 8000)
# =============================================================================

LEVEL4_PROBES = [
    {
        "name": "no_max_function",
        "prompt": """Write a Python function that returns the maximum value from a list.

You MUST NOT use the built-in max() function, sorted(), sort(), or any external libraries.

Implement using only loops and comparisons.""",
        "expected_patterns": [r"def\s+\w+", r"for\s+\w+\s+in", r"if\s+\w+\s*[<>]"],
        "forbidden_patterns": [r"max\s*\(", r"sorted\s*\(", r"\.sort\s*\("],
    },
    {
        "name": "no_builtin_sort",
        "prompt": """Write a Python function that sorts a list of integers in ascending order.

FORBIDDEN: sorted(), .sort(), heapq, numpy

Implement the sorting algorithm manually using only loops and swaps.""",
        "expected_patterns": [r"def\s+\w+", r"for\s+\w+\s+in", r"while"],
        "forbidden_patterns": [r"sorted\s*\(", r"\.sort\s*\(", r"heapq", r"numpy"],
    },
    {
        "name": "no_slice_reverse",
        "prompt": """Write a Python function that reverses a string.

FORBIDDEN: [::-1], reversed(), .reverse()

Use only a loop to build the reversed string.""",
        "expected_patterns": [r"def\s+\w+", r"for\s+\w+\s+in"],
        "forbidden_patterns": [r"\[::-1\]", r"reversed\s*\(", r"\.reverse\s*\("],
    },
    {
        "name": "no_sum_function",
        "prompt": """Write a Python function that calculates the sum of all numbers in a list.

FORBIDDEN: sum(), numpy, pandas

Use only a loop and addition.""",
        "expected_patterns": [r"def\s+\w+", r"for\s+\w+\s+in", r"\+="],
        "forbidden_patterns": [r"\bsum\s*\(", r"numpy", r"pandas"],
    },
    {
        "name": "no_len_function",
        "prompt": """Write a Python function that counts the number of elements in a list.

FORBIDDEN: len(), __len__

Use only a loop with a counter variable.""",
        "expected_patterns": [r"def\s+\w+", r"for\s+\w+\s+in", r"\+=\s*1", r"count"],
        "forbidden_patterns": [r"\blen\s*\(", r"__len__"],
    },
    # EXPANDED L4 PROBES (5 more for better coverage)
    {
        "name": "no_list_comprehension",
        "prompt": """Write a Python function that squares each element in a list and returns the result.

FORBIDDEN: list comprehensions (e.g., [x**2 for x in lst]), map()

Use only a for loop and append.""",
        "expected_patterns": [r"def\s+\w+", r"for\s+\w+\s+in", r"\.append\s*\("],
        "forbidden_patterns": [r"\[.+for.+in", r"map\s*\("],
    },
    {
        "name": "no_min_function",
        "prompt": """Write a Python function that finds the minimum value in a list.

FORBIDDEN: min(), sorted(), numpy

Implement using only loops and comparisons.""",
        "expected_patterns": [r"def\s+\w+", r"for\s+\w+\s+in", r"if\s+\w+\s*[<>]"],
        "forbidden_patterns": [r"\bmin\s*\(", r"sorted\s*\(", r"numpy"],
    },
    {
        "name": "no_enumerate",
        "prompt": """Write a Python function that prints each element with its index.

FORBIDDEN: enumerate()

Use only a manual counter variable with a for loop.""",
        "expected_patterns": [r"def\s+\w+", r"for\s+\w+\s+in", r"\+=\s*1"],
        "forbidden_patterns": [r"enumerate\s*\("],
    },
    {
        "name": "no_join_method",
        "prompt": """Write a Python function that concatenates a list of strings with a separator.

FORBIDDEN: ''.join(), str.join()

Use only a loop to build the result string.""",
        "expected_patterns": [r"def\s+\w+", r"for\s+\w+\s+in", r"\+\s*="],
        "forbidden_patterns": [r"\.join\s*\(", r"join\s*\("],
    },
    {
        "name": "no_in_operator_search",
        "prompt": """Write a Python function that checks if an element exists in a list.

FORBIDDEN: 'in' operator, any(), index(), count()

Use only a for loop to iterate and check each element.""",
        "expected_patterns": [r"def\s+\w+", r"for\s+\w+\s+in", r"if\s+\w+\s*=="],
        "forbidden_patterns": [r"\bany\s*\(", r"\.index\s*\(", r"\.count\s*\("],
    },
]


def evaluate_probe(
    model,
    tokenizer,
    probe: dict,
    level: int,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> ProbeResult:
    """Evaluate a single probe against the model."""

    # Generate response
    inputs = tokenizer(probe["prompt"], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for validation
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    response_lower = response.lower()

    # Check expected patterns
    expected_matches = 0
    for pattern in probe["expected_patterns"]:
        if re.search(pattern, response_lower, re.IGNORECASE):
            expected_matches += 1

    # Check forbidden patterns
    forbidden_violations = 0
    for pattern in probe["forbidden_patterns"]:
        if re.search(pattern, response_lower, re.IGNORECASE):
            forbidden_violations += 1

    # Calculate score
    expected_score = expected_matches / len(probe["expected_patterns"]) if probe["expected_patterns"] else 1.0
    forbidden_penalty = forbidden_violations / len(probe["forbidden_patterns"]) if probe["forbidden_patterns"] else 0.0

    # Final score: expected matches minus penalty for forbidden patterns
    score = max(0.0, expected_score - (forbidden_penalty * 0.5))
    passed = score >= 0.5 and forbidden_violations == 0

    return ProbeResult(
        probe_name=probe["name"],
        level=level,
        passed=passed,
        expected=str(probe["expected_patterns"]),
        actual=response[:500],  # Truncate for logging
        score=score,
        details=f"Expected matches: {expected_matches}/{len(probe['expected_patterns'])}, Forbidden violations: {forbidden_violations}",
    )


def run_validation(
    model,
    tokenizer,
    step: int,
    levels: list[int] = [1, 2, 3, 4],
    device: str = "cuda",
) -> ValidationReport:
    """Run full validation suite and return report."""

    report = ValidationReport(step=step)

    probe_sets = [
        (1, LEVEL1_PROBES),
        (2, LEVEL2_PROBES),
        (3, LEVEL3_PROBES),
        (4, LEVEL4_PROBES),
    ]

    for level, probes in probe_sets:
        if level not in levels:
            continue

        print(f"  Running Level {level} probes...")
        for probe in probes:
            result = evaluate_probe(model, tokenizer, probe, level, device=device)
            report.results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"    {probe['name']}: {status} (score: {result.score:.2f})")

    return report


class CPIDetector:
    """
    Context-Parametric Inversion Detector.

    Monitors A_CF (counterfactual accuracy) and A_P (parametric/standard accuracy).
    Triggers alert if A_CF declines while A_P rises.
    """

    def __init__(self, window_size: int = 5, threshold: float = 0.1):
        self.history: list[dict] = []
        self.window_size = window_size
        self.threshold = threshold

    def add_report(self, report: ValidationReport):
        """Add a validation report to history."""
        self.history.append({
            "step": report.step,
            "A_CF": report.counterfactual_accuracy,
            "A_P": report.overall_accuracy,
            "L1": report.level1_accuracy,
            "L2": report.level2_accuracy,
            "L3": report.level3_accuracy,
            "L4": report.level4_accuracy,
        })

    def check_inversion(self) -> tuple[bool, Optional[str]]:
        """
        Check for Context-Parametric Inversion.

        Returns:
            (inversion_detected, message)
        """
        if len(self.history) < self.window_size:
            return False, None

        recent = self.history[-self.window_size:]
        early = self.history[:self.window_size] if len(self.history) > self.window_size else recent[:1]

        # Calculate trends
        recent_acf = sum(h["A_CF"] for h in recent) / len(recent)
        early_acf = sum(h["A_CF"] for h in early) / len(early)

        recent_l1 = sum(h["L1"] for h in recent) / len(recent)
        early_l1 = sum(h["L1"] for h in early) / len(early)

        # Check for inversion pattern
        acf_declining = (early_acf - recent_acf) > self.threshold
        l1_declining = (early_l1 - recent_l1) > self.threshold

        if acf_declining or l1_declining:
            peak_step = max(self.history, key=lambda h: h["A_CF"])["step"]
            msg = f"CONTEXT-PARAMETRIC INVERSION DETECTED!\n"
            msg += f"  A_CF declined from {early_acf:.3f} to {recent_acf:.3f}\n"
            msg += f"  Best checkpoint was at step {peak_step}\n"
            msg += f"  RECOMMENDATION: Stop training, use checkpoint from step {peak_step}"
            return True, msg

        return False, None

    def get_best_checkpoint_step(self) -> Optional[int]:
        """Return the step with highest counterfactual accuracy."""
        if not self.history:
            return None
        return max(self.history, key=lambda h: h["A_CF"])["step"]

    def save_history(self, path: Path):
        """Save history to JSON file."""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_history(self, path: Path):
        """Load history from JSON file."""
        if path.exists():
            with open(path) as f:
                self.history = json.load(f)


def check_abort_criteria(report: ValidationReport, step: int) -> tuple[bool, Optional[str]]:
    """
    Check if training should be aborted based on validation results.

    Abort criteria:
    - No improvement on Level 1 probes after 2000 steps
    - Level 1 accuracy peaks then declines (CPI)
    """

    # After step 2000, Level 1 should have >50% accuracy
    if step >= 2000 and report.level1_accuracy < 0.5:
        return True, f"ABORT: Level 1 accuracy ({report.level1_accuracy:.2f}) below 0.5 at step {step}"

    return False, None


# =============================================================================
# CROSS-FORMAT CONSISTENCY TESTS (for DAFT validation)
# =============================================================================

@dataclass
class CrossFormatResult:
    """Result of cross-format consistency test."""
    problem_id: str
    expected_answer: str
    format_answers: dict  # domain_id -> answer
    is_consistent: bool  # All formats give same answer
    consistency_score: float  # % of formats giving most common answer


@dataclass
class FormatInvarianceReport:
    """Report on format invariance metrics."""
    cross_format_consistency: float  # % problems with consistent answers across formats
    cue_ablation_sensitivity: float  # Performance drop when format cues removed
    per_domain_accuracy: dict  # Accuracy per format domain
    worst_domain_accuracy: float
    best_domain_accuracy: float
    domain_gap: float  # best - worst

    def meets_targets(self) -> tuple[bool, list[str]]:
        """Check if metrics meet production thresholds."""
        failures = []

        if self.cross_format_consistency < 0.95:
            failures.append(f"Cross-format consistency {self.cross_format_consistency:.1%} < 95%")

        if self.cue_ablation_sensitivity > 0.10:
            failures.append(f"Cue ablation sensitivity {self.cue_ablation_sensitivity:.1%} > 10%")

        if self.worst_domain_accuracy < 0.85:
            failures.append(f"Worst domain accuracy {self.worst_domain_accuracy:.1%} < 85%")

        if self.domain_gap > 0.05:
            failures.append(f"Domain gap {self.domain_gap:.1%} > 5%")

        return len(failures) == 0, failures


# Cross-format test problems with multiple format variants
CROSS_FORMAT_L3_PROBLEMS = [
    {
        "id": "cf_undistributed_middle",
        "expected": "UNKNOWN",
        "formats": {
            0: """Premise 1: All wibbles are snorgs
Premise 2: All plonk are snorgs

Are some wibbles also plonk?""",
            1: """• All wibbles are snorgs
• All plonk are snorgs

HYPOTHESIS: Some wibbles are plonk""",
            2: """P1: wibbles ⊂ snorgs
P2: plonk ⊂ snorgs

⊢ wibbles ∩ plonk ≠ ∅?""",
            3: """Given that all wibbles are snorgs and all plonk are snorgs, are some wibbles also plonk?""",
            4: """Are some wibbles also plonk?

Given:
1. All wibbles are snorgs
2. All plonk are snorgs""",
            5: """All wibbles are snorgs All plonk are snorgs Are some wibbles plonk""",
        },
    },
    {
        "id": "cf_modus_ponens",
        "expected": "PROVED",
        "formats": {
            0: """Premise 1: All frumious are bandersnatch
Premise 2: Qar is frumious

Is Qar a bandersnatch?""",
            1: """• All frumious are bandersnatch
• Qar is frumious

HYPOTHESIS: Qar is a bandersnatch""",
            2: """P1: frumious ⊂ bandersnatch
P2: Qar ∈ frumious

⊢ Qar ∈ bandersnatch?""",
            3: """Given that all frumious are bandersnatch and Qar is frumious, is Qar a bandersnatch?""",
            4: """Is Qar a bandersnatch?

Given:
1. All frumious are bandersnatch
2. Qar is frumious""",
            5: """All frumious are bandersnatch Qar is frumious Is Qar bandersnatch""",
        },
    },
    {
        "id": "cf_affirming_consequent",
        "expected": "UNKNOWN",
        "formats": {
            0: """Premise 1: All zorplax are quindrix
Premise 2: Zyx is a quindrix

Is Zyx a zorplax?""",
            1: """• All zorplax are quindrix
• Zyx is a quindrix

HYPOTHESIS: Zyx is a zorplax""",
            2: """P1: zorplax ⊂ quindrix
P2: Zyx ∈ quindrix

⊢ Zyx ∈ zorplax?""",
            3: """Given that all zorplax are quindrix and Zyx is a quindrix, is Zyx a zorplax?""",
            4: """Is Zyx a zorplax?

Given:
1. All zorplax are quindrix
2. Zyx is a quindrix""",
            5: """All zorplax are quindrix Zyx is quindrix Is Zyx zorplax""",
        },
    },
]


def extract_logical_answer(response: str) -> str:
    """Extract PROVED/DISPROVED/UNKNOWN from response."""
    response_upper = response.upper()

    if "PROVED" in response_upper and "DISPROVED" not in response_upper:
        return "PROVED"
    elif "DISPROVED" in response_upper:
        return "DISPROVED"
    elif "UNKNOWN" in response_upper:
        return "UNKNOWN"

    # Fallback patterns
    if "VALID" in response_upper and "INVALID" not in response_upper:
        return "PROVED"
    elif "INVALID" in response_upper:
        return "UNKNOWN"

    return "UNKNOWN"


def test_cross_format_consistency(
    model,
    tokenizer,
    problems: list[dict] = None,
    device: str = "cuda",
) -> tuple[float, list[CrossFormatResult]]:
    """Test if model gives consistent answers across format variants.

    Args:
        model: The model to test
        tokenizer: Tokenizer
        problems: List of cross-format problems (default: CROSS_FORMAT_L3_PROBLEMS)
        device: Device to run on

    Returns:
        (consistency_rate, list of results)
    """
    if problems is None:
        problems = CROSS_FORMAT_L3_PROBLEMS

    results = []

    for problem in problems:
        format_answers = {}

        for domain_id, prompt in problem["formats"].items():
            # Add instruction
            full_prompt = f"""Given the following premises, determine if the conclusion can be logically derived.

Answer with PROVED, DISPROVED, or UNKNOWN.

{prompt}"""

            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            answer = extract_logical_answer(response)
            format_answers[domain_id] = answer

        # Check consistency
        unique_answers = set(format_answers.values())
        is_consistent = len(unique_answers) == 1

        # Calculate consistency score (% giving most common answer)
        from collections import Counter
        answer_counts = Counter(format_answers.values())
        most_common_count = answer_counts.most_common(1)[0][1]
        consistency_score = most_common_count / len(format_answers)

        result = CrossFormatResult(
            problem_id=problem["id"],
            expected_answer=problem["expected"],
            format_answers=format_answers,
            is_consistent=is_consistent,
            consistency_score=consistency_score,
        )
        results.append(result)

    consistency_rate = sum(1 for r in results if r.is_consistent) / len(results)
    return consistency_rate, results


def run_format_invariance_validation(
    model,
    tokenizer,
    device: str = "cuda",
) -> FormatInvarianceReport:
    """Run comprehensive format invariance validation.

    Tests:
    1. Cross-format consistency
    2. Cue ablation sensitivity
    3. Per-domain accuracy

    Args:
        model: The model to test
        tokenizer: Tokenizer
        device: Device to run on

    Returns:
        FormatInvarianceReport with all metrics
    """
    print("Running format invariance validation...")

    # 1. Cross-format consistency
    print("  Testing cross-format consistency...")
    consistency_rate, results = test_cross_format_consistency(
        model, tokenizer, device=device
    )

    # 2. Per-domain accuracy
    print("  Testing per-domain accuracy...")
    domain_correct = {d: 0 for d in range(6)}
    domain_total = {d: 0 for d in range(6)}

    for result in results:
        for domain_id, answer in result.format_answers.items():
            domain_total[domain_id] += 1
            if answer == result.expected_answer:
                domain_correct[domain_id] += 1

    per_domain_accuracy = {
        d: domain_correct[d] / max(domain_total[d], 1)
        for d in range(6)
    }

    worst_acc = min(per_domain_accuracy.values())
    best_acc = max(per_domain_accuracy.values())

    # 3. Cue ablation: compare domain 0 (full cues) vs domain 5 (minimal)
    acc_with_cues = per_domain_accuracy.get(0, 0)
    acc_without_cues = per_domain_accuracy.get(5, 0)
    cue_ablation = max(0, acc_with_cues - acc_without_cues)

    return FormatInvarianceReport(
        cross_format_consistency=consistency_rate,
        cue_ablation_sensitivity=cue_ablation,
        per_domain_accuracy=per_domain_accuracy,
        worst_domain_accuracy=worst_acc,
        best_domain_accuracy=best_acc,
        domain_gap=best_acc - worst_acc,
    )


def quick_sanity_check(model, tokenizer, device: str = "cuda") -> bool:
    """
    Quick sanity check - just run Level 1 probes.
    Returns True if model shows any context-following behavior.
    """
    report = run_validation(model, tokenizer, step=0, levels=[1], device=device)
    return report.level1_accuracy > 0.2


# =============================================================================
# MAIN - For standalone testing
# =============================================================================

if __name__ == "__main__":
    print("Validation Probes Module")
    print("=" * 50)
    print(f"Level 1 Probes: {len(LEVEL1_PROBES)}")
    print(f"Level 2 Probes: {len(LEVEL2_PROBES)}")
    print(f"Level 3 Probes: {len(LEVEL3_PROBES)}")
    print(f"Level 4 Probes: {len(LEVEL4_PROBES)}")
    print(f"Total: {len(LEVEL1_PROBES) + len(LEVEL2_PROBES) + len(LEVEL3_PROBES) + len(LEVEL4_PROBES)}")

    print("\nLevel 1 Probe Names:")
    for p in LEVEL1_PROBES:
        print(f"  - {p['name']}")

    print("\nLevel 2 Probe Names:")
    for p in LEVEL2_PROBES:
        print(f"  - {p['name']}")

    print("\nLevel 3 Probe Names:")
    for p in LEVEL3_PROBES:
        print(f"  - {p['name']}")

    print("\nLevel 4 Probe Names:")
    for p in LEVEL4_PROBES:
        print(f"  - {p['name']}")
