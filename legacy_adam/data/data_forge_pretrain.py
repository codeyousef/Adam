"""
data_forge_pretrain.py — Synthetic Pretraining Corpus Generator for Adam PoC

Generates ~6B tokens of 100% synthetic, knowledge-sparse text in JSONL format.
Every document uses {"text": "..."} structure for CLM pretraining.

Design rules (Section 2 of the research document):
  - NO named entities ("the scientist", not "Newton")
  - NO scientific constants (use abstract vars: G_var, c_var)
  - NO dates (relative ordering only: "event A preceded event B")
  - NO real-world units (use unit_length, unit_time, unit_mass)
  - All facts are mathematical truths OR explicitly marked ASSUME:
  - All reasoning uses <|begin_of_thought|> / <|end_of_thought|> tags

Corpus composition (6B token target):
  40%  Propositional logic + syllogisms   (2.4B tokens)
  30%  Synthetic arithmetic + equations   (1.8B tokens)
  20%  Instruction-following dialogues    (1.2B tokens)
  10%  Formal proof fragments             (0.6B tokens)

Usage:
    python data/data_forge_pretrain.py \
        --output-dir hope/adam_training_data/pretrain_corpus \
        --val-output hope/adam_training_data/pretrain_val.jsonl \
        --total-tokens 6000000000

Quick test:
    python data/data_forge_pretrain.py \
        --output-dir /tmp/pretrain_test \
        --val-output /tmp/pretrain_val.jsonl \
        --total-tokens 500000 \
        --workers 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, Iterator


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Nonsense predicates (same set as data_forge_adam_balanced.py for consistency)
PREDICATES = [
    "wampimuk", "zorplax", "mimsy", "blurgle", "snorflax",
    "quizzle", "frimble", "glorbax", "thistwick", "plonkish",
    "vurglax", "dweemish", "scronkle", "frumple", "zazzle",
    "blixthorn", "crumblax", "sniglet", "plorbax", "wumple",
]

# Abstract variable names (for arithmetic domain)
ABSTRACT_VARS = [
    "x", "y", "z", "a", "b", "c", "p", "q", "r", "s",
    "alpha", "beta", "gamma", "delta", "epsilon", "lambda_val",
    "mu", "nu", "xi", "phi",
]

# Abstract unit names (for arithmetic domain)
UNIT_NAMES = [
    "unit_length", "unit_time", "unit_mass", "unit_energy",
    "unit_speed", "unit_force", "unit_temp", "unit_volume",
    "unit_pressure", "unit_charge",
]

# Abstract "constant" names (never real values)
ABSTRACT_CONSTANTS = [
    "G_var", "c_var", "k_var", "h_var", "e_var",
    "R_var", "N_var", "sigma_var", "alpha_var",
]

# Connectives for logic domain
CONNECTIVES = ["AND", "OR", "NOT", "IF", "IFF"]

# Syllogism form labels
SYLLOGISM_FORMS = ["VALID", "INVALID", "UNKNOWN"]
SYLLOGISM_TYPES = [
    "Barbara", "Celarent", "Darii", "Ferio",
    "Camestres", "Cesare", "Baroco", "Festino",
    "Undetermined",
]

# Banned patterns for verification.
# Keep these SPECIFIC — avoid catch-all patterns that fire on our own output.
BANNED_ENTITY_PATTERNS = [
    # Known proper names / places
    r"\bNewton\b", r"\bEinstein\b", r"\bDarwin\b", r"\bGalileo\b",
    r"\bParis\b", r"\bLondon\b", r"\bBerlin\b", r"\bTokyo\b",
    r"\bAmerica\b", r"\bEurope\b", r"\bChina\b",
    # Month names
    r"\bJanuary\b", r"\bFebruary\b", r"\bMarch\b", r"\bApril\b",
    r"\bMay\b", r"\bJune\b", r"\bJuly\b", r"\bAugust\b",
    r"\bSeptember\b", r"\bOctober\b", r"\bNovember\b", r"\bDecember\b",
    # Real-world units
    r"\bmeters?\b", r"\bkilograms?\b", r"\bseconds?\b", r"\bhours?\b",
    r"\bNewtons?\b", r"\bJoules?\b", r"\bPascals?\b", r"\bWatts?\b",
    r"\bvolts?\b", r"\bamperes?\b",
]

CHARS_PER_TOKEN_ESTIMATE = 4.0  # conservative estimate for token counting


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_doc(text: str) -> bool:
    """Returns True if document passes entity/constant leak check."""
    for pattern in BANNED_ENTITY_PATTERNS:
        if re.search(pattern, text):
            return False
    return True


def estimate_tokens(text: str) -> int:
    """Fast approximation: chars / 4."""
    return max(1, int(len(text) / CHARS_PER_TOKEN_ESTIMATE))


# ─────────────────────────────────────────────────────────────────────────────
# Section A: Logic Domain (40%) — Propositional + Syllogistic
# ─────────────────────────────────────────────────────────────────────────────

class LogicGenerator:
    """
    Generates self-contained logic reasoning problems using nonsense predicates.
    Includes both propositional logic and Aristotelian syllogisms.
    Output format mirrors Adam's VALID/INVALID/UNKNOWN training targets.
    """

    def __init__(self, rng: random.Random):
        self.rng = rng

    def _pick(self, lst: list) -> str:
        return self.rng.choice(lst)

    def _pick_n(self, lst: list, n: int) -> list:
        pool = list(lst)
        self.rng.shuffle(pool)
        return pool[:n]

    # ── Syllogism generator ───────────────────────────────────────────────

    def _make_syllogism(self, requested_form: str | None = None) -> tuple[str, str, str, bool]:
        """
        Returns (major_premise, minor_premise, conclusion, is_valid).
        requested_form pins the syllogism structure so the CoT label always
        matches the actual premises generated. Without it, picks randomly.
        """
        preds = self._pick_n(PREDICATES, 3)
        A, B, C = preds[0], preds[1], preds[2]

        form_type = requested_form or self.rng.choice(
            ["barbara", "celarent", "darii", "ferio", "invalid"]
        )

        if form_type == "barbara":
            # All A are B. All B are C. Therefore all A are C. (VALID)
            maj = f"All {A} are {B}."
            min_ = f"All {B} are {C}."
            conc = f"All {A} are {C}."
            return maj, min_, conc, True

        elif form_type == "celarent":
            # No A are B. All C are A. Therefore no C are B. (VALID)
            maj = f"No {A} are {B}."
            min_ = f"All {C} are {A}."
            conc = f"No {C} are {B}."
            return maj, min_, conc, True

        elif form_type == "darii":
            # All A are B. Some C are A. Therefore some C are B. (VALID)
            maj = f"All {A} are {B}."
            min_ = f"Some {C} are {A}."
            conc = f"Some {C} are {B}."
            return maj, min_, conc, True

        elif form_type == "ferio":
            # No A are B. Some C are A. Therefore some C are not B. (VALID)
            maj = f"No {A} are {B}."
            min_ = f"Some {C} are {A}."
            conc = f"Some {C} are not {B}."
            return maj, min_, conc, True

        else:
            # Invalid form: affirming the consequent-style
            maj = f"All {A} are {B}."
            min_ = f"All {C} are {B}."
            conc = f"All {A} are {C}."
            return maj, min_, conc, False

    def _make_unknown_syllogism(self) -> tuple[str, str, str]:
        """Premises that don't determine the conclusion (UNKNOWN)."""
        preds = self._pick_n(PREDICATES, 3)
        A, B, C = preds[0], preds[1], preds[2]
        maj = f"Some {A} are {B}."
        min_ = f"Some {B} are {C}."
        conc = f"Some {A} are {C}."
        return maj, min_, conc

    def _syllogism_cot(
        self,
        major: str, minor: str, conclusion: str,
        is_valid: bool, form_type: str
    ) -> str:
        validity_label = "VALID" if is_valid else "INVALID"
        reason = (
            f"The argument follows {form_type} form and the conclusion follows necessarily from the premises."
            if is_valid
            else f"The conclusion does not follow necessarily from the premises."
        )
        return textwrap.dedent(f"""\
            Given: {major}
            Given: {minor}
            Question: Does it follow that: {conclusion}
            <|begin_of_thought|>
            Step 1: PARSE_PREMISES - Major: [{major}] Minor: [{minor}]
            Step 2: IDENTIFY_FORM - {form_type} syllogism
            Step 3: CHECK_VALIDITY - {reason}
            Step 4: DETERMINE_CONCLUSION - {validity_label}
            <|end_of_thought|>
            FORM: {form_type} VALIDITY: {validity_label} REASON: {reason}
        """).strip()

    def _unknown_cot(self, major: str, minor: str, conclusion: str) -> str:
        return textwrap.dedent(f"""\
            Given: {major}
            Given: {minor}
            Question: Does it follow that: {conclusion}
            <|begin_of_thought|>
            Step 1: PARSE_PREMISES - Major: [{major}] Minor: [{minor}]
            Step 2: IDENTIFY_FORM - Undetermined (existential premises only)
            Step 3: CHECK_VALIDITY - Existential premises do not guarantee existential conclusions.
            Step 4: DETERMINE_CONCLUSION - UNKNOWN
            <|end_of_thought|>
            FORM: Undetermined VALIDITY: UNKNOWN REASON: Existential premises alone do not entail the conclusion.
        """).strip()

    # ── Propositional logic generator ─────────────────────────────────────

    def _prop_var(self) -> str:
        return self.rng.choice(["P", "Q", "R", "S", "T", "U"])

    def _make_modus_ponens(self) -> str:
        P = self._prop_var()
        Q = self._prop_var()
        while Q == P:
            Q = self._prop_var()
        return textwrap.dedent(f"""\
            Given: IF {P} THEN {Q}.
            Given: {P} is TRUE.
            Question: What is the truth value of {Q}?
            <|begin_of_thought|>
            Step 1: IDENTIFY_RULE - Modus Ponens: (P → Q) ∧ P → Q
            Step 2: VERIFY_ANTECEDENT - {P} is TRUE (given)
            Step 3: APPLY_RULE - Since {P} is TRUE and {P} → {Q}, conclude {Q} is TRUE.
            <|end_of_thought|>
            CONCLUSION: {Q} is TRUE. RULE: Modus Ponens.
        """).strip()

    def _make_modus_tollens(self) -> str:
        P = self._prop_var()
        Q = self._prop_var()
        while Q == P:
            Q = self._prop_var()
        return textwrap.dedent(f"""\
            Given: IF {P} THEN {Q}.
            Given: {Q} is FALSE.
            Question: What is the truth value of {P}?
            <|begin_of_thought|>
            Step 1: IDENTIFY_RULE - Modus Tollens: (P → Q) ∧ ¬Q → ¬P
            Step 2: VERIFY_CONSEQUENT - {Q} is FALSE (given)
            Step 3: APPLY_RULE - Since {Q} is FALSE and {P} → {Q}, conclude {P} is FALSE.
            <|end_of_thought|>
            CONCLUSION: {P} is FALSE. RULE: Modus Tollens.
        """).strip()

    def _make_hypothetical_syllogism(self) -> str:
        P, Q, R = [self._prop_var() for _ in range(3)]
        # ensure distinct
        vars_used = [P]
        for v in [Q, R]:
            while v in vars_used:
                v = self._prop_var()
            vars_used.append(v)
        P, Q, R = vars_used
        return textwrap.dedent(f"""\
            Given: IF {P} THEN {Q}.
            Given: IF {Q} THEN {R}.
            Question: What can we conclude about the relationship between {P} and {R}?
            <|begin_of_thought|>
            Step 1: IDENTIFY_RULE - Hypothetical Syllogism: (P → Q) ∧ (Q → R) → (P → R)
            Step 2: CHAIN_IMPLICATIONS - {P} → {Q} → {R}
            Step 3: APPLY_RULE - Conclude: IF {P} THEN {R}.
            <|end_of_thought|>
            CONCLUSION: IF {P} THEN {R}. RULE: Hypothetical Syllogism.
        """).strip()

    def generate_document(self) -> str:
        """Generate one logic reasoning document."""
        doc_type = self.rng.choices(
            ["barbara", "celarent", "darii", "ferio", "invalid", "unknown",
             "modus_ponens", "modus_tollens", "hyp_syllogism"],
            weights=[15, 10, 10, 10, 15, 20, 7, 7, 6],
        )[0]

        if doc_type == "unknown":
            maj, min_, conc = self._make_unknown_syllogism()
            return self._unknown_cot(maj, min_, conc)
        elif doc_type in {"barbara", "celarent", "darii", "ferio", "invalid"}:
            form_map = {
                "barbara": "Barbara",
                "celarent": "Celarent",
                "darii": "Darii",
                "ferio": "Ferio",
                "invalid": "Undetermined",
            }
            # Pass doc_type so the generated premises match the CoT label exactly
            maj, min_, conc, is_valid = self._make_syllogism(requested_form=doc_type)
            return self._syllogism_cot(maj, min_, conc, is_valid, form_map[doc_type])
        elif doc_type == "modus_ponens":
            return self._make_modus_ponens()
        elif doc_type == "modus_tollens":
            return self._make_modus_tollens()
        else:
            return self._make_hypothetical_syllogism()


def generate_logic_domain(
    target_tokens: int,
    seed: int,
    verify: bool = True,
) -> Iterator[str]:
    """Yields logic documents until target_tokens is approximately reached."""
    rng = random.Random(seed)
    gen = LogicGenerator(rng)
    tokens_emitted = 0
    rejected = 0

    while tokens_emitted < target_tokens:
        text = gen.generate_document()
        if verify and not verify_doc(text):
            rejected += 1
            continue
        tokens_emitted += estimate_tokens(text)
        yield text

    if rejected > 0:
        print(f"  [logic] Rejected {rejected} docs with banned entities", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Section B: Arithmetic Domain (30%) — Expression Trees + Equations
# ─────────────────────────────────────────────────────────────────────────────

class ArithmeticGenerator:
    """
    3-rung difficulty curriculum:
      Rung 1 (33%): Direct formula, no CoT
      Rung 2 (33%): Multi-step with STEPS
      Rung 3 (33%): Full CoT with semantic labels
    10% of documents use counterfactual constants (primes L2).
    """

    def __init__(self, rng: random.Random):
        self.rng = rng

    def _randint(self, lo: int = 2, hi: int = 200) -> int:
        return self.rng.randint(lo, hi)

    def _unit(self) -> str:
        return self.rng.choice(UNIT_NAMES)

    def _var(self) -> str:
        return self.rng.choice(ABSTRACT_VARS)

    def _const_name(self) -> str:
        return self.rng.choice(ABSTRACT_CONSTANTS)

    # ── Rung 1: Direct (no CoT) ───────────────────────────────────────────

    def _rung1_distance(self) -> str:
        speed = self._randint(10, 500)
        time = self._randint(1, 100)
        distance = speed * time
        u_s = self._unit()
        u_t = self._unit()
        u_d = self._unit()
        return (
            f"Given: speed={speed} {u_s}, time={time} {u_t}.\n"
            f"Find: distance.\n"
            f"Formula: distance = speed × time = {speed} × {time} = {distance} {u_d}."
        )

    def _rung1_area(self) -> str:
        w = self._randint(2, 50)
        h = self._randint(2, 50)
        area = w * h
        u = self._unit()
        return (
            f"Given: width={w} {u}, height={h} {u}.\n"
            f"Find: area.\n"
            f"Formula: area = width × height = {w} × {h} = {area} {u}²."
        )

    def _rung1_ratio(self) -> str:
        a = self._randint(1, 100)
        b = self._randint(1, 100)
        total = a + b
        frac_a = round(a / total, 4)
        frac_b = round(b / total, 4)
        pred_a = self.rng.choice(PREDICATES)
        pred_b = self.rng.choice(PREDICATES)
        return (
            f"Given: {a} {pred_a} and {b} {pred_b} in total.\n"
            f"Find: fraction of {pred_a}.\n"
            f"Formula: {a} / {total} = {frac_a}."
        )

    # ── Rung 2: Multi-step (minimal CoT) ─────────────────────────────────

    def _rung2_compound(self) -> str:
        rate = self._randint(5, 100)
        time1 = self._randint(1, 10)
        time2 = self._randint(1, 10)
        phase1 = rate * time1
        phase2 = rate * 2 * time2
        total = phase1 + phase2
        u_r = self._unit()
        u_t = self._unit()
        return textwrap.dedent(f"""\
            Given: base_rate={rate} {u_r}, phase_1_duration={time1} {u_t}, phase_2_rate=2×base, phase_2_duration={time2} {u_t}.
            Find: total output.
            Step 1: phase_1_output = {rate} × {time1} = {phase1} {u_r}
            Step 2: phase_2_output = {rate*2} × {time2} = {phase2} {u_r}
            Step 3: total = {phase1} + {phase2} = {total} {u_r}
            Result: {total} {u_r}.
        """).strip()

    def _rung2_weighted_avg(self) -> str:
        n = self.rng.randint(2, 4)
        values = [self._randint(1, 100) for _ in range(n)]
        weights = [self._randint(1, 10) for _ in range(n)]
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        total_weight = sum(weights)
        avg = round(weighted_sum / total_weight, 3)
        u = self._unit()
        lines = [f"Given: {n} measurements with values and weights:"]
        for i, (v, w) in enumerate(zip(values, weights), 1):
            lines.append(f"  m_{i}: value={v} {u}, weight={w}")
        lines.append("Find: weighted average.")
        for i, (v, w) in enumerate(zip(values, weights), 1):
            lines.append(f"Step {i}: contribution_{i} = {v} × {w} = {v*w}")
        lines.append(f"Step {n+1}: weighted_sum = {weighted_sum}")
        lines.append(f"Step {n+2}: total_weight = {total_weight}")
        lines.append(f"Result: {weighted_sum} / {total_weight} = {avg} {u}.")
        return "\n".join(lines)

    # ── Rung 3: Full CoT (semantic labels) ───────────────────────────────

    def _rung3_derived_quantity(self, counterfactual: bool = False) -> str:
        const_name = self._const_name()
        # Standard value vs counterfactual
        standard_val = self._randint(5, 50)
        if counterfactual:
            cf_val = self._randint(1, 100)
            while cf_val == standard_val:
                cf_val = self._randint(1, 100)
            const_val = cf_val
            cf_note = f"ASSUME: In this context, {const_name}={cf_val} (overrides default {standard_val})."
        else:
            const_val = standard_val
            cf_note = f"ASSUME: {const_name}={const_val} (standard value for this domain)."

        base_quantity = self._randint(3, 200)
        exponent = self.rng.randint(1, 3)
        intermediate = base_quantity ** exponent
        result = round(const_val * intermediate, 4)
        u_b = self._unit()
        u_r = self._unit()

        return textwrap.dedent(f"""\
            {cf_note}
            Given: base_quantity={base_quantity} {u_b}, exponent={exponent}.
            Find: derived_result using {const_name}.
            <|begin_of_thought|>
            Step 1: IDENTIFY_GIVEN - base={base_quantity}, exponent={exponent}, {const_name}={const_val}
            {"Step 2: MODIFY_CONSTANT - Using context-provided value " + str(const_val) + " instead of default " + str(standard_val) if counterfactual else "Step 2: APPLY_CONSTANT - Using " + const_name + "=" + str(const_val)}
            Step 3: APPLY_FORMULA - result = {const_name} × base^exponent
            Step 4: COMPUTE_INTERMEDIATE - {base_quantity}^{exponent} = {intermediate}
            Step 5: COMPUTE_RESULT - {const_val} × {intermediate} = {result}
            Step 6: VERIFY_UNITS - input: {u_b}^{exponent}, constant: dimensionless, output: {u_r}
            Step 7: CONCLUDE - derived_result = {result} {u_r}
            <|end_of_thought|>
            RESULT: {result} {u_r}.
        """).strip()

    def generate_document(self) -> str:
        rung = self.rng.choices([1, 2, 3], weights=[33, 33, 34])[0]
        counterfactual = self.rng.random() < 0.10

        if rung == 1:
            fn = self.rng.choice([
                self._rung1_distance,
                self._rung1_area,
                self._rung1_ratio,
            ])
            return fn()
        elif rung == 2:
            fn = self.rng.choice([
                self._rung2_compound,
                self._rung2_weighted_avg,
            ])
            return fn()
        else:
            return self._rung3_derived_quantity(counterfactual=counterfactual)


def generate_arithmetic_domain(
    target_tokens: int,
    seed: int,
    verify: bool = True,
) -> Iterator[str]:
    rng = random.Random(seed + 1000)
    gen = ArithmeticGenerator(rng)
    tokens_emitted = 0
    rejected = 0

    while tokens_emitted < target_tokens:
        text = gen.generate_document()
        if verify and not verify_doc(text):
            rejected += 1
            continue
        tokens_emitted += estimate_tokens(text)
        yield text

    if rejected > 0:
        print(f"  [arithmetic] Rejected {rejected} docs with banned entities", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Section C: Instruction Domain (20%) — ASSUME: Dialogues
# ─────────────────────────────────────────────────────────────────────────────

class InstructionGenerator:
    """
    Every document uses ASSUME: markers for provided premises.
    Templates model instruction-following with explicit context override patterns.
    """

    def __init__(self, rng: random.Random):
        self.rng = rng

    def _unit(self) -> str:
        return self.rng.choice(UNIT_NAMES)

    def _randint(self, lo: int = 2, hi: int = 100) -> int:
        return self.rng.randint(lo, hi)

    def _property_name(self) -> str:
        return self.rng.choice([
            "boiling_point", "melting_point", "density",
            "conductivity", "viscosity", "refractive_index",
            "capacity", "threshold", "resistance", "efficiency",
        ])

    def _make_context_override(self) -> str:
        """Standard context override template."""
        prop = self._property_name()
        standard_val = self._randint(10, 500)
        context_val = self._randint(10, 500)
        while context_val == standard_val:
            context_val = self._randint(10, 500)
        container_val = self._randint(1, 20)
        u = self._unit()

        derived = round(context_val * container_val * 0.5, 2)

        return textwrap.dedent(f"""\
            ASSUME: In this context, {prop}={context_val} {u}.
            ASSUME: The container holds {container_val} {u}.
            Instruction: Calculate the energy required to process the container contents.
            <|begin_of_thought|>
            Step 1: IDENTIFY_CONTEXT - {prop}={context_val} {u} (from ASSUME, overrides default {standard_val})
            Step 2: IDENTIFY_QUANTITY - container={container_val} {u}
            Step 3: APPLY_FORMULA - energy = {prop} × quantity × factor(0.5)
            Step 4: COMPUTE - {context_val} × {container_val} × 0.5 = {derived}
            <|end_of_thought|>
            CONTEXT_SAYS: {prop}={context_val} OVERRIDE: standard_{prop}={standard_val} ANSWER: {derived} {u}
        """).strip()

    def _make_conditional_instruction(self) -> str:
        """Instruction with conditional premise."""
        thresh = self._randint(50, 200)
        val = self._randint(10, 300)
        u = self._unit()
        result = "ACTIVATE" if val > thresh else "STANDBY"
        rationale = (
            f"{val} > {thresh}, threshold exceeded"
            if val > thresh
            else f"{val} ≤ {thresh}, threshold not reached"
        )
        return textwrap.dedent(f"""\
            ASSUME: The system activates when the value exceeds {thresh} {u}.
            ASSUME: The observed value is {val} {u}.
            Instruction: Determine the system state.
            <|begin_of_thought|>
            Step 1: IDENTIFY_CONDITION - threshold={thresh} {u}
            Step 2: COMPARE_VALUE - observed={val} {u}
            Step 3: EVALUATE - {rationale}
            Step 4: DETERMINE_STATE - {result}
            <|end_of_thought|>
            CONDITION: value>{thresh} → ACTIVATE else STANDBY
            OBSERVED: {val} {u}
            STATE: {result}
        """).strip()

    def _make_multi_assume(self) -> str:
        """Multiple ASSUME premises, then derive conclusion."""
        a = self._randint(2, 50)
        b = self._randint(2, 50)
        rate = self._randint(1, 10)
        u1 = self._unit()
        u2 = self._unit()
        total = (a + b) * rate
        return textwrap.dedent(f"""\
            ASSUME: Source_A contributes {a} {u1} per cycle.
            ASSUME: Source_B contributes {b} {u1} per cycle.
            ASSUME: The process runs for {rate} cycles.
            Instruction: Calculate the total output from all sources combined.
            <|begin_of_thought|>
            Step 1: PARSE_ASSUMES - A={a} {u1}/cycle, B={b} {u1}/cycle, cycles={rate}
            Step 2: SUM_SOURCES - combined_rate = {a} + {b} = {a+b} {u1}/cycle
            Step 3: SCALE_BY_CYCLES - {a+b} × {rate} = {total}
            Step 4: CONCLUDE - total output = {total} {u1}
            <|end_of_thought|>
            SOURCES: A={a}, B={b}, cycles={rate}
            TOTAL_OUTPUT: {total} {u1}
        """).strip()

    def _make_value_lookup(self) -> str:
        """
        Pure value lookup: the answer IS the verbatim context value.
        Teaches span-extraction — model must copy the value directly, not compute it.
        Mirrors the CONTEXT_SAYS / OVERRIDE / ANSWER format used in SFT L1 data.
        """
        prop = self._property_name()
        val = self._randint(1, 9999)
        u = self._unit()
        return textwrap.dedent(f"""\
            ASSUME: The {prop} is {val} {u}.
            Question: What is the {prop}?
            CONTEXT_SAYS: {prop}={val} {u} OVERRIDE: context-provided value supersedes prior ANSWER: {val} {u}
        """).strip()

    def _make_entity_lookup(self) -> str:
        """
        Multi-property context, extract one specific value.
        Teaches selective span-extraction from a richer context window.
        """
        prop_pool = [
            "load_capacity", "transfer_rate", "cycle_count",
            "threshold_level", "baseline_value", "reference_constant",
            "output_factor", "decay_rate", "absorption_rate", "yield_fraction",
        ]
        self.rng.shuffle(prop_pool)
        props = prop_pool[:3]
        vals = [self._randint(1, 999) for _ in range(3)]
        units = [self._unit() for _ in range(3)]
        target_idx = self.rng.randint(0, 2)
        target_prop = props[target_idx]
        target_val = vals[target_idx]
        target_u = units[target_idx]
        params_str = ", ".join(
            f"{p}={v} {u}" for p, v, u in zip(props, vals, units)
        )
        return textwrap.dedent(f"""\
            ASSUME: System parameters: {params_str}.
            Question: Report the {target_prop}.
            CONTEXT_SAYS: {target_prop}={target_val} {target_u} OVERRIDE: use context value ANSWER: {target_val} {target_u}
        """).strip()

    def _make_override_chain(self) -> str:
        """
        A chain of two overrides: context changes a value, then question asks for it.
        Teaches that the most recent context value wins.
        """
        prop = self._property_name()
        default_val = self._randint(10, 500)
        context_val = self._randint(10, 500)
        while context_val == default_val:
            context_val = self._randint(10, 500)
        u = self._unit()
        return textwrap.dedent(f"""\
            Default assumption: {prop}={default_val} {u}.
            ASSUME: In this context, {prop}={context_val} {u}.
            Question: Given the context, what is the {prop}?
            <|begin_of_thought|>
            Step 1: IDENTIFY_DEFAULT - standard {prop}={default_val} {u}
            Step 2: IDENTIFY_CONTEXT - context sets {prop}={context_val} {u}
            Step 3: APPLY_OVERRIDE - context value supersedes default
            <|end_of_thought|>
            CONTEXT_SAYS: {prop}={context_val} {u} OVERRIDE: default was {default_val} {u} ANSWER: {context_val} {u}
        """).strip()

    def generate_document(self) -> str:
        fn = self.rng.choices(
            [
                self._make_context_override,
                self._make_conditional_instruction,
                self._make_multi_assume,
                self._make_value_lookup,
                self._make_entity_lookup,
                self._make_override_chain,
            ],
            weights=[15, 15, 15, 25, 20, 10],
        )[0]
        return fn()


def generate_instruction_domain(
    target_tokens: int,
    seed: int,
    verify: bool = True,
) -> Iterator[str]:
    rng = random.Random(seed + 2000)
    gen = InstructionGenerator(rng)
    tokens_emitted = 0
    rejected = 0

    while tokens_emitted < target_tokens:
        text = gen.generate_document()
        if verify and not verify_doc(text):
            rejected += 1
            continue
        tokens_emitted += estimate_tokens(text)
        yield text

    if rejected > 0:
        print(f"  [instruction] Rejected {rejected} docs with banned entities", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Section D: Formal Proof Domain (10%) — Natural Deduction Fragments
# ─────────────────────────────────────────────────────────────────────────────

class ProofGenerator:
    """
    Generates natural deduction proof sketches using sequent-style notation.
    """

    def __init__(self, rng: random.Random):
        self.rng = rng

    def _prop_vars(self, n: int) -> list[str]:
        pool = ["P", "Q", "R", "S", "T", "U", "V", "W"]
        self.rng.shuffle(pool)
        return pool[:n]

    def _make_hypothetical_chain(self) -> str:
        P, Q, R = self._prop_vars(3)
        return textwrap.dedent(f"""\
            Theorem: If {P} → {Q} and {Q} → {R}, then {P} → {R}.
            Proof:
            1. Assume {P}.                     [assumption]
            2. {P} → {Q}.                      [premise]
            3. {Q}.                            [modus ponens: 1, 2]
            4. {Q} → {R}.                      [premise]
            5. {R}.                            [modus ponens: 3, 4]
            6. {P} → {R}.                      [→-intro: discharge assumption 1]
            QED.
        """).strip()

    def _make_conjunction_intro(self) -> str:
        P, Q = self._prop_vars(2)
        return textwrap.dedent(f"""\
            Theorem: If {P} and {Q}, then {P} ∧ {Q}.
            Proof:
            1. {P}.                            [premise]
            2. {Q}.                            [premise]
            3. {P} ∧ {Q}.                      [∧-intro: 1, 2]
            QED.
        """).strip()

    def _make_disjunction_elim(self) -> str:
        P, Q, R = self._prop_vars(3)
        return textwrap.dedent(f"""\
            Theorem: If {P} ∨ {Q}, {P} → {R}, and {Q} → {R}, then {R}.
            Proof:
            1. {P} ∨ {Q}.                      [premise]
            2. {P} → {R}.                      [premise]
            3. {Q} → {R}.                      [premise]
            4. Case {P}:
               4a. {P}.                        [assumption]
               4b. {R}.                        [modus ponens: 4a, 2]
            5. Case {Q}:
               5a. {Q}.                        [assumption]
               5b. {R}.                        [modus ponens: 5a, 3]
            6. {R}.                            [∨-elim: 1, 4, 5]
            QED.
        """).strip()

    def _make_double_negation(self) -> str:
        P = self._prop_vars(1)[0]
        return textwrap.dedent(f"""\
            Theorem: ¬¬{P} → {P}.
            Proof:
            1. Assume ¬¬{P}.                   [assumption]
            2. {P}.                            [double negation elimination: 1]
            3. ¬¬{P} → {P}.                    [→-intro: discharge assumption 1]
            QED.
        """).strip()

    def _make_contradiction(self) -> str:
        P, Q = self._prop_vars(2)
        return textwrap.dedent(f"""\
            Theorem: {P} and ¬{P} implies {Q} (ex falso).
            Proof:
            1. {P}.                            [premise]
            2. ¬{P}.                           [premise]
            3. {P} ∧ ¬{P}.                     [∧-intro: 1, 2]
            4. ⊥.                              [contradiction: 3]
            5. {Q}.                            [ex falso (⊥-elim): 4]
            QED.
        """).strip()

    def generate_document(self) -> str:
        fn = self.rng.choice([
            self._make_hypothetical_chain,
            self._make_conjunction_intro,
            self._make_disjunction_elim,
            self._make_double_negation,
            self._make_contradiction,
        ])
        return fn()


def generate_proof_domain(
    target_tokens: int,
    seed: int,
    verify: bool = True,
) -> Iterator[str]:
    rng = random.Random(seed + 3000)
    gen = ProofGenerator(rng)
    tokens_emitted = 0
    rejected = 0

    while tokens_emitted < target_tokens:
        text = gen.generate_document()
        if verify and not verify_doc(text):
            rejected += 1
            continue
        tokens_emitted += estimate_tokens(text)
        yield text

    if rejected > 0:
        print(f"  [proof] Rejected {rejected} docs with banned entities", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Shard I/O
# ─────────────────────────────────────────────────────────────────────────────

def write_shard(
    docs: list[str],
    output_dir: str,
    shard_idx: int,
    domain: str,
) -> tuple[int, int]:
    """Write a JSONL shard file. Returns (n_docs, n_tokens_approx)."""
    output_path = Path(output_dir) / f"{domain}_shard_{shard_idx:04d}.jsonl"
    n_tokens = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for text in docs:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n_tokens += estimate_tokens(text)
    return len(docs), n_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Domain worker (for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_generate_domain(args: tuple) -> tuple[str, int, int]:
    """
    Worker function for parallel generation.
    Returns (domain_name, docs_written, tokens_written).
    """
    (
        domain, target_tokens, seed, output_dir,
        shard_size_tokens, verify,
    ) = args

    if domain == "logic":
        iterator = generate_logic_domain(target_tokens, seed, verify)
    elif domain == "arithmetic":
        iterator = generate_arithmetic_domain(target_tokens, seed, verify)
    elif domain == "instruction":
        iterator = generate_instruction_domain(target_tokens, seed, verify)
    elif domain == "proof":
        iterator = generate_proof_domain(target_tokens, seed, verify)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    buffer: list[str] = []
    buffer_tokens = 0
    total_docs = 0
    total_tokens = 0

    for text in iterator:
        buffer.append(text)
        t = estimate_tokens(text)
        buffer_tokens += t

        if buffer_tokens >= shard_size_tokens:
            n_docs, n_toks = write_shard(buffer, output_dir, shard_idx, domain)
            total_docs += n_docs
            total_tokens += n_toks
            shard_idx += 1
            buffer = []
            buffer_tokens = 0

    # Flush remaining
    if buffer:
        n_docs, n_toks = write_shard(buffer, output_dir, shard_idx, domain)
        total_docs += n_docs
        total_tokens += n_toks

    return domain, total_docs, total_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic pretraining corpus generator for Adam PoC"
    )
    parser.add_argument(
        "--output-dir",
        default="hope/adam_training_data/pretrain_corpus",
        help="Directory to write training JSONL shards",
    )
    parser.add_argument(
        "--val-output",
        default="hope/adam_training_data/pretrain_val.jsonl",
        help="Path for validation JSONL file",
    )
    parser.add_argument(
        "--total-tokens", type=int, default=6_000_000_000,
        help="Approximate total training tokens (default: 6B)",
    )
    parser.add_argument(
        "--val-tokens", type=int, default=10_000_000,
        help="Approximate validation token count (default: 10M)",
    )
    parser.add_argument(
        "--shard-size", type=int, default=100_000_000,
        help="Tokens per shard file (default: 100M → ~60 shards for 6B)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel generation workers",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Disable banned-entity verification (faster but less safe)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify = not args.no_verify

    # ── Corpus composition ─────────────────────────────────────────────────
    domain_fractions = {
        "logic": 0.40,
        "arithmetic": 0.30,
        "instruction": 0.20,
        "proof": 0.10,
    }
    training_tokens = args.total_tokens
    val_tokens = args.val_tokens

    print("=" * 70)
    print("Adam PoC — Synthetic Pretraining Corpus Generator")
    print("=" * 70)
    def _fmt_tokens(n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n/1e9:.2f}B"
        return f"{n/1e6:.1f}M"

    print(f"\nTarget tokens:    {_fmt_tokens(training_tokens)} training + "
          f"{_fmt_tokens(val_tokens)} validation")
    print(f"Output dir:       {args.output_dir}")
    print(f"Val output:       {args.val_output}")
    print(f"Shard size:       {args.shard_size / 1e6:.0f}M tokens")
    print(f"Workers:          {args.workers}")
    print(f"Verify:           {verify}")
    print(f"\nDomain split:")
    for d, frac in domain_fractions.items():
        toks = int(training_tokens * frac)
        print(f"  {d:15s} {frac*100:4.0f}%  →  {_fmt_tokens(toks)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.val_output).parent.mkdir(parents=True, exist_ok=True)

    # ── Generate validation set (single-threaded, small) ──────────────────
    print(f"\nGenerating validation set ({val_tokens / 1e6:.0f}M tokens)...")
    val_docs: list[str] = []
    # Use all 4 domains proportionally for val
    val_by_domain = {d: int(val_tokens * f) for d, f in domain_fractions.items()}
    for domain, target in val_by_domain.items():
        if domain == "logic":
            it = generate_logic_domain(target, args.seed + 99, verify)
        elif domain == "arithmetic":
            it = generate_arithmetic_domain(target, args.seed + 99, verify)
        elif domain == "instruction":
            it = generate_instruction_domain(target, args.seed + 99, verify)
        else:
            it = generate_proof_domain(target, args.seed + 99, verify)
        for text in it:
            val_docs.append(text)

    with open(args.val_output, "w", encoding="utf-8") as f:
        for text in val_docs:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    approx_val = sum(estimate_tokens(t) for t in val_docs)
    print(f"  Written {len(val_docs)} val docs (~{approx_val / 1e6:.1f}M tokens) → {args.val_output}")

    # ── Generate training shards ───────────────────────────────────────────
    print(f"\nGenerating training corpus with {args.workers} workers...")

    worker_args = [
        (
            domain,
            int(training_tokens * frac),
            args.seed + i * 100,
            args.output_dir,
            args.shard_size,
            verify,
        )
        for i, (domain, frac) in enumerate(domain_fractions.items())
    ]

    total_docs = 0
    total_tokens_written = 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker_generate_domain, wa): wa[0]
                       for wa in worker_args}
            for future in as_completed(futures):
                domain, n_docs, n_toks = future.result()
                total_docs += n_docs
                total_tokens_written += n_toks
                print(
                    f"  [{domain}] Done: {n_docs:,} docs, "
                    f"~{n_toks / 1e6:.0f}M tokens",
                    flush=True,
                )
    else:
        for wa in worker_args:
            domain, n_docs, n_toks = _worker_generate_domain(wa)
            total_docs += n_docs
            total_tokens_written += n_toks
            print(
                f"  [{domain}] Done: {n_docs:,} docs, "
                f"~{n_toks / 1e6:.0f}M tokens",
                flush=True,
            )

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Corpus generation complete.")
    print(f"  Training docs:   {total_docs:,}")
    print(f"  Training tokens: ~{total_tokens_written / 1e9:.2f}B")
    print(f"  Shards:          {args.output_dir}/")
    print(f"  Validation:      {args.val_output}")
    print(f"  Val docs:        {len(val_docs):,}")

    # Save generation metadata
    meta = {
        "total_docs": total_docs,
        "approx_tokens_training": total_tokens_written,
        "approx_tokens_val": approx_val,
        "val_docs": len(val_docs),
        "seed": args.seed,
        "domain_fractions": domain_fractions,
        "shard_size_tokens": args.shard_size,
        "verify": verify,
    }
    with open(Path(args.output_dir) / "generation_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata:        {args.output_dir}/generation_metadata.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
